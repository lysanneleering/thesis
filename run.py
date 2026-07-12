import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

EXTRACTION_MODEL = Path("./extraction_model")
LINKING_MODEL = Path("./linking_model")
OUTPUT_FOLDER = Path("./output")


def generate_html(graph: dict, html_output: Path):
    """
    Creates a standalone HTML visualisation by inserting the graph data
    into the HTML template.
    """

    # Make sure the HTML template exists.
    template_path = Path("./index.html")
    if not template_path.exists():
        print(f"[ERROR] Template not found at {template_path}")
        print("        Make sure index.html is in the current directory.")
        sys.exit(1)

    # Read the HTML template into memory and find the injection marker.
    template = template_path.read_text(encoding="utf-8")
    marker = "// __GRAPH_DATA_PLACEHOLDER__"
    if marker not in template:
        print("[ERROR] index.html is missing the injection marker.")
        sys.exit(1)

    # Replace the marker with a JavaScript variable containing the graph data.
    html_output.write_text(
        template.replace(
            marker,
            f"var graphData = {json.dumps(graph, indent=2)};"
        ),
        encoding="utf-8",
    )

    print(f"  Visualisation: {html_output}")


if __name__ == "__main__":
    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Input PDF")
    pdf = Path(parser.parse_args().pdf).resolve()
    stem = pdf.stem

    # Create the output directory if it doesn't already exist.
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Run the extraction layer.
    print("[1/3] Running extraction layer...")
    from extraction_layer import detect_environments
    raw_envs = detect_environments(
        str(pdf),
        str(EXTRACTION_MODEL),
    )

    # The extraction layer stores the environment type under the key "label",
    # but the linking layer expects it to be called "type", so we rename the
    # field during conversion to a normal dictionary.
    environments = [
        {**asdict(e), "type": asdict(e).pop("label")}
        for e in raw_envs
    ]

    print(f"  {len(environments)} environments detected.")

    # Run the linking layer.
    print("\n[2/3] Running linking layer...")

    from linking_layer import link_environments

    graph = link_environments(environments)

    # Count how many dependencies were found explicitly.
    n_exp = sum(
        1
        for edge in graph["edges"]
        if edge["method"] == "explicit"
    )

    print(
        f"  {len(graph['edges'])} edges "
        f"({n_exp} explicit, "
        f"{len(graph['edges']) - n_exp} implicit)"
    )

    # Save the results.
    print("\n[3/3] Saving outputs...")

    graph_path = OUTPUT_FOLDER / f"{stem}_graph.json"
    html_path = OUTPUT_FOLDER / f"{stem}_graph.html"

    # Save the graph as formatted JSON.
    graph_path.write_text(
        json.dumps(graph, indent=2),
        encoding="utf-8",
    )

    print(f"  Graph JSON:    {graph_path}")

    # Generate the interactive HTML visualization.
    generate_html(graph, html_path)

    print("\n✓ Done!")
    print(
        f"  Open {html_path} in a browser to explore the dependency graph."
    )
