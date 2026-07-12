import json
import re
import random
import tarfile
from pathlib import Path
from collections import defaultdict, Counter

RAW_DIR     = Path("data/raw")
OUTPUT_DIR  = Path("data")

EXTRACT_TRAIN_FILE = OUTPUT_DIR / "extraction_train_dataset.json"
EXTRACT_TEST_FILE  = OUTPUT_DIR / "extraction_test_dataset.json"

LINKING_TRAIN_FILE = OUTPUT_DIR / "linking_train_dataset.json"
LINKING_TEST_FILE  = OUTPUT_DIR / "linking_test_dataset.json"

# Implicit reference phrases searched for inside LaTeX proof bodies.
IMPLICIT_PATTERNS = [
    r"the previous (theorem|lemma|proposition|corollary|definition|result)",
    r"the above (theorem|lemma|proposition|corollary|definition|result)",
    r"the (theorem|lemma|proposition|corollary|definition|result) above",
    r"the (theorem|lemma|proposition|corollary|definition|result) below",
    r"the preceding (theorem|lemma|proposition|corollary|definition|result)",
    r"by the (previous|above|preceding) result",
    r"from the (previous|above|preceding) (argument|construction|observation)",
    r"applying the same (reasoning|argument|technique)",
    r"as (shown|proved|established|argued|noted|observed) (above|earlier|previously|before)",
    r"the result (above|stated|established) (above|earlier|previously)",
    r"this (follows|is immediate) from (the )?(previous|above|preceding)",
]
IMPLICIT_RE = re.compile("|".join(IMPLICIT_PATTERNS), re.IGNORECASE)

# Explicit \ref{label} and \label{name} patterns inside LaTeX.
LATEX_REF_RE = re.compile(r"\\ref\{([^}]+)\}")
LATEX_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")

ENV_TYPES = {"definition", "theorem", "proposition", "lemma", "corollary", "conjecture", "proof"}


def read_tex_from_tar(tarball_path: Path) -> str:
    """
    Extracts and concatenates the text of all .tex files in a tarball.
    """
    tex_contents = []
    try:
        with tarfile.open(tarball_path, "r:*") as tar:
            # Iterate through all files in the archive.
            for member in tar.getmembers():
                # Only process .tex files.
                if member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        tex_contents.append(f.read().decode(errors="ignore"))
    except Exception:
        pass
    return "\n".join(tex_contents)


def extract_latex_environments(latex_source: str) -> list[dict]:
    """
    Pulls every theorem-like environment out of a LaTeX source string.
    """
    envs = []
    order = 0

    # Process each environment type separately.
    for env_type in ENV_TYPES:
        # Match \begin{type}...\end{type}, optional * variant included.
        pattern = re.compile(
            r"\\begin\{" + re.escape(env_type) + r"\*?\}(.*?)\\end\{" + re.escape(env_type) + r"\*?\}",
            re.DOTALL | re.IGNORECASE,
        )

        # Find all instances of this environment type.
        for m in pattern.finditer(latex_source):
            body = m.group(1)  # The content inside the environment.

            # Check if the body contains a \label{} command.
            label_match = LATEX_LABEL_RE.search(body)
            label = label_match.group(1) if label_match else None

            # Store environment information.
            envs.append({
                "type": env_type.lower(),
                "label": label,
                "body": body.strip(),
                "start": m.start(),  # character offset used for relative sorting.
                "order": order,  # will be reassigned after sorting.
            })
            order += 1

    # Sort by actual visual appearance order in the source code.
    envs.sort(key=lambda e: e["start"])

    # Reassign sequential order based on sorted position/
    for i, e in enumerate(envs):
        e["order"] = i

    return envs


def build_label_to_order(latex_envs: list[dict]) -> dict[str, int]:
    """
    Maps LaTeX \label string keys to their sequential document order index.
    """
    return {e["label"]: e["order"] for e in latex_envs if e["label"]}


def mine_explicit_positives(latex_envs: list[dict], label_to_order: dict) -> list[tuple[int, int]]:
    """
    Finds explicit backward citations targeting labeled theorem structures.
    """
    positives = []

    # Check each environment's body for \ref{} commands.
    for env in latex_envs:
        refs = LATEX_REF_RE.findall(env["body"])
        for ref_label in refs:
            # Check if this label references a known environment.
            if ref_label in label_to_order:
                target_order = label_to_order[ref_label]
                # Only consider backward references.
                if target_order < env["order"]:
                    positives.append((env["order"], target_order))

    # Remove duplicates.
    return list(set(positives))


def mine_implicit_positives(latex_envs: list[dict], window: int = 5) -> list[tuple[int, int]]:
    """
    Uses keyword heuristic patterns to match contextually linked neighboring environments.
    """
    positives = []

    # Check each environment's body for implicit reference patterns.
    for env in latex_envs:
        if IMPLICIT_RE.search(env["body"]):
            i = env["order"]
            # Look at previous environments within the window.
            for j in range(max(0, i - window), i):
                positives.append((i, j))

    return list(set(positives))


def sample_negatives(
    latex_envs: list[dict],
    positives: set[tuple[int, int]],
    hard_window: int = 10,
    n_easy: int = 30,
) -> list[tuple[int, int]]:
    """
    Generates negative structural links using distance-aware sampling constraints.
    """
    n = len(latex_envs)
    negatives = []

    # Hard negatives: pairs that appear near each other but have no citation.
    for i in range(n):
        for j in range(max(0, i - hard_window), i):
            if (i, j) not in positives:
                negatives.append((i, j))

    # Easy negatives: pairs that are far apart and unlikely to be related.
    candidates = [
        (i, j)
        for i in range(n)
        for j in range(0, max(0, i - hard_window))
        if (i, j) not in positives
    ]
    random.shuffle(candidates)
    negatives.extend(candidates[:n_easy])

    return negatives


def build_dataset_for_paper(
    paper_id: str,
    latex_source: str,
    extracted_envs: list[dict],
) -> list[dict]:
    """
    Extracts structural logical links for one paper document and maps the pairs
    back into text spans extracted by the upstream pipeline layer.
    """
    # Extract environments from LaTeX source.
    latex_envs = extract_latex_environments(latex_source)
    if not latex_envs:
        return []

    # Build mapping from labels to document order.
    label_to_order = build_label_to_order(latex_envs)

    # Mine positive examples (actual dependencies).
    explicit_pos = mine_explicit_positives(latex_envs, label_to_order)
    implicit_pos = mine_implicit_positives(latex_envs)
    all_positives = set(explicit_pos + implicit_pos)

    # Sample negative examples (non-dependencies).
    negatives = sample_negatives(latex_envs, all_positives)

    def get_text(order: int, env_type: str) -> str | None:
        """
        Finds text extracted from the PDF matching the target category and order context.
        """
        # Filter extracted environments by type.
        same_type = [e for e in extracted_envs if e.get("label", "").lower() == env_type]

        if not same_type:
            # Fallback: use order as index.
            if order < len(extracted_envs):
                return extracted_envs[order]["text"]
            return None

        # Select the best match based on proximity in order.
        idx = min(range(len(same_type)), key=lambda k: abs(k - order))
        return same_type[idx]["text"]

    records = []

    # Process positive relationships.
    for (i, j) in all_positives:
        env_i = latex_envs[i]
        env_j = latex_envs[j]
        text_i = get_text(i, env_i["type"])
        text_j = get_text(j, env_j["type"])

        # Only add if we have text for both environments.
        if text_i and text_j:
            # Determine if this is implicit or explicit.
            pair_type = "implicit" if (i, j) in set(implicit_pos) else "explicit"
            records.append({
                "env_a": text_i,  # Dependent environment.
                "env_b": text_j,  # Referenced environment.
                "label": 1,  # Positive: dependency exists.
                "pair_type": pair_type,
                "paper_id": paper_id,
            })

    # Process negative relationships.
    for (i, j) in negatives:
        env_i = latex_envs[i]
        env_j = latex_envs[j]
        text_i = get_text(i, env_i["type"])
        text_j = get_text(j, env_j["type"])

        # Only add if we have text for both environments.
        if text_i and text_j:
            # Determine if hard or easy negative.
            pair_type = "hard_negative" if abs(i - j) <= 10 else "easy_negative"
            records.append({
                "env_a": text_i,  # Dependent environment.
                "env_b": text_j,  # Referenced environment.
                "label": 0,  # Negative: no dependency.
                "pair_type": pair_type,
                "paper_id": paper_id,
            })

    return records


def process_split(extract_file: Path, output_file: Path, split_name: str) -> None:
    """
    Processes all papers contained within a specific evaluation split dataset.
    """
    if not extract_file.exists():
        print(f"Error: Base extraction file '{extract_file}' missing. Run extraction script first.")
        return

    print(f"\nProcessing {split_name.upper()} data from {extract_file}...")

    # Load extraction layer records.
    with open(extract_file, "r", encoding="utf-8") as f:
        extraction_records = json.load(f)

    # Group flat extraction samples back into structural paper contexts.
    papers_dict = defaultdict(list)
    for record in extraction_records:
        # Ignore structural 'other' negatives because they represent non-environment blocks.
        if record.get("label") != "other":
            papers_dict[record["paper_id"]].append(record)

    linking_dataset = []
    total_papers = len(papers_dict)

    # Process each paper individually.
    for idx, (paper_id, extracted_envs) in enumerate(papers_dict.items(), 1):
        # Find the LaTeX source tarball for this paper.
        tar_path = RAW_DIR / f"{paper_id}.tar.gz"
        if not tar_path.exists():
            continue

        # Read the LaTeX source.
        latex_source = read_tex_from_tar(tar_path)
        if not latex_source:
            continue

        # Build linking pairs for this paper.
        records = build_dataset_for_paper(paper_id, latex_source, extracted_envs)

        # Tag each record with its split.
        for r in records:
            r["split"] = split_name
        linking_dataset.extend(records)

        # Print progress.
        if idx % 20 == 0 or idx == total_papers:
            print(f"  [{idx}/{total_papers}] papers structured... Generated {len(linking_dataset)} pairs.")

    # Save the linking dataset.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(linking_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved completed split data: {output_file}")

    # Print metrics distribution summary.
    labels = [r["label"] for r in linking_dataset]
    types = [r["pair_type"] for r in linking_dataset]
    print(f"  Summary metrics: {dict(Counter(labels))}")
    print(f"  Subtype distribution: {dict(Counter(types))}")


if __name__ == "__main__":
    # Set random seed for reproducibility.
    random.seed(42)

    print("=" * 60)
    print("Building Linking Layer Structural Reference Dataset")
    print("=" * 60)

    process_split(EXTRACT_TRAIN_FILE, LINKING_TRAIN_FILE, "train")
    process_split(EXTRACT_TEST_FILE, LINKING_TEST_FILE, "test")

    print("\nLinking Layer processing complete.")
