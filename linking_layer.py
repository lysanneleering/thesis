import re
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LINKING_MODEL = Path(__file__).parent / "linking_model"

# Maximum number of environments to look backward for implicit references.
IMPLICIT_WINDOW = 10

# Minimum confidence score required to accept an implicit prediction.
CONFIDENCE_THRESHOLD = 0.50

# Mapping of common abbreviations to canonical environment types.
CANONICAL = {
    "thm":  "theorem",  "thrm": "theorem",
    "lem":  "lemma",    "lmm":  "lemma",
    "def":  "definition", "defn": "definition",
    "prop": "proposition",
    "cor":  "corollary",  "coro": "corollary",
    "pf":   "proof",      "prf":  "proof",
}

ENV_TYPES = {"theorem", "lemma", "definition", "proposition", "corollary", "proof"}

# Pattern to match environment numbers like "2.1", "A.3", "III.4".
# Handles Roman numerals, letters, and decimal numbers.
NUMBER_PATTERN = r"(?:[A-Z]{1,4}\.)*[A-Z0-9]+(?:\.[0-9]+)*"

# Pattern to match environment type keywords.
_KW = (r"(?:theorem|lemma|definition|proposition|corollary|proof|"
       r"thm|lem|def|defn|prop|cor|coro|pf|prf)")

# Matches phrases like "by Theorem 2.1", "from Lemma 3", "using Proposition 4.2".
EXPLICIT_RE = re.compile(
    r"(?:by|from|using|see|applying|via|as in|recall|cf\.?|per)\s+"
    r"(?:the\s+)?"
    r"(?P<env_type>" + _KW + r")"
    r"s?\b\.?"
    r"\s*"
    r"(?P<ref_num>" + NUMBER_PATTERN + r")",
    re.IGNORECASE,
)

# Matches phrases that refer to previous results without explicit numbers.
IMPLICIT_RE = re.compile(
    r"(?:"
    r"the (?:previous|above|preceding|following|latter|former) "
    r"(?:theorem|lemma|definition|proposition|corollary|result|argument|construction)|"
    r"(?:as|which) (?:shown|proved|established|argued|noted|observed) "
    r"(?:above|earlier|previously|before|in the preceding)|"
    r"applying the same (?:reasoning|argument|technique)|"
    r"this (?:follows|is immediate) from the (?:previous|above|preceding)|"
    r"the result (?:above|stated|established)"
    r")",
    re.IGNORECASE,
)


def normalise_type(raw: str) -> str:
    """
    Converts environment type abbreviations to canonical full names.
    """
    key = raw.lower().rstrip(".")
    return CANONICAL.get(key, key)


def normalise_number(num: str) -> str:
    """
    Normalises reference numbers by stripping whitespace and converting to lowercase.
    """
    return num.strip().lower()


def parse_env_id(env: dict) -> tuple:
    """
    Parses an environment ID into type and number components.
    """
    env_id = env.get("id", "")
    parts = env_id.split("_")

    # First part is always the type.
    env_type = parts[0] if parts else env.get("type", "")

    # Remaining parts form the number with underscores replaced by dots.
    number = ".".join(parts[1:]) if len(parts) > 1 else ""

    return normalise_type(env_type), normalise_number(number)


def detect_explicit_references(environments: list) -> list:
    """
    Detects explicit citations using pattern matching.
    """
    type_num_to_env = {}
    for env in environments:
        env_type, number = parse_env_id(env)
        if number:  # Only environments with numbers can be referenced.
            type_num_to_env[(env_type, number)] = env

    edges = []

    for i, env in enumerate(environments):
        text = env.get("text", "")
        normalised_text = re.sub(
            r"\b(" + "|".join(re.escape(k) for k in CANONICAL) + r")s?\b\.?",
            lambda m: CANONICAL.get(m.group(1).lower(), m.group(1)),
            text,
            flags=re.IGNORECASE,
        )

        # Search for explicit reference patterns.
        for match in EXPLICIT_RE.finditer(normalised_text):
            # Extract the referenced environment type and number.
            ref_type = normalise_type(match.group("env_type"))
            ref_num = normalise_number(match.group("ref_num"))

            # Look up the referenced environment.
            target = type_num_to_env.get((ref_type, ref_num))
            if target is None:
                continue  # Reference not found in our environments.

            # Find the index of the target environment.
            target_idx = environments.index(target)

            # Only consider backward references.
            if target_idx >= i:
                continue

            # Create an edge from the referencing to the referenced environment.
            edges.append({
                "source": env["id"],
                "target": target["id"],
                "method": "explicit",
                "confidence": 1.0,
            })

    # Remove duplicate edges.
    return deduplicate_edges(edges)


def load_linking_model(model_dir: str):
    """
    Loads the trained linking classifier model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Linking model loaded on {device}")

    return model, tokenizer, device


def sliding_window_predict(model, tokenizer, env_a: str, env_b: str,
                           max_length=512, stride=128, device="cpu"):
    """
    Classifies whether env_a depends on env_b using sliding window for long texts.
    """
    ids = tokenizer(env_a, env_b, add_special_tokens=True)["input_ids"]

    if len(ids) <= max_length:
        # Single forward pass with standard tokenization.
        batch = tokenizer(
            env_a, env_b,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**batch).logits

        # Convert logits to probabilities using softmax.
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    else:
        window_probs = []

        # Slide the window over the token sequence.
        for start in range(0, len(ids), stride):
            # Extract a chunk of tokens.
            chunk = ids[start: start + max_length]

            # Skip very short chunks.
            if len(chunk) < 4:
                break

            # Pad the chunk to max_length.
            pad_len = max_length - len(chunk)
            padded = chunk + [tokenizer.pad_token_id] * pad_len
            attention = [1] * len(chunk) + [0] * pad_len

            # Create batch and run inference.
            batch = {
                "input_ids": torch.tensor([padded], device=device),
                "attention_mask": torch.tensor([attention], device=device),
            }

            with torch.no_grad():
                logits = model(**batch).logits

            # Get probabilities for this window.
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            window_probs.append(probs)

            # Stop if we've processed the entire sequence.
            if start + max_length >= len(ids):
                break

        # Average probabilities across all windows.
        probs = np.mean(window_probs, axis=0)

    # Get the predicted label and confidence score.
    label = int(np.argmax(probs))
    confidence = float(probs[label])
    return label, confidence


def detect_implicit_references(environments, model, tokenizer, device,
                                explicit_edges, window=IMPLICIT_WINDOW,
                                confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Detects implicit references using the trained classifier.
    """
    # Build set of already-resolved pairs to avoid duplicates.
    resolved = {(e["source"], e["target"]) for e in explicit_edges}
    edges = []
    n = len(environments)

    for i in range(n):
        env_a = environments[i]

        # Check if this environment contains implicit reference phrases.
        has_implicit = bool(IMPLICIT_RE.search(env_a.get("text", "")))

        # Look backward within the window.
        for j in range(max(0, i - window), i):
            env_b = environments[j]
            pair = (env_a["id"], env_b["id"])

            # Skip if already resolved.
            if pair in resolved:
                continue

            # Skip if same type and no implicit phrase.
            same_type = (normalise_type(env_a.get("type", "")) ==
                         normalise_type(env_b.get("type", "")))
            if not has_implicit and same_type:
                continue

            # Run the classifier on this pair.
            label, confidence = sliding_window_predict(
                model, tokenizer,
                env_a.get("text", ""),
                env_b.get("text", ""),
                device=device,
            )

            # Keep positive predictions above the confidence threshold.
            if label == 1 and confidence >= confidence_threshold:
                edges.append({
                    "source": env_a["id"],
                    "target": env_b["id"],
                    "method": "implicit",
                    "confidence": round(confidence, 4),
                })

    return deduplicate_edges(edges)


def deduplicate_edges(edges: list) -> list:
    """
    Removes duplicate edges, keeping the one with highest confidence.
    """
    best = {}
    for edge in edges:
        key = (edge["source"], edge["target"])
        # Keep the edge with highest confidence.
        if key not in best or edge["confidence"] > best[key]["confidence"]:
            best[key] = edge

    return list(best.values())


def merge_edges(explicit: list, implicit: list) -> list:
    """
    Merges explicit and implicit edges, preferring explicit when both exist.
    """
    # Build set of explicit pairs for quick lookup.
    explicit_pairs = {(e["source"], e["target"]) for e in explicit}

    # Filter out implicit edges that duplicate explicit ones.
    filtered_implicit = [
        e for e in implicit
        if (e["source"], e["target"]) not in explicit_pairs
    ]

    return explicit + filtered_implicit


def build_graph(environments: list, edges: list) -> dict:
    """
    Builds a dependency graph from environments and edges.
    """
    # Create nodes from all environments.
    nodes = [
        {
            "id": env.get("id"),
            "type": env.get("type"),
            "text": env.get("text"),
            "page": env.get("page"),
            "number": env.get("number", ""),
            "confidence": env.get("confidence"),
        }
        for env in environments
    ]

    return {"nodes": nodes, "edges": edges}


def link_environments(environments: list) -> dict:
    print("  [Stage 1] Detecting explicit references…")
    explicit_edges = detect_explicit_references(environments)
    print(f"            {len(explicit_edges)} explicit edges found.")

    print("  [Stage 2] Loading linking model…")
    model, tokenizer, device = load_linking_model(str(LINKING_MODEL))
    print(f"            Running implicit detection (window={IMPLICIT_WINDOW})…")
    implicit_edges = detect_implicit_references(
        environments, model, tokenizer, device,
        explicit_edges=explicit_edges,
    )
    print(f"            {len(implicit_edges)} implicit edges found.")

    all_edges = merge_edges(explicit_edges, implicit_edges)

    return build_graph(environments, all_edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a dependency graph from an extraction-layer JSON file"
    )
    parser.add_argument("input", help="Extraction output JSON (from extraction_layer.py)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    environments = data if isinstance(data, list) else data.get("environments", [])
    print(f"Loaded {len(environments)} environments from {args.input}")

    graph = link_environments(environments)

    stem = Path(args.input).stem.removesuffix("_environments")
    output = stem + "_graph.json"

    with open(output, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"Graph written to: {output}")
