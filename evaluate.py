import re
import json
import argparse
import tarfile
from pathlib import Path
from collections import defaultdict

DEFAULT_EXTRACTION_MODEL_DIR = Path("extraction_model")
DEFAULT_LINKING_MODEL_DIR = Path("linking_model")
DEFAULT_DATA_DIR = Path("data")


def data_paths(data_dir: Path) -> dict:
    """
    Returns a dictionary of all expected data file paths.
    """
    return {
        "manifest":      data_dir / "split_manifest.json",
        "test_extract":  data_dir / "extraction_test_dataset.json",
        "test_linking":  data_dir / "linking_test_dataset.json",
        "pdfs":          data_dir / "pdfs",
        "raw":           data_dir / "raw",
    }


def pdf_path_for(paper_id: str, pdfs_dir: Path) -> Path:
    """
    Resolves the on-disk path for a cached PDF given its arXiv paper_id.
    """
    # Replace slashes with underscores to handle old-style arXiv IDs.
    safe_id   = paper_id.replace("/", "_")

    # Try the versioned filename first.
    candidate = pdfs_dir / f"{safe_id}.pdf"
    if candidate.exists():
        return candidate

    # If not found, strip version suffix and try again.
    bare_id   = re.sub(r"v\d+$", "", safe_id)
    candidate = pdfs_dir / f"{bare_id}.pdf"
    if candidate.exists():
        return candidate

    return pdfs_dir / f"{safe_id}.pdf"


def tar_path_for(paper_id: str, raw_dir: Path) -> Path:
    """
    Same normalisation logic as pdf_path_for, but for .tar.gz files.
    """
    safe_id   = paper_id.replace("/", "_")
    candidate = raw_dir / f"{safe_id}.tar.gz"
    if candidate.exists():
        return candidate
    bare_id   = re.sub(r"v\d+$", "", safe_id)
    candidate = raw_dir / f"{bare_id}.tar.gz"
    if candidate.exists():
        return candidate
    return raw_dir / f"{safe_id}.tar.gz"


ENV_NAMES  = {"theorem", "lemma", "proof", "definition",
              "proposition", "corollary", "conjecture"}

# Regex to extract LaTeX environments from .tex files.
TEX_ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(ENV_NAMES) + r")\*?\}"
    r"(?:\[[^\]]*\])?"
    r"(.*?)"
    r"\\end\{\1\*?\}",
    re.DOTALL | re.IGNORECASE,
)

# Regex to find LaTeX labels.
LABEL_RE   = re.compile(r"\\label\{([^}]+)\}")

# Regex to find LaTeX references.
REF_RE     = re.compile(r"\\ref\{([^}]+)\}")


def read_tex_from_tar(tar_path: Path) -> str:
    """
    Extracts and concatenates all .tex files from a LaTeX tarball.
    """
    parts = []
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            # Iterate through all files in the archive.
            for member in tar.getmembers():
                # Only process .tex files.
                if member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        parts.append(f.read().decode(errors="ignore"))
    except Exception:
        pass
    return "\n".join(parts)


JACCARD_THRESHOLD = 0.45

# Common stopwords to remove before computing Jaccard similarity. These words
# do not carry any meaningful content for matching purposes.
STOP_TOKENS = {
    "the", "a", "an", "of", "and", "in", "is", "to", "that", "we", "let",
    "for", "with", "by", "on", "be", "it", "if", "then", "are", "have", "this",
    "from", "as", "at", "or", "not", "so", "can", "its", "which", "all", "any",
    "such", "each"
}


def tokenize(text: str) -> set:
    """
    Converts text to a set of meaningful tokens (lowercase alphanumeric only).
    """
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_TOKENS


def jaccard(a: set, b: set) -> float:
    """
    Calculates the Jaccard similarity between two token sets.
    """
    if not a and not b:
        return 1.0
    inter = len(a & b)  # Number of tokens in both sets.
    union = len(a | b)  # Number of tokens in either set.
    return inter / union if union > 0 else 0.0


def match_environments(predicted: list, ground_truth: list):
    """
    Greedy 1-to-1 matching between predicted and ground-truth environments.
    """
    scores = []

    # Compute similarity scores for all pairs.
    for pi, pred in enumerate(predicted):
        pred_tokens = tokenize(pred.get("text", ""))
        pred_type   = pred.get("type", "")

        for gi, gt in enumerate(ground_truth):
            # Only compare environments of the same type.
            if pred_type != gt.get("label", ""):
                continue

            # Calculate the Jaccard similarity.
            j = jaccard(pred_tokens, tokenize(gt.get("text", "")))

            # Only keep pairs that meet the threshold.
            if j >= JACCARD_THRESHOLD:
                scores.append((j, pi, gi))

    # Sort by similarity (highest first) for greedy matching.
    scores.sort(reverse=True)

    # Greedy assignment: each pred and gt can be used at most once.
    used_pred, used_gt, matched = set(), set(), []
    for _, pi, gi in scores:
        if pi not in used_pred and gi not in used_gt:
            matched.append((pi, gi))
            used_pred.add(pi)
            used_gt.add(gi)

    # Find unmatched indices.
    unmatched_pred = set(range(len(predicted))) - used_pred
    unmatched_gt   = set(range(len(ground_truth))) - used_gt

    return matched, unmatched_pred, unmatched_gt


def prf(tp: int, fp: int, fn: int) -> tuple:
    """
    Calculates Precision, Recall, and F1 score from confusion matrix counts.
    """
    # Precision: of all positive predictions, how many were correct?
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall: of all actual positives, how many did we find?
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1: harmonic mean of precision and recall
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return round(p, 4), round(r, 4), round(f1, 4)


def macro_f1(per_class: dict) -> float:
    """
    Calculates the macro-averaged F1 score.
    """
    scores = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def evaluate_extraction(model_dir: str, p: dict) -> dict:
    """
    Evaluates the extraction layer's performance on the test set.
    """
    from extraction_layer import detect_environments

    # Check that the test dataset exists.
    if not p["test_extract"].exists():
        raise FileNotFoundError(
            f"test_dataset.json not found at {p['test_extract']}.\n"
            "Run create_extraction_dataset.py first."
        )

    # Load ground truth data.
    with open(p["test_extract"]) as f:
        all_records = json.load(f)

    # Only evaluate on the positive classes (theorem, lemma, etc.).
    gt_by_paper: dict[str, list] = defaultdict(list)
    for rec in all_records:
        if rec.get("label") != "other":
            gt_by_paper[rec["paper_id"]].append(rec)

    paper_ids = sorted(gt_by_paper.keys())
    print(f"\nEvaluating extraction on {len(paper_ids)} test papers…\n")

    tp_per_class: dict[str, int] = defaultdict(int)
    fp_per_class: dict[str, int] = defaultdict(int)
    fn_per_class: dict[str, int] = defaultdict(int)

    paper_results = []
    skipped = 0

    for paper_id in paper_ids:
        # Find the cached PDF file for this paper.
        pdf = pdf_path_for(paper_id, p["pdfs"])
        if not pdf.exists():
            print(f"  [SKIP] PDF not found: {pdf}")
            skipped += 1
            continue

        # Get the ground truth for this paper.
        gt_records = gt_by_paper[paper_id]

        try:
            # Run the extraction layer on the PDF.
            predicted = detect_environments(str(pdf), model_dir)
        except Exception as e:
            print(f"  [ERROR] {paper_id}: {e}")
            skipped += 1
            continue

        # Convert predictions to a simpler dict format for matching.
        pred_dicts = [{"type": e.label, "text": e.text} for e in predicted]

        # Match predictions to ground truth using Jaccard similarity.
        matched, unmatched_pred, unmatched_gt = match_environments(
            pred_dicts, gt_records
        )

        for pi, gi in matched:
            tp_per_class[pred_dicts[pi]["type"]] += 1  # True positive.
        for pi in unmatched_pred:
            fp_per_class[pred_dicts[pi]["type"]] += 1  # False positive.
        for gi in unmatched_gt:
            fn_per_class[gt_records[gi]["label"]] += 1  # False negative.

        # Store per-paper statistics.
        tp = len(matched)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        paper_results.append({
            "paper": paper_id, "pdf": str(pdf),
            "gt": len(gt_records), "pred": len(pred_dicts),
            "tp": tp, "fp": fp, "fn": fn,
        })

        # Print progress.
        print(f"  {paper_id:40s}  gt={len(gt_records):3d}  pred={len(pred_dicts):3d}  "
              f"tp={tp:3d}  fp={fp:3d}  fn={fn:3d}")

    if skipped:
        print(f"\n  ({skipped} papers skipped — PDF not found or extraction error)")

    # Compute per-class metrics.
    all_types = set(tp_per_class) | set(fp_per_class) | set(fn_per_class)
    per_class = {}
    for t in sorted(all_types):
        tp = tp_per_class[t]
        fp = fp_per_class[t]
        fn = fn_per_class[t]
        pr, rc, f1 = prf(tp, fp, fn)
        per_class[t] = {
            "precision": pr, "recall": rc, "f1": f1,
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Compute overall metrics.
    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())
    overall_p, overall_r, overall_f = prf(total_tp, total_fp, total_fn)

    return {
        "per_class":     per_class,
        "macro_f1":      macro_f1(per_class),
        "micro":         {"precision": overall_p, "recall": overall_r, "f1": overall_f},
        "paper_results": paper_results,
    }


def evaluate_linking(linking_model_dir: str, p: dict) -> dict:
    """
    Evaluates the linking layer's performance on the test set.
    """
    from linking_layer import (
        EXPLICIT_RE,
        load_linking_model,
        sliding_window_predict,
    )

    # Check that the test dataset exists.
    if not p["test_linking"].exists():
        raise FileNotFoundError(f"Missing {p['test_linking']}")

    # Load ground truth pairs.
    with open(p["test_linking"]) as f:
        dataset = json.load(f)

    total = len(dataset)

    print("\nEvaluating linking on test pairs…\n")

    # Load the linking model.
    print("Loading linking model...")
    model, tokenizer, device = load_linking_model(linking_model_dir)
    print(f"Model loaded on {device}\n")

    # Initialize counters for confusion matrix.
    tp_all = fp_all = fn_all = 0
    tp_exp = fp_exp = fn_exp = 0
    tp_imp = fp_imp = fn_imp = 0

    # Per-paper statistics.
    paper_stats = defaultdict(
        lambda: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
    )

    # Process each test pair.
    for idx, rec in enumerate(dataset):
        env_a = rec["env_a"]
        env_b = rec["env_b"]
        gt = int(rec["label"])
        pair_type = rec["pair_type"]
        paper = rec["paper_id"]

        pred = 0
        method = "none"

        # Strategy 1: Check for explicit citation in env_a.
        if EXPLICIT_RE.search(env_a):
            pred = 1
            method = "explicit"

        # Strategy 2: Use the trained classifier for implicit dependencies.
        elif model is not None:
            pred, _ = sliding_window_predict(
                model,
                tokenizer,
                env_a,
                env_b,
                device=device,
            )
            if pred == 1:
                method = "implicit"

        # Update per-paper stats.
        if gt:
            paper_stats[paper]["gt"] += 1
        if pred:
            paper_stats[paper]["pred"] += 1

        # Update confusion matrix.
        if pred == 1 and gt == 1:
            tp_all += 1
            paper_stats[paper]["tp"] += 1
            if pair_type == "explicit":
                tp_exp += 1
            else:
                tp_imp += 1

        elif pred == 1 and gt == 0:
            fp_all += 1
            paper_stats[paper]["fp"] += 1
            if method == "explicit":
                fp_exp += 1
            else:
                fp_imp += 1

        elif pred == 0 and gt == 1:
            fn_all += 1
            paper_stats[paper]["fn"] += 1
            if pair_type == "explicit":
                fn_exp += 1
            else:
                fn_imp += 1

        # Print progress every 100 pairs or at the end.
        if idx % 100 == 0 or idx == total - 1:
            progress = idx / total * 100
            print(
                f"  [{idx:6d}/{total}] "
                f"{progress:5.1f}% | "
                f"TP={tp_all:4d} FP={fp_all:4d} FN={fn_all:4d}"
            )

    print("\nFinished evaluation\n")

    # Compute metrics for each category.
    overall = prf(tp_all, fp_all, fn_all)
    explicit = prf(tp_exp, fp_exp, fn_exp)
    implicit = prf(tp_imp, fp_imp, fn_imp)

    # Return results in the same format as extraction evaluation.
    return {
        "overall": {
            "precision": overall[0],
            "recall": overall[1],
            "f1": overall[2],
        },
        "explicit": {
            "precision": explicit[0],
            "recall": explicit[1],
            "f1": explicit[2],
        },
        "implicit": {
            "precision": implicit[0],
            "recall": implicit[1],
            "f1": implicit[2],
        },
        "paper_results": [
            {"paper": k, **v}
            for k, v in paper_stats.items()
        ],
    }


def print_extraction_report(results: dict):
    """
    Prints formatted extraction evaluation results.
    """
    print("\n" + "=" * 60)
    print("EXTRACTION LAYER RESULTS")
    print("=" * 60)
    print(f"\n{'Class':<16} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print("-" * 46)

    # Print each class's metrics.
    for cls, m in results["per_class"].items():
        print(f"{cls:<16} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8d}")

    # Print micro-averaged metrics.
    print("-" * 46)
    mi = results["micro"]
    print(f"{'micro avg':<16} {mi['precision']:>6.3f} {mi['recall']:>6.3f} "
          f"{mi['f1']:>6.3f}")
    print(f"\nMacro F1: {results['macro_f1']:.3f}")


def print_linking_report(results: dict):
    """
    Prints formatted linking evaluation results.
    """
    print("\n" + "=" * 60)
    print("LINKING LAYER RESULTS")
    print("=" * 60)
    print(f"\n{'Category':<16} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 36)

    # Print results for each category.
    for category in ("overall", "explicit", "implicit"):
        m = results[category]
        print(f"{category:<16} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")

    # Print per-paper summary.
    print("\n" + "-" * 36)
    print("\nPer-paper summary:")
    print(f"{'Paper ID':<40} {'GT':>4} {'Pred':>4} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 66)

    for paper_result in results.get("paper_results", []):
        print(
            f"{paper_result['paper']:<40} "
            f"{paper_result['gt']:>4d} "
            f"{paper_result['pred']:>4d} "
            f"{paper_result['tp']:>4d} "
            f"{paper_result['fp']:>4d} "
            f"{paper_result['fn']:>4d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate extraction and linking layers"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help=f"Root data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Path to save results JSON (default: evaluation_results.json)")
    args = parser.parse_args()

    p = data_paths(Path(args.data_dir))
    all_results = {}

    # Run extraction evaluation using default model path.
    print("\n" + "=" * 60)
    print("EXTRACTION EVALUATION")
    print("=" * 60)
    ext_results = evaluate_extraction(str(DEFAULT_EXTRACTION_MODEL_DIR), p)
    print_extraction_report(ext_results)
    all_results["extraction"] = ext_results

    # Run linking evaluation using default model path.
    print("\n" + "=" * 60)
    print("LINKING EVALUATION")
    print("=" * 60)
    lnk_results = evaluate_linking(str(DEFAULT_LINKING_MODEL_DIR), p)
    print_linking_report(lnk_results)
    all_results["linking"] = lnk_results

    # Save full results as JSON for further analysis.
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {args.output}")
