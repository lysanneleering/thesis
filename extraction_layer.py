import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import fitz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EXTRACTION_MODEL = Path(__file__).parent / "extraction_model"

# Set of theorem-like environment types that this tool recognizes.
# Keys and values are identical here. The dictionary just doubles as an ordered
# set that is easy to interpolate into regex patterns below.
ENVIRONMENTS = {
    "definition":  "definition",
    "theorem":     "theorem",
    "proposition": "proposition",
    "lemma":       "lemma",
    "corollary":   "corollary",
    "conjecture":  "conjecture",
    "proof":       "proof",
}

ENV_PATTERN = "|".join(ENVIRONMENTS)

# Matches a line that starts with an environment keyword.
DECLARATION_PATTERN = re.compile(
    rf"""
    ^
    \s*
    ({ENV_PATTERN})        # the environment keyword itself (captured)
    \s*
    (
        \d+(\.\d+)*         # optional numbering like "2" or "2.1.3"
    )?
    [\.:]?                  # optional trailing period or colon
    """,
    re.I | re.X,
)

# Matches the alternate "number-first" header style used in some papers.
NUMBERED_HEADER_PATTERN = re.compile(
    rf"""
    ^
    \s*
    (\d+(\.\d+)*)           # leading number, e.g. "2.1"
    \s+
    ({ENV_PATTERN})         # followed by the environment keyword
    """,
    re.I | re.X,
)

CONFIDENCE_THRESHOLD = 0.50

MAX_MERGE_BLOCKS = 25

@dataclass
class PDFBlock:
    raw_text: str  # original text as it appeared on the page.
    norm_text: str  # normalized version of raw_text.
    page: int  # page number the block came from.
    x0: float  # left edge of the block's bounding box.
    y0: float  # top edge of the block's bounding box.
    x1: float  # right edge of the block's bounding box.
    y1: float  # bottom edge of the block's bounding box.
    bold_ratio: float  # fraction of characters in the block that are bold.
    avg_font_size: float  # average font size across all spans in the block.
    is_bold: bool  # True if bold_ratio >= 0.5

@dataclass
class Environment:
    id: str  # e.g. "theorem_2_1" or "proof_3."
    label: str  # environment type, e.g. "theorem", "proof."
    text: str  # the full merged text of the environment.
    page: int  # page number where the environment starts.
    confidence: float  # classifier confidence for the assigned label.
    number: str  # e.g. "2.1" or "" for unnumbered proofs.


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_number_from_header(text: str) -> str:
    """
    Pulls '2.1' out of 'Theorem 2.1' or '2.1 Theorem'.
    """
    # Try the "keyword-first" pattern first.
    m = DECLARATION_PATTERN.match(text)
    if m and m.group(2):
        return m.group(2).strip()
    # If that does not work, fall back to the "number-first" pattern.
    m2 = NUMBERED_HEADER_PATTERN.match(text)
    if m2 and m2.group(1):
        return m2.group(1).strip()
    # Return "" if no number is found at all.
    return ""


def make_env_id(label: str, number: str, counters: dict) -> str:
    # Numbered environments get a deterministic id from their own number.
    if number:
        num_part = number.replace(".", "_")
        return f"{label}_{num_part}"
    # Unnumbered environments get the next available sequential index for that
    # label.
    else:
        counters[label] += 1
        return f"{label}_{counters[label]}"


def compute_block_stats(block):
    total_chars = 0
    bold_chars  = 0
    font_sizes  = []
    texts       = []

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            txt = span.get("text", "")
            texts.append(txt)
            n = len(txt)
            total_chars += n
            font_sizes.append(span.get("size", 0))
            # Bit 4 of the span "flags" field indicates bold text in PyMuPDF's
            # font-flag bitmask.
            if span.get("flags", 0) & (1 << 4):
                bold_chars += n

    raw_text   = " ".join(texts).strip()
    bold_ratio = bold_chars / total_chars if total_chars > 0 else 0
    avg_font   = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    return raw_text, bold_ratio, avg_font


def extract_blocks(pdf_path: str):
    doc    = fitz.open(pdf_path)
    blocks = []

    for page_num, page in enumerate(doc, start=1):
        # "dict" mode gives structured access to blocks/lines/spans,
        # including bounding boxes and font info (needed for header scoring).
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            # type 0 = text block. Other types (e.g. 1 = image) are skipped.
            if block.get("type") != 0:
                continue
            raw_text, bold_ratio, avg_font = compute_block_stats(block)
            # Skip near-empty blocks.
            if len(raw_text.strip()) < 5:
                continue
            x0, y0, x1, y1 = block["bbox"]
            blocks.append(PDFBlock(
                raw_text=raw_text,
                norm_text=normalize(raw_text),
                page=page_num,
                x0=x0, y0=y0, x1=x1, y1=y1,
                bold_ratio=bold_ratio,
                avg_font_size=avg_font,
                is_bold=(bold_ratio >= 0.5),
            ))

    doc.close()

    def sort_key(b):
        # Approximate multi-column reading order.
        column = round(b.x0 / 300)
        return (b.page, column, b.y0)

    blocks.sort(key=sort_key)
    return blocks


def header_score(block: PDFBlock) -> int:
    """
    Heuristic score estimating how likely a block is to be the header line of a
    theorem-like environment.
    """
    score = 0
    text  = block.raw_text
    if DECLARATION_PATTERN.search(text):
        score += 5   # strongest signal: text matches "Theorem 2.1." style.
    if NUMBERED_HEADER_PATTERN.search(text):
        score += 4   # also strong: matches "2.1 Theorem" style.
    if block.is_bold:
        score += 2   # headers are often bolded.
    if len(text.split()) <= 12:
        score += 1   # headers tend to be short lines, not full paragraphs.
    if block.avg_font_size >= 11:
        score += 1   # headers are sometimes set in a slightly larger font.
    return score


def looks_like_environment_start(block: PDFBlock) -> bool:
    """
    A block is treated as the start of a candidate environment if its
    header_score clears this threshold.
    """
    return header_score(block) >= 4


def should_stop_merge(block: PDFBlock) -> bool:
    """
    While merging the body text that follows an environment header,
    stop as soon as we hit another block that itself looks like a new
    environment declaration.
    """
    text = block.raw_text
    return bool(
        DECLARATION_PATTERN.search(text)
        or NUMBERED_HEADER_PATTERN.search(text)
    )


def extract_candidates(pdf_path: str):
    blocks     = extract_blocks(pdf_path)
    candidates = []
    i = 0

    while i < len(blocks):
        block = blocks[i]
        # Skip blocks that don't look like an environment header at all.
        if not looks_like_environment_start(block):
            i += 1
            continue

        # Start a new candidate from this header block.
        merged_blocks = [block.raw_text]
        start_page    = block.page
        j             = i + 1
        merge_count   = 0

        # Greedily absorb following blocks into this candidate until we hit a
        # stopping condition.
        while j < len(blocks) and merge_count < MAX_MERGE_BLOCKS:
            nxt = blocks[j]
            # Stop if the next block looks like the start of a new environment.
            if should_stop_merge(nxt):
                break
            # Stop if there's a large vertical gap to the previous block.
            vertical_gap = nxt.y0 - blocks[j - 1].y1
            if vertical_gap > 80:
                break
            merged_blocks.append(nxt.raw_text)
            merge_count += 1
            j += 1

        merged_text = "\n".join(merged_blocks)
        number      = extract_number_from_header(block.raw_text)
        candidates.append((merged_text, start_page, number))
        # Resume scanning right after the last block we merged in, so blocks
        # are not considered as candidates twice.
        i = j

    return candidates


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] Extraction model loaded on {device}")
    return tokenizer, model, device


def predict_text(text, tokenizer, model, device):
    """
    Classifies a (possibly long) candidate text.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        stride=128,  # overlap between windows, to avoid splitting key phrases at a boundary.
        return_overflowing_tokens=True,  # produce multiple windows if text is long.
        padding=True,
        return_tensors="pt",
    )

    encoding = {
        k: v.to(device)
        for k, v in encoding.items()
        if k != "overflow_to_sample_mapping"
    }
    with torch.no_grad():
        logits = model(**encoding).logits
    probs      = torch.softmax(logits, dim=-1)
    mean_probs = probs.mean(dim=0)
    confidence, pred_id = mean_probs.max(dim=0)
    return pred_id.item(), confidence.item()


def classify_candidates(candidates, tokenizer, model, device):
    """
    Runs the classifier over every candidate, drops anything predicted as
    "other" or below the confidence threshold, and assigns each surviving
    candidate a stable id via make_env_id().
    """
    id2label = model.config.id2label
    results  = []
    counters = defaultdict(int)

    for text, page, number in candidates:
        pred_id, confidence = predict_text(text, tokenizer, model, device)
        label = id2label[pred_id]

        if label == "other":
            continue
        # Low-confidence predictions are treated as too unreliable to keep.
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        env_id = make_env_id(label, number, counters)
        results.append(Environment(
            id=env_id,
            label=label,
            text=text,
            page=page,
            confidence=round(confidence, 4),
            number=number,
        ))

    return results


def detect_environments(pdf_path: str, model_dir: str) -> list:
    print(f"[INFO] Extracting candidates from {pdf_path}")
    candidates = extract_candidates(pdf_path)
    print(f"[INFO] {len(candidates)} candidates found")

    tokenizer, model, device = load_model(model_dir)
    print("[INFO] Running classifier")

    environments = classify_candidates(candidates, tokenizer, model, device)
    print(f"[INFO] {len(environments)} environments detected")

    return environments


def save_results(environments: list, output_path: str):
    data = []
    for e in environments:
        d = asdict(e)
        d["type"] = d.pop("label")
        data.append(d)
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect theorem-like environments in PDFs")
    parser.add_argument("pdf", help="Input PDF path")
    args = parser.parse_args()

    environments = detect_environments(args.pdf, str(EXTRACTION_MODEL))

    output = Path(args.pdf).stem + "_environments.json"
    save_results(environments, output)

    from collections import Counter
    counts = Counter(e.label for e in environments)
    print("\nDetected environments:")
    for label, count in sorted(counts.items()):
        print(f"  {label:<15} {count}")
    print(f"\n  Total: {len(environments)}")
