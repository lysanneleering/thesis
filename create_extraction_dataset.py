import json
import re
import time
import fitz
import tarfile
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from collections import Counter

RAW_DIR     = Path("data/raw")
PDF_DIR     = Path("data/pdfs")
OUTPUT_DIR  = Path("data")

TRAIN_FILE    = OUTPUT_DIR / "extraction_train_dataset.json"
TEST_FILE     = OUTPUT_DIR / "extraction_test_dataset.json"
MANIFEST_FILE = OUTPUT_DIR / "split_manifest.json"

NUMBER_OF_PAPERS = 500
TEST_PER_CATEGORY = 25

# Jaccard similarity threshold for accepting a PDF block as a verified.
MATCH_THRESHOLD = 0.45

# A verified block must contain at least this many whitespace-separated tokens
# after normalisation.
MIN_MATCH_TOKENS = 10

# Maximum number of plain "other" examples collected per paper.
OTHER_PER_PAPER = 15

# After a keyword-prefix block, merge at most this many tokens of following
# blocks into it.
MAX_MERGE_TOKENS = 500

CATEGORIES = [
    "math.CO",
    "math.LO",
    "math.NT",
    "cs.CC"
]

ENVIRONMENTS = {
    "definition" : "definition",
    "theorem"    : "theorem",
    "proposition": "proposition",
    "lemma"      : "lemma",
    "corollary"  : "corollary",
    "conjecture" : "conjecture",
    "proof"      : "proof"
}
ENV_PATTERN = "|".join(ENVIRONMENTS)

# Matches a PDF block whose very first word is a theorem keyword.
KEYWORD_PREFIX = re.compile(
    rf"^({ENV_PATTERN})\b",
    re.IGNORECASE,  # Case insensitive: Theorem, THEOREM, theorem all match
)

# Common words filtered from Jaccard token sets.
# These words don't carry meaningful content for matching purposes.
STOP_TOKENS = {
    "the", "a", "an", "of", "and", "in", "is", "to", "that", "we",
    "let", "for", "with", "by", "on", "be", "it", "if", "then",
    "are", "have", "this", "from", "as", "at", "or", "not", "so",
    "can", "its", "which", "all", "any", "such", "each",
}

ARXIV_API   = "https://export.arxiv.org/api/query"  # For searching papers.
ARXIV_LATEX = "https://arxiv.org/e-print"  # For downloading LaTeX source.
ARXIV_PDF   = "https://arxiv.org/pdf"  # For downloading PDFs.


def fetch(url: str, retries: int = 3) -> bytes | None:
    """
    Downloads raw bytes from *url*, retrying on timeout.
    """
    # Create a request with a proper User-Agent (required by arXiv).
    request = Request(url, headers={"User-Agent": "MathDatasetBuilder/1.0"})

    for attempt in range(retries):
        try:
            # Download with 60-second timeout to prevent hanging.
            return urlopen(request, timeout=60).read()
        except TimeoutError:
            # Wait longer each time (5, 10, 15 seconds).
            time.sleep((attempt + 1) * 5)
        except (URLError, HTTPError):
            # Network errors are usually not recoverable.
            return None
    return None


def search_arxiv(category: str, limit: int) -> list[str]:
    """
    Returns up to *limit* paper IDs from *category* via the arXiv API.
    """
    url = (
        f"{ARXIV_API}?search_query=cat:{category}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={limit}"
    )

    # Download the API response.
    data = fetch(url)
    if not data:
        return []

    # Parse the XML response.
    root = ET.fromstring(data)
    ids = []

    # Extract paper IDs from Atom feed entries.
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        paper_id = entry.find("{http://www.w3.org/2005/Atom}id")
        if paper_id is not None:
            ids.append(paper_id.text.split("/abs/")[-1])

    return ids


def collect_and_split(
    n_total: int,
    test_per_category: int,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Collects paper IDs from arXiv and splits them into train / test sets
    stratified by category.
    """
    rng = random.Random(seed)

    # Calculate how many papers to fetch per category.
    per_category_total = n_total // len(CATEGORIES)

    # Fetch a few extras to cover any de-duplication losses.
    fetch_limit = per_category_total + 10

    train_ids: dict[str, list[str]] = {}
    test_ids:  dict[str, list[str]] = {}

    # Process each category independently.
    for category in CATEGORIES:
        print(f"  Searching {category}...")
        ids = search_arxiv(category, fetch_limit)

        # De-duplicate within this category fetch.
        ids = list(dict.fromkeys(ids))

        # Warn if we didn't get enough papers.
        if len(ids) < per_category_total:
            print(f"  Warning: only {len(ids)} papers found for {category}, "
                  f"wanted {per_category_total}.")

        # Shuffle and split.
        rng.shuffle(ids)
        test_ids[category]  = ids[:test_per_category]  # First 25 go to test.
        train_ids[category] = ids[test_per_category: per_category_total]

        print(f"    train={len(train_ids[category])}  test={len(test_ids[category])}")

        time.sleep(3)

    return train_ids, test_ids


def flatten_split(
    split: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """
    Flattens { category: [id, ...] } → [(category, id), ...].
    """
    return [
        (category, paper_id)
        for category, ids in split.items()
        for paper_id in ids
    ]


def read_tex_files(tarball: Path) -> list[str]:
    """
    Extracts and returns the text of every .tex file in a tar.gz archive.
    """
    tex_files = []
    try:
        with tarfile.open(tarball, "r:*") as tar:
            # Iterate through all files in the archive.
            for member in tar.getmembers():
                # Only process .tex files.
                if member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        tex_files.append(f.read().decode(errors="ignore"))
    except Exception:
        pass
    return tex_files


def fetch_pdf_cached(paper_id: str) -> bytes | None:
    """
    Returns PDF bytes for *paper_id*, downloading and caching to disk if
    not already present.
    """
    # Check if PDF is already cached.
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if pdf_path.exists():
        return pdf_path.read_bytes()

    # Download and cache.
    pdf_bytes = fetch(f"{ARXIV_PDF}/{paper_id}.pdf")
    if pdf_bytes:
        pdf_path.write_bytes(pdf_bytes)

    return pdf_bytes


def normalize(text: str) -> str:
    """
    Lowercases and strips everything except letters, digits, and spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_latex(text: str) -> str:
    """
    Removes LaTeX syntax from source text.
    """
    # Remove LaTeX comments (from % to end of line).
    text = re.sub(r"%[^\n]*", " ", text)
    # Remove \begin{...} and \end{...} commands.
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    # Replace simple commands with their arguments.
    text = re.sub(r"\\[a-zA-Z]+\{([^}]{0,200})\}", r"\1", text)
    # Remove other LaTeX commands without arguments.
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    # Remove display math ($$ ... $$).
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    # Remove inline math ($ ... $).
    text = re.sub(r"\$.*?\$", " ", text)
    # Remove curly braces that may be left over.
    text = re.sub(r"[{}]", " ", text)
    # Normalize the remaining text.
    return normalize(text)


def tokenize(text: str) -> set[str]:
    """
    Returns meaningful tokens from *text*, excluding stop words.
    """
    return set(normalize(text).split()) - STOP_TOKENS


def similarity(a: str, b: str) -> float:
    """
    Jaccard similarity over filtered token sets of *a* and *b*.
    """
    ta, tb = tokenize(a), tokenize(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def extract_latex_blocks(tex: str) -> list[dict]:
    """
    Extracts theorem-like environments and plain paragraphs from LaTeX source.
    """
    blocks = []

    pattern = re.compile(
        rf"\\begin\{{({ENV_PATTERN})\*?\}}(.*?)\\end\{{({ENV_PATTERN})\*?\}}",
        re.DOTALL | re.IGNORECASE,  # DOTALL: . matches newlines too
    )

    # Extract all named environments.
    for match in pattern.finditer(tex):
        # Get the environment type (theorem, lemma, etc.).
        label = ENVIRONMENTS.get(match.group(1).lower())
        # Get the content and strip LaTeX syntax.
        text = strip_latex(match.group(2))

        # Only keep if it's a valid environment and has reasonable length.
        if label and 30 < len(text) < 1500:
            blocks.append({"label": label, "text": text})

    # Extract plain paragraphs as "other" examples.
    other = 0
    # Split the text by environment blocks.
    chunks = re.split(
        rf"\\begin\{{(?:{ENV_PATTERN})\*?\}}.*?\\end\{{(?:{ENV_PATTERN})\*?\}}",
        tex, flags=re.DOTALL | re.IGNORECASE,
    )

    # Process each chunk between environments.
    for chunk in chunks:
        if other >= OTHER_PER_PAPER:
            break
        # Split into paragraphs.
        for paragraph in re.split(r"\n{2,}", chunk):
            text = strip_latex(paragraph)
            # Only keep paragraphs of reasonable length.
            if 50 < len(text) < 600:
                blocks.append({"label": "other", "text": text})
                other += 1
                break  # Only take one paragraph per chunk.

    return blocks


def extract_pdf_blocks(
    pdf_bytes: bytes,
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Extracts text from the PDF and returns two separate pools.
    """
    raw_blocks: list[str] = []
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract text from each page.
        for page in document:
            # Get text blocks from the page.
            for block in page.get_text("blocks"):
                # block[6] is the block type: 0 = text, 1 = image.
                if block[6] != 0:
                    continue  # Skip non-text blocks.

                # Normalize the text.
                norm = normalize(block[4])

                # Only keep blocks with at least 10 characters.
                if len(norm) >= 10:
                    raw_blocks.append(norm)

        document.close()

    except Exception:
        pass  # Return empty lists if PDF parsing fails.

    theorem_candidates: list[tuple[str, str]] = []
    other_candidates: list[str] = []

    i = 0
    while i < len(raw_blocks):
        norm = raw_blocks[i]

        # Check if this block starts with a theorem keyword.
        m = KEYWORD_PREFIX.match(norm)

        if m:
            # Get the canonical label for this environment type.
            label = ENVIRONMENTS.get(m.group(1).lower())

            if label:
                # Start merging consecutive blocks.
                merged = norm
                j = i + 1

                # Merge subsequent blocks until we hit a limit or another keyword.
                while (
                    j < len(raw_blocks)
                    and len(merged.split()) < MAX_MERGE_TOKENS
                ):
                    nxt = raw_blocks[j]

                    # Stop if we hit another environment start.
                    if KEYWORD_PREFIX.match(nxt):
                        break

                    # Add this block to the merged text.
                    merged += " " + nxt
                    j += 1

                # Store as a theorem candidate.
                theorem_candidates.append((label, merged))
                i = j  # Skip the merged blocks.
                continue

        # If not a theorem keyword, add to other candidates.
        other_candidates.append(norm)
        i += 1

    return theorem_candidates, other_candidates


def verify_and_collect(
    theorem_candidates: list[tuple[str, str]],
    latex_blocks:       list[dict],
    paper_id:           str,
) -> tuple[list[dict], list[dict]]:
    """
    Verifies each PDF theorem candidate against the LaTeX oracle.
    """
    verified: list[dict] = []
    hard_negatives: list[dict] = []

    # Group LaTeX blocks by label for efficient lookup.
    latex_by_label: dict[str, list[str]] = {}
    for b in latex_blocks:
        if b["label"] != "other":
            latex_by_label.setdefault(b["label"], []).append(b["text"])

    # Process each theorem candidate.
    for (pdf_label, pdf_text) in theorem_candidates:
        # Find LaTeX blocks with the same label.
        candidates = latex_by_label.get(pdf_label, [])

        # If no matching LaTeX block, it's a hard negative.
        if not candidates:
            hard_negatives.append({
                "text"    : pdf_text,
                "label"   : "other",
                "paper_id": paper_id,
            })
            continue

        # Find the best match among all LaTeX blocks of this type.
        best_score = max(similarity(pdf_text, c) for c in candidates)

        # Check if the match meets the threshold.
        if (
            best_score >= MATCH_THRESHOLD
            and len(pdf_text.split()) >= MIN_MATCH_TOKENS
        ):
            # Good match -> positive example.
            verified.append({
                "text"    : pdf_text,
                "label"   : pdf_label,
                "paper_id": paper_id,
            })
        else:
            # Poor match -> hard negative.
            hard_negatives.append({
                "text"    : pdf_text,
                "label"   : "other",
                "paper_id": paper_id,
            })

    return verified, hard_negatives


def collect_other_examples(
    other_candidates: list[str],
    latex_blocks:     list[dict],
    paper_id:         str,
) -> list[dict]:
    """
    Samples plain "other" training examples from non-keyword PDF blocks.
    """
    # Collect all theorem texts from LaTeX oracle.
    all_theorem_texts = [
        b["text"] for b in latex_blocks if b["label"] != "other"
    ]

    results: list[dict] = []

    # Process each candidate until we have enough.
    for pdf_text in other_candidates:
        if len(results) >= OTHER_PER_PAPER:
            break

        # Check if this text is too similar to any theorem.
        # This prevents false negatives.
        if all_theorem_texts:
            max_sim = max(similarity(pdf_text, t) for t in all_theorem_texts)
            if max_sim >= 0.55:
                continue  # Skip this candidate, it might be a theorem.

        # Add as a plain "other" example.
        results.append({
            "text"    : pdf_text,
            "label"   : "other",
            "paper_id": paper_id,
        })

    return results


def process_paper(paper_id: str) -> list[dict] | None:
    """
    Runs the full pipeline for a single paper.
    """
    tar_path = RAW_DIR / f"{paper_id}.tar.gz"

    # Download LaTeX source.
    if not tar_path.exists():
        latex_bytes = fetch(f"{ARXIV_LATEX}/{paper_id}")
        # Check if we got LaTeX or PDF.
        if not latex_bytes or latex_bytes[:4] == b"%PDF":
            print("    Skipped: no LaTeX source available")
            return None
        tar_path.write_bytes(latex_bytes)

    # Build LaTeX oracle.
    latex_blocks: list[dict] = []
    for tex in read_tex_files(tar_path):
        latex_blocks.extend(extract_latex_blocks(tex))

    # Skip if no theorem-like environments are found in LaTeX.
    if not latex_blocks:
        print("    Skipped: no theorem-like environments found in LaTeX")
        return None

    # Download PDF.
    pdf = fetch_pdf_cached(paper_id)
    if not pdf:
        print("    Skipped: PDF download failed")
        return None

    # Extract PDF blocks.
    theorem_candidates, other_candidates = extract_pdf_blocks(pdf)
    if not theorem_candidates:
        print("    Skipped: no keyword-prefixed blocks found in PDF")
        return None

    # Verify candidates and collect hard negatives.
    verified, hard_negatives = verify_and_collect(
        theorem_candidates, latex_blocks, paper_id
    )

    # Collect other examples.
    other_egs = collect_other_examples(
        other_candidates, latex_blocks, paper_id
    )

    # Return all records combined.
    return verified + hard_negatives + other_egs


def print_summary(label: str, records: list[dict]) -> None:
    """
    Prints a summary of the dataset with class distribution.
    """
    print(f"\n{label} summary:")
    counts = Counter(d["label"] for d in records)
    total = len(records)
    for lbl, count in sorted(counts.items()):
        pct = count / total * 100 if total else 0
        print(f"  {lbl:<15} {count:>6}  ({pct:.1f}%)")
    print(f"  {'Total':<15} {total:>6}")


if __name__ == "__main__":
    # Create necessary directories.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect IDs and split before any downloading.
    print("Collecting arXiv paper IDs and splitting train/test...\n")
    train_split, test_split = collect_and_split(
        n_total=NUMBER_OF_PAPERS,
        test_per_category=TEST_PER_CATEGORY,
    )

    # Flatten the splits into lists of (category, paper_id) pairs.
    train_pairs = flatten_split(train_split)
    test_pairs  = flatten_split(test_split)

    print(f"\nSplit complete:")
    print(f"  Train papers : {len(train_pairs)}  (target: {NUMBER_OF_PAPERS - TEST_PER_CATEGORY * len(CATEGORIES)})")
    print(f"  Test papers  : {len(test_pairs)}  (target: {TEST_PER_CATEGORY * len(CATEGORIES)})")

    # Save the manifest so the split is reproducible.
    manifest = {
        "train": {cat: ids for cat, ids in train_split.items()},
        "test":  {cat: ids for cat, ids in test_split.items()},
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSplit manifest saved to: {MANIFEST_FILE}")

    # Process train papers.
    print("\n" + "=" * 60)
    print("Processing TRAIN papers...")
    print("=" * 60)

    train_dataset: list[dict] = []
    for i, (category, paper_id) in enumerate(train_pairs, 1):
        print(f"[train {i}/{len(train_pairs)}] {paper_id}  ({category})")

        records = process_paper(paper_id)
        if records:
            # Tag each record with its split and category.
            for r in records:
                r["category"] = category
                r["split"] = "train"
            train_dataset.extend(records)

            # Print statistics for this paper.
            verified_count = sum(1 for r in records if r["label"] != "other")
            other_count = sum(1 for r in records if r["label"] == "other")
            print(f"    {verified_count} verified  |  {other_count} other")

        time.sleep(1)

    # Process test papers.
    print("\n" + "=" * 60)
    print("Processing TEST papers...")
    print("=" * 60)

    test_dataset: list[dict] = []
    for i, (category, paper_id) in enumerate(test_pairs, 1):
        print(f"[test {i}/{len(test_pairs)}] {paper_id}  ({category})")

        records = process_paper(paper_id)
        if records:
            # Tag each record with its split and category.
            for r in records:
                r["category"] = category
                r["split"] = "test"
            test_dataset.extend(records)

            # Print statistics for this paper.
            verified_count = sum(1 for r in records if r["label"] != "other")
            other_count = sum(1 for r in records if r["label"] == "other")
            print(f"    {verified_count} verified  |  {other_count} other")

        time.sleep(1)

    # Save the datasets.
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)

    # Print summary statistics.
    print_summary("Train dataset", train_dataset)
    print_summary("Test dataset",  test_dataset)

    print(f"\nFiles saved:")
    print(f"  {TRAIN_FILE}")
    print(f"  {TEST_FILE}")
    print(f"  {MANIFEST_FILE}")
