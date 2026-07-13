
Given a math paper as a PDF, this project detects theorem-like environments
(theorems, lemmas, proofs, definitions, propositions, corollaries,
conjectures), figures out which ones depend on which, and renders the result
as an interactive dependency graph you can explore in a browser.

It works in two stages:

1. **Extraction layer** — parses the PDF, finds blocks of text that look like
   theorem-style environments, and classifies each one by type.
2. **Linking layer** — for every environment, finds which earlier
   environments it depends on, using both regex-based citation matching
   ("by Theorem 2.1…") and a learned classifier for implicit references
   ("as shown above…").

The output is a graph (`nodes` = environments, `edges` = dependencies) saved
as JSON and rendered as a force-directed HTML visualization.

## How it works

```
                ┌─────────────────┐        ┌─────────────────┐
   paper.pdf ─▶ │ extraction_layer │  ─▶   │  linking_layer   │ ─▶ graph.json / graph.html
                │  (SciBERT clf)   │        │  (regex + clf)   │
                └─────────────────┘        └─────────────────┘
```

**Extraction** (`extraction_layer.py`): PyMuPDF pulls structured text blocks
(with font size / bold / position info) out of the PDF. Blocks that look like
the start of an environment (bold, short, matches a pattern like `Theorem
2.1.`) are merged with the text that follows into candidates, which are then
classified by a fine-tuned SciBERT sequence classifier into one of the
environment types or `other`.

**Linking** (`linking_layer.py`): explicit references ("by Lemma 3.2",
"see Theorem 1") are found first using regex and resolved against known
environment numbers. Everything else is checked pairwise, within a sliding
window of nearby environments, by a second SciBERT classifier trained to
predict whether one environment depends on another. This catches implicit
references like "the result above" that don't cite a number.

**Visualization** (`index.html` + `run.py`): the resulting graph is injected
into a D3.js force-directed graph template. Click a node to see its text,
page number, confidence, and incoming/outgoing references.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline on a PDF

```bash
python run.py path/to/paper.pdf
```

This produces, in `./output/`:

- `<paper>_graph.json` — the dependency graph (nodes + edges).
- `<paper>_graph.html` — an interactive visualization (open in a browser).

### Run a layer standalone

```bash
# Extraction only
python extraction_layer.py path/to/paper.pdf
# -> writes <paper>_environments.json

# Linking only, from a previous extraction output
python linking_layer.py <paper>_environments.json
# -> writes <paper>_graph.json
```

## Training your own models

1. **Build the extraction dataset**

   ```bash
   python create_extraction_dataset.py
   ```

   This downloads LaTeX sources and PDFs into `data/raw/` and `data/pdfs/`,
   and writes `data/extraction_train_dataset.json` and
   `data/extraction_test_dataset.json`.

2. **Train the extraction classifier**:

   ```bash
   python train_extraction.py
   ```

   Saves the fine-tuned model to `extraction_model/final/`.

3. **Build the linking dataset**

   ```bash
   python create_linking_dataset.py
   ```

   Writes `data/linking_train_dataset.json` and
   `data/linking_test_dataset.json`.

4. **Train the linking classifier**:

   ```bash
   python train_linking.py
   ```

   Saves the fine-tuned model to `linking_model/final/`.

Rename or symlink `extraction_model/final` → `extraction_model` and
`linking_model/final` → `linking_model` (or adjust the paths in
`extraction_layer.py` / `linking_layer.py`) so `run.py` can find them.

## Evaluation

```bash
python evaluate.py --data-dir data --output evaluation_results.json
```

Results from a prior run are saved in `extraction_evaluation_results.json`
and `linking_evaluation_results.json`.
