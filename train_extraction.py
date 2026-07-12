import json
import numpy as np

from pathlib import Path
from collections import Counter
from datasets import Dataset

# Used for creating a train/test split.
from sklearn.model_selection import train_test_split
# Evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

DATASET_FILE = Path("data/extraction_train_dataset.json")
BASE_MODEL = "allenai/scibert_scivocab_uncased"

# Make sure the output folder exists.
MODEL_DIR = Path("extraction_model")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Number of times the model sees the entire training dataset.
NUM_EPOCHS = 3

# Number of examples processed at once.
BATCH_SIZE = 16

# Maximum number of tokens per input text.
MAX_LENGTH = 256

# Percentage of data that is reserved for evaluation.
TEST_SIZE = 0.2


def tokenize(batch):
    """
    Converts raw text into transformer input IDs.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        # Pad shorter examples.
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids

    # Select the class with the highest probability.
    y_pred = np.argmax(
        eval_pred.predictions,
        axis=1
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
    }


if __name__ == "__main__":
    print("\nLoading dataset...\n")

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        raw_data = json.load(f)

    print(
        f"Raw samples loaded: {len(raw_data)}"
    )

    # Only keep the fields that are necessary for training.
    raw_data = [
        {
            "text": item["text"],
            "label": item["label"],
            "paper_id": item["paper_id"],
        }
        for item in raw_data
    ]

    print(
        f"Samples after filtering: {len(raw_data)}"
    )

    # Count how many examples exist for each class.
    counts = Counter(
        item["label"]
        for item in raw_data
    )

    print("\nClass distribution:")

    for label, count in sorted(counts.items()):
        print(
            f"  {label:<15} {count:>5}"
        )

    # Convert labels to numbers.
    labels = sorted(
        counts.keys()
    )
    label2id = {
        label: index
        for index, label in enumerate(labels)
    }
    id2label = {
        index: label
        for label, index in label2id.items()
    }
    print(
        "\nLabel mapping:"
    )
    print(label2id)

    # Replace text labels with integer IDs.
    for item in raw_data:
        item["label"] = label2id[
            item["label"]
        ]

    # Split the train and test set using stratification so that the same class
    # balance is used in both training and testing sets.
    train_data, test_data = train_test_split(
        raw_data,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=[
            item["label"]
            for item in raw_data
        ],
    )

    print(
        f"\nTrain samples: {len(train_data)}"
    )

    print(
        f"Test samples:  {len(test_data)}"
    )


    # Convert Python lists into HuggingFace datasets.
    train_dataset = Dataset.from_list(
        train_data
    )

    test_dataset = Dataset.from_list(
        test_data
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    print(
        "Tokenizing dataset..."
    )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True
    )

    # The model does not need the original text anymore.
    train_dataset = train_dataset.remove_columns(
        ["text"]
    )

    test_dataset = test_dataset.remove_columns(
        ["text"]
    )

    # Trainer expects the target column to be called "labels."
    train_dataset = train_dataset.rename_column(
        "label",
        "labels"
    )

    test_dataset = test_dataset.rename_column(
        "label",
        "labels"
    )

    # Convert data into PyTorch tensors.
    train_dataset.set_format(
        "torch"
    )

    test_dataset.set_format(
        "torch"
    )

    print(
        "\nLoading transformer model..."
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Training configuration.
    training_args = TrainingArguments(
        output_dir=str(
            MODEL_DIR / "checkpoints"
        ),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )

    # Dynamically pads batches efficiently.
    data_collator = DataCollatorWithPadding(
        tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(
        "\nStarting training...\n"
    )

    trainer.train()

    print(
        "\nRunning final evaluation...\n"
    )

    results = trainer.predict(
        test_dataset
    )

    y_pred = np.argmax(
        results.predictions,
        axis=1
    )

    y_true = results.label_ids

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
        )
    )

    # Save the model.
    save_path = MODEL_DIR / "final"

    print(
        "\nSaving model..."
    )

    trainer.save_model(
        str(save_path)
    )

    tokenizer.save_pretrained(
        str(save_path)
    )

    print(
        f"\nModel saved to: {save_path}"
    )

    print(
        """
You can load the model using:

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="extraction_model/final"
)

"""
    )
