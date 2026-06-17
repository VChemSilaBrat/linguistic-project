#!/usr/bin/env python3
"""
export_hf_dataset.py
Converts the unified BIO TSV (data/bio/bio_merged.tsv) into a HuggingFace
DatasetDict suitable for ruBERT token classification.
Usage:
    python scripts/exporters/export_hf_dataset.py \
        --input data/bio/bio_merged.tsv \
        --output data/exported/hf_dataset
Each example in the resulting dataset has:
    tokens:      list[str]  -- the words, in original order
    ner_tags:    list[int]  -- 0=O, 1=B-MATH, 2=I-MATH
    sentence_id: str
Splits produced: train_clean, train_noisy, val, test (taken verbatim from
the `split` column of the input TSV).
"""
import argparse
import csv
import sys
from collections import defaultdict

LABEL_TO_ID = {"O": 0, "B-MATH": 1, "I-MATH": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
EXPECTED_SPLITS = ["train_clean", "train_llm", "train_noisy", "val", "test"]


def read_bio_tsv(path):
    raw = defaultdict(list)
    sent_split = {}
    order_per_split = defaultdict(list)
    seen_sentence_ids = set()
    n_rows = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required_cols = {"sentence_id", "token_index", "token", "bio_tag", "source", "split"}
        missing = required_cols - set(reader.fieldnames or [])
        assert not missing, f"Input TSV is missing required columns: {missing}"

        for row in reader:
            n_rows += 1
            sid = row["sentence_id"]
            split = row["split"]
            tok_idx = int(row["token_index"])
            token = row["token"]
            bio_tag = row["bio_tag"]

            assert bio_tag in LABEL_TO_ID, (
                f"Unexpected bio_tag '{bio_tag}' for sentence_id={sid}, "
                f"token_index={tok_idx}. Expected one of {list(LABEL_TO_ID)}."
            )

            if sid not in sent_split:
                sent_split[sid] = split
                order_per_split[split].append(sid)
            else:
                assert sent_split[sid] == split, (
                    f"Inconsistent split for sentence_id={sid}: "
                    f"saw '{sent_split[sid]}' and '{split}'."
                )

            raw[sid].append((tok_idx, token, bio_tag))
            seen_sentence_ids.add(sid)

    sentences = defaultdict(list)
    for split, sids in order_per_split.items():
        for sid in sids:
            rows = sorted(raw[sid], key=lambda r: r[0])
            tokens = [r[1] for r in rows]
            ner_tags = [LABEL_TO_ID[r[2]] for r in rows]
            sentences[split].append(
                {"sentence_id": sid, "tokens": tokens, "ner_tags": ner_tags}
            )

    return sentences, n_rows, seen_sentence_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to bio_merged.tsv")
    parser.add_argument("--output", required=True, help="Output directory for the HF dataset")
    args = parser.parse_args()

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as e:
        print(
            "ERROR: the `datasets` package (HuggingFace) is required. "
            "Install with `pip install datasets`.",
            file=sys.stderr,
        )
        raise e

    sentences, n_rows, seen_sentence_ids = read_bio_tsv(args.input)

    found_splits = set(sentences.keys())
    unexpected = found_splits - set(EXPECTED_SPLITS)
    assert not unexpected, f"Unexpected split values found in input: {unexpected}"

    dataset_dict = {}
    total_examples = 0
    total_tokens_out = 0

    for split in EXPECTED_SPLITS:
        examples = sentences.get(split, [])
        if not examples:
            print(f"WARNING: split '{split}' has 0 sentences.", file=sys.stderr)
        cols = {
            "sentence_id": [ex["sentence_id"] for ex in examples],
            "tokens": [ex["tokens"] for ex in examples],
            "ner_tags": [ex["ner_tags"] for ex in examples],
        }
        ds = Dataset.from_dict(cols)
        dataset_dict[split] = ds
        total_examples += len(examples)
        total_tokens_out += sum(len(ex["tokens"]) for ex in examples)

    dd = DatasetDict(dataset_dict)

    written_sentence_ids = set()
    for split, examples in sentences.items():
        for ex in examples:
            written_sentence_ids.add(ex["sentence_id"])

    assert written_sentence_ids == seen_sentence_ids, (
        "Sentence drop detected: "
        f"input had {len(seen_sentence_ids)} unique sentence_ids, "
        f"output has {len(written_sentence_ids)}."
    )

    dd.save_to_disk(args.output)

    print("=== export_hf_dataset.py summary ===")
    print(f"Input rows (tokens) read: {n_rows}")
    print(f"Unique sentence_ids: {len(seen_sentence_ids)}")
    for split in EXPECTED_SPLITS:
        n = len(dataset_dict[split])
        ntok = sum(len(ex) for ex in dataset_dict[split]["tokens"])
        print(f"  split={split}: sentences={n}, tokens={ntok}")
    print(f"Total sentences across splits: {total_examples}")
    print(f"Total tokens across splits: {total_tokens_out}")
    print(f"Saved DatasetDict to: {args.output}")


if __name__ == "__main__":
    main()
