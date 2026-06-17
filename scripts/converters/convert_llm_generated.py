#!/usr/bin/env python3
"""Convert Groq-generated [MATH]-tagged sentences to unified BIO TSV format.

Input:  data/generated/llm_sentences.txt   (one [MATH]...[/MATH]-tagged sentence per line,
                                             produced by scripts/generators/generate_groq.py)
Output: data/bio/bio_llm_generated.tsv     (sentence_id, token_index, token, bio_tag, source)

Tokenization and tag-parsing follow the same convention as
scripts/converters/convert_synthetic.py (whitespace split, [MATH]/[/MATH] markers ->
B-MATH / I-MATH, no blank lines between sentences — sentences are distinguished by
sentence_id only, matching the other bio_*.tsv files in this repo).
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


def parse_math_tags(tagged_sentence):
    """Parse [MATH]...[/MATH] tags and return list of (token, bio_tag) pairs.

    Same algorithm as convert_synthetic.py: split on the tag markers, odd-indexed
    parts (1, 3, 5, ...) are inside a MATH span.
    """
    parts = re.split(r"\[MATH\]|\[/MATH\]", tagged_sentence)

    tokens_tags = []
    for i, part in enumerate(parts):
        is_math = (i % 2 == 1)
        words = part.split()
        for j, word in enumerate(words):
            if is_math:
                tag = "B-MATH" if j == 0 else "I-MATH"
            else:
                tag = "O"
            tokens_tags.append((word, tag))

    return tokens_tags


PUNCT_ONLY = re.compile(r"^[.,;!?]+$")

# "спан" (span) is jargon from our own generation prompt — real Russian lecture
# transcripts never use this anglicism. If the LLM's response leaks its own planning/
# reasoning about the task ("Первое требование: 2 предложения с одним коротким
# спаном...", "**2 предложения с длинными спанами**...") instead of producing an
# actual lecture sentence, that leaked text reliably contains this word. Treating any
# line that contains it as non-lecture text is a simple, low-risk filter: it costs a
# few percent of lines but guarantees no meta-commentary slips into training data.
META_LEAK_RE = re.compile(r"спан", re.IGNORECASE)


def force_punct_to_o(tokens_tags):
    """Pure-punctuation tokens are always O, even if they fell inside [MATH] tags.

    If a punctuation-only token is forced from B/I to O, the following I-MATH
    token (if any) is promoted to B-MATH so the BIO sequence stays consistent.
    """
    fixed = []
    promote_next = False
    for tok, tag in tokens_tags:
        if PUNCT_ONLY.match(tok):
            if tag != "O":
                promote_next = True
            fixed.append((tok, "O"))
            continue
        if promote_next and tag == "I-MATH":
            tag = "B-MATH"
        fixed.append((tok, tag))
        promote_next = False
    return fixed


def has_nested_or_unbalanced_tags(line):
    """Detect nested [MATH] tags (a [MATH] opening before the prior one closed)."""
    depth = 0
    for tok in re.findall(r"\[MATH\]|\[/MATH\]", line):
        if tok == "[MATH]":
            if depth > 0:
                return True
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return True
    return depth != 0


def validate_bio(tokens_tags):
    """Check BIO consistency: no I-MATH after O without a preceding B-MATH."""
    for i, (tok, tag) in enumerate(tokens_tags):
        if tag == "I-MATH" and (i == 0 or tokens_tags[i - 1][1] == "O"):
            return False
    return True


def span_lengths(tokens_tags):
    """Return list of span lengths (in tokens) for all B-MATH...I-MATH* spans."""
    lengths = []
    cur = 0
    for tok, tag in tokens_tags:
        if tag == "B-MATH":
            if cur:
                lengths.append(cur)
            cur = 1
        elif tag == "I-MATH":
            cur += 1
        else:
            if cur:
                lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return lengths


def convert_llm_generated(input_path, output_path):
    rows_out = []
    n_skipped = 0
    n_lines = 0

    with open(input_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            n_lines += 1

            n_open, n_close = line.count("[MATH]"), line.count("[/MATH]")
            if n_open != n_close or n_open == 0:
                print(f"WARNING: malformed tags in line {n_lines}, skipping: {line[:80]!r}")
                n_skipped += 1
                continue

            if has_nested_or_unbalanced_tags(line):
                print(f"WARNING: nested/unbalanced MATH tags in line {n_lines}, skipping: {line[:80]!r}")
                n_skipped += 1
                continue

            if META_LEAK_RE.search(line):
                print(f"WARNING: leaked task-planning text (contains 'спан') in line "
                      f"{n_lines}, skipping: {line[:80]!r}")
                n_skipped += 1
                continue

            tokens_tags = parse_math_tags(line)
            tokens_tags = force_punct_to_o(tokens_tags)

            if not validate_bio(tokens_tags):
                print(f"WARNING: BIO inconsistency in line {n_lines}, skipping: {line[:80]!r}")
                n_skipped += 1
                continue

            sent_idx = len(set(r["sentence_id"] for r in rows_out))
            sentence_id = f"llm_{sent_idx:04d}"
            for tok_idx, (token, tag) in enumerate(tokens_tags):
                rows_out.append({
                    "sentence_id": sentence_id,
                    "token_index": tok_idx,
                    "token": token,
                    "bio_tag": tag,
                    "source": "llm_generated",
                })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sentence_id", "token_index", "token", "bio_tag", "source"])
        for r in rows_out:
            writer.writerow([r["sentence_id"], r["token_index"], r["token"], r["bio_tag"], r["source"]])

    # --- Stats ---
    sentence_ids = sorted(set(r["sentence_id"] for r in rows_out))
    n_sentences = len(sentence_ids)
    n_tokens = len(rows_out)
    tag_counts = Counter(r["bio_tag"] for r in rows_out)
    n_spans = tag_counts.get("B-MATH", 0)

    all_lengths = []
    for sid in sentence_ids:
        toks = [(r["token"], r["bio_tag"]) for r in rows_out if r["sentence_id"] == sid]
        all_lengths.extend(span_lengths(toks))

    length_hist = Counter(all_lengths)
    avg_len = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0

    print(f"\nLLM-generated: {n_lines} input lines, {n_skipped} skipped "
          f"({n_lines - n_skipped} kept)")
    print(f"Output: {n_sentences} sentences, {n_tokens} tokens, {n_spans} math spans")
    print(f"Tokens per tag: O={tag_counts.get('O', 0)}, "
          f"B-MATH={tag_counts.get('B-MATH', 0)}, I-MATH={tag_counts.get('I-MATH', 0)}")
    print(f"Avg span length: {avg_len:.2f} tokens")
    print("Span length distribution (tokens -> count):")
    for length in sorted(length_hist):
        print(f"  {length:>3} -> {length_hist[length]}")

    # Diversity check: how many sentences have 1 / 2 / 3+ spans
    spans_per_sentence = Counter()
    for sid in sentence_ids:
        n = sum(1 for r in rows_out if r["sentence_id"] == sid and r["bio_tag"] == "B-MATH")
        spans_per_sentence[n] += 1
    print("Spans per sentence (n_spans -> n_sentences):")
    for n in sorted(spans_per_sentence):
        print(f"  {n} -> {spans_per_sentence[n]}")

    # Sample sentences
    print("\n--- Sample sentences ---")
    for sid in sentence_ids[:3]:
        toks = [(r["token"], r["bio_tag"]) for r in rows_out if r["sentence_id"] == sid]
        print(f"\n{sid}:")
        for tok, tag in toks:
            marker = f" <{tag}>" if tag != "O" else ""
            print(f"  {tok}{marker}")

    return rows_out


def append_to_merged(rows_out, merged_path):
    """Append LLM-generated rows to bio_merged.tsv with split='train_llm'.

    Does not touch any other existing rows/splits. Only run with --append-to-merged,
    since the project convention is not to silently modify existing files.
    """
    merged_path = Path(merged_path)
    assert merged_path.exists(), f"{merged_path} does not exist; run merge_all.py first"

    with open(merged_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        existing_rows = list(reader)

    assert header == ["sentence_id", "token_index", "token", "bio_tag", "source", "split"], (
        f"Unexpected bio_merged.tsv header: {header}"
    )

    with open(merged_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(existing_rows)
        for r in rows_out:
            writer.writerow([r["sentence_id"], r["token_index"], r["token"],
                              r["bio_tag"], r["source"], "train_llm"])

    print(f"\nAppended {len(rows_out)} rows ({len(set(r['sentence_id'] for r in rows_out))} "
          f"sentences) to {merged_path} with split=train_llm")


def main():
    base = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="Convert Groq-generated [MATH]-tagged sentences to BIO TSV."
    )
    parser.add_argument("--input", type=str,
                         default=str(base / "data" / "generated" / "llm_sentences.txt"))
    parser.add_argument("--output", type=str,
                         default=str(base / "data" / "bio" / "bio_llm_generated.tsv"))
    parser.add_argument("--append-to-merged", action="store_true",
                         help="Also append the converted rows to data/bio/bio_merged.tsv "
                              "with split=train_llm (off by default; does not touch "
                              "existing rows).")
    parser.add_argument("--merged-path", type=str,
                         default=str(base / "data" / "bio" / "bio_merged.tsv"))
    args = parser.parse_args()

    input_path = Path(args.input)
    assert input_path.exists(), f"Input file not found: {input_path}"

    rows_out = convert_llm_generated(input_path, args.output)

    if args.append_to_merged:
        append_to_merged(rows_out, args.merged_path)


if __name__ == "__main__":
    main()
