#!/usr/bin/env python3
"""Convert Mathematicon WebAnno TSV 3.3 to unified BIO TSV format.

Input:  data/hse/mathcorp-webtsv/annotation/*/admin.tsv
Output: data/bio/bio_mathematicon.tsv

Heuristic for span formation:
  - Tokens with math entity annotations (non-empty Name or inception reference) are MATH
  - Consecutive MATH tokens with gap ≤ 2 non-math tokens are merged into one span
  - Spans are NOT merged across sentence-breaking punctuation (. ! ? ;)
  - Metadata sentences (title:, yb_link:, etc.) are skipped
"""

import csv
import os
import re
from pathlib import Path


METADATA_PREFIXES = ('title:', 'yb_link:', 'branch:', 'level:',
                     'timecode_start:', 'timecode_end:', 'text:',
                     'si=', 'http')
SENT_BREAK_PUNCT = set('.!?;')
MAX_GAP = 2


def parse_webtsv(filepath):
    """Parse a WebAnno TSV 3.3 file, yield (sentence_id, sentence_text, tokens) tuples.

    tokens is a list of dicts: {index, token, is_math}
    """
    sentences = []
    current_sent_id = None
    current_text = None
    current_tokens = []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')

            if line.startswith('#Sentence.id='):
                # Save previous sentence
                if current_sent_id is not None and current_tokens:
                    sentences.append((current_sent_id, current_text, current_tokens))
                current_sent_id = line.split('=', 1)[1]
                current_text = None
                current_tokens = []

            elif line.startswith('#Text='):
                current_text = line[6:]

            elif line.startswith('#') or not line.strip():
                continue

            else:
                fields = line.split('\t')
                if len(fields) < 4:
                    continue

                token_id = fields[0]  # e.g., "3-10"
                token_text = fields[2]

                # Check for math annotation
                name = fields[3] if len(fields) > 3 else '_'
                reference = fields[6] if len(fields) > 6 else '_'

                is_math = (
                    (name not in ('_', '*', '')) or
                    (reference not in ('_', '*', '') and 'inception' in reference)
                )

                # Extract token index within sentence
                parts = token_id.split('-')
                if len(parts) == 2:
                    tok_idx = int(parts[1]) - 1  # 0-based
                else:
                    tok_idx = len(current_tokens)

                current_tokens.append({
                    'index': tok_idx,
                    'token': token_text,
                    'is_math': is_math
                })

    # Save last sentence
    if current_sent_id is not None and current_tokens:
        sentences.append((current_sent_id, current_text, current_tokens))

    return sentences


def is_metadata_sentence(text):
    """Check if a sentence is metadata (title, link, etc.)."""
    if text is None:
        return True
    text_stripped = text.strip().strip('"').strip()
    for prefix in METADATA_PREFIXES:
        if text_stripped.lower().startswith(prefix.lower()):
            return True
    return False


def apply_gap_merge(tokens, max_gap=MAX_GAP):
    """Apply gap-merge heuristic: merge math tokens within gap ≤ max_gap,
    but don't merge across sentence-breaking punctuation."""

    n = len(tokens)
    is_math = [t['is_math'] for t in tokens]

    # Find math token positions
    math_positions = [i for i in range(n) if is_math[i]]
    if not math_positions:
        return ['O'] * n

    # Group math tokens into spans using gap merge
    spans = []  # list of (start, end) inclusive
    current_start = math_positions[0]
    current_end = math_positions[0]

    for pos in math_positions[1:]:
        gap = pos - current_end - 1

        # Check if gap contains sentence-breaking punctuation
        has_break = False
        for g in range(current_end + 1, pos):
            tok_text = tokens[g]['token']
            if any(c in SENT_BREAK_PUNCT for c in tok_text):
                has_break = True
                break

        if gap <= max_gap and not has_break:
            current_end = pos
        else:
            spans.append((current_start, current_end))
            current_start = pos
            current_end = pos

    spans.append((current_start, current_end))

    # Convert spans to BIO tags
    bio = ['O'] * n
    for start, end in spans:
        bio[start] = 'B-MATH'
        for i in range(start + 1, end + 1):
            bio[i] = 'I-MATH'

    return bio


def convert_mathematicon(input_dir, output_path):
    """Convert all Mathematicon WebAnno TSV files to unified BIO TSV."""
    rows_out = []
    total_files = 0
    total_sentences = 0
    skipped_metadata = 0
    skipped_no_math = 0

    annotation_dir = Path(input_dir)
    for doc_dir in sorted(annotation_dir.iterdir()):
        admin_path = doc_dir / 'admin.tsv'
        if not admin_path.is_file():
            continue

        total_files += 1
        doc_name = doc_dir.name.replace('.txt', '').replace('.conllu', '')

        sentences = parse_webtsv(admin_path)

        for sent_id, sent_text, tokens in sentences:
            # Skip metadata sentences
            if is_metadata_sentence(sent_text):
                skipped_metadata += 1
                continue

            # Check if sentence has any math tokens
            has_math = any(t['is_math'] for t in tokens)
            if not has_math:
                skipped_no_math += 1
                # Still include non-math sentences (they're valid O-only examples)
                # But for train_noisy we'll filter later to "at least one math span"

            total_sentences += 1

            # Apply gap-merge heuristic
            bio_tags = apply_gap_merge(tokens)

            # Generate unique sentence ID
            sid = f'math_{doc_name}_{sent_id}'

            for tok_idx, (tok, tag) in enumerate(zip(tokens, bio_tags)):
                rows_out.append({
                    'sentence_id': sid,
                    'token_index': tok_idx,
                    'token': tok['token'],
                    'bio_tag': tag,
                    'source': 'mathematicon'
                })

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['sentence_id', 'token_index', 'token', 'bio_tag', 'source'])
        for r in rows_out:
            writer.writerow([r['sentence_id'], r['token_index'],
                             r['token'], r['bio_tag'], r['source']])

    # Stats
    unique_sents = sorted(set(r['sentence_id'] for r in rows_out))
    n_tokens = len(rows_out)
    n_math = sum(1 for r in rows_out if r['bio_tag'] != 'O')
    n_spans = sum(1 for r in rows_out if r['bio_tag'] == 'B-MATH')
    sents_with_math = len(set(r['sentence_id'] for r in rows_out if r['bio_tag'] != 'O'))

    print(f"Mathematicon: {total_files} files processed")
    print(f"  {total_sentences} sentences total (skipped {skipped_metadata} metadata)")
    print(f"  {n_tokens} tokens, {n_math} math tokens, {n_spans} math spans")
    print(f"  {sents_with_math} sentences with at least one math span")

    # Print samples
    print("\n--- Sample sentences ---")
    sample_sents = [s for s in unique_sents if any(
        r['bio_tag'] != 'O' for r in rows_out if r['sentence_id'] == s
    )][:3]

    for sid in sample_sents:
        toks = [(r['token'], r['bio_tag']) for r in rows_out if r['sentence_id'] == sid]
        print(f"\n{sid}:")
        for tok, tag in toks:
            marker = f' <{tag}>' if tag != 'O' else ''
            print(f"  {tok}{marker}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent.parent
    convert_mathematicon(
        base / 'data' / 'hse' / 'mathcorp-webtsv' / 'annotation',
        base / 'data' / 'bio' / 'bio_mathematicon.tsv'
    )
