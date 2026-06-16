#!/usr/bin/env python3
"""Merge all BIO TSV files and assign train/val/test splits.

Splits:
  - test:        50 sentences from Gosha (random, seed=42)
  - val:         31 sentences from Gosha (remaining)
  - train_clean: 185 synthetic sentences
  - train_noisy: all Mathematicon sentences with at least one math span

Input:  data/bio/bio_gosha.tsv, bio_synthetic.tsv, bio_mathematicon.tsv
Output: data/bio/bio_merged.tsv (with 'split' column)
"""

import csv
import random
from pathlib import Path


def load_tsv(path):
    """Load BIO TSV file, return list of row dicts."""
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows


def get_sentence_ids(rows):
    """Get unique sentence IDs in order."""
    seen = set()
    ids = []
    for r in rows:
        sid = r['sentence_id']
        if sid not in seen:
            seen.add(sid)
            ids.append(sid)
    return ids


def has_math_span(rows, sentence_id):
    """Check if a sentence has at least one math span (B-MATH tag)."""
    return any(r['bio_tag'] == 'B-MATH' for r in rows if r['sentence_id'] == sentence_id)


def merge_all(bio_dir, output_path):
    bio_dir = Path(bio_dir)

    # Load all sources
    gosha_rows = load_tsv(bio_dir / 'bio_gosha.tsv')
    synth_rows = load_tsv(bio_dir / 'bio_synthetic.tsv')
    math_rows = load_tsv(bio_dir / 'bio_mathematicon.tsv')

    # --- Assign splits ---

    # Gosha: 50 test + 31 val
    gosha_sids = get_sentence_ids(gosha_rows)
    random.seed(42)
    random.shuffle(gosha_sids)
    test_sids = set(gosha_sids[:50])
    val_sids = set(gosha_sids[50:])
    print(f"Gosha: {len(test_sids)} test, {len(val_sids)} val")

    # Synthetic: all train_clean
    synth_sids = set(get_sentence_ids(synth_rows))
    print(f"Synthetic: {len(synth_sids)} train_clean")

    # Mathematicon: only sentences with math spans → train_noisy
    math_all_sids = get_sentence_ids(math_rows)
    math_noisy_sids = set(
        sid for sid in math_all_sids if has_math_span(math_rows, sid)
    )
    print(f"Mathematicon: {len(math_noisy_sids)} train_noisy "
          f"(of {len(math_all_sids)} total)")

    # --- Merge with split column ---
    merged = []

    for r in gosha_rows:
        split = 'test' if r['sentence_id'] in test_sids else 'val'
        merged.append({**r, 'split': split})

    for r in synth_rows:
        merged.append({**r, 'split': 'train_clean'})

    for r in math_rows:
        if r['sentence_id'] in math_noisy_sids:
            merged.append({**r, 'split': 'train_noisy'})

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['sentence_id', 'token_index', 'token', 'bio_tag',
                          'source', 'split'])
        for r in merged:
            writer.writerow([r['sentence_id'], r['token_index'],
                             r['token'], r['bio_tag'], r['source'], r['split']])

    # --- Validation & stats ---
    split_counts = {}
    for r in merged:
        s = r['split']
        if s not in split_counts:
            split_counts[s] = {'sentences': set(), 'tokens': 0, 'math_tokens': 0,
                               'b_math': 0}
        split_counts[s]['sentences'].add(r['sentence_id'])
        split_counts[s]['tokens'] += 1
        if r['bio_tag'] != 'O':
            split_counts[s]['math_tokens'] += 1
        if r['bio_tag'] == 'B-MATH':
            split_counts[s]['b_math'] += 1

    print(f"\n{'Split':<14} {'Sents':>6} {'Tokens':>7} {'Math':>6} {'Spans':>6}")
    print('-' * 45)
    for split in ['test', 'val', 'train_clean', 'train_noisy']:
        c = split_counts.get(split, {'sentences': set(), 'tokens': 0,
                                      'math_tokens': 0, 'b_math': 0})
        print(f"{split:<14} {len(c['sentences']):>6} {c['tokens']:>7} "
              f"{c['math_tokens']:>6} {c['b_math']:>6}")

    total_sents = sum(len(c['sentences']) for c in split_counts.values())
    total_tokens = sum(c['tokens'] for c in split_counts.values())
    print(f"{'TOTAL':<14} {total_sents:>6} {total_tokens:>7}")

    # BIO consistency check
    print("\n--- BIO consistency check ---")
    errors = 0
    prev_tag = 'O'
    prev_sid = None
    for r in merged:
        if r['sentence_id'] != prev_sid:
            prev_tag = 'O'
            prev_sid = r['sentence_id']
        tag = r['bio_tag']
        if tag == 'I-MATH' and prev_tag == 'O':
            errors += 1
            if errors <= 3:
                print(f"  ERROR: I-MATH after O in {r['sentence_id']} "
                      f"token {r['token_index']}: '{r['token']}'")
        prev_tag = tag

    if errors == 0:
        print("  All BIO tags consistent!")
    else:
        print(f"  {errors} BIO consistency errors found")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent.parent
    merge_all(base / 'data' / 'bio',
              base / 'data' / 'bio' / 'bio_merged.tsv')
