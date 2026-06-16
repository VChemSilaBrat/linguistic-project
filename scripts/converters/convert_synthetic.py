#!/usr/bin/env python3
"""Convert synthetic dataset (dataset_clean.csv) to unified BIO TSV format.

Input:  data/mine/prompts/dataset_clean.csv
        Columns: sentence_clean, sentence_tagged, num_formulas, f1_pron, f1_latex, ...
        Tags: [MATH]...[/MATH] in sentence_tagged

Output: data/bio/bio_synthetic.tsv
"""

import csv
import re
from pathlib import Path


def parse_math_tags(tagged_sentence):
    """Parse [MATH]...[/MATH] tags and return list of (token, bio_tag) pairs."""
    # Split into segments: text outside MATH tags and text inside
    parts = re.split(r'\[MATH\]|\[/MATH\]', tagged_sentence)
    # Odd-indexed parts are inside MATH tags (0-indexed: 0=outside, 1=inside, 2=outside, ...)

    tokens_tags = []
    for i, part in enumerate(parts):
        is_math = (i % 2 == 1)
        words = part.split()
        for j, word in enumerate(words):
            if is_math:
                tag = 'B-MATH' if j == 0 else 'I-MATH'
            else:
                tag = 'O'
            tokens_tags.append((word, tag))

    return tokens_tags


def validate_bio(tokens_tags):
    """Check BIO consistency: no I-MATH after O without B-MATH."""
    for i, (tok, tag) in enumerate(tokens_tags):
        if tag == 'I-MATH' and (i == 0 or tokens_tags[i - 1][1] == 'O'):
            return False
    return True


def convert_synthetic(input_path, output_path):
    """Convert synthetic CSV to BIO TSV."""
    rows_out = []

    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for sent_idx, row in enumerate(reader):
            tagged = row['sentence_tagged']
            tokens_tags = parse_math_tags(tagged)

            if not validate_bio(tokens_tags):
                print(f"WARNING: BIO inconsistency in sentence {sent_idx}")

            for tok_idx, (token, tag) in enumerate(tokens_tags):
                rows_out.append({
                    'sentence_id': f'synth_{sent_idx:04d}',
                    'token_index': tok_idx,
                    'token': token,
                    'bio_tag': tag,
                    'source': 'synthetic'
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
    n_sentences = len(set(r['sentence_id'] for r in rows_out))
    n_tokens = len(rows_out)
    n_math = sum(1 for r in rows_out if r['bio_tag'] != 'O')
    n_spans = sum(1 for r in rows_out if r['bio_tag'] == 'B-MATH')
    print(f"Synthetic: {n_sentences} sentences, {n_tokens} tokens, "
          f"{n_math} math tokens, {n_spans} math spans")

    # Print samples
    print("\n--- Sample sentences ---")
    for sid in [f'synth_{i:04d}' for i in range(3)]:
        toks = [(r['token'], r['bio_tag']) for r in rows_out if r['sentence_id'] == sid]
        text_parts = []
        for tok, tag in toks:
            if tag == 'B-MATH':
                text_parts.append(f'[{tok}')
            elif tag == 'I-MATH':
                text_parts.append(tok)
            else:
                if text_parts and text_parts[-1] and not text_parts[-1].startswith('['):
                    pass
                text_parts.append(tok)
        # Reconstruct with span markers
        print(f"\n{sid}:")
        in_math = False
        for tok, tag in toks:
            if tag == 'B-MATH' and not in_math:
                print(f"  >>> MATH START")
                in_math = True
            elif tag == 'O' and in_math:
                print(f"  <<< MATH END")
                in_math = False
            marker = f' <{tag}>' if tag != 'O' else ''
            print(f"  {tok}{marker}")
        if in_math:
            print(f"  <<< MATH END")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent.parent
    convert_synthetic(base / 'data' / 'mine' / 'prompts' / 'dataset_clean.csv',
                      base / 'data' / 'bio' / 'bio_synthetic.tsv')
