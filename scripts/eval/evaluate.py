#!/usr/bin/env python3
"""Universal evaluation script for MathSpan BIO pipeline.

Metrics:
  - Span-level F1 (seqeval, strict)
  - Token-level F1 (B-MATH + I-MATH vs O)
  - Exact match (% sentences with all spans matched perfectly)

Usage:
  python scripts/eval/evaluate.py --pred predictions.tsv --gold data/bio/bio_merged.tsv --split test
  python scripts/eval/evaluate.py --pred predictions.tsv --gold data/bio/bio_merged.tsv  # all splits

Input format (both pred and gold):
  TSV with columns: sentence_id, token_index, token, bio_tag, [source, split]
  Pred file needs at minimum: sentence_id, token_index, bio_tag

Output: JSON with metrics + human-readable summary to stdout.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict


def load_bio_tsv(path, split_filter=None):
    """Load BIO TSV, return dict {sentence_id: [(token, bio_tag), ...]}."""
    sentences = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if split_filter and row.get('split', '') != split_filter:
                continue
            sid = row['sentence_id']
            tag = row['bio_tag']
            token = row.get('token', '')
            sentences[sid].append((token, tag))
    return dict(sentences)


def extract_spans(bio_tags):
    """Extract spans from BIO tag sequence. Returns list of (start, end) inclusive."""
    spans = []
    start = None
    for i, tag in enumerate(bio_tags):
        if tag == 'B-MATH':
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == 'O':
            if start is not None:
                spans.append((start, i - 1))
                start = None
        # I-MATH continues the current span
    if start is not None:
        spans.append((start, len(bio_tags) - 1))
    return spans


def span_f1(gold_spans, pred_spans):
    """Compute strict span-level precision, recall, F1."""
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


def token_f1(gold_tags, pred_tags):
    """Compute token-level precision, recall, F1 for MATH class."""
    tp = fp = fn = tn = 0
    for g, p in zip(gold_tags, pred_tags):
        g_math = g != 'O'
        p_math = p != 'O'
        if g_math and p_math:
            tp += 1
        elif not g_math and p_math:
            fp += 1
        elif g_math and not p_math:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def exact_match(gold_sentences, pred_sentences):
    """Compute exact match: % of sentences where all span boundaries match."""
    total = 0
    matches = 0
    for sid in gold_sentences:
        if sid not in pred_sentences:
            continue
        total += 1
        gold_tags = [t[1] for t in gold_sentences[sid]]
        pred_tags = [t[1] for t in pred_sentences[sid]]

        gold_spans = extract_spans(gold_tags)
        pred_spans = extract_spans(pred_tags)

        if gold_spans == pred_spans:
            matches += 1

    return matches / total if total > 0 else 0.0


def evaluate(gold_path, pred_path, split_filter=None):
    """Run full evaluation, return metrics dict."""
    gold = load_bio_tsv(gold_path, split_filter)
    pred = load_bio_tsv(pred_path, split_filter)

    # Align sentences
    common_sids = sorted(set(gold.keys()) & set(pred.keys()))
    if not common_sids:
        print("ERROR: No common sentence IDs between gold and pred!")
        print(f"  Gold has {len(gold)} sentences, pred has {len(pred)}")
        print(f"  Gold sample: {list(gold.keys())[:3]}")
        print(f"  Pred sample: {list(pred.keys())[:3]}")
        return None

    missing_in_pred = set(gold.keys()) - set(pred.keys())
    if missing_in_pred:
        print(f"WARNING: {len(missing_in_pred)} gold sentences missing from predictions")

    # Collect all tags and spans
    all_gold_tags = []
    all_pred_tags = []
    all_gold_spans = []
    all_pred_spans = []
    length_mismatches = 0

    for sid in common_sids:
        g_tags = [t[1] for t in gold[sid]]
        p_tags = [t[1] for t in pred[sid]]

        if len(g_tags) != len(p_tags):
            length_mismatches += 1
            # Truncate to min length
            min_len = min(len(g_tags), len(p_tags))
            g_tags = g_tags[:min_len]
            p_tags = p_tags[:min_len]

        all_gold_tags.extend(g_tags)
        all_pred_tags.extend(p_tags)

        g_spans = extract_spans(g_tags)
        p_spans = extract_spans(p_tags)
        all_gold_spans.extend(g_spans)
        all_pred_spans.extend(p_spans)

    # Compute metrics
    span_metrics = span_f1(all_gold_spans, all_pred_spans)
    token_metrics = token_f1(all_gold_tags, all_pred_tags)
    em = exact_match(gold, pred)

    results = {
        'split': split_filter or 'all',
        'n_sentences': len(common_sids),
        'n_tokens': len(all_gold_tags),
        'length_mismatches': length_mismatches,
        'span_f1': span_metrics,
        'token_f1': token_metrics,
        'exact_match': em
    }

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    if results is None:
        return

    print(f"\n{'='*50}")
    print(f"  Evaluation Results — split: {results['split']}")
    print(f"{'='*50}")
    print(f"  Sentences: {results['n_sentences']}")
    print(f"  Tokens:    {results['n_tokens']}")
    if results['length_mismatches'] > 0:
        print(f"  ⚠ Length mismatches: {results['length_mismatches']}")

    sf = results['span_f1']
    print(f"\n  Span-level:")
    print(f"    Precision: {sf['precision']:.4f}")
    print(f"    Recall:    {sf['recall']:.4f}")
    print(f"    F1:        {sf['f1']:.4f}")
    print(f"    (TP={sf['tp']}, FP={sf['fp']}, FN={sf['fn']})")

    tf = results['token_f1']
    print(f"\n  Token-level:")
    print(f"    Precision: {tf['precision']:.4f}")
    print(f"    Recall:    {tf['recall']:.4f}")
    print(f"    F1:        {tf['f1']:.4f}")
    print(f"    (TP={tf['tp']}, FP={tf['fp']}, FN={tf['fn']}, TN={tf['tn']})")

    print(f"\n  Exact Match: {results['exact_match']:.4f}")
    print(f"{'='*50}")


def self_test(gold_path, split_filter=None):
    """Run sanity tests: perfect predictions should give F1=1.0."""
    import tempfile
    import shutil

    gold = load_bio_tsv(gold_path, split_filter)
    if not gold:
        print("No data for self-test")
        return

    # Test 1: Perfect predictions
    print("\n--- Self-test: Perfect predictions ---")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False,
                                      encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        # Include split column so filtering works on pred too
        writer.writerow(['sentence_id', 'token_index', 'token', 'bio_tag', 'split'])
        for sid, tokens in gold.items():
            for i, (tok, tag) in enumerate(tokens):
                writer.writerow([sid, i, tok, tag, split_filter or ''])
        perfect_path = f.name

    results = evaluate(gold_path, perfect_path, split_filter)
    print_results(results)
    assert results['span_f1']['f1'] == 1.0, f"Perfect preds should give span-F1=1.0, got {results['span_f1']['f1']}"
    assert results['token_f1']['f1'] == 1.0, f"Perfect preds should give token-F1=1.0, got {results['token_f1']['f1']}"
    assert results['exact_match'] == 1.0, f"Perfect preds should give EM=1.0, got {results['exact_match']}"
    print("  ✓ Perfect prediction test PASSED")

    # Test 2: All-O predictions
    print("\n--- Self-test: All-O predictions ---")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False,
                                      encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['sentence_id', 'token_index', 'token', 'bio_tag', 'split'])
        for sid, tokens in gold.items():
            for i, (tok, tag) in enumerate(tokens):
                writer.writerow([sid, i, tok, 'O', split_filter or ''])
        all_o_path = f.name

    results = evaluate(gold_path, all_o_path, split_filter)
    print_results(results)

    # All-O should have 0 precision (no predictions), 0 recall, 0 F1
    has_math = any(tag != 'O' for tokens in gold.values() for _, tag in tokens)
    if has_math:
        assert results['span_f1']['f1'] == 0.0, f"All-O should give span-F1=0.0, got {results['span_f1']['f1']}"
        assert results['token_f1']['recall'] == 0.0, f"All-O should give token-recall=0.0"
        print("  ✓ All-O prediction test PASSED")
    else:
        print("  ⚠ No math tokens in gold, skipping All-O assertion")

    # Test 3: Random predictions
    print("\n--- Self-test: Random predictions ---")
    import random
    random.seed(42)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False,
                                      encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['sentence_id', 'token_index', 'token', 'bio_tag', 'split'])
        for sid, tokens in gold.items():
            prev = 'O'
            for i, (tok, tag) in enumerate(tokens):
                r = random.random()
                if r < 0.1:
                    pred_tag = 'B-MATH'
                elif r < 0.2 and prev != 'O':
                    pred_tag = 'I-MATH'
                else:
                    pred_tag = 'O'
                writer.writerow([sid, i, tok, pred_tag, split_filter or ''])
                prev = pred_tag
        random_path = f.name

    results = evaluate(gold_path, random_path, split_filter)
    print_results(results)
    print(f"  Random F1 ≈ {results['span_f1']['f1']:.3f} (expected near 0)")
    print("  ✓ Random prediction test PASSED")

    # Cleanup
    import os
    for p in [perfect_path, all_o_path, random_path]:
        os.unlink(p)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MathSpan BIO predictions')
    parser.add_argument('--pred', required=False, help='Predictions TSV file')
    parser.add_argument('--gold', required=True, help='Gold standard TSV file')
    parser.add_argument('--split', default=None, help='Filter by split (test/val/train_clean/train_noisy)')
    parser.add_argument('--json', default=None, help='Output JSON file path')
    parser.add_argument('--self-test', action='store_true', help='Run sanity self-tests')
    args = parser.parse_args()

    if args.self_test:
        self_test(args.gold, args.split)
        return

    if not args.pred:
        parser.error('--pred is required unless --self-test is used')

    results = evaluate(args.gold, args.pred, args.split)
    if results is None:
        sys.exit(1)

    print_results(results)

    if args.json:
        # Convert for JSON serialization
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.json}")


if __name__ == '__main__':
    main()
