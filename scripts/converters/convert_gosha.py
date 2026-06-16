#!/usr/bin/env python3
"""Convert Gosha's binary-mask CSV to unified BIO TSV format.

Input:  data/gosha/train.csv  (sentence, mask)
Output: data/bio/bio_gosha.tsv

Gosha's mask uses a custom tokenization where punctuation (commas, periods,
etc.) is sometimes split off as separate tokens, and alphanumeric compounds
like "2Y" may be kept as one token or split. We use razdel as a base
tokenizer (which oversplits), then DP-align to the mask length, choosing
which punct/compound tokens to merge back based on mask-value consistency.
"""

import csv
import re
from pathlib import Path
from razdel import tokenize as razdel_tokenize


MATH_WORDS = {
    'минус', 'плюс', 'больше', 'меньше', 'равно', 'равняться', 'равен', 'равна',
    'деленное', 'умножить', 'умноженное', 'деленная', 'делённое',
    'квадрат', 'куб', 'корень', 'степени', 'степень', 'модуль', 'модуля', 'модулю',
    'интеграл', 'производная', 'производной', 'предел', 'пределом', 'предела',
    'sin', 'cos', 'tg', 'ctg', 'ln', 'log',
    'скалярно', 'параллельная',
    'лагарифм', 'логарифм',
    'функция', 'функций', 'функции',
    'дельта', 'эпсилон', 'эпселен', 'лямбда', 'альфа', 'тета', 'сигма',
    'вектор', 'вектора', 'векторы', 'матрица',
}


def is_math_like(token):
    """Score how likely a token is to be part of a math expression."""
    t = token.strip('.,!?;:²³')
    if not t:
        return -10
    if re.match(r'^[0-9]+$', t):
        return 5
    if re.match(r'^[a-zA-Z][0-9]?$', t):
        return 5
    if (re.match(r'^[0-9]+[a-zA-Zа-яА-ЯёЁ]{1,2}$', t) or
            re.match(r'^[a-zA-Zа-яА-ЯёЁ]{1,2}[0-9]+$', t)):
        return 4
    if t.lower() in MATH_WORDS:
        return 3
    if len(t) <= 2 and t in ('и', 'от', 'на', 'к', 'до', 'в', 'не'):
        return 0
    return -2


def is_punct_token(tok):
    return bool(re.match(r'^[,\.!?;:]+$', tok))


def align_razdel_to_mask(text, mask):
    """Align razdel tokens to mask by DP-selecting which punct to merge back.

    razdel oversplits (all punct separate). We merge some back so that
    len(tokens) == len(mask), choosing merges that keep math tokens clean.
    """
    rtokens = [t.text for t in razdel_tokenize(text)]
    target = len(mask)
    N = len(rtokens)

    if N == target:
        return rtokens
    if N < target:
        return None

    # Identify which tokens can be merged into their predecessor
    mergeable = [False] * N
    for i in range(1, N):
        prev, curr = rtokens[i - 1], rtokens[i]
        # Punct after word
        if is_punct_token(curr) and not is_punct_token(prev):
            mergeable[i] = True
        # Short alphanumeric compounds: "2"+"Y", "a"+"1"
        if len(prev) <= 3 and len(curr) <= 3:
            if (re.match(r'^[0-9]+$', prev) and
                    re.match(r'^[a-zA-Zа-яА-ЯёЁ]+$', curr)):
                mergeable[i] = True
            if (re.match(r'^[a-zA-Zа-яА-ЯёЁ]+$', prev) and
                    re.match(r'^[0-9]+$', curr)):
                mergeable[i] = True

    # DP: dp[i][j] = best score reaching razdel pos i, mask pos j
    INF = float('-inf')
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    parent = [[None] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0

    for i in range(N):
        for j in range(target + 1):
            if dp[i][j] == INF:
                continue

            # Option 1: map token i to mask position j
            if j < target:
                score = dp[i][j]
                ms = is_math_like(rtokens[i])
                if mask[j] == '1':
                    score += ms  # math-like at mask=1: good
                elif ms > 2:
                    score -= ms  # strong math word at mask=0: bad

                if score > dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = score
                    parent[i + 1][j + 1] = ('map', i, j)

            # Option 2: merge token i into previous (doesn't consume mask pos)
            if mergeable[i]:
                score = dp[i][j]
                # Penalize merging punct into math tokens
                if j > 0 and mask[j - 1] == '1' and is_punct_token(rtokens[i]):
                    score -= 8
                elif j > 0 and mask[j - 1] == '0' and is_punct_token(rtokens[i]):
                    score += 1

                if score > dp[i + 1][j]:
                    dp[i + 1][j] = score
                    parent[i + 1][j] = ('merge', i, j)

    if dp[N][target] == INF:
        return None

    # Backtrack to find merged tokens
    merged_set = set()
    i, j = N, target
    while i > 0:
        action, pi, pj = parent[i][j]
        if action == 'merge':
            merged_set.add(pi)
        i, j = pi, pj

    # Build result
    result = []
    for i in range(N):
        if i in merged_set:
            if result:
                result[-1] += rtokens[i]
        else:
            result.append(rtokens[i])

    return result


def mask_to_bio(mask_str):
    """Convert binary mask string to BIO tag list."""
    tags = []
    for i, bit in enumerate(mask_str):
        if bit == '1':
            if i == 0 or mask_str[i - 1] == '0':
                tags.append('B-MATH')
            else:
                tags.append('I-MATH')
        else:
            tags.append('O')
    return tags


def convert_gosha(input_path, output_path):
    """Convert Gosha CSV to BIO TSV."""
    rows_out = []
    errors = []

    with open(input_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for sent_idx, row in enumerate(reader):
            sentence = row['sentence']
            mask = row['mask']

            tokens = align_razdel_to_mask(sentence, mask)
            if tokens is None or len(tokens) != len(mask):
                errors.append(sent_idx)
                print(f"WARNING: sent {sent_idx} alignment failed "
                      f"(got {len(tokens) if tokens else 'None'}, "
                      f"expected {len(mask)})")
                # Fallback: space-split with proportional alignment
                tokens = sentence.split()
                bio_tags = []
                n_w, n_m = len(tokens), len(mask)
                for i in range(n_w):
                    start = round(i * n_m / n_w)
                    end = round((i + 1) * n_m / n_w)
                    bits = [int(mask[j]) for j in range(start, min(end, n_m))]
                    label = 1 if sum(bits) > len(bits) / 2 else 0
                    if label == 1:
                        if i == 0 or bio_tags[-1] == 'O':
                            bio_tags.append('B-MATH')
                        else:
                            bio_tags.append('I-MATH')
                    else:
                        bio_tags.append('O')
            else:
                bio_tags = mask_to_bio(mask)

            for tok_idx, (token, tag) in enumerate(zip(tokens, bio_tags)):
                rows_out.append({
                    'sentence_id': f'gosha_{sent_idx:04d}',
                    'token_index': tok_idx,
                    'token': token,
                    'bio_tag': tag,
                    'source': 'gosha'
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
    print(f"Gosha: {n_sentences} sentences, {n_tokens} tokens, "
          f"{n_math} math tokens, {n_spans} math spans")
    if errors:
        print(f"  WARNING: {len(errors)} sentences used fallback: {errors}")

    # Print samples
    print("\n--- Sample sentences ---")
    for sid in [f'gosha_{i:04d}' for i in range(3)]:
        toks = [(r['token'], r['bio_tag']) for r in rows_out
                if r['sentence_id'] == sid]
        print(f"\n{sid}:")
        for tok, tag in toks:
            marker = f' <{tag}>' if tag != 'O' else ''
            print(f"  {tok}{marker}")


if __name__ == '__main__':
    base = Path(__file__).resolve().parent.parent.parent
    convert_gosha(base / 'data' / 'gosha' / 'train.csv',
                  base / 'data' / 'bio' / 'bio_gosha.tsv')
