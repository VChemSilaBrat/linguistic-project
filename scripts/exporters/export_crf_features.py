#!/usr/bin/env python3
"""
export_crf_features.py
Converts the unified BIO TSV (data/bio/bio_merged.tsv) into per-token CRF
feature dicts + label lists, ready for sklearn-crfsuite.
Usage:
    python scripts/exporters/export_crf_features.py \
        --input data/bio/bio_merged.tsv \
        --output data/exported/crf_features.pkl
Output: a pickle of
    {
        'train_clean': [(features_list, labels_list), ...],
        'train_noisy': [...],
        'val':         [...],
        'test':        [...],
    }
"""
import argparse
import csv
import pickle
import re
import sys
import os
from collections import defaultdict

EXPECTED_SPLITS = ["train_clean", "train_noisy", "val", "test"]

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
    'дробь', 'дроби', 'числитель', 'знаменатель',
    'сумма', 'разность', 'произведение', 'частное',
    'уравнение', 'уравнения', 'неравенство', 'неравенства',
    'множество', 'множества', 'подмножество',
    'факториал', 'бином', 'перестановка', 'сочетание',
    'синус', 'косинус', 'тангенс', 'котангенс',
    'арксинус', 'арккосинус', 'арктангенс',
    'экспонента', 'экспоненты',
    # expansion
    'график', 'графика', 'графики', 'отрезок', 'отрезка', 'интервал', 'интервала',
    'точка', 'точки', 'прямая', 'прямой', 'окружность', 'окружности',
    'треугольник', 'треугольника', 'площадь', 'площади', 'объем', 'объём',
    'вероятность', 'вероятности', 'среднее', 'коэффициент', 'коэффициента',
    'переменная', 'переменной', 'выражение', 'выражения', 'формула', 'формулы',
    'дифференциал', 'дифференциала', 'дифференцирование', 'интегрирование',
    'координата', 'координаты', 'координат', 'плоскость', 'плоскости',
    'гипотенуза', 'катет', 'катета', 'радиус', 'диаметр', 'периметр',
    'делитель', 'делителя', 'делимое', 'остаток', 'остатка',
    'минимум', 'максимум', 'экстремум', 'асимптота',
}

PUNCT_STRIP_RE = re.compile(r'^[^\w]+|[^\w]+$', re.UNICODE)
LATIN_RE = re.compile(r'^[a-zA-Z]+$')
DIGIT_RE = re.compile(r'^\d+$')


def strip_punct(token):
    return PUNCT_STRIP_RE.sub('', token)


def is_math_word(token, math_words=MATH_WORDS):
    cleaned = strip_punct(token).lower()
    return cleaned in math_words


def word_shape(token):
    out = []
    for ch in token:
        if ch.isupper():
            out.append('X')
        elif ch.islower():
            out.append('x')
        elif ch.isdigit():
            out.append('d')
        else:
            out.append(ch)
    return ''.join(out)


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
            assert bio_tag in ("O", "B-MATH", "I-MATH"), (
                f"Unexpected bio_tag '{bio_tag}' for sentence_id={sid}, token_index={tok_idx}"
            )
            if sid not in sent_split:
                sent_split[sid] = split
                order_per_split[split].append(sid)
            else:
                assert sent_split[sid] == split, f"Inconsistent split for sentence_id={sid}"
            raw[sid].append((tok_idx, token, bio_tag))
            seen_sentence_ids.add(sid)

    sentences = defaultdict(list)
    for split, sids in order_per_split.items():
        for sid in sids:
            rows = sorted(raw[sid], key=lambda r: r[0])
            tokens = [r[1] for r in rows]
            labels = [r[2] for r in rows]
            sentences[split].append({"sentence_id": sid, "tokens": tokens, "labels": labels})

    return sentences, n_rows, seen_sentence_ids


def build_pos_tagger():
    try:
        import stanza
    except Exception as e:
        return None, f"Stanza could not be imported ({type(e).__name__}: {e}); POS tagging disabled, postag='UNK' for all tokens."

    try:
        try:
            nlp = stanza.Pipeline(
                lang="ru",
                processors="tokenize,pos",
                tokenize_pretokenized=True,
                verbose=False,
            )
        except Exception:
            stanza.download("ru")
            nlp = stanza.Pipeline(
                lang="ru",
                processors="tokenize,pos",
                tokenize_pretokenized=True,
                verbose=False,
            )
    except Exception as e:
        return None, f"Stanza ru model could not be loaded/downloaded ({e}); POS tagging disabled, postag='UNK' for all tokens."

    def tag_fn(sentences_tokens):
        results = []
        for tokens in sentences_tokens:
            if not tokens:
                results.append([])
                continue
            try:
                text = " ".join(tokens)
                doc = nlp(text)
                words = [w for sent in doc.sentences for w in sent.words]
                if len(words) == len(tokens):
                    results.append([w.upos if w.upos else "UNK" for w in words])
                else:
                    results.append(["UNK"] * len(tokens))
            except Exception:
                results.append(["UNK"] * len(tokens))
        return results

    return tag_fn, "Stanza ru POS tagging active."


def base_token_features(token, postag):
    return {
        'bias': 1.0,
        'word.lower': token.lower(),
        'word[-3:]': token[-3:],
        'word[:3]': token[:3],
        'word.is_digit': bool(DIGIT_RE.match(token)),
        'word.is_latin': bool(LATIN_RE.match(token)),
        'word.is_upper': token.isupper(),
        'word.is_title': token[:1].isupper() if token else False,
        'word.len': len(token),
        'word.is_math_word': is_math_word(token),
        'word.shape': word_shape(token),
        'postag': postag,
    }


def build_sentence_features(tokens, postags):
    n = len(tokens)
    base = [base_token_features(tokens[i], postags[i]) for i in range(n)]
    is_math_flags = [base[i]['word.is_math_word'] for i in range(n)]

    sentence_features = []
    for i in range(n):
        feat = dict(base[i])
        feat['BOS'] = (i == 0)
        feat['EOS'] = (i == n - 1)

        for offset in (-2, -1, 1, 2):
            j = i + offset
            prefix = f"{offset}:"
            if 0 <= j < n:
                feat[f"{prefix}word.lower"] = tokens[j].lower()
                feat[f"{prefix}postag"] = postags[j]
                feat[f"{prefix}is_math_word"] = is_math_flags[j]
            else:
                marker = "BOS" if j < 0 else "EOS"
                feat[f"{prefix}word.lower"] = marker
                feat[f"{prefix}postag"] = marker
                feat[f"{prefix}is_math_word"] = False

        sentence_features.append(feat)

    return sentence_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to bio_merged.tsv")
    parser.add_argument("--output", required=True, help="Output pickle path")
    args = parser.parse_args()

    sentences, n_rows, seen_sentence_ids = read_bio_tsv(args.input)

    found_splits = set(sentences.keys())
    unexpected = found_splits - set(EXPECTED_SPLITS)
    assert not unexpected, f"Unexpected split values found in input: {unexpected}"

    tag_fn, pos_status = build_pos_tagger()
    print(f"[POS] {pos_status}")

    output = {}
    n_sentences_total = 0
    n_tokens_total = 0
    written_sentence_ids = set()

    for split in EXPECTED_SPLITS:
        examples = sentences.get(split, [])
        if not examples:
            print(f"WARNING: split '{split}' has 0 sentences.", file=sys.stderr)
            output[split] = []
            continue

        all_tokens = [ex["tokens"] for ex in examples]
        if tag_fn is not None:
            all_postags = tag_fn(all_tokens)
        else:
            all_postags = [["UNK"] * len(toks) for toks in all_tokens]

        split_out = []
        for ex, postags in zip(examples, all_postags):
            assert len(postags) == len(ex["tokens"])
            feats = build_sentence_features(ex["tokens"], postags)
            assert len(feats) == len(ex["labels"]) == len(ex["tokens"])
            split_out.append((feats, ex["labels"]))
            written_sentence_ids.add(ex["sentence_id"])
            n_sentences_total += 1
            n_tokens_total += len(ex["tokens"])

        output[split] = split_out

    assert written_sentence_ids == seen_sentence_ids

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    print("=== export_crf_features.py summary ===")
    print(f"Input rows (tokens) read: {n_rows}")
    print(f"Unique sentence_ids: {len(seen_sentence_ids)}")
    for split in EXPECTED_SPLITS:
        n_sent = len(output[split])
        n_tok = sum(len(f) for f, _ in output[split])
        print(f"  split={split}: sentences={n_sent}, tokens={n_tok}")
    print(f"Total sentences across splits: {n_sentences_total}")
    print(f"Total tokens across splits: {n_tokens_total}")
    print(f"Saved CRF feature pickle to: {args.output}")


if __name__ == "__main__":
    main()
