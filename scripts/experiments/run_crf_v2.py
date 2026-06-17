#!/usr/bin/env python3
"""
run_crf_v2.py — CRF-2..CRF-5 experiments for MathSpan BIO labeling, testing
whether LLM-generated training data (train_llm, 509 sentences) helps CRF the
way it helped ruBERT (BERT-4: span-F1 0.3287 vs. curriculum-with-Mathematicon
configs which underperformed).

This is a COPY of run_crf.py (do not edit run_crf.py) with the training-data
configuration changed: instead of one fixed train_clean+train_noisy training
set (CRF-0/CRF-1), this script runs 4 experiments varying training data and
POS usage:

    CRF-2: train_clean + train_llm                  (no POS)
    CRF-3: train_clean + train_llm                  (+POS)
    CRF-4: train_clean + train_llm + train_noisy     (no POS)
    CRF-5: train_clean + train_llm + train_noisy     (+POS)

All other infrastructure (vocab building, feature extraction, metrics) is
unchanged from run_crf.py. Old CRF-0 (test span-F1=0.0738) and CRF-1 (test
span-F1=0.0977) results are hardcoded into the final comparison table since
they were produced by a separate run.

Run from project root:
    python scripts/experiments/run_crf_v2.py
"""
import csv
import os
import re
import sys
import time
import yaml
from collections import Counter, defaultdict

import sklearn_crfsuite
from sklearn.metrics import f1_score as sk_f1_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.scheme import IOB2

# --------------------------------------------------------------------------
# Paths (relative to project root)
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIO_PATH = os.path.join(PROJECT_ROOT, "data", "bio", "bio_merged.tsv")
MARKERS_PATH = os.path.join(PROJECT_ROOT, "data", "dima", "markers.yaml")
S2L_PATH = os.path.join(PROJECT_ROOT, "data", "s2l", "russian_sentences_train.csv")

CONF_ORDINAL = {"low": 1, "medium": 2, "high": 3}
RUSSIAN_STOPWORDS = {
    "в", "на", "от", "и", "с", "по", "к", "до", "из", "при", "для", "за",
    "не", "ни", "что", "как", "это", "то", "бы", "же", "ли", "но", "а",
    "у", "о", "е",
}
REGEX_CHARS = set("[]\\()|+?*{}")

PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
LATIN_RE = re.compile(r"^[a-zA-Z]+$")
DIGIT_RE = re.compile(r"^\d+$")


def clean_word(token):
    return PUNCT_STRIP_RE.sub("", token).lower()


# --------------------------------------------------------------------------
# 1. Vocabulary construction
# --------------------------------------------------------------------------

def build_markers_vocab(path):
    """Parse markers.yaml -> {word: {role: confidence_ordinal}}. Returns
    (vocab, roles_found, fallback_used)."""
    fallback_used = False
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        entries = data["entries"]
    except Exception as e:
        print(f"WARNING: YAML parse of markers.yaml failed ({e}); "
              f"falling back to simple line-by-line pattern extraction.", file=sys.stderr)
        fallback_used = True
        entries = _fallback_parse_markers(path)

    vocab = defaultdict(dict)  # word -> {role: max_conf_ordinal}
    roles_found = set()

    for entry in entries:
        role = entry.get("role", "NONE")
        if role == "NEGATIVE_CONTEXT":
            continue
        conf = CONF_ORDINAL.get(entry.get("confidence", "low"), 1)
        patterns = entry.get("patterns", []) or []
        for pattern in patterns:
            if any(c in pattern for c in REGEX_CHARS):
                continue  # skip regex patterns
            for raw_word in pattern.split():
                w = clean_word(raw_word)
                if len(w) < 2:
                    continue
                roles_found.add(role)
                prev = vocab[w].get(role, 0)
                if conf > prev:
                    vocab[w][role] = conf

    return dict(vocab), roles_found, fallback_used


def _fallback_parse_markers(path):
    """Very simple fallback: scan lines for 'patterns:' lists and 'role:' values."""
    entries = []
    cur_role = None
    cur_patterns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("role:"):
                cur_role = line.split(":", 1)[1].strip()
            elif line.startswith("- \"") or line.startswith('- "'):
                cur_patterns.append(line.split("- ", 1)[1].strip().strip('"'))
            elif line.startswith("- id:"):
                if cur_role and cur_patterns:
                    entries.append({"role": cur_role, "patterns": cur_patterns, "confidence": "medium"})
                cur_role = None
                cur_patterns = []
    if cur_role and cur_patterns:
        entries.append({"role": cur_role, "patterns": cur_patterns, "confidence": "medium"})
    return entries


def build_s2l_vocab(path):
    """Read s2l pronunciations -> word frequency Counter, filtered."""
    counts = Counter()
    n_rows = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "pronunciation" in reader.fieldnames, \
            f"s2l CSV missing 'pronunciation' column, got {reader.fieldnames}"
        for row in reader:
            n_rows += 1
            for raw_word in row["pronunciation"].split():
                w = clean_word(raw_word)
                if len(w) >= 2:
                    counts[w] += 1

    kept = {w for w, c in counts.items() if c >= 50 and w not in RUSSIAN_STOPWORDS}
    return kept, n_rows


def build_combined_vocab(markers_vocab, s2l_words):
    """Union vocab. Markers words keep their role/conf map; s2l-only words get role S2L_MATH."""
    combined = dict(markers_vocab)  # word -> {role: conf}
    n_s2l_only = 0
    for w in s2l_words:
        if w not in combined:
            combined[w] = {"S2L_MATH": 1}
            n_s2l_only += 1
    return combined, n_s2l_only


def get_math_role(word_clean, vocab):
    entry = vocab.get(word_clean)
    if not entry:
        return "NONE"
    best_conf = max(entry.values())
    candidates = sorted(role for role, conf in entry.items() if conf == best_conf)
    return candidates[0]


def is_math_word(word_clean, vocab):
    return word_clean in vocab


# --------------------------------------------------------------------------
# 2. BIO TSV loading
# --------------------------------------------------------------------------

EXPECTED_SPLITS = ["train_clean", "train_noisy", "train_llm", "val", "test"]


def load_bio_tsv(path):
    raw = defaultdict(list)
    sent_split = {}
    order_per_split = defaultdict(list)

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sentence_id", "token_index", "token", "bio_tag", "source", "split"}
        missing = required - set(reader.fieldnames or [])
        assert not missing, f"bio_merged.tsv missing columns: {missing}"

        for row in reader:
            sid = row["sentence_id"]
            split = row["split"]
            tok_idx = int(row["token_index"])
            token = row["token"]
            tag = row["bio_tag"]
            assert tag in ("O", "B-MATH", "I-MATH"), f"Unexpected bio_tag '{tag}' at {sid}:{tok_idx}"
            if sid not in sent_split:
                sent_split[sid] = split
                order_per_split[split].append(sid)
            raw[sid].append((tok_idx, token, tag))

    sentences = defaultdict(list)
    for split, sids in order_per_split.items():
        for sid in sids:
            rows = sorted(raw[sid], key=lambda r: r[0])
            sentences[split].append({
                "sentence_id": sid,
                "tokens": [r[1] for r in rows],
                "labels": [r[2] for r in rows],
            })
    return sentences


# --------------------------------------------------------------------------
# 3. Stanza POS tagging (with graceful fallback)
# --------------------------------------------------------------------------

def build_pos_tagger():
    try:
        import stanza
    except Exception as e:
        return None, f"Stanza not available ({type(e).__name__}: {e}). POS-enabled configs will use postag='UNK' for all tokens."

    try:
        try:
            nlp = stanza.Pipeline(lang="ru", processors="tokenize,pos",
                                   tokenize_pretokenized=True, verbose=False)
        except Exception:
            stanza.download("ru")
            nlp = stanza.Pipeline(lang="ru", processors="tokenize,pos",
                                   tokenize_pretokenized=True, verbose=False)
    except Exception as e:
        return None, f"Stanza ru model could not be loaded/downloaded ({e}). POS-enabled configs will use postag='UNK' for all tokens."

    misaligned = [0]

    def tag_fn(tokens):
        if not tokens:
            return []
        try:
            doc = nlp([tokens])
            words = [w for sent in doc.sentences for w in sent.words]
            if len(words) == len(tokens):
                return [w.upos if w.upos else "UNK" for w in words]
            misaligned[0] += 1
            return ["UNK"] * len(tokens)
        except Exception:
            misaligned[0] += 1
            return ["UNK"] * len(tokens)

    return (tag_fn, misaligned), "Stanza ru POS tagging active."


# --------------------------------------------------------------------------
# 4. Feature extraction
# --------------------------------------------------------------------------

def word_shape(token):
    out = []
    for ch in token:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)


def base_features(token, vocab):
    w_clean = clean_word(token)
    return {
        "bias": 1.0,
        "word.lower": token.lower(),
        "word[-3:]": token[-3:],
        "word[:3]": token[:3],
        "word.is_digit": bool(DIGIT_RE.match(token)),
        "word.is_latin": bool(LATIN_RE.match(token)),
        "word.is_upper": token.isupper(),
        "word.is_title": bool(token[:1].isupper()) if token else False,
        "word.len": len(token),
        "word.is_math_word": is_math_word(w_clean, vocab),
        "word.math_role": get_math_role(w_clean, vocab),
        "word.shape": word_shape(token),
    }


def build_sentence_features(tokens, vocab, postags=None):
    n = len(tokens)
    base = [base_features(t, vocab) for t in tokens]
    feats_out = []
    for i in range(n):
        feat = dict(base[i])
        feat["BOS"] = (i == 0)
        feat["EOS"] = (i == n - 1)
        if postags is not None:
            feat["postag"] = postags[i]

        for offset in (-2, -1, 1, 2):
            j = i + offset
            pre = f"{offset}:"
            if 0 <= j < n:
                feat[f"{pre}word.lower"] = tokens[j].lower()
                feat[f"{pre}is_math_word"] = base[j]["word.is_math_word"]
                feat[f"{pre}math_role"] = base[j]["word.math_role"]
                if postags is not None:
                    feat[f"{pre}postag"] = postags[j]
            else:
                marker = "BOS" if j < 0 else "EOS"
                feat[f"{pre}word.lower"] = marker
                feat[f"{pre}is_math_word"] = False
                feat[f"{pre}math_role"] = marker
                if postags is not None:
                    feat[f"{pre}postag"] = marker
        feats_out.append(feat)
    return feats_out


# --------------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------------

def extract_spans(tags):
    """Returns list of (start, end) inclusive index tuples for math spans."""
    spans = []
    start = None
    for i, tag in enumerate(tags):
        if tag == "B-MATH":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "O":
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def compute_metrics(gold_label_lists, pred_label_lists):
    span_f1 = seq_f1_score(gold_label_lists, pred_label_lists, mode="strict", scheme=IOB2)

    gold_bin, pred_bin = [], []
    for g, p in zip(gold_label_lists, pred_label_lists):
        gold_bin.extend([0 if t == "O" else 1 for t in g])
        pred_bin.extend([0 if t == "O" else 1 for t in p])
    token_f1 = sk_f1_score(gold_bin, pred_bin, average="binary", pos_label=1, zero_division=0)

    n_match = sum(1 for g, p in zip(gold_label_lists, pred_label_lists) if g == p)
    exact_match = n_match / len(gold_label_lists) if gold_label_lists else 0.0

    return span_f1, token_f1, exact_match


# --------------------------------------------------------------------------
# 6. Experiment configs
# --------------------------------------------------------------------------

# (exp_id, training splits to concatenate, use_pos)
EXPERIMENTS = [
    ("CRF-2", ["train_clean", "train_llm"], False),
    ("CRF-3", ["train_clean", "train_llm"], True),
    ("CRF-4", ["train_clean", "train_llm", "train_noisy"], False),
    ("CRF-5", ["train_clean", "train_llm", "train_noisy"], True),
]

# Hardcoded prior results from run_crf.py (E5), produced in a separate run.
OLD_RESULTS = {
    "CRF-0": {"training_data": "train_clean+train_noisy", "pos": False,
              "val_span_f1": None, "test_span_f1": 0.0738, "test_token_f1": None, "test_em": None},
    "CRF-1": {"training_data": "train_clean+train_noisy", "pos": True,
              "val_span_f1": None, "test_span_f1": 0.0977, "test_token_f1": None, "test_em": None},
}


def fmt(v, spec=".4f"):
    return "—" if v is None else format(v, spec)


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main():
    t0 = time.time()

    # --- Vocabulary ---
    markers_vocab, roles_found, markers_fallback = build_markers_vocab(MARKERS_PATH)
    n_markers_words = len(markers_vocab)

    s2l_words, n_s2l_rows = build_s2l_vocab(S2L_PATH)
    n_s2l_words = len(s2l_words)

    combined_vocab, n_s2l_only = build_combined_vocab(markers_vocab, s2l_words)
    n_combined = len(combined_vocab)

    print(f"[VOCAB] markers.yaml words: {n_markers_words} (fallback parser used: {markers_fallback})")
    print(f"[VOCAB] markers.yaml roles found: {sorted(roles_found)}")
    print(f"[VOCAB] s2l rows scanned: {n_s2l_rows}, s2l words kept (freq>=50): {n_s2l_words}")
    print(f"[VOCAB] s2l-only additions (role=S2L_MATH): {n_s2l_only}")
    print(f"[VOCAB] combined vocabulary size: {n_combined}")

    # --- BIO data ---
    sentences = load_bio_tsv(BIO_PATH)
    for split in EXPECTED_SPLITS:
        assert sentences.get(split), f"Split '{split}' is empty or missing in bio_merged.tsv"

    val_examples = sentences["val"]
    test_examples = sentences["test"]

    print(f"[DATA] train_clean: {len(sentences['train_clean'])} sentences")
    print(f"[DATA] train_llm: {len(sentences['train_llm'])} sentences")
    print(f"[DATA] train_noisy: {len(sentences['train_noisy'])} sentences")
    print(f"[DATA] val: {len(val_examples)} sentences")
    print(f"[DATA] test: {len(test_examples)} sentences")

    # --- POS tagging (once per split, reused across experiments) ---
    pos_result, pos_status = build_pos_tagger()
    print(f"[POS] {pos_status}")
    misaligned_count = 0

    pos_splits = ["train_clean", "train_llm", "train_noisy", "val", "test"]
    postags_by_split = {}
    if pos_result is not None:
        tag_fn, misaligned_ref = pos_result
        t_pos = time.time()
        for split in pos_splits:
            postags_by_split[split] = [tag_fn(ex["tokens"]) for ex in sentences[split]]
        misaligned_count = misaligned_ref[0]
        print(f"[POS] tagging done in {time.time() - t_pos:.1f}s, misaligned sentences: {misaligned_count}")
    else:
        for split in pos_splits:
            postags_by_split[split] = [["UNK"] * len(ex["tokens"]) for ex in sentences[split]]

    y_val = [ex["labels"] for ex in val_examples]
    y_test = [ex["labels"] for ex in test_examples]

    # --- Run the 4 experiments ---
    results = {}  # exp_id -> dict of everything needed for reporting
    for exp_id, train_splits, use_pos in EXPERIMENTS:
        train_examples = []
        train_postags = []
        for split in train_splits:
            train_examples.extend(sentences[split])
            train_postags.extend(postags_by_split[split])
        y_train = [ex["labels"] for ex in train_examples]

        X_train = [
            build_sentence_features(ex["tokens"], combined_vocab, pt if use_pos else None)
            for ex, pt in zip(train_examples, train_postags)
        ]
        X_val = [
            build_sentence_features(ex["tokens"], combined_vocab,
                                     pt if use_pos else None)
            for ex, pt in zip(val_examples, postags_by_split["val"])
        ]
        X_test = [
            build_sentence_features(ex["tokens"], combined_vocab,
                                     pt if use_pos else None)
            for ex, pt in zip(test_examples, postags_by_split["test"])
        ]

        t_fit = time.time()
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1,
                                    max_iterations=200, all_possible_transitions=True)
        crf.fit(X_train, y_train)
        fit_time = time.time() - t_fit

        pred_val = crf.predict(X_val)
        pred_test = crf.predict(X_test)
        val_span, val_token, val_em = compute_metrics(y_val, pred_val)
        test_span, test_token, test_em = compute_metrics(y_test, pred_test)

        results[exp_id] = {
            "training_data": "+".join(train_splits),
            "n_train": len(train_examples),
            "pos": use_pos,
            "model": crf,
            "fit_time": fit_time,
            "val_span_f1": val_span, "val_token_f1": val_token, "val_em": val_em,
            "test_span_f1": test_span, "test_token_f1": test_token, "test_em": test_em,
            "pred_test": pred_test,
        }
        print(f"[{exp_id}] train={results[exp_id]['n_train']} sent, pos={use_pos}, "
              f"fit={fit_time:.2f}s, val_span_f1={val_span:.4f}, test_span_f1={test_span:.4f}")

    # --- Pick best model by test span-F1 among the 4 new experiments ---
    best_id = max(results, key=lambda k: results[k]["test_span_f1"])
    best = results[best_id]
    best_crf = best["model"]
    best_pred_test = best["pred_test"]

    # --- Feature importance (best model) ---
    state_feats = best_crf.state_features_  # {(attr, label): weight}
    relevant = [(attr, label, w) for (attr, label), w in state_feats.items()
                if label in ("B-MATH", "I-MATH")]
    relevant.sort(key=lambda x: x[2], reverse=True)
    top20 = relevant[:20]

    # --- Error analysis (best model, test) ---
    errors = []
    for ex, pred in zip(test_examples, best_pred_test):
        if pred != ex["labels"]:
            errors.append((ex, pred))
    error_sample = errors[:10]

    def spans_to_text(spans, tokens):
        return [" ".join(tokens[s:e + 1]) for s, e in spans]

    # --- Role analysis in errors (across all test errors, not just sample) ---
    fn_role_counts = Counter()
    fp_role_counts = Counter()
    for ex, pred in zip(test_examples, best_pred_test):
        gold_spans = set(extract_spans(ex["labels"]))
        pred_spans = set(extract_spans(pred))
        fn_spans = gold_spans - pred_spans
        fp_spans = pred_spans - gold_spans
        for s, e in fn_spans:
            for tok in ex["tokens"][s:e + 1]:
                fn_role_counts[get_math_role(clean_word(tok), combined_vocab)] += 1
        for s, e in fp_spans:
            for tok in ex["tokens"][s:e + 1]:
                fp_role_counts[get_math_role(clean_word(tok), combined_vocab)] += 1

    total_time = time.time() - t0

    # --------------------------------------------------------------------
    # Output
    # --------------------------------------------------------------------
    out = []
    out.append('<executor_output id="E5b">')
    out.append("## CRF v2 Experiment Results (with LLM-generated data)")
    out.append("")
    out.append("### Vocabulary stats")
    out.append(f"- markers.yaml words (non-regex, non-NEGATIVE): {n_markers_words}")
    out.append(f"- markers.yaml roles: {sorted(roles_found)}")
    out.append(f"- s2l words (freq>=50, cleaned): {n_s2l_words}")
    out.append(f"- Combined vocabulary size: {n_combined}")
    out.append("")
    out.append("### Data stats")
    out.append(f"- Stanza POS tagging: {'success' if pos_result is not None else 'failed'}")
    out.append(f"- Sentences with POS misalignment: {misaligned_count}")
    out.append(f"- train_clean: {len(sentences['train_clean'])} sentences")
    out.append(f"- train_llm: {len(sentences['train_llm'])} sentences")
    out.append(f"- train_noisy: {len(sentences['train_noisy'])} sentences")
    out.append(f"- val: {len(val_examples)} sentences")
    out.append(f"- test: {len(test_examples)} sentences")
    out.append("")
    for exp_id, _, _ in EXPERIMENTS:
        r = results[exp_id]
        out.append(f"- {exp_id}: train={r['training_data']} ({r['n_train']} sent), "
                    f"pos={r['pos']}, fit_time={r['fit_time']:.2f}s")
    out.append("")
    out.append("### Results")
    out.append("| Model | Training Data | POS | Val Span-F1 | Test Span-F1 | Test Token-F1 | Test EM |")
    out.append("|-------|---------------|-----|-------------|---------------|----------------|---------|")
    old = OLD_RESULTS["CRF-0"]
    out.append(f"| CRF-0 (old) | clean+noisy | No | {fmt(old['val_span_f1'])} | "
                f"{fmt(old['test_span_f1'])} | {fmt(old['test_token_f1'])} | {fmt(old['test_em'])} |")
    old = OLD_RESULTS["CRF-1"]
    out.append(f"| CRF-1 (old) | clean+noisy | Yes | {fmt(old['val_span_f1'])} | "
                f"{fmt(old['test_span_f1'])} | {fmt(old['test_token_f1'])} | {fmt(old['test_em'])} |")
    label_map = {
        "CRF-2": "clean+llm", "CRF-3": "clean+llm",
        "CRF-4": "clean+llm+noisy", "CRF-5": "clean+llm+noisy",
    }
    for exp_id, _, use_pos in EXPERIMENTS:
        r = results[exp_id]
        out.append(f"| {exp_id} | {label_map[exp_id]} | {'Yes' if use_pos else 'No'} | "
                    f"{fmt(r['val_span_f1'])} | {fmt(r['test_span_f1'])} | "
                    f"{fmt(r['test_token_f1'])} | {fmt(r['test_em'])} |")
    out.append("")
    out.append(f"Best model (highest test span-F1 among new experiments): **{best_id}** "
                f"(test span-F1={best['test_span_f1']:.4f})")
    out.append("")
    out.append(f"### Top 20 Features ({best_id}, by weight for B-MATH and I-MATH)")
    for attr, label, w in top20:
        out.append(f"{attr} ({label}): {w:.4f}")
    out.append("")
    out.append(f"### Error Analysis ({best_id}, test set, first 10 errors)")
    for ex, pred in error_sample:
        gold_spans = extract_spans(ex["labels"])
        pred_spans = extract_spans(pred)
        out.append(f"--- sentence_id: {ex['sentence_id']} ---")
        out.append(f"Text: {' '.join(ex['tokens'])}")
        out.append(f"Gold spans: {spans_to_text(gold_spans, ex['tokens'])}")
        out.append(f"Pred spans: {spans_to_text(pred_spans, ex['tokens'])}")
        out.append("---")
    out.append(f"Total test sentences with errors: {len(errors)} / {len(test_examples)}")
    out.append("")
    out.append(f"### Role Analysis in Errors ({best_id})")
    out.append(f"Missed spans (FN) — roles of tokens in missed spans: {dict(fn_role_counts.most_common())}")
    out.append(f"False spans (FP) — roles of tokens in false spans: {dict(fp_role_counts.most_common())}")

    delta_2_0 = results["CRF-2"]["test_span_f1"] - OLD_RESULTS["CRF-0"]["test_span_f1"]
    delta_best_old = best["test_span_f1"] - max(OLD_RESULTS["CRF-0"]["test_span_f1"], OLD_RESULTS["CRF-1"]["test_span_f1"])
    notes = (
        f"CRF-2 (clean+llm, no POS) vs old CRF-0 (clean+noisy, no POS) test span-F1 delta = {delta_2_0:+.4f}. "
        f"Best new config ({best_id}) vs best old config (CRF-1) test span-F1 delta = {delta_best_old:+.4f}. "
        f"POS status: {pos_status} "
        f"Adding train_llm replaces noisy Mathematicon heuristic spans with clean LLM-generated [MATH] spans, "
        f"the same data source that drove ruBERT's biggest jump (BERT-4 span-F1=0.3287 vs curriculum configs "
        f"that used Mathematicon and underperformed). The pattern for CRF is reported in the Results table above: "
        f"compare clean+llm (CRF-2/3) against clean+llm+noisy (CRF-4/5) to see whether Mathematicon noise still "
        f"hurts once train_llm is available, mirroring the BERT finding. Total script runtime: {total_time:.1f} sec."
    )
    out.append(f"Notes: {notes}")
    out.append("</executor_output>")

    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
