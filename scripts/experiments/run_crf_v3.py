#!/usr/bin/env python3
"""
run_crf_v3.py — CRF-6..CRF-11 experiments for MathSpan BIO labeling, testing
whether Dima's Stanza-based dependency features help CRF on top of the best
clean+llm training configuration (CRF-3: test span-F1=0.3355).

This is a COPY of run_crf_v2.py (do not edit run_crf_v2.py) that adds two new
per-token features sourced from Dima's standalone detector, precomputed by
scripts/features/extract_dima_features.py into data/features/dima_features.tsv:

    - dep_label : Stanza dependency relation (nsubj, obj, obl, conj, ...)
    - dima_bio  : Dima's rule-based BIO prediction (B-MATH/I-MATH/O)

Training data is fixed to train_clean + train_llm (694 sentences), the best
configuration from v2. Six experiments toggle POS / dep_label / dima_bio:

    CRF-6:  dep_label                       (no POS, no dima_bio)
    CRF-7:  POS + dep_label                 (no dima_bio)
    CRF-8:  dima_bio                        (no POS, no dep_label)
    CRF-9:  POS + dima_bio                  (no dep_label)
    CRF-10: dep_label + dima_bio            (no POS)
    CRF-11: POS + dep_label + dima_bio

All other infrastructure (vocab, base features, metrics) is unchanged from
run_crf_v2.py. CRF hyperparameters are identical (lbfgs, c1=0.1, c2=0.1,
max_iterations=200). Old results CRF-2 (0.3232) and CRF-3 (0.3355) are
hardcoded into the comparison table.

Run from project root (after extract_dima_features.py has produced the
features file):
    python scripts/experiments/run_crf_v3.py
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
DIMA_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "dima_features.tsv")

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
# 1. Vocabulary construction (unchanged from v2)
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
# 2. BIO TSV loading (keeps token_index so we can join Dima features)
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
                "token_indices": [r[0] for r in rows],
                "tokens": [r[1] for r in rows],
                "labels": [r[2] for r in rows],
            })
    return sentences


# --------------------------------------------------------------------------
# 2b. Dima feature loading
# --------------------------------------------------------------------------

def load_dima_features(path):
    """Load data/features/dima_features.tsv -> {(sid, token_index): (dep, bio)}."""
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: Dima features file not found at {path}\n"
            f"Run scripts/features/extract_dima_features.py first (requires Stanza)."
        )
    lookup = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sentence_id", "token_index", "dep_label", "dima_bio"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: dima_features.tsv missing columns: {missing}")
        for row in reader:
            key = (row["sentence_id"], int(row["token_index"]))
            lookup[key] = (row["dep_label"] or "UNK", row["dima_bio"] or "O")
    return lookup


def dima_arrays_for(example, lookup):
    """Return (dep_labels, dima_bios) parallel to example tokens, fallback UNK/O."""
    deps, bios = [], []
    for ti in example["token_indices"]:
        dep, bio = lookup.get((example["sentence_id"], ti), ("UNK", "O"))
        deps.append(dep)
        bios.append(bio)
    return deps, bios


# --------------------------------------------------------------------------
# 3. Stanza POS tagging (with graceful fallback) — unchanged from v2
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
# 4. Feature extraction (extended with dep_label / dima_bio)
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


def build_sentence_features(tokens, vocab, postags=None, dep_labels=None, dima_bios=None):
    n = len(tokens)
    base = [base_features(t, vocab) for t in tokens]
    feats_out = []
    for i in range(n):
        feat = dict(base[i])
        feat["BOS"] = (i == 0)
        feat["EOS"] = (i == n - 1)
        if postags is not None:
            feat["postag"] = postags[i]
        if dep_labels is not None:
            feat["word.dep_label"] = dep_labels[i]
        if dima_bios is not None:
            feat["word.dima_bio"] = dima_bios[i]

        for offset in (-2, -1, 1, 2):
            j = i + offset
            pre = f"{offset}:"
            if 0 <= j < n:
                feat[f"{pre}word.lower"] = tokens[j].lower()
                feat[f"{pre}is_math_word"] = base[j]["word.is_math_word"]
                feat[f"{pre}math_role"] = base[j]["word.math_role"]
                if postags is not None:
                    feat[f"{pre}postag"] = postags[j]
                if dep_labels is not None:
                    feat[f"{pre}dep_label"] = dep_labels[j]
                if dima_bios is not None:
                    feat[f"{pre}dima_bio"] = dima_bios[j]
            else:
                marker = "BOS" if j < 0 else "EOS"
                feat[f"{pre}word.lower"] = marker
                feat[f"{pre}is_math_word"] = False
                feat[f"{pre}math_role"] = marker
                if postags is not None:
                    feat[f"{pre}postag"] = marker
                if dep_labels is not None:
                    feat[f"{pre}dep_label"] = marker
                if dima_bios is not None:
                    feat[f"{pre}dima_bio"] = marker
        feats_out.append(feat)
    return feats_out


# --------------------------------------------------------------------------
# 5. Metrics (unchanged from v2)
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

TRAIN_SPLITS = ["train_clean", "train_llm"]

# (exp_id, use_pos, use_dep, use_dima)
EXPERIMENTS = [
    ("CRF-6", False, True, False),
    ("CRF-7", True, True, False),
    ("CRF-8", False, False, True),
    ("CRF-9", True, False, True),
    ("CRF-10", False, True, True),
    ("CRF-11", True, True, True),
]

# Hardcoded prior results from run_crf_v2.py (E5b).
OLD_RESULTS = {
    "CRF-2": {"training_data": "clean+llm", "features": "base (no POS)",
              "test_span_f1": 0.3232},
    "CRF-3": {"training_data": "clean+llm", "features": "base +POS",
              "test_span_f1": 0.3355},
}


def feat_label(use_pos, use_dep, use_dima):
    parts = []
    if use_pos:
        parts.append("POS")
    if use_dep:
        parts.append("dep")
    if use_dima:
        parts.append("dima_bio")
    return "+".join(parts) if parts else "base"


def fmt(v, spec=".4f"):
    return "—" if v is None else format(v, spec)


# --------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------

def main():
    t0 = time.time()

    # --- Dima features (fail fast if missing) ---
    dima_lookup = load_dima_features(DIMA_FEATURES_PATH)
    print(f"[DIMA] loaded {len(dima_lookup)} (sentence_id, token_index) feature rows "
          f"from {DIMA_FEATURES_PATH}")

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
    print(f"[DATA] val: {len(val_examples)} sentences")
    print(f"[DATA] test: {len(test_examples)} sentences")

    # --- Coverage check: how many tokens have Dima features in each used split ---
    used_splits = TRAIN_SPLITS + ["val", "test"]
    for split in used_splits:
        n_tok = sum(len(ex["tokens"]) for ex in sentences[split])
        n_hit = sum(
            1 for ex in sentences[split] for ti in ex["token_indices"]
            if (ex["sentence_id"], ti) in dima_lookup
        )
        cov = 100.0 * n_hit / n_tok if n_tok else 0.0
        print(f"[DIMA] {split}: {n_hit}/{n_tok} tokens covered ({cov:.1f}%)")

    # --- POS tagging (once per used split, reused across experiments) ---
    pos_result, pos_status = build_pos_tagger()
    print(f"[POS] {pos_status}")
    misaligned_count = 0

    postags_by_split = {}
    if pos_result is not None:
        tag_fn, misaligned_ref = pos_result
        t_pos = time.time()
        for split in used_splits:
            postags_by_split[split] = [tag_fn(ex["tokens"]) for ex in sentences[split]]
        misaligned_count = misaligned_ref[0]
        print(f"[POS] tagging done in {time.time() - t_pos:.1f}s, misaligned sentences: {misaligned_count}")
    else:
        for split in used_splits:
            postags_by_split[split] = [["UNK"] * len(ex["tokens"]) for ex in sentences[split]]

    # --- Precompute Dima dep/bio arrays per used split ---
    dep_by_split = {}
    dima_by_split = {}
    for split in used_splits:
        deps_list, bios_list = [], []
        for ex in sentences[split]:
            deps, bios = dima_arrays_for(ex, dima_lookup)
            deps_list.append(deps)
            bios_list.append(bios)
        dep_by_split[split] = deps_list
        dima_by_split[split] = bios_list

    y_val = [ex["labels"] for ex in val_examples]
    y_test = [ex["labels"] for ex in test_examples]

    # --- Build training arrays (fixed training set across experiments) ---
    train_examples = []
    train_postags = []
    train_deps = []
    train_dima = []
    for split in TRAIN_SPLITS:
        train_examples.extend(sentences[split])
        train_postags.extend(postags_by_split[split])
        train_deps.extend(dep_by_split[split])
        train_dima.extend(dima_by_split[split])
    y_train = [ex["labels"] for ex in train_examples]
    print(f"[DATA] training set (clean+llm): {len(train_examples)} sentences")

    # --- Run the 6 experiments ---
    results = {}
    for exp_id, use_pos, use_dep, use_dima in EXPERIMENTS:
        X_train = [
            build_sentence_features(
                ex["tokens"], combined_vocab,
                pt if use_pos else None,
                dp if use_dep else None,
                db if use_dima else None,
            )
            for ex, pt, dp, db in zip(train_examples, train_postags, train_deps, train_dima)
        ]
        X_val = [
            build_sentence_features(
                ex["tokens"], combined_vocab,
                pt if use_pos else None,
                dp if use_dep else None,
                db if use_dima else None,
            )
            for ex, pt, dp, db in zip(
                val_examples, postags_by_split["val"], dep_by_split["val"], dima_by_split["val"])
        ]
        X_test = [
            build_sentence_features(
                ex["tokens"], combined_vocab,
                pt if use_pos else None,
                dp if use_dep else None,
                db if use_dima else None,
            )
            for ex, pt, dp, db in zip(
                test_examples, postags_by_split["test"], dep_by_split["test"], dima_by_split["test"])
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
            "features": feat_label(use_pos, use_dep, use_dima),
            "use_pos": use_pos, "use_dep": use_dep, "use_dima": use_dima,
            "model": crf,
            "fit_time": fit_time,
            "val_span_f1": val_span, "val_token_f1": val_token, "val_em": val_em,
            "test_span_f1": test_span, "test_token_f1": test_token, "test_em": test_em,
            "pred_test": pred_test,
        }
        print(f"[{exp_id}] feats={results[exp_id]['features']}, "
              f"fit={fit_time:.2f}s, val_span_f1={val_span:.4f}, test_span_f1={test_span:.4f}")

    # --- Pick best model by test span-F1 among the 6 new experiments ---
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

    # --- Role analysis in errors (across all test errors) ---
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
    out.append('<executor_output id="E6">')
    out.append("## CRF v3 Experiment Results (with Dima dependency features)")
    out.append("")
    out.append("### Setup")
    out.append(f"- Training data: train_clean + train_llm ({len(train_examples)} sentences)")
    out.append(f"- Combined vocabulary size: {n_combined}")
    out.append(f"- Stanza POS tagging: {'success' if pos_result is not None else 'failed (postag=UNK)'}")
    out.append(f"- POS misaligned sentences: {misaligned_count}")
    out.append(f"- Dima feature rows loaded: {len(dima_lookup)}")
    out.append(f"- val: {len(val_examples)} sentences, test: {len(test_examples)} sentences")
    out.append("")
    out.append("### Results")
    out.append("| Model | Features | Val Span-F1 | Test Span-F1 | Test Token-F1 | Test EM |")
    out.append("|-------|----------|-------------|---------------|----------------|---------|")
    out.append(f"| CRF-2 (old) | {OLD_RESULTS['CRF-2']['features']} | — | "
               f"{fmt(OLD_RESULTS['CRF-2']['test_span_f1'])} | — | — |")
    out.append(f"| CRF-3 (old, best v2) | {OLD_RESULTS['CRF-3']['features']} | — | "
               f"{fmt(OLD_RESULTS['CRF-3']['test_span_f1'])} | — | — |")
    for exp_id, _, _, _ in EXPERIMENTS:
        r = results[exp_id]
        out.append(f"| {exp_id} | {r['features']} | "
                   f"{fmt(r['val_span_f1'])} | {fmt(r['test_span_f1'])} | "
                   f"{fmt(r['test_token_f1'])} | {fmt(r['test_em'])} |")
    out.append("")
    out.append(f"Best new model (highest test span-F1): **{best_id}** "
               f"(features={best['features']}, test span-F1={best['test_span_f1']:.4f})")
    delta_vs_crf3 = best["test_span_f1"] - OLD_RESULTS["CRF-3"]["test_span_f1"]
    out.append(f"Delta vs old best CRF-3 (0.3355): {delta_vs_crf3:+.4f}")
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
    out.append("")
    notes = (
        f"All 6 experiments train on clean+llm ({len(train_examples)} sentences) with identical CRF "
        f"hyperparameters (lbfgs, c1=0.1, c2=0.1, max_iterations=200) to isolate the effect of the two "
        f"new Dima-derived features: word.dep_label (Stanza deprel) and word.dima_bio (Dima's rule-based "
        f"BIO prediction), both also added to the ±2 context window. Compare against the v2 baselines "
        f"CRF-2 (base, 0.3232) and CRF-3 (base+POS, 0.3355): the dep_label-only runs (CRF-6/7) test the "
        f"syntactic signal, the dima_bio-only runs (CRF-8/9) test stacking Dima's detector as a feature, "
        f"and the combined runs (CRF-10/11) test both. POS status: {pos_status} "
        f"Total script runtime: {total_time:.1f} sec."
    )
    out.append(f"Notes: {notes}")
    out.append("</executor_output>")

    print("\n" + "\n".join(out))


if __name__ == "__main__":
    main()
