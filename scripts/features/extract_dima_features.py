#!/usr/bin/env python3
"""
extract_dima_features.py — extract two per-token features from Dima's standalone
Stanza-based math-span detector (scripts/team_scripts/formula_bio.py) for every
token in data/bio/bio_merged.tsv:

    - dep_label : Stanza dependency relation (nsubj, obj, obl, conj, ...)
    - dima_bio  : BIO tag predicted by Dima's rule-based detector (B-MATH/I-MATH/O)

For each sentence in bio_merged.tsv we:
  1. Reconstruct raw text as " ".join(tokens).
  2. parse_text(text)            -> Dima/Stanza Token objects (with .deprel)
  3. detect_from_tokens(tokens)  -> DetectionResult with .bio predictions
  4. Align Dima's *word* tokens (mask_index is not None) with our tokens.
       - If token counts match: direct 1:1 mapping.
       - Otherwise: character-level alignment over word characters.
       - Unaligned tokens get fallback dep_label="UNK", dima_bio="O".
  5. Write data/features/dima_features.tsv with columns:
       sentence_id, token_index, token, dep_label, dima_bio

This script REQUIRES Stanza (and the ru model). It is meant to be run locally
by Pasha where Stanza is available:

    python scripts/features/extract_dima_features.py

NOTE: this script does not modify any existing files; it only creates
data/features/dima_features.tsv.
"""
import csv
import os
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Paths (relative to project root) + make formula_bio importable
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIO_PATH = os.path.join(PROJECT_ROOT, "data", "bio", "bio_merged.tsv")
MARKERS_PATH = os.path.join(PROJECT_ROOT, "data", "dima", "markers.yaml")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "features")
OUT_PATH = os.path.join(OUT_DIR, "dima_features.tsv")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "team_scripts"))
import formula_bio  # noqa: E402


# --------------------------------------------------------------------------
# 1. Load bio_merged.tsv, grouped by sentence (preserving appearance order)
# --------------------------------------------------------------------------

def load_bio_sentences(path):
    """Return list of {sentence_id, token_indices: [int], tokens: [str]} in the
    order sentences first appear in the file, tokens sorted by token_index."""
    raw = defaultdict(list)  # sid -> [(tok_idx, token)]
    order = []
    seen = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sentence_id", "token_index", "token"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"ERROR: bio_merged.tsv missing columns: {missing}")
        for row in reader:
            sid = row["sentence_id"]
            if sid not in seen:
                seen.add(sid)
                order.append(sid)
            raw[sid].append((int(row["token_index"]), row["token"]))

    sentences = []
    for sid in order:
        rows = sorted(raw[sid], key=lambda r: r[0])
        sentences.append({
            "sentence_id": sid,
            "token_indices": [r[0] for r in rows],
            "tokens": [r[1] for r in rows],
        })
    return sentences


# --------------------------------------------------------------------------
# 2. Alignment helpers
# --------------------------------------------------------------------------

def _word_chars(token):
    """Lowercased alphanumeric (word) characters of a token, punctuation removed.
    Used to build a comparable character stream for char-level alignment."""
    return [c for c in token.lower() if c.isalnum()]


def char_align(our_tokens, dima_texts):
    """Two-pointer character-level alignment over word characters.

    Returns a list `mapping` of length len(our_tokens), where mapping[i] is the
    index into dima_texts that best covers our_tokens[i], or None if unaligned.
    """
    # Build (char, owner_index) streams over word characters only.
    our_stream = []
    for oi, tok in enumerate(our_tokens):
        for c in _word_chars(tok):
            our_stream.append((c, oi))
    dima_stream = []
    for di, txt in enumerate(dima_texts):
        for c in _word_chars(txt):
            dima_stream.append((c, di))

    votes = defaultdict(Counter)  # our_idx -> Counter(dima_idx)
    i = j = 0
    while i < len(our_stream) and j < len(dima_stream):
        oc, oi = our_stream[i]
        dc, di = dima_stream[j]
        if oc == dc:
            votes[oi][di] += 1
            i += 1
            j += 1
        else:
            # Characters disagree (rare: Stanza re-tokenization / normalization).
            # Advance both in lockstep; unresolved tokens fall back to UNK/O.
            i += 1
            j += 1

    mapping = [None] * len(our_tokens)
    for oi, counter in votes.items():
        if counter:
            mapping[oi] = counter.most_common(1)[0][0]
    return mapping


def align_sentence(our_tokens, dima_words):
    """Align our tokens with Dima's word tokens.

    dima_words is a list of (text, dep_label, dima_bio).
    Returns (dep_labels, dima_bios, n_fallback) parallel to our_tokens.
    """
    n = len(our_tokens)
    dep_labels = ["UNK"] * n
    dima_bios = ["O"] * n
    n_fallback = 0

    if len(dima_words) == n:
        # Direct 1:1 mapping.
        for i in range(n):
            _, dep, bio = dima_words[i]
            dep_labels[i] = dep if dep else "UNK"
            dima_bios[i] = bio if bio else "O"
        return dep_labels, dima_bios, 0

    # Lengths differ -> character-level alignment.
    dima_texts = [w[0] for w in dima_words]
    mapping = char_align(our_tokens, dima_texts)
    for i in range(n):
        di = mapping[i]
        if di is None:
            n_fallback += 1
            continue
        _, dep, bio = dima_words[di]
        dep_labels[i] = dep if dep else "UNK"
        dima_bios[i] = bio if bio else "O"
    return dep_labels, dima_bios, n_fallback


# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------

def main():
    if not os.path.exists(BIO_PATH):
        sys.exit(f"ERROR: bio_merged.tsv not found at {BIO_PATH}")
    if not os.path.exists(MARKERS_PATH):
        sys.exit(f"ERROR: markers.yaml not found at {MARKERS_PATH}")

    sentences = load_bio_sentences(BIO_PATH)
    print(f"[DATA] loaded {len(sentences)} sentences from bio_merged.tsv")

    # Load markers once (explicit path; formula_bio's default is 'markers.yaml').
    try:
        markers = formula_bio.load_markers(MARKERS_PATH)
    except Exception as e:
        sys.exit(f"ERROR: failed to load markers from {MARKERS_PATH}: {e}")
    print(f"[MARKERS] loaded {len(markers.entries)} marker entries from {MARKERS_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    n_aligned = 0          # sentences with a clean 1:1 (no fallback) alignment
    n_misaligned = 0       # sentences where token counts differed
    n_with_fallback = 0    # sentences where >=1 token used fallback
    total_tokens = 0
    total_fallback_tokens = 0
    dep_dist = Counter()
    bio_dist = Counter()

    rows_out = []
    for s_no, sent in enumerate(sentences):
        sid = sent["sentence_id"]
        our_tokens = sent["tokens"]
        text = " ".join(our_tokens)

        try:
            dima_tokens = formula_bio.parse_text(text)
            result = formula_bio.detect_from_tokens(dima_tokens, markers)
        except Exception as e:
            print(f"[WARN] {sid}: Dima pipeline failed ({type(e).__name__}: {e}); "
                  f"using fallback UNK/O for all {len(our_tokens)} tokens", file=sys.stderr)
            for ti, tok in zip(sent["token_indices"], our_tokens):
                rows_out.append((sid, ti, tok, "UNK", "O"))
                dep_dist["UNK"] += 1
                bio_dist["O"] += 1
            total_tokens += len(our_tokens)
            total_fallback_tokens += len(our_tokens)
            n_misaligned += 1
            n_with_fallback += 1
            continue

        # Dima's word tokens (mask_index is not None) -> (text, deprel, bio).
        dima_words = [
            (t.text, t.deprel or "UNK", result.bio[t.mask_index])
            for t in result.tokens
            if t.mask_index is not None
        ]

        if len(dima_words) != len(our_tokens):
            n_misaligned += 1
            print(f"[WARN] {sid}: token-count mismatch "
                  f"(ours={len(our_tokens)}, dima={len(dima_words)}); "
                  f"using character-level alignment", file=sys.stderr)

        dep_labels, dima_bios, n_fb = align_sentence(our_tokens, dima_words)
        if n_fb == 0 and len(dima_words) == len(our_tokens):
            n_aligned += 1
        if n_fb > 0:
            n_with_fallback += 1

        for ti, tok, dep, bio in zip(sent["token_indices"], our_tokens, dep_labels, dima_bios):
            rows_out.append((sid, ti, tok, dep, bio))
            dep_dist[dep] += 1
            bio_dist[bio] += 1

        total_tokens += len(our_tokens)
        total_fallback_tokens += n_fb

        if (s_no + 1) % 200 == 0:
            print(f"[PROGRESS] {s_no + 1}/{len(sentences)} sentences processed")

    # Write output TSV.
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sentence_id", "token_index", "token", "dep_label", "dima_bio"])
        writer.writerows(rows_out)

    # Summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output written to: {OUT_PATH}")
    print(f"Total sentences:               {len(sentences)}")
    print(f"  clean 1:1 aligned:           {n_aligned}")
    print(f"  token-count mismatched:      {n_misaligned}")
    print(f"  sentences using >=1 fallback:{n_with_fallback}")
    print(f"Total tokens:                  {total_tokens}")
    print(f"  fallback (UNK/O) tokens:     {total_fallback_tokens} "
          f"({100.0 * total_fallback_tokens / total_tokens:.2f}%)" if total_tokens else "")
    print("\ndep_label distribution (top 25):")
    for dep, c in dep_dist.most_common(25):
        print(f"  {dep:<16} {c}")
    print("\ndima_bio distribution:")
    for bio, c in bio_dist.most_common():
        print(f"  {bio:<10} {c}")


if __name__ == "__main__":
    main()
