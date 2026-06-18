#!/usr/bin/env python3
"""Standalone Stanza-based Russian math-span detector -> BIO rows.

This file intentionally does not import ``mathspan_detector``. It is a
single-file port of the current src detector: Stanza tokenization/POS/lemma/
dependency parsing, markers.yaml matching, dependency span expansion, span
merge/trim, and BIO export.

Examples:
    python3 standalone_bio/formula_bio.py --text "икс равно нулю"
    python3 standalone_bio/formula_bio.py --input lecture.txt --output lecture_bio.tsv
    python3 standalone_bio/formula_bio.py --download-stanza-ru
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required: install package dependencies first.") from exc


DEFAULT_TEXT = "Поэтому можем расписать, что Z здесь будет равняться X минус Y и минус, раз X больше Y."
MARKERS_PATH = "markers.yaml"
GAZETTEER_PATH = Path(__file__).resolve().parents[1] / "generated" / "pronunciation_gazetteer.yaml"
DELIMITER = " "
WRITE_HEADER = True

RU_PUNCT = "«»„“”…—–"
PUNCT_TABLE = str.maketrans("", "", string.punctuation + RU_PUNCT)
WORD_RE = re.compile(r"[\wА-Яа-яЁё]+", re.UNICODE)

POSITIVE_ROLES = {
    "RELATION",
    "OPERATOR",
    "STRUCTURE",
    "OBJECT",
    "VARIABLE",
    "CONSTANT",
    "QUANTIFIER",
    "CONNECTIVE",
}
BOUNDARY_ROLES = {"NEGATIVE_CONTEXT", "BOUNDARY", "BOUNDARY_AMBIGUOUS", "END_CONTEXT"}
MATH_HEAD_DEPRELS = {
    "root",
    "conj",
    "acl",
    "advcl",
    "ccomp",
    "xcomp",
    "nsubj",
    "obj",
    "obl",
    "nmod",
    "appos",
}
EXPANSION_DEPRELS = {"nmod", "obl", "appos", "flat", "fixed", "conj", "amod", "compound", "case"}
RELATION_LEMMAS = {
    "равно",
    "равный",
    "равняться",
    "равен",
    "больше",
    "меньше",
    "принадлежать",
    "совпадать",
    "содержаться",
}
RELATION_CHILD_DEPRELS = {"nsubj", "obj", "obl", "xcomp", "conj", "csubj"}
VARIABLE_WORDS = {
    "икс",
    "игрек",
    "зет",
    "дзета",
    "эта",
    "эн",
    "эпсилон",
    "дельта",
    "альфа",
    "бета",
    "бэ",
    "гамма",
    "лямбда",
    "мю",
    "ню",
    "фи",
    "тау",
}
STRONG_BOUNDARY_PUNCT = {".", "?", "!", ";", ":"}
LEADING_NON_ANCHOR_UPOS = {"ADP", "SCONJ", "PART", "CCONJ"}
LECTURE_TOPIC_PREPOSITIONS = {"про", "о", "об"}
FORMULA_CORE_ROLES = {"RELATION", "OPERATOR", "STRUCTURE", "CONSTANT", "QUANTIFIER", "CONNECTIVE"}
NON_VARIABLE_UPOS = {"ADP", "CCONJ", "SCONJ", "PART", "PRON"}
MAX_ANCHOR_CLUSTER_GAP = 3
TOPIC_OBJECT_ROLES = {"OBJECT", "VARIABLE", "CONNECTIVE"}
BOUNDARY_MARKER_ROLES = BOUNDARY_ROLES | {"DISCOURSE_START"}


class StanzaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    lemma: str = ""
    upos: str = ""
    head: int | None = None
    deprel: str = ""
    start_char: int | None = None
    end_char: int | None = None
    mask_index: int | None = None
    sentence_index: int = 0

    @property
    def is_punct(self) -> bool:
        return self.mask_index is None


@dataclass(frozen=True)
class MarkerMatch:
    start: int
    end: int
    role: str
    entry_id: str
    pattern: str
    match_type: str
    confidence: str = "medium"


@dataclass(frozen=True)
class MarkerEntry:
    entry_id: str
    role: str
    match_type: str
    confidence: str
    pattern: str
    token_pattern: tuple[str, ...]
    regex: re.Pattern[str] | None = None


@dataclass(frozen=True)
class MarkerSet:
    entries: tuple[MarkerEntry, ...]

    @property
    def positive_entries(self) -> tuple[MarkerEntry, ...]:
        return tuple(entry for entry in self.entries if entry.role in POSITIVE_ROLES)

    @property
    def boundary_entries(self) -> tuple[MarkerEntry, ...]:
        return tuple(entry for entry in self.entries if entry.role in BOUNDARY_ROLES)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    start_char: int | None = None
    end_char: int | None = None
    roles: tuple[str, ...] = ()
    anchors: tuple[MarkerMatch, ...] = ()

    def overlaps_or_touches(self, other: "Span") -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class DetectionResult:
    tokens: list[Token]
    mask: str
    bio: list[str]
    spans: list[Span]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": [token.__dict__ for token in self.tokens],
            "mask": self.mask,
            "bio": self.bio,
            "spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "roles": list(span.roles),
                    "anchors": [anchor.__dict__ for anchor in span.anchors],
                }
                for span in self.spans
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BioTokenRow:
    sentence_id: str
    token_index: int
    token: str
    dep_label: str
    bio_tag: str


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


def normalize_token(text: str) -> str:
    return normalize_text(text).translate(PUNCT_TABLE)


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    normalized = normalize_text(phrase)
    return tuple(normalize_token(match.group(0)) for match in WORD_RE.finditer(normalized) if normalize_token(match.group(0)))


@lru_cache(maxsize=1)
def _pipeline():
    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except ModuleNotFoundError as exc:
        raise StanzaUnavailableError(
            "stanza is not installed. Install dependencies and run "
            "`python3 standalone_bio/formula_bio.py --download-stanza-ru`."
        ) from exc

    try:
        return stanza.Pipeline(
            lang="ru",
            processors="tokenize,pos,lemma,depparse",
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=False,
            verbose=False,
        )
    except Exception as exc:
        raise StanzaUnavailableError(
            "Russian Stanza model is not available. Run "
            "`python3 standalone_bio/formula_bio.py --download-stanza-ru` and retry."
        ) from exc


def download_ru_model() -> None:
    try:
        import stanza
    except ModuleNotFoundError as exc:
        raise StanzaUnavailableError("stanza is not installed. Install package dependencies first.") from exc
    stanza.download("ru")


def parse_text(text: str) -> list[Token]:
    doc = _pipeline()(text)
    tokens: list[Token] = []
    mask_index = 0
    for sentence_index, sentence in enumerate(doc.sentences):
        sentence_start = len(tokens)
        for word in sentence.words:
            upos = word.upos or ""
            current_mask_index = None if upos == "PUNCT" and not is_wordlike_token(word.text) else mask_index
            if current_mask_index is not None:
                mask_index += 1
            head = None if word.head == 0 else sentence_start + word.head - 1
            tokens.append(
                Token(
                    index=len(tokens),
                    text=word.text,
                    lemma=word.lemma or word.text,
                    upos=upos,
                    head=head,
                    deprel=word.deprel or "",
                    start_char=getattr(word, "start_char", None),
                    end_char=getattr(word, "end_char", None),
                    mask_index=current_mask_index,
                    sentence_index=sentence_index,
                )
            )
    return tokens


def is_wordlike_token(text: str) -> bool:
    return bool(re.fullmatch(r"[\wА-Яа-яЁё]+", text, re.UNICODE))


def load_markers(
    path: str | Path | None = None,
    include_gazetteer: bool = False,
    gazetteer_path: str | Path | None = None,
) -> MarkerSet:
    marker_path = Path(path) if path is not None else MARKERS_PATH
    with marker_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    entries: list[MarkerEntry] = []
    for raw_entry in raw.get("entries", []):
        match_type = raw_entry["match_type"]
        for pattern in raw_entry.get("patterns", []):
            regex = re.compile(pattern, re.IGNORECASE) if match_type == "regex" else None
            entries.append(
                MarkerEntry(
                    entry_id=raw_entry["id"],
                    role=raw_entry["role"],
                    match_type=match_type,
                    confidence=raw_entry.get("confidence", "medium"),
                    pattern=pattern,
                    token_pattern=phrase_tokens(pattern),
                    regex=regex,
                )
            )
    if include_gazetteer:
        entries.extend(_load_gazetteer_entries(gazetteer_path))
    return MarkerSet(tuple(entries))


def _load_gazetteer_entries(path: str | Path | None = None) -> list[MarkerEntry]:
    gazetteer_path = Path(path) if path is not None else GAZETTEER_PATH
    if not gazetteer_path.exists():
        return []
    with gazetteer_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    entries: list[MarkerEntry] = []
    for idx, raw_entry in enumerate(raw.get("entries", [])):
        phrase = raw_entry.get("phrase", "")
        token_pattern = phrase_tokens(phrase)
        if not token_pattern:
            continue
        entries.append(
            MarkerEntry(
                entry_id=f"gazetteer_{idx}",
                role="STRUCTURE",
                match_type="phrase",
                confidence="medium",
                pattern=phrase,
                token_pattern=token_pattern,
            )
        )
    return entries


def find_marker_matches(tokens: list[Token], markers: MarkerSet) -> list[MarkerMatch]:
    words = [token for token in tokens if token.mask_index is not None]
    surface = [normalize_token(token.text) for token in words]
    lemma = [normalize_token(token.lemma or token.text) for token in words]
    matches: list[MarkerMatch] = []
    occupied_positive: set[int] = set()
    occupied_boundary: set[int] = set()

    phrase_entries = [
        entry
        for entry in markers.entries
        if entry.match_type in {"phrase", "exact", "lemma"} and entry.token_pattern
    ]
    phrase_entries.sort(key=lambda entry: len(entry.token_pattern), reverse=True)

    for entry in phrase_entries:
        haystack = lemma if entry.match_type == "lemma" else surface
        size = len(entry.token_pattern)
        for start in range(0, len(words) - size + 1):
            positions = set(range(start, start + size))
            occupied = occupied_boundary if entry.role in BOUNDARY_MARKER_ROLES else occupied_positive
            if positions & occupied:
                continue
            if tuple(haystack[start : start + size]) == entry.token_pattern:
                matches.append(_match_from_words(words, start, start + size, entry))
                occupied.update(positions)

    matches.extend(_regex_marker_matches(words, surface, markers.entries, occupied_positive))

    matches.sort(key=lambda match: (match.start, -(match.end - match.start), match.entry_id))
    return matches


def _regex_marker_matches(
    words: list[Token],
    surface: list[str],
    entries: tuple[MarkerEntry, ...],
    occupied_positive: set[int],
) -> list[MarkerMatch]:
    matches: list[MarkerMatch] = []
    joined, spans = _joined_token_spans(surface)
    starts = {start for start, _ in spans}
    ends = {end for _, end in spans}
    for entry in entries:
        if entry.match_type != "regex" or entry.regex is None:
            continue
        for regex_match in entry.regex.finditer(joined):
            if regex_match.start() not in starts or regex_match.end() not in ends:
                continue
            start = next(index for index, span in enumerate(spans) if span[0] == regex_match.start())
            end = next(index + 1 for index, span in enumerate(spans) if span[1] == regex_match.end())
            positions = set(range(start, end))
            if entry.role not in BOUNDARY_MARKER_ROLES and positions & occupied_positive:
                continue
            matches.append(_match_from_words(words, start, end, entry))
            if entry.role not in BOUNDARY_MARKER_ROLES:
                occupied_positive.update(positions)
    return matches


def _joined_token_spans(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        spans.append((start, cursor))
    return "".join(pieces), spans


def _match_from_words(words: list[Token], start: int, end: int, entry: MarkerEntry) -> MarkerMatch:
    return MarkerMatch(
        start=words[start].index,
        end=words[end - 1].index + 1,
        role=entry.role,
        entry_id=entry.entry_id,
        pattern=entry.pattern,
        match_type=entry.match_type,
        confidence=entry.confidence,
    )


def detect(
    text: str,
    markers_path: str | None = None,
    include_gazetteer: bool = False,
    gazetteer_path: str | None = None,
) -> DetectionResult:
    tokens = parse_text(text)
    markers = load_markers(markers_path, include_gazetteer=include_gazetteer, gazetteer_path=gazetteer_path)
    return detect_from_tokens(tokens, markers)


def detect_from_tokens(tokens: list[Token], markers: MarkerSet | None = None) -> DetectionResult:
    markers = markers or load_markers()
    matches = find_marker_matches(tokens, markers)
    positive = [match for match in matches if match.role in POSITIVE_ROLES and not _is_non_math_variable_match(tokens, match)]
    boundaries = [match for match in matches if match.role in BOUNDARY_MARKER_ROLES]
    spans = build_dependency_spans(tokens, positive)
    spans = trim_spans(tokens, spans, boundaries)
    spans = merge_spans(tokens, spans)
    spans = split_spans_by_anchor_gaps(tokens, spans)
    spans = suppress_context_false_positives(tokens, spans)
    mask, bio = spans_to_mask_bio(tokens, spans)
    return DetectionResult(tokens=tokens, mask=mask, bio=bio, spans=spans, metadata={"markers": len(matches)})


def build_dependency_spans(tokens: list[Token], anchors: list[MarkerMatch]) -> list[Span]:
    children = _children(tokens)
    anchor_indices = {idx for anchor in anchors for idx in range(anchor.start, anchor.end)}
    spans: list[Span] = []

    for anchor in anchors:
        head = _math_head(tokens, anchor, anchor_indices)
        indices = _subtree(children, head)
        indices.update(range(anchor.start, anchor.end))
        indices.update(_relation_expansion(tokens, children, head))
        indices.update(_anchored_expansion(tokens, children, indices, anchor_indices))
        indices = _filter_variable_noise(tokens, indices, anchor_indices)
        span = _span_from_indices(tokens, indices, [anchor])
        if span is not None:
            spans.append(span)
    return spans


def trim_spans(tokens: list[Token], spans: list[Span], boundary_matches: list[MarkerMatch]) -> list[Span]:
    trimmed: list[Span] = []
    boundary_indices = {idx for match in boundary_matches for idx in range(match.start, match.end)}
    punct_indices = {token.index for token in tokens if token.text in STRONG_BOUNDARY_PUNCT}

    for span in spans:
        indices = [
            token.index
            for token in tokens
            if token.mask_index is not None and span.start <= token.mask_index < span.end
        ]
        if not indices:
            continue
        kept = [idx for idx in indices if idx not in boundary_indices]
        kept = _drop_after_internal_punct(tokens, kept, punct_indices)
        new_span = _span_from_indices(tokens, set(kept), list(span.anchors))
        if new_span is not None:
            trimmed.append(new_span)
    return trimmed


def merge_spans(tokens: list[Token], spans: list[Span]) -> list[Span]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[Span] = [spans[0]]
    for span in spans[1:]:
        previous = merged[-1]
        if previous.end > span.start or (previous.end == span.start and _can_merge_gap(tokens, previous, span)) or _can_merge_gap(tokens, previous, span):
            merged[-1] = _merge_two(previous, span)
        else:
            merged.append(span)
    return merged


def suppress_context_false_positives(tokens: list[Token], spans: list[Span]) -> list[Span]:
    kept: list[Span] = []
    for span in spans:
        if _is_lecture_topic_object_span(tokens, span):
            continue
        kept.append(span)
    return kept


def split_spans_by_anchor_gaps(tokens: list[Token], spans: list[Span]) -> list[Span]:
    split: list[Span] = []
    for span in spans:
        clusters = _anchor_clusters(span.anchors)
        for cluster in clusters:
            start = min(anchor.start for anchor in cluster)
            end = max(anchor.end for anchor in cluster)
            cluster_span = _span_from_indices(tokens, set(range(start, end)), list(cluster))
            if cluster_span is not None:
                split.append(cluster_span)
    return split


def _anchor_clusters(anchors: tuple[MarkerMatch, ...]) -> list[list[MarkerMatch]]:
    ordered = sorted(
        (anchor for anchor in anchors if anchor.role != "CONNECTIVE"),
        key=lambda anchor: (anchor.start, anchor.end, anchor.entry_id),
    )
    if not ordered:
        return []
    clusters: list[list[MarkerMatch]] = [[ordered[0]]]
    current_end = ordered[0].end
    for anchor in ordered[1:]:
        if anchor.start - current_end > MAX_ANCHOR_CLUSTER_GAP:
            clusters.append([anchor])
        else:
            clusters[-1].append(anchor)
        current_end = max(current_end, anchor.end)
    return clusters


def spans_to_mask_bio(tokens: list[Token], spans: list[Span]) -> tuple[str, list[str]]:
    word_count = sum(1 for token in tokens if token.mask_index is not None)
    mask = ["0"] * word_count
    bio = ["O"] * word_count
    for span in spans:
        for pos in range(max(0, span.start), min(word_count, span.end)):
            mask[pos] = "1"
            bio[pos] = "B-MATH" if pos == span.start else "I-MATH"
    return "".join(mask), bio


def _children(tokens: list[Token]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)
    for token in tokens:
        if token.head is not None:
            children[token.head].append(token.index)
    return children


def _math_head(tokens: list[Token], anchor: MarkerMatch, anchor_indices: set[int]) -> int:
    current = anchor.start
    seen: set[int] = set()
    best = current
    while current is not None and current not in seen:
        seen.add(current)
        token = tokens[current]
        if token.deprel in MATH_HEAD_DEPRELS or _is_math_token(token):
            best = current
        head = token.head
        if head is None:
            return best
        if _is_boundary_between(tokens, current, head):
            return best
        if _is_math_token(tokens[head]) or head in anchor_indices:
            current = head
            continue
        return best
    return best


def _is_non_math_variable_match(tokens: list[Token], match: MarkerMatch) -> bool:
    if match.role != "VARIABLE":
        return False
    return any(tokens[idx].upos in NON_VARIABLE_UPOS for idx in range(match.start, match.end))


def _subtree(children: dict[int, list[int]], head: int) -> set[int]:
    found: set[int] = set()
    queue: deque[int] = deque([head])
    while queue:
        idx = queue.popleft()
        if idx in found:
            continue
        found.add(idx)
        queue.extend(children.get(idx, []))
    return found


def _relation_expansion(tokens: list[Token], children: dict[int, list[int]], head: int) -> set[int]:
    token = tokens[head]
    if not _is_relation_token(token):
        return set()
    indices = {head}
    for child in children.get(head, []):
        if tokens[child].deprel in RELATION_CHILD_DEPRELS:
            indices.update(_subtree(children, child))
    return indices


def _anchored_expansion(
    tokens: list[Token], children: dict[int, list[int]], seed_indices: set[int], anchor_indices: set[int]
) -> set[int]:
    indices = set(seed_indices)
    changed = True
    while changed:
        changed = False
        for parent, child_list in children.items():
            for child in child_list:
                if child in indices:
                    continue
                if tokens[child].deprel not in EXPANSION_DEPRELS:
                    continue
                subtree = _subtree(children, child)
                if subtree & anchor_indices or any(_is_variable_token(tokens[idx]) for idx in subtree):
                    before = len(indices)
                    indices.update(subtree)
                    if parent in indices:
                        indices.add(parent)
                    changed = changed or len(indices) != before
    return indices


def _filter_variable_noise(tokens: list[Token], indices: set[int], anchor_indices: set[int]) -> set[int]:
    if len(indices) <= 1:
        return indices
    kept: set[int] = set()
    for idx in indices:
        token = tokens[idx]
        if not _is_single_letter_variable(token):
            kept.add(idx)
            continue
        head_ok = token.head in indices and token.deprel in {"nmod", "appos", "conj", "flat", "obj", "nsubj", "obl"}
        relation_near = any(abs(idx - anchor_idx) <= 3 for anchor_idx in anchor_indices)
        if head_ok or relation_near or idx in anchor_indices:
            kept.add(idx)
    return kept


def _drop_after_internal_punct(tokens: list[Token], indices: list[int], punct_indices: set[int]) -> list[int]:
    if not indices:
        return []
    kept: list[int] = []
    for idx in indices:
        if kept and any(punct in punct_indices for punct in range(kept[-1] + 1, idx)):
            break
        kept.append(idx)
    return kept


def _span_from_indices(tokens: list[Token], indices: set[int], anchors: list[MarkerMatch]) -> Span | None:
    word_indices = sorted(idx for idx in indices if 0 <= idx < len(tokens) and tokens[idx].mask_index is not None)
    anchor_indices = {idx for anchor in anchors for idx in range(anchor.start, anchor.end)}
    while len(word_indices) > 1 and word_indices[0] not in anchor_indices and tokens[word_indices[0]].upos in LEADING_NON_ANCHOR_UPOS:
        word_indices.pop(0)
    if not word_indices:
        return None
    mask_positions = [tokens[idx].mask_index for idx in word_indices if tokens[idx].mask_index is not None]
    roles = tuple(sorted({anchor.role for anchor in anchors}))
    start_char = next((tokens[idx].start_char for idx in word_indices if tokens[idx].start_char is not None), None)
    end_char = next((tokens[idx].end_char for idx in reversed(word_indices) if tokens[idx].end_char is not None), None)
    return Span(
        start=min(mask_positions),
        end=max(mask_positions) + 1,
        start_char=start_char,
        end_char=end_char,
        roles=roles,
        anchors=tuple(anchors),
    )


def _is_lecture_topic_object_span(tokens: list[Token], span: Span) -> bool:
    roles = set(span.roles)
    if roles <= {"CONNECTIVE"}:
        return True
    if not roles <= TOPIC_OBJECT_ROLES:
        if roles & FORMULA_CORE_ROLES:
            return False
        return False
    previous = _previous_word_token(tokens, span.start)
    return previous is not None and normalize_token(previous.text) in LECTURE_TOPIC_PREPOSITIONS


def _previous_word_token(tokens: list[Token], mask_index: int) -> Token | None:
    candidates = [
        token
        for token in tokens
        if token.mask_index is not None and token.mask_index < mask_index
    ]
    return candidates[-1] if candidates else None


def _merge_two(left: Span, right: Span) -> Span:
    anchors = tuple(sorted(left.anchors + right.anchors, key=lambda anchor: (anchor.start, anchor.end, anchor.entry_id)))
    roles = tuple(sorted(set(left.roles) | set(right.roles)))
    return Span(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        start_char=left.start_char if left.start <= right.start else right.start_char,
        end_char=right.end_char if right.end >= left.end else left.end_char,
        roles=roles,
        anchors=anchors,
    )


def _can_merge_gap(tokens: list[Token], left: Span, right: Span) -> bool:
    if right.start - left.end > 2:
        return False
    if _different_sentence(tokens, left, right):
        return False
    if _strong_punct_in_mask_gap(tokens, left, right):
        return False
    gap_tokens = [token for token in tokens if token.mask_index is not None and left.end <= token.mask_index < right.start]
    if any(normalize_token(token.text) in {"а", "но", "однако", "зато"} for token in gap_tokens):
        return False
    return True


def _different_sentence(tokens: list[Token], left: Span, right: Span) -> bool:
    left_tokens = [token for token in tokens if token.mask_index is not None and left.start <= token.mask_index < left.end]
    right_tokens = [token for token in tokens if token.mask_index is not None and right.start <= token.mask_index < right.end]
    if not left_tokens or not right_tokens:
        return False
    return left_tokens[-1].sentence_index != right_tokens[0].sentence_index


def _strong_punct_in_mask_gap(tokens: list[Token], left: Span, right: Span) -> bool:
    left_tokens = [token for token in tokens if token.mask_index is not None and left.start <= token.mask_index < left.end]
    right_tokens = [token for token in tokens if token.mask_index is not None and right.start <= token.mask_index < right.end]
    if not left_tokens or not right_tokens:
        return False
    return any(token.text in STRONG_BOUNDARY_PUNCT for token in tokens[left_tokens[-1].index + 1 : right_tokens[0].index])


def _is_boundary_between(tokens: list[Token], left: int, right: int) -> bool:
    start, end = sorted((left, right))
    return any(token.text in STRONG_BOUNDARY_PUNCT for token in tokens[start + 1 : end])


def _is_math_token(token: Token) -> bool:
    return _is_relation_token(token) or _is_variable_token(token)


def _is_relation_token(token: Token) -> bool:
    value = normalize_token(token.lemma or token.text)
    surface = normalize_token(token.text)
    return value in RELATION_LEMMAS or surface in RELATION_LEMMAS


def _is_variable_token(token: Token) -> bool:
    if token.upos in NON_VARIABLE_UPOS:
        return False
    value = normalize_token(token.text)
    return value in VARIABLE_WORDS or _is_single_letter_variable(token)


def _is_single_letter_variable(token: Token) -> bool:
    value = normalize_token(token.text)
    return len(value) == 1 and value.isalpha()


def bio_rows_from_detection(sentence_id: str, result: DetectionResult, *, include_punct: bool = False) -> list[BioTokenRow]:
    rows: list[BioTokenRow] = []
    for token in result.tokens:
        if token.mask_index is None:
            if include_punct:
                rows.append(BioTokenRow(sentence_id, len(rows), token.text, token.deprel, "O"))
            continue
        rows.append(BioTokenRow(sentence_id, len(rows), token.text, token.deprel, result.bio[token.mask_index]))
    return rows


def detect_bio_rows(
    sentence_id: str,
    sentence: str,
    markers: MarkerSet,
    *,
    include_punct: bool = False,
) -> list[BioTokenRow]:
    result = detect_from_tokens(parse_text(sentence), markers)
    return bio_rows_from_detection(sentence_id, result, include_punct=include_punct)


def bio_rows_by_stanza_sentence(
    result: DetectionResult,
    first_sentence_number: int,
    args: argparse.Namespace,
    *,
    include_punct: bool = False,
) -> tuple[list[BioTokenRow], int]:
    rows: list[BioTokenRow] = []
    sentence_indices = sorted({token.sentence_index for token in result.tokens})
    next_sentence_number = first_sentence_number
    for sentence_index in sentence_indices:
        sentence_id = sentence_id_for(args, next_sentence_number)
        next_sentence_number += 1
        token_index = 0
        for token in result.tokens:
            if token.sentence_index != sentence_index:
                continue
            if token.mask_index is None:
                if include_punct:
                    rows.append(BioTokenRow(sentence_id, token_index, token.text, token.deprel, "O"))
                    token_index += 1
                continue
            rows.append(BioTokenRow(sentence_id, token_index, token.text, token.deprel, result.bio[token.mask_index]))
            token_index += 1
    return rows, next_sentence_number


def split_input_sentences(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_input_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input is not None:
        with open(args.input, encoding="utf-8") as fh:
            return fh.read()
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        if stdin_text.strip():
            return stdin_text
    return DEFAULT_TEXT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Stanza Russian transcript -> BIO math-span tagger")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Input sentence/transcript")
    source.add_argument("--input", help="UTF-8 text file; each non-empty line is treated as one sentence")
    parser.add_argument("--output", help="Output BIO path; stdout by default")
    parser.add_argument(
        "--sentence-id-prefix",
        help="Optional sentence_id prefix override; otherwise --input uses file stem and --text/stdin uses a number",
    )
    parser.add_argument("--markers", default=str(MARKERS_PATH), help="Path to markers.yaml")
    parser.add_argument("--include-gazetteer", action="store_true", help="Also load generated pronunciation gazetteer")
    parser.add_argument("--gazetteer", default=str(GAZETTEER_PATH), help="Path to generated gazetteer YAML")
    parser.add_argument("--include-punct", action="store_true", help="Emit punctuation rows as O")
    parser.add_argument("--json", action="store_true", help="Print full DetectionResult JSON per input line")
    parser.add_argument("--download-stanza-ru", action="store_true", help="Download Stanza Russian model and exit")
    return parser.parse_args()


def sentence_id_for(args: argparse.Namespace, sentence_number: int) -> str:
    if args.sentence_id_prefix:
        return f"{args.sentence_id_prefix}_{sentence_number:04d}"
    if args.input:
        return f"{safe_sentence_id_part(Path(args.input).stem)}_{sentence_number:04d}"
    return str(sentence_number)


def safe_sentence_id_part(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", value.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "input"


def main() -> int:
    args = parse_args()
    try:
        if args.download_stanza_ru:
            download_ru_model()
            return 0

        markers = load_markers(args.markers, include_gazetteer=args.include_gazetteer, gazetteer_path=args.gazetteer)
        sentences = split_input_sentences(read_input_text(args))

        if args.json:
            output_lines = []
            for sentence in sentences:
                result = detect(
                    sentence,
                    markers_path=args.markers,
                    include_gazetteer=args.include_gazetteer,
                    gazetteer_path=args.gazetteer,
                )
                output_lines.append(json.dumps(result.to_dict(), ensure_ascii=False))
            output_text = "\n".join(output_lines) + "\n"
        else:
            output_lines = []
            if WRITE_HEADER:
                output_lines.append(DELIMITER.join(["sentence_id", "token_index", "token", "dep_label", "bio_tag"]))
            next_sentence_number = 0
            for sentence_number, sentence in enumerate(sentences):
                result = detect_from_tokens(parse_text(sentence), markers)
                rows, next_sentence_number = bio_rows_by_stanza_sentence(
                    result,
                    next_sentence_number,
                    args,
                    include_punct=args.include_punct,
                )
                for row in rows:
                    output_lines.append(
                        DELIMITER.join([row.sentence_id, str(row.token_index), row.token, row.dep_label, row.bio_tag])
                    )
            output_text = "\n".join(output_lines) + "\n"

        if args.output:
            with open(args.output, "w", encoding="utf-8", newline="") as fh:
                fh.write(output_text)
        else:
            sys.stdout.write(output_text)
        return 0
    except StanzaUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
