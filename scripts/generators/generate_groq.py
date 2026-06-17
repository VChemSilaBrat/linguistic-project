#!/usr/bin/env python3
"""Generate ~500 lecture-style Russian sentences with [MATH]...[/MATH] spans via Groq API.

Why this script exists: BERT-0 (trained on 185 clean synthetic sentences) plateaus at
span-F1=0.19. The 185 sentences are too uniform (always 2-3 clean isolated formulas).
Real lecture data (Gosha test set) has single-token spans, discourse markers touching
math spans, mixed short/long spans in the same sentence, and lecturer speech patterns
(corrections, asides, repetitions). This script asks an LLM to generate more sentences
that match that real distribution, using s2l formula pronunciations as raw material.

Input:  data/s2l/russian_sentences_train.csv  (columns: pronunciation, sentence_normalized)
Output: data/generated/llm_sentences.txt       (one [MATH]-tagged sentence per line)
        data/generated/llm_sentences_raw.jsonl (full API responses, for debugging)

Usage:
    export GROQ_API_KEY=...
    python scripts/generators/generate_groq.py --num_batches 50 --batch_size 10
    python scripts/generators/generate_groq.py --dry-run   # print prompt, no API call
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Few-shot examples: real lecture sentences from Gosha's manual dataset.
# These show the target style — short/long spans mixed, single-token spans,
# discourse markers touching or inside math spans, lecturer speech patterns.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    "Итак, число B называется [MATH]пределом функций х от двух переменных х и грек в точке A[/MATH] . "
    "Если [MATH]существует такое дельта[/MATH] , только все-таки начнем с другого.",

    "Тут [MATH]единицы или минус единицы[/MATH] , верно? Потому что если я [MATH]фиксирую[/MATH] у, "
    "это любое число отличное [MATH]от[/MATH] нуля, [MATH]и устремляю х к[/MATH] нулю, что получается?",

    "Ну да. Длина [MATH]1 на 4[/MATH] . Так, значит сколько там будет [MATH]10 четвертых , 5 вторых плюс 5[/MATH] . "
    "Это значит сколько, это [MATH]15 вторых[/MATH] , да?",

    "Раз [MATH]логарифм[/MATH] , то все. Мы же физики. Для нас натуральный, значит. "
    "От [MATH]тангенса х куб[/MATH] . Вот. Возьмите эту производную.",

    "Есть плотность [MATH]P от х тета относительно некоторой меры μ[/MATH] .",

    "Давайте я выберу в пространстве [MATH]V[/MATH] базис, состоящий из двух векторов, которые перпендикулярны "
    "друг другу и имеют длину [MATH]1[/MATH] , а в пространстве [MATH]W[/MATH] , которое совпадает с пространством "
    "[MATH]V[/MATH] , я выберу тот же самый базис.",
]

# What NOT to produce — too clean, always exactly 2 isolated spans with generic connectors.
BAD_EXAMPLE = (
    "Рассмотрим [MATH]предел при эн стремящемся к бесконечности от один делить на эн[/MATH] и "
    "[MATH]сумма от один делить на эн[/MATH] ."
)

MATH_TOPICS = [
    "матанализ", "линейная алгебра", "теория вероятностей", "дифуры",
    "дискретная математика", "мат. статистика",
]

DISCOURSE_MARKERS = ["ну вот", "значит", "смотрите", "то есть", "так", "ну да", "хорошо"]

PROMPT_TEMPLATE = """Ты помогаешь собрать обучающие данные для модели разметки математических \
выражений в расшифровках русскоязычных лекций. Нужно сгенерировать {batch_size} предложений \
в стиле РЕАЛЬНОЙ устной лекции, где текстовые описания математических формул помечены тегами \
[MATH]...[/MATH].

КРИТИЧЕСКИ ВАЖНО — разнообразие. В этой партии из {batch_size} предложений должно быть:
- 2 предложения с РОВНО ОДНИМ математическим спаном, КОРОТКИМ (1-3 токена), например "[MATH]тета[/MATH]" или "[MATH]V[/MATH]"
- 2 предложения с РОВНО ОДНИМ математическим спаном, ДЛИННЫМ (8+ токенов)
- 2 предложения с 2 РАЗДЕЛЬНЫМИ спанами разной длины (один короткий, один длинный)
- 2 предложения с 3+ спанами, среди которых хотя бы один из ОДНОГО токена
- 2 предложения, где речевые маркеры ("ну вот", "значит", "то есть", "смотрите") стоят ВПЛОТНУЮ \
к спану или прямо ВНУТРИ него

Стилистические требования:
- Это устная речь лектора: используй маркеры типа "ну вот", "значит", "смотрите", "то есть", \
"так", а также самокоррекции и повторы ("это... это будет", "не, подождите, давайте по-другому")
- Математические спаны — это РУССКИЕ СЛОВЕСНЫЕ описания формул (никакого LaTeX, никаких \
символов вроде = или ^), например "икс в квадрате плюс два" или "предел при эн стремящемся к бесконечности"
- Спаны могут быть одним токеном: переменной ("[MATH]V[/MATH]", "[MATH]тета[/MATH]"), числом, \
или одним словом ("[MATH]логарифм[/MATH]")
- Разные предложения — разные темы: {topics}
- НЕ делай все предложения одинаковыми по структуре — избегай шаблона "ровно 2 изолированных \
спана через generic-связку", вот пример ТОГО, ЧЕГО НЕ НАДО:
  {bad_example}

Вот примеры РЕАЛЬНОГО стиля лекций (из расшифровок), на которые нужно ориентироваться:
{few_shot}

Используй как "сырьё" следующие произнесённые формулы (вплетай их в предложения свободно, \
меняя обёртку и контекст, не обязательно использовать все и можно менять формулировку):
{formulas}

Можешь также придумывать свои простые математические отсылки (одна переменная, "больше нуля", \
"стремится к", "в степени", "больше или равно" и т.п.), не обязательно беря их из списка выше.

Выведи РОВНО {batch_size} предложений, каждое на отдельной строке, без нумерации, без \
дополнительных комментариев, без markdown — просто текст предложений с тегами [MATH]...[/MATH]."""


def load_s2l_pronunciations(s2l_path):
    """Load formula pronunciations from the s2l CSV (column: pronunciation)."""
    pronunciations = []
    with open(s2l_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pronunciation" not in reader.fieldnames:
            raise AssertionError(
                f"Expected 'pronunciation' column in {s2l_path}, got {reader.fieldnames}"
            )
        for row in reader:
            p = row["pronunciation"].strip()
            if p:
                pronunciations.append(p)
    if not pronunciations:
        raise AssertionError(f"No pronunciations loaded from {s2l_path}")
    return pronunciations


def build_prompt(batch_size, sampled_formulas):
    formulas_block = "\n".join(f"- {f}" for f in sampled_formulas)
    few_shot_block = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(FEW_SHOT_EXAMPLES))
    return PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        topics=", ".join(MATH_TOPICS),
        bad_example=BAD_EXAMPLE,
        few_shot=few_shot_block,
        formulas=formulas_block,
    )


# "спан" (span) only appears in our own prompt's instructions — never in real Russian
# lecture speech. Some models occasionally leak their planning/reasoning about the
# task as one or more of the "10 lines" (e.g. "Первое требование: 2 предложения с
# одним коротким спаном...", "**2 предложения с длинными спанами**..."). Any line
# containing this word is that leakage, not a generated sentence, and is dropped here
# so it never reaches llm_sentences.txt in the first place.
META_LEAK_RE = re.compile(r"спан", re.IGNORECASE)


def parse_response_text(text):
    """Split raw LLM output into candidate sentences and validate [MATH] tags.

    Returns (valid_sentences, warnings).
    """
    valid = []
    warnings = []
    lines = [ln.strip() for ln in text.splitlines()]
    for ln in lines:
        if not ln:
            continue
        # Strip leading numbering like "1. " or "1) " if the model added it anyway.
        ln = re.sub(r"^\d+[\.\)]\s*", "", ln)

        n_open = ln.count("[MATH]")
        n_close = ln.count("[/MATH]")
        if n_open != n_close:
            warnings.append(f"Mismatched MATH tags, skipping: {ln[:80]!r}")
            continue
        if n_open == 0:
            warnings.append(f"No MATH span, skipping: {ln[:80]!r}")
            continue
        if META_LEAK_RE.search(ln):
            warnings.append(f"Leaked task-planning text (contains 'спан'), skipping: {ln[:80]!r}")
            continue

        # Check for nesting: a second [MATH] must not appear before the
        # matching [/MATH] closes.
        depth = 0
        nested = False
        for tok in re.findall(r"\[MATH\]|\[/MATH\]", ln):
            if tok == "[MATH]":
                if depth > 0:
                    nested = True
                    break
                depth += 1
            else:
                depth -= 1
                if depth < 0:
                    nested = True
                    break
        if nested or depth != 0:
            warnings.append(f"Nested/unbalanced MATH tags, skipping: {ln[:80]!r}")
            continue

        valid.append(ln)
    return valid, warnings


def call_groq_with_retry(client, model, prompt, max_retries=5, base_delay=5):
    """Call the Groq chat completion API with exponential backoff on rate limits."""
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=2048,
            )
            return response
        except Exception as e:  # noqa: BLE001 - we want to retry on any API error
            last_err = e
            err_str = str(e).lower()
            is_rate_limit = "rate limit" in err_str or "429" in err_str
            delay = base_delay * (2 ** attempt) if is_rate_limit else base_delay
            print(
                f"  WARNING: API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"Groq API call failed after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate lecture-style Russian [MATH]-tagged sentences via Groq API."
    )
    parser.add_argument("--num_batches", type=int, default=50,
                         help="Number of API calls to make (default: 50)")
    parser.add_argument("--batch_size", type=int, default=10,
                         help="Sentences requested per API call (default: 10)")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile",
                         help="Groq model name. If unavailable, pass e.g. "
                              "llama-3.1-70b-versatile or check `groq models list`.")
    parser.add_argument("--s2l_path", type=str, default=None,
                         help="Path to s2l russian_sentences_train.csv "
                              "(default: data/s2l/russian_sentences_train.csv from repo root)")
    parser.add_argument("--output", type=str, default=None,
                         help="Output path for tagged sentences "
                              "(default: data/generated/llm_sentences.txt)")
    parser.add_argument("--raw_output", type=str, default=None,
                         help="Output path for raw JSONL API responses "
                              "(default: data/generated/llm_sentences_raw.jsonl)")
    parser.add_argument("--sleep", type=float, default=2.0,
                         help="Seconds to sleep between API calls (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling formulas")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the prompt for one batch and exit, no API call")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent
    s2l_path = Path(args.s2l_path) if args.s2l_path else base / "data" / "s2l" / "russian_sentences_train.csv"
    output_path = Path(args.output) if args.output else base / "data" / "generated" / "llm_sentences.txt"
    raw_output_path = Path(args.raw_output) if args.raw_output else base / "data" / "generated" / "llm_sentences_raw.jsonl"

    assert s2l_path.exists(), f"s2l file not found: {s2l_path}"

    random.seed(args.seed)
    pronunciations = load_s2l_pronunciations(s2l_path)
    print(f"Loaded {len(pronunciations)} formula pronunciations from {s2l_path}")

    if args.dry_run:
        sampled = random.sample(pronunciations, min(args.batch_size, len(pronunciations)))
        prompt = build_prompt(args.batch_size, sampled)
        print("\n" + "=" * 80)
        print("DRY RUN — prompt for one batch (no API call made):")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print(f"\nWould make {args.num_batches} batches x {args.batch_size} sentences "
              f"= {args.num_batches * args.batch_size} target sentences.")
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.", file=sys.stderr)
        print("Set it with: export GROQ_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("ERROR: groq package not installed. Run: pip install groq", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_valid = 0
    total_warnings = 0

    # Open in append mode so partial progress survives a crash / rate-limit abort.
    with open(output_path, "a", encoding="utf-8") as out_f, \
         open(raw_output_path, "a", encoding="utf-8") as raw_f:

        for batch_idx in range(args.num_batches):
            sampled = random.sample(pronunciations, min(args.batch_size, len(pronunciations)))
            prompt = build_prompt(args.batch_size, sampled)

            print(f"[batch {batch_idx + 1}/{args.num_batches}] calling Groq ({args.model})...")
            try:
                response = call_groq_with_retry(client, args.model, prompt)
            except RuntimeError as e:
                print(f"  ERROR: giving up on batch {batch_idx + 1}: {e}", file=sys.stderr)
                continue

            raw_f.write(json.dumps({
                "batch_idx": batch_idx,
                "model": args.model,
                "sampled_formulas": sampled,
                "response": response.model_dump() if hasattr(response, "model_dump") else str(response),
            }, ensure_ascii=False) + "\n")
            raw_f.flush()

            text = response.choices[0].message.content
            valid, warnings = parse_response_text(text)

            for w in warnings:
                print(f"  WARNING: {w}")
            total_warnings += len(warnings)

            for sent in valid:
                out_f.write(sent + "\n")
            out_f.flush()
            total_valid += len(valid)

            print(f"  -> {len(valid)} valid sentences, {len(warnings)} skipped")

            if batch_idx < args.num_batches - 1:
                time.sleep(args.sleep)

    print(f"\nDone. {total_valid} valid sentences written to {output_path} "
          f"({total_warnings} malformed sentences skipped).")
    print(f"Raw API responses logged to {raw_output_path}")


if __name__ == "__main__":
    main()
