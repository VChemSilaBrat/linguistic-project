#!/usr/bin/env python3
"""
Парсер ответов LLM для сборки датасета.

Использование:
  1. Сохрани ответы модели в папку responses/ как response_01.txt, response_02.txt, ...
  2. Запусти: python parse_responses.py
  3. Результат: dataset.csv

Формат ожидаемого ввода (от LLM):
---1---
SENTENCE: Рассмотрим <F1>предел...</F1>, который равен <F2>нулю</F2>.
F1: \lim_{x\to 0}...
F2: 0
"""

import re, csv, os, glob, sys

def parse_response(text):
    """Parse one LLM response into list of samples."""
    samples = []
    # Split by ---N--- markers
    blocks = re.split(r'---\s*\d+\s*---', text)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Extract SENTENCE line
        sent_match = re.search(r'SENTENCE:\s*(.+?)(?:\n|$)', block, re.DOTALL)
        if not sent_match:
            continue
        
        sentence = sent_match.group(1).strip()
        
        # Extract formula tags from sentence
        formula_spans = []
        for m in re.finditer(r'<F(\d+)>(.*?)</F\1>', sentence):
            formula_spans.append({
                'id': int(m.group(1)),
                'text': m.group(2).strip(),
                'start': m.start(),
                'end': m.end(),
            })
        
        if len(formula_spans) < 2:
            print(f"  WARN: <2 formulas found in: {sentence[:60]}...")
            continue
        
        # Extract LaTeX lines
        latex_map = {}
        for m in re.finditer(r'F(\d+):\s*(.+?)(?:\n|$)', block):
            fid = int(m.group(1))
            latex = m.group(2).strip()
            # Remove markdown backticks if present
            latex = re.sub(r'^`+|`+$', '', latex)
            latex = latex.strip()
            latex_map[fid] = latex
        
        # Build clean sentence (without tags)
        clean_sentence = sentence
        for m in re.finditer(r'<F\d+>(.*?)</F\d+>', sentence):
            pass  # we'll do replacement below
        clean_sentence = re.sub(r'<F\d+>', '', clean_sentence)
        clean_sentence = re.sub(r'</F\d+>', '', clean_sentence)
        clean_sentence = clean_sentence.strip()
        
        # Build tagged sentence (with [MATH]...[/MATH] markers)
        tagged = re.sub(r'<F\d+>', '[MATH]', sentence)
        tagged = re.sub(r'</F\d+>', '[/MATH]', tagged)
        tagged = tagged.strip()
        
        # Collect formulas in order
        formulas = []
        for span in sorted(formula_spans, key=lambda s: s['id']):
            fid = span['id']
            formulas.append({
                'pronunciation': span['text'],
                'latex': latex_map.get(fid, '???'),
            })
        
        samples.append({
            'sentence_clean': clean_sentence,
            'sentence_tagged': tagged,
            'num_formulas': len(formulas),
            'f1_pron': formulas[0]['pronunciation'] if len(formulas) > 0 else '',
            'f1_latex': formulas[0]['latex'] if len(formulas) > 0 else '',
            'f2_pron': formulas[1]['pronunciation'] if len(formulas) > 1 else '',
            'f2_latex': formulas[1]['latex'] if len(formulas) > 1 else '',
            'f3_pron': formulas[2]['pronunciation'] if len(formulas) > 2 else '',
            'f3_latex': formulas[2]['latex'] if len(formulas) > 2 else '',
        })
    
    return samples


def main():
    resp_dir = os.path.join(os.path.dirname(__file__), 'responses')
    if not os.path.isdir(resp_dir):
        os.makedirs(resp_dir)
        print(f"Создана папка {resp_dir}/")
        print("Положи туда файлы response_01.txt, response_02.txt, ...")
        print("Затем запусти скрипт повторно.")
        return
    
    files = sorted(glob.glob(os.path.join(resp_dir, '*.txt')))
    if not files:
        print(f"В папке {resp_dir}/ нет .txt файлов!")
        return
    
    all_samples = []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        samples = parse_response(text)
        print(f"  {fname}: {len(samples)} samples parsed")
        all_samples.extend(samples)
    
    # Write CSV
    out_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sentence_clean', 'sentence_tagged', 'num_formulas',
            'f1_pron', 'f1_latex', 'f2_pron', 'f2_latex', 'f3_pron', 'f3_latex',
        ])
        writer.writeheader()
        writer.writerows(all_samples)
    
    print(f"\n=== ИТОГО: {len(all_samples)} строк → {out_path} ===")
    
    # Stats
    n2 = sum(1 for s in all_samples if s['num_formulas'] == 2)
    n3 = sum(1 for s in all_samples if s['num_formulas'] == 3)
    print(f"  С 2 формулами: {n2}")
    print(f"  С 3 формулами: {n3}")


if __name__ == '__main__':
    main()
