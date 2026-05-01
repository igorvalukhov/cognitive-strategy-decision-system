"""
convert_semeval2010_to_bioes.py
Конвертирует SemEval-2010 Task 8 в BIOES-разметку.
"""

import os
import re

# ─── Ручные пути (для Google Drive) ────────────────────────────────────────
GOOGLE_DRIVE_BASE = "/content/drive/MyDrive/Colab Notebooks/Project2"

LOCAL_TRAIN_FILE = os.path.join(GOOGLE_DRIVE_BASE, "TRAIN_FILE.TXT")
LOCAL_TEST_FILE  = os.path.join(GOOGLE_DRIVE_BASE, "TEST_FILE_FULL.TXT")

OUTPUT_TRAIN = os.path.join(GOOGLE_DRIVE_BASE, "datasets", "semeval2010_train_bioes.txt")
OUTPUT_TEST  = os.path.join(GOOGLE_DRIVE_BASE, "datasets", "semeval2010_test_bioes.txt")
# ────────────────────────────────────────────────────────────────────────────

def tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text)

def find_span_indices(tokens, span_str):
    if not span_str:
        return []
    span_tokens = tokenize(span_str)
    n = len(span_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == span_tokens:
            return list(range(i, i + n))
    return []

def bioes_tag(indices, label, tags):
    if not indices:
        return
    if len(indices) == 1:
        tags[indices[0]] = f"S-{label}"
    else:
        tags[indices[0]] = f"B-{label}"
        for idx in indices[1:-1]:
            tags[idx] = f"I-{label}"
        tags[indices[-1]] = f"E-{label}"

def extract_entity(text: str, tag: str):
    start = text.find(f'<{tag}>')
    end   = text.find(f'</{tag}>')
    if start == -1 or end == -1:
        return None
    return text[start + len(f'<{tag}>'):end].strip()

def clean_text(raw_line: str) -> str:
    no_num = re.sub(r'^\d+\s*', '', raw_line.strip())
    if no_num.startswith('"') and no_num.endswith('"'):
        no_num = no_num[1:-1]
    for tag in ['<e1>', '</e1>', '<e2>', '</e2>']:
        no_num = no_num.replace(tag, '')
    return no_num.strip()

def parse_relation(relation_line: str):
    m = re.match(r'(\w[\w-]*)\((\w+),(\w+)\)', relation_line.strip())
    if not m:
        return None
    rel_type = m.group(1)
    arg1     = m.group(2)
    arg2     = m.group(3)
    if rel_type != 'Cause-Effect':
        return None
    return rel_type, arg1, arg2

def convert_semeval(input_file: str, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total      = 0
    causal     = 0
    no_match   = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = [l.rstrip('\n') for l in fin]
        i = 0
        while i < len(lines):
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            
            text_line     = lines[i];     i += 1
            relation_line = lines[i] if i < len(lines) else '';  i += 1
            comment_line  = lines[i] if i < len(lines) else '';  i += 1
            total += 1
            
            clean = clean_text(text_line)
            tokens = tokenize(clean)
            tags   = ['O'] * len(tokens)
            
            rel_info = parse_relation(relation_line)
            is_causal = False
            
            if rel_info:
                _, cause_tag, effect_tag = rel_info
                cause_str  = extract_entity(text_line, cause_tag)
                effect_str = extract_entity(text_line, effect_tag)
                c_idxs = find_span_indices(tokens, cause_str)
                e_idxs = find_span_indices(tokens, effect_str)
                
                if c_idxs or e_idxs:
                    bioes_tag(c_idxs,  'CAUSE',  tags)
                    bioes_tag(e_idxs,  'EFFECT', tags)
                    is_causal = True
                else:
                    no_match += 1
            
            if is_causal:
                causal += 1
            
            line_out = ' '.join(f'{w}/{t}' for w, t in zip(tokens, tags))
            fout.write(line_out + '\n')
    
    non_causal = total - causal
    print()
    print("=" * 60)
    print(f"📊 СТАТИСТИКА — SemEval-2010 Task 8  [{os.path.basename(input_file)}]")
    print("=" * 60)
    print(f"  Всего предложений:        {total:>6}")
    print(f"  С каузальными парами:     {causal:>6}")
    print(f"  Без каузальных связей:    {non_causal:>6}")
    if total:
        print(f"  Доля каузальных:          {causal/total:.2%}")
    print(f"  Не найдено span-ов:       {no_match:>6}  (записаны как O)")
    print(f"  Результат сохранён в:     {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs(os.path.join(GOOGLE_DRIVE_BASE, "datasets"), exist_ok=True)
    
    train_path = LOCAL_TRAIN_FILE
    test_path  = LOCAL_TEST_FILE
    
    converted = False
    
    if train_path and os.path.exists(train_path):
        print(f"✅ Конвертируем TRAIN: {train_path}")
        convert_semeval(train_path, OUTPUT_TRAIN)
        converted = True
    else:
        print(f"⚠️ TRAIN файл не найден: {train_path}")
    
    if test_path and os.path.exists(test_path):
        print(f"✅ Конвертируем TEST: {test_path}")
        convert_semeval(test_path, OUTPUT_TEST)
        converted = True
    else:
        print(f"⚠️ TEST файл не найден: {test_path}")
    
    if converted:
        print("\n✅ Конвертация завершена!")