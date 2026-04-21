"""
convert_eventstoryline_to_bioes.py
Конвертирует EventStoryLine v1.5 в BIOES-разметку.

Использование:
    python convert_eventstoryline_to_bioes.py
    
Укажи путь к датасету в переменной DATASET_FOLDER ниже.
Датасет: https://github.com/tommasoc80/EventStoryLine
"""

import os
import xml.etree.ElementTree as ET

# ─── Укажи путь к папке с аннотированными данными ──────────────────────────
DATASET_FOLDER = "EventStoryLine/annotated_data/v1.5"
OUTPUT_FILE    = "/content/drive/MyDrive/Colab Notebooks/Project2/datasets/eventstoryline_bioes.txt"
# ────────────────────────────────────────────────────────────────────────────

def find_all_xml_files(folder_path):
    xml_files = []
    for rootdir, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith('.xml') or fname.endswith('.xml.xml'):
                xml_files.append(os.path.join(rootdir, fname))
    return sorted(xml_files)


def bioes_tag(indices, label, tags):
    """Устанавливает BIOES-метки для списка индексов."""
    if not indices:
        return
    if len(indices) == 1:
        tags[indices[0]] = f"S-{label}"
    else:
        tags[indices[0]] = f"B-{label}"
        for idx in indices[1:-1]:
            tags[idx] = f"I-{label}"
        tags[indices[-1]] = f"E-{label}"


def parse_eventstoryline(folder_path, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_sentences  = 0
    causal_sentences = 0
    skipped_files    = 0

    with open(output_file, 'w', encoding='utf-8') as fout:
        xml_files = find_all_xml_files(folder_path)
        print(f"Найдено XML-файлов: {len(xml_files)}")

        for fpath in xml_files:
            try:
                tree = ET.parse(fpath)
                root = tree.getroot()

                # ── Токены ──────────────────────────────────────────────
                tokens_by_sent = {}   # sent_id → [(t_id, word), ...]
                tokens_by_tid  = {}   # t_id    → (sent_id, word)

                for token in root.findall('.//token'):
                    sent_id = token.attrib.get('sentence', '')
                    word    = token.text or ''
                    tid     = token.attrib.get('t_id', '')
                    if not sent_id or not tid:
                        continue
                    tokens_by_sent.setdefault(sent_id, []).append((tid, word))
                    tokens_by_tid[tid] = (sent_id, word)

                # ── События (ACTION_*) ───────────────────────────────────
                events = {}   # m_id → [t_id, ...]
                markables = root.find('.//Markables')
                if markables is not None:
                    for markable in markables:
                        if markable.tag.startswith('ACTION_'):
                            m_id  = markable.attrib.get('m_id', '')
                            t_ids = [ta.attrib['t_id']
                                     for ta in markable.findall('.//token_anchor')
                                     if 't_id' in ta.attrib]
                            if m_id:
                                events[m_id] = t_ids

                # ── Каузальные связи ─────────────────────────────────────
                causal_by_sent = {}   # sent_id → [(source_tids, target_tids), ...]

                for plink in root.findall('.//PLOT_LINK'):
                    causes    = plink.attrib.get('CAUSES',    'FALSE') == 'TRUE'
                    caused_by = plink.attrib.get('CAUSED_BY', 'FALSE') == 'TRUE'
                    if not (causes or caused_by):
                        continue

                    src_el = plink.find('source')
                    tgt_el = plink.find('target')
                    if src_el is None or tgt_el is None:
                        continue

                    source_id  = src_el.attrib.get('m_id', '')
                    target_id  = tgt_el.attrib.get('m_id', '')
                    source_tids = events.get(source_id, [])
                    target_tids = events.get(target_id, [])
                    if not source_tids or not target_tids:
                        continue

                    # Определяем предложение по первому токену источника
                    sent_id = None
                    for tid in source_tids:
                        if tid in tokens_by_tid:
                            sent_id = tokens_by_tid[tid][0]
                            break
                    if not sent_id:
                        continue

                    # Если CAUSED_BY - меняем роли
                    if caused_by and not causes:
                        source_tids, target_tids = target_tids, source_tids

                    causal_by_sent.setdefault(sent_id, []).append(
                        (source_tids, target_tids))

                # ── Формируем BIOES-строки ───────────────────────────────
                for sent_id, token_list in sorted(tokens_by_sent.items(),
                                                   key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                    total_sentences += 1
                    tokens = [w for _, w in token_list]
                    tags   = ['O'] * len(tokens)
                    has_causal = False

                    if sent_id in causal_by_sent:
                        for source_tids, target_tids in causal_by_sent[sent_id]:
                            def get_indices(tid_list):
                                idxs = []
                                for tid in tid_list:
                                    for i, (tok_tid, _) in enumerate(token_list):
                                        if tok_tid == tid:
                                            idxs.append(i)
                                return sorted(set(idxs))

                            c_idxs = get_indices(source_tids)
                            e_idxs = get_indices(target_tids)

                            if c_idxs:
                                bioes_tag(c_idxs, 'CAUSE', tags)
                                has_causal = True
                            if e_idxs:
                                bioes_tag(e_idxs, 'EFFECT', tags)
                                has_causal = True

                    if has_causal:
                        causal_sentences += 1

                    line = ' '.join(f'{w}/{t}' for w, t in zip(tokens, tags))
                    fout.write(line + '\n')

            except Exception as e:
                print(f"  ⚠ Ошибка в {fpath}: {e}")
                skipped_files += 1

    # ── Статистика ──────────────────────────────────────────────────────────
    non_causal = total_sentences - causal_sentences
    print()
    print("=" * 60)
    print("📊 СТАТИСТИКА - EventStoryLine")
    print("=" * 60)
    print(f"  XML-файлов с ошибками:    {skipped_files}")
    print(f"  Всего предложений:        {total_sentences:>6}")
    print(f"  С каузальными парами:     {causal_sentences:>6}")
    print(f"  Без каузальных связей:    {non_causal:>6}")
    if total_sentences:
        print(f"  Доля каузальных:          {causal_sentences/total_sentences:.2%}")
    print(f"  Результат сохранён в:     {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    if not os.path.exists(DATASET_FOLDER):
        print(f"⚠  Папка не найдена: {DATASET_FOLDER}")
        print("   Скачай датасет: https://github.com/tommasoc80/EventStoryLine")
        print(f"   и распакуй рядом со скриптом так, чтобы путь совпадал с DATASET_FOLDER")
    else:
        parse_eventstoryline(DATASET_FOLDER, OUTPUT_FILE)
