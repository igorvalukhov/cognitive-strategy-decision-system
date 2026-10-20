const $ = (id) => document.getElementById(id);
const input = $('text');
const percent = (value) => new Intl.NumberFormat('ru-RU', {style: 'percent', maximumFractionDigits: 1}).format(value);
const updateCounter = () => { $('counter').textContent = `${input.value.length.toLocaleString('ru-RU')} / 200 000`; };
input.addEventListener('input', updateCounter);
$('example').addEventListener('click', () => {
  input.value = 'The sudden increase in interest rates led to a massive outflow of foreign investment. Political instability in the region caused a stock market crash. The Senate reviewed the fiscal reform report but deferred action until next quarter.';
  updateCounter();
  input.focus();
});

function cell(row, text, className) {
  const element = document.createElement('td');
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  row.append(element);
  return element;
}

function fragments(row, spans) {
  const container = cell(row);
  for (const span of spans) {
    const fragment = document.createElement('p');
    fragment.className = 'fragment';
    fragment.textContent = span.text;
    const score = document.createElement('small');
    score.textContent = percent(span.confidence);
    fragment.append(score);
    container.append(fragment);
  }
}

function render(data) {
  $('source-label').textContent = data.source ? `Источник: ${data.source}` : 'Источник: текстовое поле';
  $('file-warnings').textContent = (data.warnings || []).join(' ');
  $('file-warnings').hidden = !(data.warnings || []).length;
  const stats = data.statistics;
  $('total').textContent = stats.total_sentences;
  $('causal').textContent = stats.causal_sentences;
  $('non-causal').textContent = stats.non_causal_sentences;
  $('skipped-count').textContent = stats.skipped_sentences;
  $('result-caption').textContent = `Обработано ${stats.analyzed_sentences} из ${stats.total_sentences}`;
  $('rows').replaceChildren();
  for (const result of data.rows) {
    const row = document.createElement('tr');
    cell(row, result.sentence_number);
    const sentence = cell(row, result.sentence);
    if (result.ambiguous) {
      const note = document.createElement('span');
      note.className = 'ambiguous';
      note.textContent = 'Несколько фрагментов: соответствие пар не определено';
      sentence.append(note);
    }
    fragments(row, result.causes);
    fragments(row, result.effects);
    cell(row, percent(result.confidence), 'confidence');
    $('rows').append(row);
  }
  $('empty').hidden = data.rows.length > 0;
  $('skipped').replaceChildren();
  for (const item of data.skipped) {
    const paragraph = document.createElement('p');
    paragraph.textContent = `Предложение ${item.sentence_number} пропущено: ${item.reason} Текст: ${item.sentence}`;
    $('skipped').append(paragraph);
  }
  $('skipped').hidden = data.skipped.length === 0;
  $('results').hidden = false;
}

$('analysis-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('error').hidden = true;
  $('results').hidden = true;
  const file = $('document').files[0];
  if (!file && !input.value.trim()) {
    $('error').textContent = 'Введите текст для анализа.';
    $('error').hidden = false;
    return;
  }
  $('analyze').disabled = true;
  $('example').disabled = true;
  input.disabled = true;
  $('document').disabled = true;
  $('clear-file').disabled = true;
  $('analysis-form').setAttribute('aria-busy', 'true');
  $('status').textContent = 'Анализируем… Первая загрузка модели может занять несколько минут.';
  try {
    let response;
    if (file) {
      if (file.size > 10 * 1024 * 1024) throw new Error('Максимальный размер файла — 10 МБ.');
      const form = new FormData();
      form.append('file', file);
      response = await fetch('/api/analyze-file', {method: 'POST', body: form});
    } else {
      response = await fetch('/api/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: input.value})});
    }
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Проверьте текст: максимум 200 000 символов.');
    render(data);
    $('status').textContent = 'Анализ завершён.';
  } catch (error) {
    $('error').textContent = error instanceof TypeError ? 'Нет связи с сервером. Проверьте, что приложение запущено.' : error.message;
    $('error').hidden = false;
    $('status').textContent = 'Анализ не выполнен.';
  } finally {
    $('analyze').disabled = false;
    $('example').disabled = false;
    input.disabled = false;
    $('document').disabled = false;
    $('clear-file').disabled = false;
    $('analysis-form').removeAttribute('aria-busy');
  }
});

$('clear-file').addEventListener('click', () => { $('document').value = ''; });
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(tab => { tab.classList.toggle('active', tab === button); tab.setAttribute('aria-pressed', String(tab === button)); });
  document.querySelectorAll('.tab-panel').forEach(panel => { panel.hidden = panel.id !== button.dataset.panel; });
}));

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Проверьте параметры запроса.');
  return data;
}

let generationTimer;
function showJob(job) {
  const active = ['running', 'cancelling'].includes(job.status);
  $('generate').disabled = active;
  $('cancel-generation').hidden = !active;
  $('cancel-generation').disabled = job.status === 'cancelling';
  ['provider', 'domain', 'sample-count', 'negative-ratio', 'seed'].forEach(id => { $(id).disabled = active; });
  $('generation-progress').max = job.count || 2000;
  $('generation-progress').value = job.accepted || 0;
  $('generation-status').textContent = job.message || 'Генерация ещё не запускалась.';
  $('generation-detail').textContent = job.id ? `Запуск ${job.id} · ${job.accepted || 0}/${job.count} примеров · ${job.attempts || 0} попыток · ${job.status}` : '';
  clearTimeout(generationTimer);
  if (active) generationTimer = setTimeout(pollJob, 2500);
}
async function pollJob() {
  try { showJob(await api('/api/generator/status')); }
  catch (error) { $('generation-status').textContent = 'Нет связи с сервером. Состояние генерации неизвестно.'; generationTimer = setTimeout(pollJob, 5000); }
}
$('generation-form').addEventListener('submit', async event => {
  event.preventDefault();
  $('generate').disabled = true;
  try {
    showJob(await api('/api/generator/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
      provider: $('provider').value, domain: $('domain').value.trim(), count: Number($('sample-count').value), negative_ratio: Number($('negative-ratio').value)/100, seed: Number($('seed').value)
    })}));
  } catch (error) { $('generation-status').textContent = error.message; $('generate').disabled = false; }
});
$('cancel-generation').addEventListener('click', async () => {
  try { showJob(await api('/api/generator/cancel', {method: 'POST'})); }
  catch (error) { $('generation-status').textContent = error.message; }
});
$('check-gigachat').addEventListener('click', async () => {
  $('check-gigachat').disabled = true;
  $('gigachat-status').textContent = 'Проверка подключения…';
  try { const health = await api('/api/generator/health?provider=' + encodeURIComponent($('provider').value)); $('gigachat-status').textContent = health.available ? `Подключён: ${health.models.join(', ')} · ${health.base_url}` : health.message; }
  catch (error) { $('gigachat-status').textContent = error.message; }
  finally { $('check-gigachat').disabled = false; }
});
$('download-bundle').addEventListener('click', async () => {
  $('download-bundle').disabled = true;
  try {
    const response = await fetch('/api/colab/bundle?dataset=' + encodeURIComponent($('training-dataset').value.trim()));
    if (!response.ok) { const data = await response.json(); throw new Error(data.detail); }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a'); link.href = url; link.download = 'module1-training.zip'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    $('training-status').textContent = 'Пакет скачан. Загрузите его из блокнота Colab.';
  } catch (error) { $('training-status').textContent = error.message; }
  finally { $('download-bundle').disabled = false; }
});
pollJob();

$('provider').addEventListener('change', () => {
  const name = $('provider').value === 'qwen' ? 'Qwen' : 'GigaChat';
  $('check-gigachat').textContent = 'Проверить ' + name;
  $('generate').textContent = 'Генерировать через ' + name + ' →';
  $('gigachat-status').textContent = 'Проверьте подключение к ' + name + '.';
});
