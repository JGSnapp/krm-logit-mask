const $ = s => document.querySelector(s);
const state = {profiles: [], catalogs: [], details: {}, result: null};

async function init() {
  try {
    const [health, catalogs, profiles] = await Promise.all([
      fetch('/api/health').then(r => r.json()),
      fetch('/api/catalogs').then(r => r.json()),
      fetch('/profiles.json').then(r => r.json()),
    ]);
    state.profiles = profiles;
    state.catalogs = catalogs.catalogs;
    $('#catalogSelect').innerHTML = catalogs.catalogs.map(c =>
      `<option value="${esc(c.id)}" ${c.id === catalogs.default_catalog_id ? 'selected' : ''}>${esc(c.name)} · ${c.competencies} компетенций</option>`).join('');
    $('#modelSelect').innerHTML = (health.models || []).map(m =>
      `<option value="${esc(m.id)}" ${m.id === (health.active_model_id || health.default_model_id) ? 'selected' : ''}>${esc(m.name)} · ${esc(m.description)}</option>`).join('');
    profiles.forEach(p => $('#profilePreset').insertAdjacentHTML('beforeend',
      `<option value="${esc(p.id)}">${esc(p.title)} · ${esc(p.domain)}</option>`));
    if (profiles.length) { $('#profilePreset').value = profiles[0].id; $('#candidateText').value = profiles[0].text; }
    $('.status').classList.add('ok');
    $('#statusText').textContent = health.model_loaded ? 'модель загружена · CUDA ready' : 'runtime готов · модель загрузится по запросу';
  } catch (error) {
    $('#statusText').textContent = 'backend недоступен';
    console.error(error);
  }
}

function insertProfile() {
  const profile = state.profiles.find(p => p.id === $('#profilePreset').value);
  if (profile) $('#candidateText').value = profile.text;
}

async function run() {
  const text = $('#candidateText').value.trim();
  if (text.length < 3) return;
  const catalogId = $('#catalogSelect').value;
  const catalog = state.catalogs.find(c => c.id === catalogId);
  $('#runBtn').disabled = true;
  $('#errorBox').classList.add('hidden');
  $('#loadingText').textContent = catalog && !catalog.ready
    ? 'Первый запуск на этом каталоге: загружаем модель и строим словарь каталога…'
    : 'Модель ищет кандидатов, проверяет их и подбирает цитаты…';
  $('#loadingBar').classList.add('active');
  try {
    const response = await fetch('/api/select', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text, catalog_id: catalogId, model_id: $('#modelSelect').value || null,
        max_competencies: Number($('#maxCompetencies').value), pool: Number($('#poolSize').value),
      }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    state.result = await response.json();
    if (catalog) catalog.ready = true;
    await loadDetails(catalogId);
    render();
    $('#statusText').textContent = `${state.result.model?.name || 'Модель'} загружена · CUDA ready`;
    $('#resultSection').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    $('#errorBox').textContent = `Не удалось выполнить подбор: ${error.message}`;
    $('#errorBox').classList.remove('hidden');
  } finally {
    $('#runBtn').disabled = false;
    $('#loadingBar').classList.remove('active');
  }
}

async function loadDetails(catalogId) {
  if (state.details[catalogId]) return;
  try { state.details[catalogId] = (await fetch(`/api/catalogs/${encodeURIComponent(catalogId)}`).then(r => r.json())).competencies; }
  catch { state.details[catalogId] = {}; }
}

function picks() { return (state.result?.selected || []).filter(p => p.id !== '__NONE__'); }

function render() {
  const r = state.result, chosen = picks(), t = r.timings || {};
  $('#resultSection').classList.remove('hidden');
  $('#candidateSection').classList.remove('hidden');
  const confirmed = r.candidates.filter(c => c.confirmed).length;
  $('#summaryStrip').innerHTML = [
    ['Подобрано', chosen.length, true],
    ['Кандидатов проверено', r.candidates.length],
    ['Ответ «ДА»', confirmed],
    ['Каталог', r.catalog?.name || '—'],
    ['Поиск · проверка · цитаты', `${fmt(t.search_s)} · ${fmt(t.check_s)} · ${fmt(t.quote_s)}`],
    ['Всего', fmt(t.request_total_s)],
  ].map(([label, value, accent]) => `<div class="summary-item ${accent ? 'accent' : ''}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join('');
  renderPicks(chosen);
  renderSource(chosen);
  renderCandidates();
}

function renderPicks(chosen) {
  const details = state.details[state.result.catalog?.id] || {};
  if (!chosen.length) {
    $('#picks').innerHTML = `<article class="final-card sel-empty"><div class="final-card-head"><div class="final-title"><h3>Подходящих компетенций не найдено</h3><p>Ни один кандидат поиска не получил ответ «ДА». Список проверенных кандидатов — ниже.</p></div></div></article>`;
    return;
  }
  $('#picks').innerHTML = chosen.map((p, index) => {
    const info = details[p.id] || {}, quote = p.quote || {};
    return `<article class="final-card sel-card">
      <div class="final-card-head">
        <div class="final-code sel-code">${esc(p.id)}</div>
        <div class="final-title"><h3>${esc(p.name)}</h3><p>место в поиске ${p.search_rank} · уверенность ${signed(p.margin)}</p></div>
        <div class="sel-yes" title="Вероятность ответа «ДА» среди ДА/НЕТ">${percent(p.p_yes)}<small>ДА</small></div>
      </div>
      ${quote.text ? `<blockquote class="final-quote sel-quote" data-index="${index + 1}">${esc(quote.text)}<footer><span>дословно из резюме</span><span>доля среди предложений ${percent(quote.share)}</span></footer></blockquote>` : '<p class="sel-muted">Цитата не получена.</p>'}
      <div class="sel-signals"><span class="sel-label">Почему найдено</span>${(p.signals || []).map(s => `<span class="chip">${esc(s.join(' + '))}</span>`).join('') || '<span class="sel-muted">—</span>'}</div>
      ${info.indicators ? `<details class="sel-details"><summary>Индикаторы компетенции · ${info.indicators.length}</summary><ul>${info.indicators.map(i => `<li>${esc(i)}</li>`).join('')}</ul></details>` : ''}
    </article>`;
  }).join('');
}

function renderSource(chosen) {
  const text = state.result.text;
  const spans = chosen.map((p, i) => ({...(p.quote || {}), n: i + 1})).filter(q => Number.isInteger(q.start) && Number.isInteger(q.end)).sort((a, b) => a.start - b.start || b.end - a.end);
  let html = '', cursor = 0;
  spans.forEach(span => {
    if (span.start < cursor) return;
    html += esc(text.slice(cursor, span.start)) + `<mark data-n="${span.n}">${esc(text.slice(span.start, span.end))}</mark>`;
    cursor = span.end;
  });
  html += esc(text.slice(cursor));
  $('#sourceText').innerHTML = html;
  $('#quoteCount').textContent = `${spans.length} цитат`;
}

function renderCandidates() {
  const chosen = new Set(picks().map(p => p.id));
  const margins = state.result.candidates.map(c => Math.abs(c.margin));
  const scale = Math.max(1, ...margins);
  $('#candidateRows').innerHTML = state.result.candidates.map(c => {
    const width = Math.min(50, Math.abs(c.margin) / scale * 50);
    const status = chosen.has(c.id) ? ['выбрана', 'is-picked'] : c.confirmed ? ['ДА, вне лимита', 'is-yes'] : ['НЕТ', 'is-no'];
    return `<tr class="${status[1]}">
      <td class="sel-rank">${c.search_rank}</td>
      <td><b>${esc(c.id)}</b> <span>${esc(c.name)}</span></td>
      <td><div class="sel-bar" title="${signed(c.margin)}"><i class="${c.margin >= 0 ? 'pos' : 'neg'}" style="width:${width}%;${c.margin >= 0 ? 'left:50%' : `left:${50 - width}%`}"></i></div><small>${signed(c.margin)}</small></td>
      <td><span class="sel-status">${status[0]}</span></td>
      <td class="sel-signal-cell">${(c.signals || []).map(s => esc(s.join(' + '))).join(' · ')}</td>
    </tr>`;
  }).join('');
}

function report() {
  const r = state.result, lines = ['Подбор компетенций по резюме', `Каталог: ${r.catalog?.name || '—'}`, `Время: ${fmt(r.timings?.request_total_s)}`, ''];
  picks().forEach(p => {
    lines.push(`${p.id} — ${p.name} (ДА ${percent(p.p_yes)}, уверенность ${signed(p.margin)})`);
    if (p.quote?.text) lines.push(`Цитата: «${p.quote.text}»`);
    lines.push('');
  });
  if (!picks().length) lines.push('Подходящих компетенций не найдено.');
  return lines.join('\n');
}

async function copyReport() {
  await navigator.clipboard.writeText(report());
  const button = $('#copyReportBtn');
  button.textContent = 'Скопировано ✓';
  setTimeout(() => button.textContent = 'Копировать отчёт', 1600);
}

function exportJson() {
  const blob = new Blob([JSON.stringify(state.result, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url;
  link.download = `krm-selection-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function fmt(v) { return Number.isFinite(Number(v)) ? `${Number(v).toFixed(1)} с` : '—'; }
function percent(v) { return Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(Number(v) > 0.995 && Number(v) < 1 ? 1 : 0)}%` : '—'; }
function signed(v) { return Number.isFinite(Number(v)) ? `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}` : '—'; }
function esc(s) { return String(s ?? '').replace(/[&<>'"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[c])); }

$('#profilePreset').onchange = insertProfile;
$('#runBtn').onclick = run;
$('#copyReportBtn').onclick = copyReport;
$('#exportJsonBtn').onclick = exportJson;
init();
