const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const state = {profiles: [], catalogs: [], details: {}, result: null};
const WEAK_YES = 2; // margin below this: the model says ДА, but not with confidence

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

function mode() { return $('input[name="mode"]:checked')?.value || 'quote_first'; }

function syncMode() {
  const first = mode() === 'quote_first';
  $('#quotesField').classList.toggle('hidden', !first);
  $('#poolField').classList.toggle('hidden', first);
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
    : mode() === 'quote_first'
      ? 'Модель выбирает цитаты, ищет к ним компетенции и проверяет их…'
      : 'Модель ищет кандидатов, проверяет их и подбирает цитаты…';
  $('#loadingBar').classList.add('active');
  try {
    const response = await fetch('/api/select', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text, mode: mode(), catalog_id: catalogId, model_id: $('#modelSelect').value || null,
        max_competencies: Number($('#maxCompetencies').value),
        pool: Number($('#poolSize').value), quotes: Number($('#quoteCountSelect').value),
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
function quoteFirst() { return state.result?.mode === 'quote_first'; }

function render() {
  const r = state.result, chosen = picks(), t = r.timings || {};
  $('#resultSection').classList.remove('hidden');
  $('#candidateSection').classList.remove('hidden');
  $('#quoteSection').classList.toggle('hidden', !quoteFirst());
  $('#candidateSectionNo').textContent = quoteFirst() ? '04' : '03';
  const confirmed = r.candidates.filter(c => c.confirmed).length;
  const stages = quoteFirst()
    ? ['Цитаты · поиск · проверка', `${fmt(t.quotes_s)} · ${fmt(t.search_s)} · ${fmt(t.check_s)}`]
    : ['Поиск · проверка · цитаты', `${fmt(t.search_s)} · ${fmt(t.check_s)} · ${fmt(t.quote_s)}`];
  $('#summaryStrip').innerHTML = [
    ['Подобрано', chosen.length, true],
    ['Кандидатов проверено', r.candidates.length],
    ['Ответ «ДА»', confirmed],
    ['Подход', quoteFirst() ? 'Цитата сначала' : 'Цитата в конце'],
    stages,
    ['Всего', fmt(t.request_total_s)],
  ].map(([label, value, accent]) => `<div class="summary-item ${accent ? 'accent' : ''}"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join('');
  renderPicks(chosen);
  renderSource(chosen);
  if (quoteFirst()) renderQuotes(chosen);
  renderCandidates();
}

function confidence(margin) {
  const weak = margin < WEAK_YES;
  return `<div class="sel-yes ${weak ? 'weak' : ''}" title="Уверенность = log P(ДА) − log P(НЕТ)">${signed(margin)}<small>${weak ? 'слабое ДА' : 'уверенность ДА'}</small></div>`;
}

function renderPicks(chosen) {
  const details = state.details[state.result.catalog?.id] || {};
  if (!chosen.length) {
    $('#picks').innerHTML = `<article class="final-card sel-empty"><div class="final-card-head"><div class="final-title"><h3>Подходящих компетенций не найдено</h3><p>Ни один кандидат не получил ответ «ДА». Список проверенных кандидатов — ниже.</p></div></div></article>`;
    return;
  }
  $('#picks').innerHTML = chosen.map((p, index) => {
    const info = details[p.id] || {}, quote = p.quote || {};
    const where = quoteFirst() ? `найдено по цитате ${p.quote_rank}` : `место в поиске ${p.search_rank}`;
    const number = quoteFirst() ? quote.rank : index + 1;
    const footer = quoteFirst() ? 'цитата, по которой найдена компетенция' : `дословно из резюме · доля среди предложений ${percent(quote.share)}`;
    return `<article class="final-card sel-card">
      <div class="final-card-head">
        <div class="final-code sel-code">${esc(p.id)}</div>
        <div class="final-title"><h3>${esc(p.name)}</h3><p>${where} · P(ДА) ${percent(p.p_yes)}</p></div>
        ${confidence(p.margin)}
      </div>
      ${quote.text ? `<blockquote class="final-quote sel-quote" data-index="${number}">${esc(quote.text)}<footer><span>${footer}</span></footer></blockquote>` : '<p class="sel-muted">Цитата не получена.</p>'}
      <div class="sel-signals"><span class="sel-label">Почему найдено</span>${(p.signals || []).map(s => `<span class="chip">${esc(s.join(' + '))}</span>`).join('') || '<span class="sel-muted">—</span>'}</div>
      ${info.indicators ? `<details class="sel-details"><summary>Индикаторы компетенции · ${info.indicators.length}</summary><ul>${info.indicators.map(i => `<li>${esc(i)}</li>`).join('')}</ul></details>` : ''}
    </article>`;
  }).join('');
}

function renderSource(chosen) {
  const text = state.result.text;
  const quotes = quoteFirst()
    ? (state.result.quotes || []).map(q => ({...q, n: q.rank}))
    : chosen.map((p, i) => ({...(p.quote || {}), n: i + 1}));
  const spans = quotes.filter(q => Number.isInteger(q.start) && Number.isInteger(q.end)).sort((a, b) => a.start - b.start || b.end - a.end);
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

function renderQuotes(chosen) {
  const picked = new Set(chosen.map(p => p.id));
  const byId = Object.fromEntries(state.result.candidates.map(c => [c.id, c]));
  $('#quoteCards').innerHTML = (state.result.quotes || []).map(q => `
    <article class="trace-card sel-quote-card">
      <header><span>ЦИТАТА ${q.rank}</span><span>${(q.candidates || []).length} компетенций</span></header>
      <blockquote>${esc(q.text)}</blockquote>
      <ul class="sel-quote-list">${(q.candidates || []).map(id => {
        const c = byId[id] || {id, name: '', margin: 0};
        const cls = picked.has(id) ? 'is-picked' : c.margin > 0 ? 'is-yes' : 'is-no';
        return `<li class="${cls}"><b>${esc(id)}</b><span>${esc(c.name)}</span><small>${signed(c.margin)}</small></li>`;
      }).join('') || '<li class="is-no"><span>компетенций не найдено</span></li>'}</ul>
    </article>`).join('');
}

function renderCandidates() {
  const chosen = new Set(picks().map(p => p.id));
  const margins = state.result.candidates.map(c => Math.abs(c.margin));
  const scale = Math.max(1, ...margins);
  $('#candidateRows').innerHTML = state.result.candidates.map(c => {
    const width = Math.min(50, Math.abs(c.margin) / scale * 50);
    const status = chosen.has(c.id) ? ['выбрана', 'is-picked'] : c.confirmed ? ['ДА, вне лимита', 'is-yes'] : ['НЕТ', 'is-no'];
    const place = quoteFirst() ? `ц. ${(c.quotes || [c.quote_rank]).join(', ')}` : c.search_rank;
    return `<tr class="${status[1]}">
      <td class="sel-rank">${place}</td>
      <td><b>${esc(c.id)}</b> <span>${esc(c.name)}</span></td>
      <td><div class="sel-bar" title="${signed(c.margin)}"><i class="${c.margin >= 0 ? 'pos' : 'neg'}" style="width:${width}%;${c.margin >= 0 ? 'left:50%' : `left:${50 - width}%`}"></i></div><small>${signed(c.margin)}</small></td>
      <td><span class="sel-status">${status[0]}</span></td>
      <td class="sel-signal-cell">${(c.signals || []).map(s => esc(s.join(' + '))).join(' · ')}</td>
    </tr>`;
  }).join('');
}

function report() {
  const r = state.result;
  const lines = ['Подбор компетенций по резюме', `Подход: ${quoteFirst() ? 'цитата сначала' : 'цитата в конце'}`,
    `Каталог: ${r.catalog?.name || '—'}`, `Время: ${fmt(r.timings?.request_total_s)}`, ''];
  picks().forEach(p => {
    lines.push(`${p.id} — ${p.name} (уверенность ${signed(p.margin)})`);
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
$$('input[name="mode"]').forEach(input => input.onchange = syncMode);
syncMode();
init();
