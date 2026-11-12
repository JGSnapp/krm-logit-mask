// Shared controls for both extraction routes. State is sent with each request.
const discToggle = document.querySelector('#discToggle');
discToggle.addEventListener('click', () => {
  const enabled = discToggle.getAttribute('aria-pressed') !== 'true';
  discToggle.setAttribute('aria-pressed', String(enabled));
  discToggle.textContent = `DISC: ${enabled ? 'включён' : 'выключен'}`;
  document.querySelector('#discBudget').disabled = !enabled;
});
function discRequestOptions() {
  return {
    use_disc: discToggle.getAttribute('aria-pressed') === 'true',
    disc_budget: Number(document.querySelector('#discBudget').value),
  };
}
function renderDiscTrace(trace, event, target) {
  const summary = (trace || []).find(step => step.disc)?.disc;
  if (!summary) return;
  const selected = summary.attempts[summary.selected_attempt - 1];
  const states = {selected:'выбран', rejected:'отвергнут', sampled:'резерв', invalid:'незавершён'};
  const safe = text => String(text ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const step = event.forced ? 'Автоподстановка: вызов модели пропущен, вес шага = 1.'
    : Number.isFinite(event.disc_allowed_mass) ? `Масса разрешённых токенов до маски: ${(event.disc_allowed_mass * 100).toFixed(2)}%.` : '';
  document.querySelector(target).insertAdjacentHTML('beforeend', `
    <div class="disc-summary"><b>DISC · ${summary.mode === 'accepted' ? 'принятие' : 'взвешенный резерв'}</b>
    <div>Проходов ${summary.attempt_count} / ${summary.budget * 2}; выбран №${summary.selected_attempt}.
    Вес пути ${(summary.weight * 100).toPrecision(3)}% (log w ${summary.log_weight.toFixed(3)}).
    Неоднозначных шагов ${selected.decision_count}; автотокенов ${selected.skipped_token_count}.</div>
    <div>${step}</div>
    <details><summary>Проходы DISC</summary>${summary.attempts.map((a, i) => `<div>№${i + 1} · ${a.phase === 'fallback' ? 'резерв' : 'принятие'} · ${safe(a.competency_id || '—')} · ${states[a.status] || safe(a.status)} · ${Number.isFinite(a.log_weight) ? `log w ${a.log_weight.toFixed(3)}` : safe(a.reason)}<br>${safe((a.selected_phrases || []).join(' → '))}</div>`).join('')}</details>
    <small>Вес учитывает неоднозначные шаги; это не вероятность правильности компетенции.</small></div>`);
}
