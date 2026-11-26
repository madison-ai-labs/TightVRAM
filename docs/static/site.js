// Experiment registry: search and filter. Without JavaScript the full list is shown.
(function () {
  var list = document.getElementById('exps');
  if (!list || !document.getElementById('q')) return;
  var rows = Array.prototype.slice.call(list.children);
  var q = document.getElementById('q');
  var count = document.getElementById('count');
  var empty = document.getElementById('empty');
  var clear = document.getElementById('clear');
  var buttons = Array.prototype.slice.call(document.querySelectorAll('button[data-filter]'));
  var state = { program: '', evidence: '', verdict: '' };

  var params = new URLSearchParams(location.search);
  Object.keys(state).forEach(function (k) { state[k] = params.get(k) || ''; });
  q.value = params.get('q') || '';

  function apply() {
    var words = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    var shown = 0;
    rows.forEach(function (r) {
      var ok = Object.keys(state).every(function (k) { return !state[k] || r.dataset[k] === state[k]; }) &&
        words.every(function (w) { return r.dataset.text.indexOf(w) !== -1; });
      r.hidden = !ok;
      if (ok) shown++;
    });
    buttons.forEach(function (b) {
      b.setAttribute('aria-pressed', String(state[b.dataset.filter] === b.dataset.value));
    });
    count.textContent = shown;
    empty.hidden = shown !== 0;
    var active = words.length || Object.keys(state).some(function (k) { return state[k]; });
    clear.hidden = !active;
    var p = new URLSearchParams();
    Object.keys(state).forEach(function (k) { if (state[k]) p.set(k, state[k]); });
    if (q.value.trim()) p.set('q', q.value.trim());
    var s = p.toString();
    history.replaceState(null, '', location.pathname + (s ? '?' + s : ''));
  }

  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      var k = b.dataset.filter;
      state[k] = state[k] === b.dataset.value ? '' : b.dataset.value;
      apply();
    });
  });
  q.addEventListener('input', apply);
  clear.addEventListener('click', function () {
    Object.keys(state).forEach(function (k) { state[k] = ''; });
    q.value = '';
    apply();
    q.focus();
  });
  apply();
})();
