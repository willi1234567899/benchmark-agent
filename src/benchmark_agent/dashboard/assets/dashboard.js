'use strict';
(function () {
  const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
  const OPS = DATA.operations;
  const STATUS_META = {
    ok: { label: 'ok', tone: 'good', icon: '✓' },
    execution_failed: { label: 'Ausführung fehlgeschlagen', tone: 'serious', icon: '✕' },
    validation_failed: { label: 'Validierung fehlgeschlagen', tone: 'critical', icon: '✕' },
    generation_failed: { label: 'Generierung fehlgeschlagen', tone: 'critical', icon: '✕' },
    infrastructure_failed: { label: 'Infrastruktur-Fehler', tone: 'warning', icon: '!' },
    not_measured: { label: 'nicht gemessen', tone: 'muted', icon: '○' },
    not_evaluated: { label: 'nicht ausgewertet', tone: 'muted', icon: '○' },
  };
  const PROMPT_LABEL = { zero_shot: 'Zero-shot', multi_shot: 'Multi-shot' };
  const SEQ = {
    light: ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'],
    dark: ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'],
  };
  const ORDINAL = { light: ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'], dark: ['#184f95', '#256abf', '#3987e5', '#86b6ef'] };
  const SVG_NS = 'http://www.w3.org/2000/svg';

  function h(tag, attrs, children) {
    const node = document.createElement(tag);
    setAttrs(node, attrs);
    append(node, children);
    return node;
  }
  function s(tag, attrs, children) {
    const node = document.createElementNS(SVG_NS, tag);
    setAttrs(node, attrs);
    append(node, children);
    return node;
  }
  function setAttrs(node, attrs) {
    if (!attrs) return;
    for (const key of Object.keys(attrs)) {
      const value = attrs[key];
      if (value === null || value === undefined) continue;
      if (key === 'text') node.textContent = String(value);
      else if (key === 'class') node.setAttribute('class', value);
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, String(value));
    }
  }
  function append(node, children) {
    if (!children) return;
    for (const child of [].concat(children)) {
      if (child === null || child === undefined || child === false) continue;
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    }
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function fmt(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return '–';
    const d = digits === undefined ? 1 : digits;
    return Number(value).toLocaleString('de-DE', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pct(value) { return value === null || value === undefined ? '–' : fmt(value * 100, 0) + ' %'; }
  function fmtP(p) {
    if (p === null || p === undefined) return '–';
    if (p < 0.001) return '< 0,001';
    return fmt(p, 3);
  }
  function fmtTime(iso) {
    if (!iso) return '–';
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' });
  }
  function median(values) {
    if (!values.length) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
  function quantile(sorted, p) {
    if (!sorted.length) return null;
    if (sorted.length === 1) return sorted[0];
    const rank = p * (sorted.length - 1);
    const lo = Math.floor(rank);
    const hi = Math.min(lo + 1, sorted.length - 1);
    return sorted[lo] * (1 - (rank - lo)) + sorted[hi] * (rank - lo);
  }
  function iqr(values) {
    if (values.length < 2) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    return quantile(sorted, 0.75) - quantile(sorted, 0.25);
  }
  function mean(values) { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null; }
  function groupBy(items, keyFn) {
    const map = new Map();
    for (const item of items) {
      const key = keyFn(item);
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(item);
    }
    return map;
  }
  function niceTicks(min, max, count) {
    if (max <= min) max = min + 1;
    const span = max - min;
    const rough = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const norm = rough / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(Math.round(v * 1e9) / 1e9);
    return ticks;
  }
  function luminance(hex) {
    const c = hex.replace('#', '');
    const r = parseInt(c.slice(0, 2), 16) / 255, g = parseInt(c.slice(2, 4), 16) / 255, b = parseInt(c.slice(4, 6), 16) / 255;
    const lin = (v) => (v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  }
  function rampColor(t, theme) {
    const ramp = SEQ[theme];
    const idx = Math.max(0, Math.min(ramp.length - 1, Math.round(t * (ramp.length - 1))));
    return ramp[idx];
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function stageB(run, source, op, metric) {
    if (!run.executed_ok) return null;
    const stats = run.measurements && run.measurements[source] && run.measurements[source][op];
    if (!stats) return null;
    if ((stats.n_requests - stats.error_count) <= 0) return null;
    const value = stats[metric];
    return value === null || value === undefined ? null : Number(value);
  }
  function promptLabel(key) { return PROMPT_LABEL[key] || key || '?'; }
  function ceilingStat(source, op, field) {
    const byOp = DATA.ceiling && DATA.ceiling[source];
    const entry = byOp && byOp[op];
    if (!entry || entry[field] === null || entry[field] === undefined) return null;
    return Number(entry[field]);
  }
  function ceilingLine(inner, py, label) {
    const g = s('g', { class: 'ceiling' });
    g.appendChild(s('line', { class: 'ceiling-line', x1: inner.x, x2: inner.x + inner.w, y1: py, y2: py }));
    g.appendChild(s('text', { class: 'value-label', x: inner.x + inner.w, y: py - 5, 'text-anchor': 'end', text: label }));
    return g;
  }
  function statusMeta(status) { return STATUS_META[status] || { label: status || '?', tone: 'muted', icon: '?' }; }
  function statusBadge(status) {
    const meta = statusMeta(status);
    return h('span', { class: 'status' }, [h('span', { class: 'icon ' + meta.tone, text: meta.icon }), meta.label]);
  }

  const THEMES = ['auto', 'light', 'dark'];
  function currentTheme() {
    const explicit = document.documentElement.getAttribute('data-theme');
    if (explicit === 'light' || explicit === 'dark') return explicit;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function applyTheme(choice) {
    if (choice === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', choice);
    document.getElementById('theme-label').textContent = choice;
    try { localStorage.setItem('benchmark-agent-theme', choice); } catch (e) { }
  }
  let themeChoice = 'auto';
  try { themeChoice = localStorage.getItem('benchmark-agent-theme') || 'auto'; } catch (e) { }
  if (!THEMES.includes(themeChoice)) themeChoice = 'auto';
  applyTheme(themeChoice);
  document.getElementById('theme-toggle').addEventListener('click', () => {
    themeChoice = THEMES[(THEMES.indexOf(themeChoice) + 1) % THEMES.length];
    applyTheme(themeChoice);
    renderAll();
  });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => renderAll());
  document.getElementById('reload').addEventListener('click', () => window.location.reload());

  const tooltip = document.getElementById('tooltip');
  function showTip(event, title, rows) {
    clear(tooltip);
    if (title) tooltip.appendChild(h('div', { class: 'tt-title', text: title }));
    for (const row of rows) {
      const key = h('span', { class: 'k' });
      if (row.color) key.appendChild(h('i', { style: 'background:' + row.color }));
      key.appendChild(document.createTextNode(row.label));
      tooltip.appendChild(h('div', { class: 'tt-row' }, [key, h('span', { class: 'v', text: row.value })]));
    }
    tooltip.hidden = false;
    moveTip(event);
  }
  function moveTip(event) {
    const pad = 14;
    let x = event.clientX + pad, y = event.clientY + pad;
    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }
  function hideTip() { tooltip.hidden = true; }
  function hover(node, title, rowsFn) {
    node.setAttribute('tabindex', '0');
    node.addEventListener('pointerenter', (e) => showTip(e, title, rowsFn()));
    node.addEventListener('pointermove', moveTip);
    node.addEventListener('pointerleave', hideTip);
    node.addEventListener('focus', (e) => { const r = node.getBoundingClientRect(); showTip({ clientX: r.left + r.width / 2, clientY: r.top }, title, rowsFn()); });
    node.addEventListener('blur', hideTip);
  }

  const state = {
    levels: new Set(DATA.levels),
    models: new Set(DATA.models),
    promptings: new Set(DATA.promptings),
    statuses: new Set(DATA.statuses),
    epochs: new Set([DATA.current_epoch || (DATA.epochs || ['1']).slice(-1)[0]]),
    source: 'proxy',
    operation: 'write',
    includeFixtures: false,
    heatMetric: 'success_rate',
    scatterMetric: 'n_requests',
    sort: { key: 'evaluated_at', dir: 'desc' },
    selectedRun: null,
  };
  const hasFixtures = DATA.runs.some((r) => r.is_fixture);
  if (!DATA.runs.some((r) => !r.is_fixture) && hasFixtures) state.includeFixtures = true;

  function filteredRuns() {
    return DATA.runs.filter((run) =>
      state.levels.has(run.guidance_level || '?') &&
      state.models.has(run.model || '?') &&
      state.promptings.has(run.prompting || '?') &&
      state.statuses.has(run.status) &&
      state.epochs.has(String(run.epoch || 1)) &&
      (state.includeFixtures || !run.is_fixture));
  }
  function evaluatedRuns(runs) { return runs.filter((r) => r.status !== 'not_evaluated'); }

  function chipGroup(label, values, selected, labelFn, onChange) {
    const chips = h('div', { class: 'chips' });
    for (const value of values) {
      const chip = h('button', { type: 'button', class: 'chip', 'aria-pressed': String(selected.has(value)), text: labelFn ? labelFn(value) : value });
      chip.addEventListener('click', (event) => {
        if (event.altKey || event.shiftKey) { selected.clear(); selected.add(value); }
        else if (selected.has(value)) { if (selected.size > 1) selected.delete(value); }
        else selected.add(value);
        onChange();
      });
      chips.appendChild(chip);
    }
    const all = h('button', { type: 'button', class: 'chip', text: 'alle' });
    all.addEventListener('click', () => { for (const v of values) selected.add(v); onChange(); });
    chips.appendChild(all);
    return h('div', { class: 'filter-group' }, [h('span', { class: 'label', text: label }), chips]);
  }
  function radioGroup(label, values, current, labelFn, onPick) {
    const chips = h('div', { class: 'chips' });
    for (const value of values) {
      const chip = h('button', { type: 'button', class: 'chip', 'aria-pressed': String(current === value), text: labelFn ? labelFn(value) : value });
      chip.addEventListener('click', () => onPick(value));
      chips.appendChild(chip);
    }
    return h('div', { class: 'filter-group' }, [h('span', { class: 'label', text: label }), chips]);
  }
  function renderFilters() {
    const root = document.getElementById('filters');
    clear(root);
    const rerender = () => renderAll();
    root.appendChild(chipGroup('Guidance-Level', DATA.levels, state.levels, null, rerender));
    root.appendChild(chipGroup('Modell', DATA.models, state.models, null, rerender));
    root.appendChild(chipGroup('Prompting', DATA.promptings, state.promptings, promptLabel, rerender));
    root.appendChild(chipGroup('Status', DATA.statuses, state.statuses, (v) => statusMeta(v).label, rerender));
    if ((DATA.epochs || []).length > 1) root.appendChild(chipGroup('Epoche (nie poolen)', DATA.epochs, state.epochs, (v) => 'Epoche ' + v, rerender));
    root.appendChild(radioGroup('Messquelle', DATA.measured_sources, state.source, (v) => (v === 'proxy' ? 'Proxy (primär)' : '/metrics (Cross-Check)'), (v) => { state.source = v; renderAll(); }));
    root.appendChild(radioGroup('Operation (Stufe B)', OPS, state.operation, null, (v) => { state.operation = v; renderAll(); }));
    if (hasFixtures) {
      const toggle = h('button', { type: 'button', class: 'chip', 'aria-pressed': String(state.includeFixtures), text: 'Fixtures/Referenz einbeziehen' });
      toggle.addEventListener('click', () => { state.includeFixtures = !state.includeFixtures; renderAll(); });
      root.appendChild(h('div', { class: 'filter-group' }, [h('span', { class: 'label', text: 'Sonderfälle' }), h('div', { class: 'chips' }, toggle)]));
    }
    const runs = filteredRuns();
    root.appendChild(h('div', { class: 'filter-summary', text: runs.length + ' von ' + DATA.runs.length + ' Runs im Filter · Alt+Klick = nur dieser Wert' }));
  }

  const cards = [];
  function registerCard(spec) { cards.push(spec); }
  function renderCards(runs) {
    const root = document.getElementById('charts');
    clear(root);
    for (const spec of cards) {
      const body = h('div', { class: 'card-body' });
      const foot = h('div', { class: 'card-foot' });
      const chartBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'true', text: 'Diagramm' });
      const tableBtn = h('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', text: 'Tabelle' });
      const card = h('section', { class: 'card' + (spec.wide ? ' wide' : '') }, [
        h('div', { class: 'card-head' }, [
          h('div', {}, [h('h2', { text: spec.title }), h('div', { class: 'sub', text: typeof spec.subtitle === 'function' ? spec.subtitle() : spec.subtitle })]),
          h('div', { class: 'card-tools' }, [chartBtn, tableBtn]),
        ]),
        body, foot,
      ]);
      root.appendChild(card);
      const draw = (mode) => {
        clear(body); clear(foot);
        chartBtn.setAttribute('aria-pressed', String(mode === 'chart'));
        tableBtn.setAttribute('aria-pressed', String(mode === 'table'));
        try {
          if (mode === 'chart') spec.chart(body, foot, runs, card.clientWidth - 30);
          else { const t = spec.table(runs); renderTable(body, t.columns, t.rows); }
        } catch (error) {
          body.appendChild(h('p', { class: 'note', text: 'Darstellung fehlgeschlagen: ' + error.message }));
        }
      };
      chartBtn.addEventListener('click', () => draw('chart'));
      tableBtn.addEventListener('click', () => draw('table'));
      draw('chart');
    }
  }
  function renderTable(container, columns, rows, options) {
    const opts = options || {};
    const wrap = h('div', { class: 'table-wrap' });
    const table = h('table', { class: 'data' });
    const head = h('tr');
    for (const col of columns) {
      const th = h('th', { class: col.num ? 'num' : '', text: col.label });
      if (opts.sortable) {
        if (opts.sort && opts.sort.key === col.key) th.setAttribute('aria-sort', opts.sort.dir === 'asc' ? 'ascending' : 'descending');
        th.addEventListener('click', () => opts.onSort(col.key));
      }
      head.appendChild(th);
    }
    table.appendChild(h('thead', {}, head));
    const tbody = h('tbody');
    for (const row of rows) {
      const tr = h('tr', { class: (opts.rowClass ? opts.rowClass(row) : '') });
      for (const col of columns) {
        const value = col.render ? col.render(row) : row[col.key];
        const td = h('td', { class: col.num ? 'num' : '' });
        if (value instanceof Node) td.appendChild(value); else td.textContent = value === null || value === undefined ? '–' : String(value);
        tr.appendChild(td);
      }
      if (opts.onRow) tr.addEventListener('click', () => opts.onRow(row));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    if (!rows.length) wrap.appendChild(h('p', { class: 'note', text: 'Keine Zeilen im aktuellen Filter.' }));
    container.appendChild(wrap);
  }
  function legend(container, entries) {
    const box = h('div', { class: 'legend' });
    for (const entry of entries) {
      box.appendChild(h('span', { class: 'key' }, [h('span', { class: 'swatch ' + (entry.kind || ''), style: 'background:' + entry.color }), entry.label]));
    }
    container.appendChild(box);
  }
  function emptyChart(body, message) { body.appendChild(h('p', { class: 'note', text: message || 'Keine Daten im aktuellen Filter.' })); }

  function frame(width, height, margin) {
    const svg = s('svg', { viewBox: '0 0 ' + width + ' ' + height, width: width, height: height, role: 'img' });
    const inner = { x: margin.left, y: margin.top, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
    return { svg, inner };
  }
  function yAxis(svg, inner, scale, ticks, formatter) {
    for (const t of ticks) {
      const y = scale(t);
      svg.appendChild(s('line', { class: 'gridline', x1: inner.x, x2: inner.x + inner.w, y1: y, y2: y }));
      svg.appendChild(s('text', { class: 'tick', x: inner.x - 6, y: y + 4, 'text-anchor': 'end', text: formatter ? formatter(t) : fmt(t, 0) }));
    }
  }
  function xBandLabels(svg, inner, band, labels, labelFn) {
    for (const label of labels) {
      const x = band(label) + band.bandwidth / 2;
      svg.appendChild(s('text', { class: 'tick', x, y: inner.y + inner.h + 16, 'text-anchor': 'middle', text: labelFn ? labelFn(label) : label }));
    }
    svg.appendChild(s('line', { class: 'baseline', x1: inner.x, x2: inner.x + inner.w, y1: inner.y + inner.h, y2: inner.y + inner.h }));
  }
  function band(domain, range, padding) {
    const pad = padding === undefined ? 0.25 : padding;
    const n = Math.max(1, domain.length);
    const step = (range[1] - range[0]) / n;
    const bw = step * (1 - pad);
    const fn = (value) => range[0] + domain.indexOf(value) * step + (step - bw) / 2;
    fn.bandwidth = bw;
    fn.step = step;
    return fn;
  }
  function linear(domain, range) {
    const [d0, d1] = domain, [r0, r1] = range;
    const span = d1 - d0 || 1;
    return (value) => r0 + ((value - d0) / span) * (r1 - r0);
  }
  function roundedBarPath(x, y, w, h, r, direction) {
    const rr = Math.min(r, w / 2, h);
    if (h <= 0 || w <= 0) return '';
    if (direction === 'up') {
      return 'M' + x + ',' + (y + h) + ' V' + (y + rr) + ' a' + rr + ',' + rr + ' 0 0 1 ' + rr + ',-' + rr + ' H' + (x + w - rr) + ' a' + rr + ',' + rr + ' 0 0 1 ' + rr + ',' + rr + ' V' + (y + h) + ' Z';
    }
    return 'M' + x + ',' + y + ' H' + (x + w - rr) + ' a' + rr + ',' + rr + ' 0 0 1 ' + rr + ',' + rr + ' V' + (y + h - rr) + ' a' + rr + ',' + rr + ' 0 0 1 -' + rr + ',' + rr + ' H' + x + ' Z';
  }

  function renderKpis(runs) {
    const root = document.getElementById('kpis');
    clear(root);
    const evaluated = evaluatedRuns(runs);
    const n = evaluated.length;
    const success = evaluated.filter((r) => r.success).length;
    const syntax = evaluated.filter((r) => r.syntax_ok).length;
    const executed = evaluated.filter((r) => r.executed_ok).length;
    const coverages = evaluated.map((r) => r.coverage_ratio).filter((v) => v !== null && v !== undefined);
    const throughput = evaluated.map((r) => stageB(r, state.source, state.operation, 'throughput_ops_s')).filter((v) => v !== null);
    const p95 = evaluated.map((r) => stageB(r, state.source, state.operation, 'latency_p95_ms')).filter((v) => v !== null);
    const feedback = evaluated.map((r) => r.feedback_iterations || 0);
    const tile = (label, value, sub, ratio, hero) => {
      const node = h('div', { class: 'kpi' }, [h('div', { class: 'label', text: label }), h('div', { class: 'value' + (hero ? ' hero' : ''), text: value }), h('div', { class: 'sub', text: sub })]);
      if (ratio !== null && ratio !== undefined) node.appendChild(h('div', { class: 'meter' }, h('i', { style: 'width:' + Math.round(ratio * 100) + '%' })));
      return node;
    };
    root.appendChild(tile('Erfolgsquote (Stufe A)', n ? pct(success / n) : '–', success + ' von ' + n + ' ausgewerteten Runs', n ? success / n : null, true));
    root.appendChild(tile('Runs', String(runs.length), n + ' ausgewertet · ' + (runs.length - n) + ' offen', null));
    root.appendChild(tile('Syntax OK', n ? pct(syntax / n) : '–', syntax + ' / ' + n, n ? syntax / n : null));
    root.appendChild(tile('Ausgeführt', n ? pct(executed / n) : '–', executed + ' / ' + n, n ? executed / n : null));
    root.appendChild(tile('Ø Coverage', coverages.length ? pct(mean(coverages)) : '–', 'write / read / delete', coverages.length ? mean(coverages) : null));
    root.appendChild(tile('Median Throughput', throughput.length ? fmt(median(throughput), 1) : '–', state.operation + ' · ' + state.source + ' · ops/s · n=' + throughput.length, null));
    root.appendChild(tile('Median p95-Latenz', p95.length ? fmt(median(p95), 1) + ' ms' : '–', state.operation + ' · n=' + p95.length, null));
    root.appendChild(tile('Ø Feedback-Runden', n ? fmt(mean(feedback), 2) : '–', 'nur L4 iteriert', null));
  }

  registerCard({
    title: 'Trichter Stufe A',
    subtitle: 'ausgewertet → Syntax OK → ausgeführt → Semantik OK (= Erfolg)',
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs);
      if (!evaluated.length) return emptyChart(body);
      const stages = funnelStages(evaluated);
      const theme = currentTheme();
      const height = 30 * stages.length + 30;
      const { svg, inner } = frame(width, height, { top: 8, right: 60, bottom: 20, left: 120 });
      const x = linear([0, stages[0].value || 1], [inner.x, inner.x + inner.w]);
      stages.forEach((stage, i) => {
        const y = inner.y + i * 30 + 4;
        const w = x(stage.value) - inner.x;
        const color = ORDINAL[theme][i];
        svg.appendChild(s('text', { x: inner.x - 8, y: y + 15, 'text-anchor': 'end', text: stage.label }));
        const bar = s('path', { class: 'mark', d: roundedBarPath(inner.x, y, Math.max(w, 0), 20, 4, 'right'), fill: color });
        hover(bar, stage.label, () => [{ label: 'Runs', value: String(stage.value), color }, { label: 'Anteil', value: pct(stage.value / (stages[0].value || 1)) }]);
        svg.appendChild(bar);
        svg.appendChild(s('text', { class: 'value-label', x: inner.x + Math.max(w, 0) + 6, y: y + 15, text: stage.value + ' (' + pct(stage.value / (stages[0].value || 1)) + ')' }));
      });
      body.appendChild(svg);
      foot.textContent = 'Jede Stufe setzt die vorige voraus; Runs mit Status "nicht ausgewertet" sind ausgeschlossen.';
    },
    table(runs) {
      const stages = funnelStages(evaluatedRuns(runs));
      return { columns: [{ key: 'label', label: 'Stufe' }, { key: 'value', label: 'Runs', num: true }, { key: 'share', label: 'Anteil', num: true }], rows: stages.map((st) => ({ label: st.label, value: st.value, share: pct(st.value / (stages[0].value || 1)) })) };
    },
  });
  function funnelStages(evaluated) {
    return [
      { label: 'ausgewertet', value: evaluated.length },
      { label: 'Syntax OK', value: evaluated.filter((r) => r.syntax_ok).length },
      { label: 'ausgeführt', value: evaluated.filter((r) => r.syntax_ok && r.executed_ok).length },
      { label: 'Semantik OK', value: evaluated.filter((r) => r.success).length },
    ];
  }

  registerCard({
    title: 'Erfolgsquote nach Guidance-Level',
    subtitle: 'Anteil erfolgreicher Runs je Level, getrennt nach Prompting-Strategie',
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs);
      if (!evaluated.length) return emptyChart(body);
      const levels = DATA.levels.filter((l) => state.levels.has(l));
      const promptings = DATA.promptings.filter((p) => state.promptings.has(p));
      const colors = [cssVar('--s1'), cssVar('--s2')];
      const height = 240;
      const { svg, inner } = frame(width, height, { top: 14, right: 12, bottom: 28, left: 44 });
      const outer = band(levels, [inner.x, inner.x + inner.w], 0.3);
      const innerBand = band(promptings, [0, outer.bandwidth], 0.1);
      const y = linear([0, 1], [inner.y + inner.h, inner.y]);
      yAxis(svg, inner, y, [0, 0.25, 0.5, 0.75, 1], (t) => pct(t));
      xBandLabels(svg, inner, outer, levels);
      for (const level of levels) {
        promptings.forEach((prompting, pi) => {
          const cell = evaluated.filter((r) => r.guidance_level === level && r.prompting === prompting);
          if (!cell.length) return;
          const rate = cell.filter((r) => r.success).length / cell.length;
          const bx = outer(level) + innerBand(prompting);
          const bw = Math.min(innerBand.bandwidth, 24);
          const x0 = bx + (innerBand.bandwidth - bw) / 2;
          const top = y(rate);
          const bar = s('path', { class: 'mark', d: roundedBarPath(x0, top, bw, inner.y + inner.h - top, 4, 'up'), fill: colors[pi % colors.length] });
          hover(bar, level + ' · ' + promptLabel(prompting), () => [
            { label: 'Erfolgsquote', value: pct(rate), color: colors[pi % colors.length] },
            { label: 'Runs', value: cell.filter((r) => r.success).length + ' / ' + cell.length },
          ]);
          svg.appendChild(bar);
          svg.appendChild(s('text', { class: 'value-label', x: x0 + bw / 2, y: top - 4, 'text-anchor': 'middle', text: cell.filter((r) => r.success).length + '/' + cell.length }));
        });
      }
      body.appendChild(svg);
      legend(foot, promptings.map((p, i) => ({ label: promptLabel(p), color: colors[i % colors.length] })));
    },
    table(runs) {
      const evaluated = evaluatedRuns(runs);
      const rows = [];
      for (const level of DATA.levels) for (const prompting of DATA.promptings) {
        const cell = evaluated.filter((r) => r.guidance_level === level && r.prompting === prompting);
        if (cell.length) rows.push({ level, prompting: promptLabel(prompting), n: cell.length, k: cell.filter((r) => r.success).length, rate: pct(cell.filter((r) => r.success).length / cell.length) });
      }
      return { columns: [{ key: 'level', label: 'Level' }, { key: 'prompting', label: 'Prompting' }, { key: 'n', label: 'n', num: true }, { key: 'k', label: 'Erfolge', num: true }, { key: 'rate', label: 'Quote', num: true }], rows };
    },
  });

  const HEAT_METRICS = {
    success_rate: { label: 'Erfolgsquote', fn: (cell) => cell.length ? cell.filter((r) => r.success).length / cell.length : null, format: pct, max: 1 },
    coverage: { label: 'Ø Coverage', fn: (cell) => { const v = cell.map((r) => r.coverage_ratio).filter((x) => x !== null && x !== undefined); return v.length ? mean(v) : null; }, format: pct, max: 1 },
    throughput: { label: 'Median Throughput', fn: (cell) => { const v = cell.map((r) => stageB(r, state.source, state.operation, 'throughput_ops_s')).filter((x) => x !== null); return v.length ? median(v) : null; }, format: (v) => fmt(v, 1), max: null },
    n: { label: 'Anzahl Runs', fn: (cell) => cell.length || null, format: (v) => fmt(v, 0), max: null },
  };
  registerCard({
    title: 'Heatmap Level × Modell',
    subtitle: () => HEAT_METRICS[state.heatMetric].label + ' je Zelle (beide Prompting-Strategien zusammen)',
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs);
      const picker = h('div', { class: 'chips', style: 'margin-bottom:6px' });
      for (const key of Object.keys(HEAT_METRICS)) {
        const chip = h('button', { type: 'button', class: 'chip', 'aria-pressed': String(state.heatMetric === key), text: HEAT_METRICS[key].label });
        chip.addEventListener('click', () => { state.heatMetric = key; renderAll(); });
        picker.appendChild(chip);
      }
      body.appendChild(picker);
      if (!evaluated.length) return emptyChart(body);
      const metric = HEAT_METRICS[state.heatMetric];
      const levels = DATA.levels.filter((l) => state.levels.has(l));
      const models = DATA.models.filter((m) => state.models.has(m));
      const theme = currentTheme();
      const cellH = 40;
      const height = cellH * models.length + 40;
      const { svg, inner } = frame(width, height, { top: 8, right: 8, bottom: 28, left: 90 });
      const xb = band(levels, [inner.x, inner.x + inner.w], 0.06);
      const values = [];
      const grid = [];
      for (const model of models) for (const level of levels) {
        const cell = evaluated.filter((r) => r.guidance_level === level && r.model === model);
        const value = metric.fn(cell);
        grid.push({ model, level, cell, value });
        if (value !== null) values.push(value);
      }
      const max = metric.max !== null ? metric.max : (values.length ? Math.max.apply(null, values) : 1);
      models.forEach((model, mi) => {
        const y0 = inner.y + mi * cellH + 2;
        svg.appendChild(s('text', { x: inner.x - 8, y: y0 + cellH / 2 + 2, 'text-anchor': 'end', text: model }));
        for (const level of levels) {
          const entry = grid.find((g) => g.model === model && g.level === level);
          const x0 = xb(level);
          const w = xb.bandwidth;
          const fill = entry.value === null ? 'transparent' : rampColor(max ? entry.value / max : 0, theme);
          const rect = s('rect', { class: 'mark', x: x0, y: y0, width: w, height: cellH - 4, rx: 4, fill, stroke: entry.value === null ? cssVar('--grid') : 'none' });
          hover(rect, level + ' · ' + model, () => [
            { label: metric.label, value: entry.value === null ? '–' : metric.format(entry.value) },
            { label: 'Runs', value: String(entry.cell.length) },
            { label: 'Erfolge', value: String(entry.cell.filter((r) => r.success).length) },
          ]);
          svg.appendChild(rect);
          const ink = entry.value === null ? cssVar('--muted') : (luminance(fill) > 0.45 ? '#0b0b0b' : '#ffffff');
          svg.appendChild(s('text', { x: x0 + w / 2, y: y0 + cellH / 2 + 4, 'text-anchor': 'middle', style: 'fill:' + ink + ';font-weight:600', text: entry.value === null ? '–' : metric.format(entry.value) + ' · n=' + entry.cell.length }));
        }
      });
      levels.forEach((level) => svg.appendChild(s('text', { class: 'tick', x: xb(level) + xb.bandwidth / 2, y: inner.y + inner.h + 16, 'text-anchor': 'middle', text: level })));
      body.appendChild(svg);
      foot.textContent = 'Sequenzielle Skala (heller = niedriger). Leere Zellen: keine Runs im Filter.';
    },
    table(runs) {
      const evaluated = evaluatedRuns(runs);
      const metric = HEAT_METRICS[state.heatMetric];
      const rows = [];
      for (const model of DATA.models) for (const level of DATA.levels) {
        const cell = evaluated.filter((r) => r.guidance_level === level && r.model === model);
        if (cell.length) rows.push({ model, level, n: cell.length, value: metric.format(metric.fn(cell)) });
      }
      return { columns: [{ key: 'model', label: 'Modell' }, { key: 'level', label: 'Level' }, { key: 'n', label: 'n', num: true }, { key: 'value', label: metric.label, num: true }], rows };
    },
  });

  registerCard({
    title: 'Throughput je Guidance-Level (Stufe B)',
    subtitle: () => state.operation + ' · ' + state.source + ' · ein Punkt je Run, Balken = Median, Fehlläufe als NA',
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs);
      const levels = DATA.levels.filter((l) => state.levels.has(l));
      const points = [];
      for (const run of evaluated) {
        const v = stageB(run, state.source, state.operation, 'throughput_ops_s');
        if (v !== null && levels.includes(run.guidance_level)) points.push({ run, value: v });
      }
      if (!points.length) return emptyChart(body, 'Kein Run mit Stufe-B-Wert für ' + state.operation + '/' + state.source + ' im Filter.');
      const ceiling = ceilingStat(state.source, state.operation, 'median_throughput_ops_s');
      const height = 260;
      const { svg, inner } = frame(width, height, { top: 14, right: 12, bottom: 28, left: 56 });
      const maxV = Math.max.apply(null, points.map((p) => p.value).concat(ceiling !== null ? [ceiling] : []));
      const ticks = niceTicks(0, maxV * 1.05, 5);
      const y = linear([0, ticks[ticks.length - 1]], [inner.y + inner.h, inner.y]);
      const xb = band(levels, [inner.x, inner.x + inner.w], 0.3);
      yAxis(svg, inner, y, ticks, (t) => fmt(t, 0));
      xBandLabels(svg, inner, xb, levels);
      svg.appendChild(s('text', { class: 'axis-label', x: inner.x, y: inner.y - 4, text: 'ops/s' }));
      if (ceiling !== null) svg.appendChild(ceilingLine(inner, y(ceiling), 'bddbench ' + fmt(ceiling, 1)));
      const dot = cssVar('--s1');
      for (const level of levels) {
        const group = points.filter((p) => p.run.guidance_level === level);
        if (!group.length) continue;
        const cx = xb(level) + xb.bandwidth / 2;
        const med = median(group.map((p) => p.value));
        group.forEach((p, i) => {
          const jitter = ((i % 7) - 3) * (xb.bandwidth / 16);
          const px = cx + jitter;
          const py = y(p.value);
          svg.appendChild(s('circle', { cx: px, cy: py, r: 6, fill: cssVar('--surface') }));
          const c = s('circle', { class: 'mark', cx: px, cy: py, r: 4.5, fill: dot });
          const hit = s('circle', { class: 'hit', cx: px, cy: py, r: 12 });
          hover(hit, p.run.run_id, () => [
            { label: 'Throughput', value: fmt(p.value, 1) + ' ops/s', color: dot },
            { label: 'Modell', value: p.run.model + ' · ' + promptLabel(p.run.prompting) },
            { label: 'Requests', value: String(p.run.measurements[state.source][state.operation].n_requests) },
          ]);
          svg.appendChild(c);
          svg.appendChild(hit);
        });
        svg.appendChild(s('line', { class: 'median-line', x1: cx - xb.bandwidth / 2, x2: cx + xb.bandwidth / 2, y1: y(med), y2: y(med) }));
        svg.appendChild(s('text', { class: 'value-label', x: cx + xb.bandwidth / 2 + 4, y: y(med) + 4, text: fmt(med, 1) }));
      }
      body.appendChild(svg);
      const legendEntries = [{ label: 'Run (Stufe B)', color: dot, kind: 'dot' }, { label: 'Median je Level', color: cssVar('--ink'), kind: 'line' }];
      if (ceiling !== null) legendEntries.push({ label: 'bddbench-Referenz (Ceiling, Median)', color: cssVar('--s2'), kind: 'line' });
      legend(foot, legendEntries);
    },
    table(runs) {
      const evaluated = evaluatedRuns(runs);
      const rows = [];
      for (const level of DATA.levels) {
        const values = evaluated.filter((r) => r.guidance_level === level).map((r) => stageB(r, state.source, state.operation, 'throughput_ops_s')).filter((v) => v !== null);
        const all = evaluated.filter((r) => r.guidance_level === level).length;
        if (all) rows.push({ level, n: all, k: values.length, median: fmt(median(values), 1), iqr: fmt(iqr(values), 1), min: values.length ? fmt(Math.min.apply(null, values), 1) : '–', max: values.length ? fmt(Math.max.apply(null, values), 1) : '–' });
      }
      const ceiling = ceilingStat(state.source, state.operation, 'median_throughput_ops_s');
      if (ceiling !== null) rows.push({ level: 'bddbench (Ceiling)', n: ceilingStat(state.source, state.operation, 'n') || 0, k: '–', median: fmt(ceiling, 1), iqr: '–', min: '–', max: '–' });
      return { columns: [{ key: 'level', label: 'Level' }, { key: 'n', label: 'Runs', num: true }, { key: 'k', label: 'Stufe B', num: true }, { key: 'median', label: 'Median ops/s', num: true }, { key: 'iqr', label: 'IQR', num: true }, { key: 'min', label: 'Min', num: true }, { key: 'max', label: 'Max', num: true }], rows };
    },
  });

  registerCard({
    title: 'Latenz je Operation',
    subtitle: () => 'Median über Stufe-B-Runs von p50 / p95 / p99 · ' + state.source,
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs);
      const percentiles = ['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms'];
      const labels = ['p50', 'p95', 'p99'];
      const theme = currentTheme();
      const colors = [ORDINAL[theme][0], ORDINAL[theme][1], ORDINAL[theme][2]];
      const ceilingFields = ['median_latency_p50_ms', 'median_latency_p95_ms', 'median_latency_p99_ms'];
      const cells = OPS.map((op) => ({ op, values: percentiles.map((m) => { const v = evaluated.map((r) => stageB(r, state.source, op, m)).filter((x) => x !== null); return v.length ? { median: median(v), n: v.length } : null; }), ceiling: ceilingFields.map((f) => ceilingStat(state.source, op, f)) }));
      const flat = cells.flatMap((c) => c.values.filter(Boolean).map((v) => v.median)).concat(cells.flatMap((c) => c.ceiling.filter((v) => v !== null)));
      if (!flat.length) return emptyChart(body, 'Keine Latenzwerte im Filter (Server-Zeilen tragen keine Latenz).');
      const height = 240;
      const { svg, inner } = frame(width, height, { top: 14, right: 12, bottom: 28, left: 56 });
      const ticks = niceTicks(0, Math.max.apply(null, flat) * 1.05, 5);
      const y = linear([0, ticks[ticks.length - 1]], [inner.y + inner.h, inner.y]);
      const outer = band(OPS, [inner.x, inner.x + inner.w], 0.3);
      const innerBand = band(labels, [0, outer.bandwidth], 0.1);
      yAxis(svg, inner, y, ticks, (t) => fmt(t, 0));
      xBandLabels(svg, inner, outer, OPS);
      svg.appendChild(s('text', { class: 'axis-label', x: inner.x, y: inner.y - 4, text: 'ms' }));
      let anyCeiling = false;
      for (const cell of cells) {
        cell.values.forEach((entry, i) => {
          if (!entry) return;
          const bw = Math.min(innerBand.bandwidth, 24);
          const x0 = outer(cell.op) + innerBand(labels[i]) + (innerBand.bandwidth - bw) / 2;
          const top = y(entry.median);
          const bar = s('path', { class: 'mark', d: roundedBarPath(x0, top, bw, inner.y + inner.h - top, 4, 'up'), fill: colors[i] });
          hover(bar, cell.op + ' · ' + labels[i], () => [{ label: 'Median', value: fmt(entry.median, 1) + ' ms', color: colors[i] }, { label: 'Runs', value: String(entry.n) }, ...(cell.ceiling[i] !== null ? [{ label: 'bddbench', value: fmt(cell.ceiling[i], 1) + ' ms', color: cssVar('--s2') }] : [])]);
          svg.appendChild(bar);
        });
        const p95 = cell.values[1];
        if (p95) svg.appendChild(s('text', { class: 'value-label', x: outer(cell.op) + innerBand(labels[1]) + innerBand.bandwidth / 2, y: y(p95.median) - 4, 'text-anchor': 'middle', text: fmt(p95.median, 0) }));
        cell.ceiling.forEach((v, i) => {
          if (v === null) return;
          anyCeiling = true;
          const bw = Math.min(innerBand.bandwidth, 24);
          const x0 = outer(cell.op) + innerBand(labels[i]) + (innerBand.bandwidth - bw) / 2;
          svg.appendChild(s('line', { class: 'ceiling-line', x1: x0, x2: x0 + bw, y1: y(v), y2: y(v) }));
        });
      }
      body.appendChild(svg);
      const legendEntries = labels.map((l, i) => ({ label: l, color: colors[i] }));
      if (anyCeiling) legendEntries.push({ label: 'bddbench-Referenz (Ceiling)', color: cssVar('--s2'), kind: 'line' });
      legend(foot, legendEntries);
    },
    table(runs) {
      const evaluated = evaluatedRuns(runs);
      const rows = OPS.map((op) => {
        const row = { op };
        for (const m of ['latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']) {
          const v = evaluated.map((r) => stageB(r, state.source, op, m)).filter((x) => x !== null);
          row[m] = v.length ? fmt(median(v), 1) : '–';
          row.n = v.length;
        }
        return row;
      });
      return { columns: [{ key: 'op', label: 'Operation' }, { key: 'n', label: 'Stufe B', num: true }, { key: 'latency_p50_ms', label: 'p50 ms', num: true }, { key: 'latency_p95_ms', label: 'p95 ms', num: true }, { key: 'latency_p99_ms', label: 'p99 ms', num: true }], rows };
    },
  });

  registerCard({
    title: 'Coverage je Operation',
    subtitle: 'Anteil ausgewerteter Runs, in denen die Operation abgedeckt wurde (Proxy) · rechts: Request-Fehlerquote',
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs);
      if (!evaluated.length) return emptyChart(body);
      const stats = coverageStats(evaluated);
      const height = 200;
      const { svg, inner } = frame(width, height, { top: 14, right: 12, bottom: 28, left: 44 });
      const y = linear([0, 1], [inner.y + inner.h, inner.y]);
      const xb = band(OPS, [inner.x, inner.x + inner.w], 0.45);
      yAxis(svg, inner, y, [0, 0.25, 0.5, 0.75, 1], (t) => pct(t));
      xBandLabels(svg, inner, xb, OPS);
      const color = cssVar('--s1');
      for (const st of stats) {
        const bw = Math.min(xb.bandwidth, 24);
        const x0 = xb(st.op) + (xb.bandwidth - bw) / 2;
        const top = y(st.share);
        const bar = s('path', { class: 'mark', d: roundedBarPath(x0, top, bw, inner.y + inner.h - top, 4, 'up'), fill: color });
        hover(bar, st.op, () => [{ label: 'abgedeckt', value: st.covered + ' / ' + st.n + ' (' + pct(st.share) + ')', color }, { label: 'Requests', value: String(st.requests) }, { label: 'Fehlerquote', value: st.requests ? pct(st.errors / st.requests) : '–' }]);
        svg.appendChild(bar);
        svg.appendChild(s('text', { class: 'value-label', x: x0 + bw / 2, y: top - 4, 'text-anchor': 'middle', text: pct(st.share) }));
      }
      body.appendChild(svg);
      foot.textContent = stats.map((st) => st.op + ': ' + st.requests + ' Requests, ' + (st.requests ? pct(st.errors / st.requests) : '–') + ' Fehler').join(' · ');
    },
    table(runs) {
      const rows = coverageStats(evaluatedRuns(runs)).map((st) => ({ op: st.op, n: st.n, covered: st.covered, share: pct(st.share), requests: st.requests, errors: st.errors, rate: st.requests ? pct(st.errors / st.requests) : '–' }));
      return { columns: [{ key: 'op', label: 'Operation' }, { key: 'n', label: 'Runs', num: true }, { key: 'covered', label: 'abgedeckt', num: true }, { key: 'share', label: 'Anteil', num: true }, { key: 'requests', label: 'Requests', num: true }, { key: 'errors', label: 'Fehler', num: true }, { key: 'rate', label: 'Fehlerquote', num: true }], rows };
    },
  });
  function coverageStats(evaluated) {
    return OPS.map((op) => {
      let covered = 0, requests = 0, errors = 0;
      for (const run of evaluated) {
        if (run.covered_operations && run.covered_operations.includes(op)) covered += 1;
        const st = run.measurements && run.measurements.proxy && run.measurements.proxy[op];
        if (st) { requests += st.n_requests; errors += st.error_count; }
      }
      return { op, n: evaluated.length, covered, share: evaluated.length ? covered / evaluated.length : 0, requests, errors };
    });
  }

  registerCard({
    title: 'Proxy vs. /metrics (Cross-Check)',
    subtitle: () => (state.scatterMetric === 'n_requests' ? 'Request-Zählung' : 'Throughput') + ' je Run für ' + state.operation + ' · Linie = Gleichheit',
    chart(body, foot, runs, width) {
      const picker = h('div', { class: 'chips', style: 'margin-bottom:6px' });
      for (const [key, label] of [['n_requests', 'Requests'], ['throughput_ops_s', 'Throughput']]) {
        const chip = h('button', { type: 'button', class: 'chip', 'aria-pressed': String(state.scatterMetric === key), text: label });
        chip.addEventListener('click', () => { state.scatterMetric = key; renderAll(); });
        picker.appendChild(chip);
      }
      body.appendChild(picker);
      const pairs = scatterPairs(evaluatedRuns(runs));
      if (!pairs.length) return emptyChart(body, 'Keine Runs mit Proxy- und Server-Zeilen für ' + state.operation + '.');
      const height = 260;
      const { svg, inner } = frame(width, height, { top: 14, right: 16, bottom: 32, left: 56 });
      const maxV = Math.max.apply(null, pairs.flatMap((p) => [p.x, p.y])) * 1.05 || 1;
      const ticks = niceTicks(0, maxV, 5);
      const top = ticks[ticks.length - 1];
      const x = linear([0, top], [inner.x, inner.x + inner.w]);
      const y = linear([0, top], [inner.y + inner.h, inner.y]);
      yAxis(svg, inner, y, ticks, (t) => fmt(t, 0));
      for (const t of ticks) svg.appendChild(s('text', { class: 'tick', x: x(t), y: inner.y + inner.h + 16, 'text-anchor': 'middle', text: fmt(t, 0) }));
      svg.appendChild(s('line', { class: 'baseline', x1: inner.x, x2: inner.x + inner.w, y1: inner.y + inner.h, y2: inner.y + inner.h }));
      svg.appendChild(s('line', { class: 'ref-line', x1: x(0), y1: y(0), x2: x(top), y2: y(top) }));
      svg.appendChild(s('text', { class: 'axis-label', x: inner.x + inner.w, y: inner.y + inner.h + 30, 'text-anchor': 'end', text: 'Proxy' }));
      svg.appendChild(s('text', { class: 'axis-label', x: inner.x, y: inner.y - 4, text: '/metrics' }));
      const color = cssVar('--s1');
      for (const p of pairs) {
        svg.appendChild(s('circle', { cx: x(p.x), cy: y(p.y), r: 6, fill: cssVar('--surface') }));
        svg.appendChild(s('circle', { class: 'mark', cx: x(p.x), cy: y(p.y), r: 4.5, fill: color }));
        const hit = s('circle', { class: 'hit', cx: x(p.x), cy: y(p.y), r: 12 });
        hover(hit, p.run.run_id, () => [{ label: 'Proxy', value: fmt(p.x, state.scatterMetric === 'n_requests' ? 0 : 1), color }, { label: '/metrics', value: fmt(p.y, state.scatterMetric === 'n_requests' ? 0 : 1) }, { label: 'Δ relativ', value: p.x ? pct(Math.abs(p.x - p.y) / p.x) : '–' }]);
        svg.appendChild(hit);
      }
      body.appendChild(svg);
      const deltas = pairs.map((p) => (p.x ? Math.abs(p.x - p.y) / p.x : 0));
      foot.textContent = pairs.length + ' Run(s) · Median |Δ| relativ = ' + pct(median(deltas)) + ' · Der Proxy ist die primäre Messung; /metrics zählt nur Requests.';
    },
    table(runs) {
      const rows = scatterPairs(evaluatedRuns(runs)).map((p) => ({ run: p.run.run_id, proxy: fmt(p.x, 1), server: fmt(p.y, 1), delta: p.x ? pct(Math.abs(p.x - p.y) / p.x) : '–' }));
      return { columns: [{ key: 'run', label: 'Run' }, { key: 'proxy', label: 'Proxy', num: true }, { key: 'server', label: '/metrics', num: true }, { key: 'delta', label: '|Δ| rel.', num: true }], rows };
    },
  });
  function scatterPairs(evaluated) {
    const pairs = [];
    for (const run of evaluated) {
      const proxy = run.measurements && run.measurements.proxy && run.measurements.proxy[state.operation];
      const server = run.measurements && run.measurements.server && run.measurements.server[state.operation];
      if (!proxy || !server) continue;
      const a = proxy[state.scatterMetric], b = server[state.scatterMetric];
      if (a === null || a === undefined || b === null || b === undefined) continue;
      if (state.scatterMetric === 'n_requests' && a === 0 && b === 0) continue;
      pairs.push({ run, x: Number(a), y: Number(b) });
    }
    return pairs;
  }

  registerCard({
    title: 'Verlauf der Runs',
    subtitle: () => 'Zeitachse der Auswertung · Punkt = Throughput ' + state.operation + '/' + state.source + ' · Markierung auf der Nulllinie = Fehllauf',
    wide: true,
    chart(body, foot, runs, width) {
      const evaluated = evaluatedRuns(runs).filter((r) => r.evaluated_at);
      if (!evaluated.length) return emptyChart(body);
      const items = evaluated.map((r) => ({ run: r, t: new Date(r.evaluated_at).getTime(), v: stageB(r, state.source, state.operation, 'throughput_ops_s') })).filter((i) => !Number.isNaN(i.t)).sort((a, b) => a.t - b.t);
      const height = 220;
      const { svg, inner } = frame(width, height, { top: 14, right: 16, bottom: 30, left: 56 });
      const t0 = items[0].t, t1 = items[items.length - 1].t;
      const span = Math.max(t1 - t0, 60000);
      const x = linear([t0 - span * 0.03, t1 + span * 0.03], [inner.x, inner.x + inner.w]);
      const values = items.map((i) => i.v).filter((v) => v !== null);
      const ticks = niceTicks(0, (values.length ? Math.max.apply(null, values) : 1) * 1.05, 4);
      const y = linear([0, ticks[ticks.length - 1]], [inner.y + inner.h, inner.y]);
      yAxis(svg, inner, y, ticks, (t) => fmt(t, 0));
      svg.appendChild(s('line', { class: 'baseline', x1: inner.x, x2: inner.x + inner.w, y1: inner.y + inner.h, y2: inner.y + inner.h }));
      const nLabels = Math.max(2, Math.min(6, Math.floor(inner.w / 140)));
      for (let i = 0; i < nLabels; i++) {
        const t = t0 + (span * i) / (nLabels - 1);
        svg.appendChild(s('text', { class: 'tick', x: x(t), y: inner.y + inner.h + 16, 'text-anchor': i === 0 ? 'start' : i === nLabels - 1 ? 'end' : 'middle', text: fmtTime(new Date(t).toISOString()) }));
      }
      const okColor = cssVar('--s1');
      const tones = { good: cssVar('--good'), serious: cssVar('--serious'), critical: cssVar('--critical'), warning: cssVar('--warning'), muted: cssVar('--muted') };
      for (const item of items) {
        const px = x(item.t);
        if (item.v !== null) {
          svg.appendChild(s('circle', { cx: px, cy: y(item.v), r: 6, fill: cssVar('--surface') }));
          svg.appendChild(s('circle', { class: 'mark', cx: px, cy: y(item.v), r: 4.5, fill: okColor }));
          const hit = s('circle', { class: 'hit', cx: px, cy: y(item.v), r: 12 });
          hover(hit, item.run.run_id, () => [{ label: 'Throughput', value: fmt(item.v, 1) + ' ops/s', color: okColor }, { label: 'Status', value: statusMeta(item.run.status).label }, { label: 'Zeit', value: fmtTime(item.run.evaluated_at) }]);
          svg.appendChild(hit);
        } else {
          const tone = statusMeta(item.run.status).tone;
          const py = inner.y + inner.h - 6;
          const mark = s('path', { class: 'mark', d: 'M' + (px - 4) + ',' + (py - 4) + ' L' + (px + 4) + ',' + (py + 4) + ' M' + (px - 4) + ',' + (py + 4) + ' L' + (px + 4) + ',' + (py - 4), stroke: tones[tone], 'stroke-width': 2, 'stroke-linecap': 'round' });
          const hit = s('circle', { class: 'hit', cx: px, cy: py, r: 12 });
          hover(hit, item.run.run_id, () => [{ label: 'Status', value: statusMeta(item.run.status).label, color: tones[tone] }, { label: 'Fehler', value: (item.run.error || item.run.skip_reason || '–').slice(0, 80) }, { label: 'Zeit', value: fmtTime(item.run.evaluated_at) }]);
          svg.appendChild(mark);
          svg.appendChild(hit);
        }
      }
      body.appendChild(svg);
      legend(foot, [{ label: 'Stufe-B-Run (ops/s)', color: okColor, kind: 'dot' }, { label: '✕ Ausführung fehlgeschlagen', color: tones.serious }, { label: '✕ Validierung/Generierung fehlgeschlagen', color: tones.critical }, { label: '! Infrastruktur', color: tones.warning }, { label: '○ nicht gemessen', color: tones.muted }]);
    },
    table(runs) {
      const rows = evaluatedRuns(runs).filter((r) => r.evaluated_at).sort((a, b) => a.evaluated_at.localeCompare(b.evaluated_at)).map((r) => ({ time: fmtTime(r.evaluated_at), run: r.run_id, status: statusMeta(r.status).label, value: fmt(stageB(r, state.source, state.operation, 'throughput_ops_s'), 1) }));
      return { columns: [{ key: 'time', label: 'Zeit' }, { key: 'run', label: 'Run' }, { key: 'status', label: 'Status' }, { key: 'value', label: 'ops/s', num: true }], rows };
    },
  });

  registerCard({
    title: 'Feedback-Runden & Versuche',
    subtitle: 'Verteilung der Feedback-Iterationen (L4) über alle Runs im Filter',
    chart(body, foot, runs, width) {
      if (!runs.length) return emptyChart(body);
      const counts = groupBy(runs, (r) => r.feedback_iterations || 0);
      const keys = Array.from(counts.keys()).sort((a, b) => a - b);
      const height = 180;
      const { svg, inner } = frame(width, height, { top: 14, right: 12, bottom: 28, left: 44 });
      const maxN = Math.max.apply(null, keys.map((k) => counts.get(k).length));
      const ticks = niceTicks(0, maxN, 4);
      const y = linear([0, ticks[ticks.length - 1] || 1], [inner.y + inner.h, inner.y]);
      const xb = band(keys.map(String), [inner.x, inner.x + inner.w], 0.45);
      yAxis(svg, inner, y, ticks, (t) => fmt(t, 0));
      xBandLabels(svg, inner, xb, keys.map(String), (k) => k + ' Runden');
      const color = cssVar('--s1');
      for (const k of keys) {
        const n = counts.get(k).length;
        const bw = Math.min(xb.bandwidth, 24);
        const x0 = xb(String(k)) + (xb.bandwidth - bw) / 2;
        const top = y(n);
        const bar = s('path', { class: 'mark', d: roundedBarPath(x0, top, bw, inner.y + inner.h - top, 4, 'up'), fill: color });
        const ok = counts.get(k).filter((r) => r.success).length;
        hover(bar, k + ' Feedback-Runden', () => [{ label: 'Runs', value: String(n), color }, { label: 'davon erfolgreich', value: String(ok) }]);
        svg.appendChild(bar);
        svg.appendChild(s('text', { class: 'value-label', x: x0 + bw / 2, y: top - 4, 'text-anchor': 'middle', text: String(n) }));
      }
      body.appendChild(svg);
      const l4 = runs.filter((r) => r.guidance_level === 'L4');
      foot.textContent = l4.length ? 'L4: ' + l4.length + ' Runs, Ø ' + fmt(mean(l4.map((r) => r.feedback_iterations || 0)), 2) + ' Runden, max ' + Math.max.apply(null, l4.map((r) => r.feedback_iterations || 0)) : 'Keine L4-Runs im Filter: andere Level iterieren nie.';
    },
    table(runs) {
      const counts = groupBy(runs, (r) => r.feedback_iterations || 0);
      const rows = Array.from(counts.keys()).sort((a, b) => a - b).map((k) => ({ rounds: k, n: counts.get(k).length, ok: counts.get(k).filter((r) => r.success).length }));
      return { columns: [{ key: 'rounds', label: 'Feedback-Runden', num: true }, { key: 'n', label: 'Runs', num: true }, { key: 'ok', label: 'erfolgreich', num: true }], rows };
    },
  });

  function renderStatistics(runs) {
    const root = document.getElementById('statistics');
    clear(root);
    const key = state.operation + '|' + state.source;
    const report = DATA.stats[key];
    const evaluated = evaluatedRuns(runs);
    root.appendChild(h('div', { class: 'section-head' }, [h('h2', { text: 'Statistik' }), h('span', { class: 'sub', text: 'Deskriptiv: aktueller Filter · Inferenz: alle ausgewerteten Runs ohne Fixtures (' + state.operation + ' · ' + state.source + ')' })]));
    const two = h('div', { class: 'two' });

    const cells = [];
    for (const level of DATA.levels) for (const model of DATA.models) for (const prompting of DATA.promptings) {
      const cell = evaluated.filter((r) => r.guidance_level === level && r.model === model && r.prompting === prompting);
      if (!cell.length) continue;
      const thr = cell.map((r) => stageB(r, state.source, state.operation, 'throughput_ops_s')).filter((v) => v !== null);
      const p95 = cell.map((r) => stageB(r, state.source, state.operation, 'latency_p95_ms')).filter((v) => v !== null);
      const cov = cell.map((r) => r.coverage_ratio).filter((v) => v !== null && v !== undefined);
      cells.push({
        level, model, prompting: promptLabel(prompting), n: cell.length,
        syntax: cell.filter((r) => r.syntax_ok).length, executed: cell.filter((r) => r.executed_ok).length, success: cell.filter((r) => r.success).length,
        rate: pct(cell.filter((r) => r.success).length / cell.length), coverage: cov.length ? pct(mean(cov)) : '–', k: thr.length,
        median: fmt(median(thr), 1), iqr: fmt(iqr(thr), 1), naive: fmt(thr.reduce((a, b) => a + b, 0) / cell.length, 1), p95: fmt(median(p95), 1),
        feedback: fmt(mean(cell.map((r) => r.feedback_iterations || 0)), 2),
        classes: (() => { const counts = {}; for (const r of cell) { if (r.failure_class && r.failure_class !== 'none') counts[r.failure_class] = (counts[r.failure_class] || 0) + 1; } return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => k + '×' + v).join(', ') || '–'; })(),
      });
    }
    const descCard = h('section', { class: 'card' }, [h('div', { class: 'card-head' }, [h('div', {}, [h('h2', { text: 'Zellen Level × Modell × Prompting' }), h('div', { class: 'sub', text: 'Stufe A (Quoten) und Stufe B (' + state.operation + ', ' + state.source + '; Median/IQR nur über ausgeführte Runs, "naiv" = Fehlläufe als 0)' })])])]);
    const descBody = h('div', { class: 'card-body' });
    renderTable(descBody, [
      { key: 'level', label: 'Level' }, { key: 'model', label: 'Modell' }, { key: 'prompting', label: 'Prompting' }, { key: 'n', label: 'n', num: true },
      { key: 'syntax', label: 'Syntax', num: true }, { key: 'executed', label: 'ausgef.', num: true }, { key: 'success', label: 'Erfolg', num: true }, { key: 'rate', label: 'Quote', num: true },
      { key: 'coverage', label: 'Ø Cov.', num: true }, { key: 'k', label: 'Stufe B', num: true }, { key: 'median', label: 'Median ops/s', num: true }, { key: 'iqr', label: 'IQR', num: true }, { key: 'naive', label: 'naiv ops/s', num: true }, { key: 'p95', label: 'p95 ms', num: true }, { key: 'feedback', label: 'Ø Feedback', num: true }, { key: 'classes', label: 'Fehlerklassen' },
    ], cells);
    descCard.appendChild(descBody);
    two.appendChild(descCard);

    const testCard = h('section', { class: 'card' }, [h('div', { class: 'card-head' }, [h('div', {}, [h('h2', { text: 'Inferenzstatistik (nicht-parametrisch)' }), h('div', { class: 'sub', text: report ? 'n = ' + report.n_measured + ' ausgewertete Runs · berechnet mit scipy beim Erzeugen der Seite' : 'kein Report' })])])]);
    const testBody = h('div', { class: 'card-body' });
    if (report) {
      renderTable(testBody, [
        { key: 'name', label: 'Test' }, { key: 'description', label: 'Fragestellung' },
        { key: 'groups', label: 'Gruppen (n)', render: (t) => t.groups.map((g, i) => g + (t.group_sizes[i] !== undefined ? ' (' + t.group_sizes[i] + ')' : '')).join(', ') || '–' },
        { key: 'statistic', label: 'Statistik', num: true, render: (t) => (t.statistic === null || t.statistic === undefined ? '–' : t.statistic_name + ' = ' + fmt(t.statistic, 3)) },
        { key: 'p_value', label: 'p', num: true, render: (t) => { const span = h('span', { class: t.p_value !== null && t.p_value !== undefined && t.p_value < 0.05 ? 'p-sig' : 'p-ns', text: fmtP(t.p_value) }); return span; } },
        { key: 'p_adjusted', label: 'p (Holm)', num: true, render: (t) => { if (t.p_adjusted === null || t.p_adjusted === undefined) return '–'; const span = h('span', { class: t.p_adjusted < 0.05 ? 'p-sig' : 'p-ns', text: fmtP(t.p_adjusted) + (t.adjust_method ? ' ' + t.adjust_method.replace('Holm ', '') : '') }); return span; } },
        { key: 'effect', label: 'Effekt', num: true, render: (t) => (t.effect === null || t.effect === undefined ? '–' : fmt(t.effect, 3) + ' (' + (t.effect_name || '') + ')') },
        { key: 'note', label: 'Hinweis', render: (t) => (t.applicable ? (t.note || '') : 'nicht anwendbar: ' + t.note) },
      ], report.tests);
      for (const note of report.notes || []) testBody.appendChild(h('p', { class: 'note warn', text: note }));
    }
    testCard.appendChild(testBody);
    testCard.appendChild(h('div', { class: 'card-foot', text: 'Signifikanz bei p < 0,05 (grün). Tests werden erst bei ausreichender Gruppengröße gerechnet; Fehlläufe gehen als NA ein, nie als 0.' }));
    two.appendChild(testCard);
    root.appendChild(two);
  }

  const RUN_COLUMNS = [
    { key: 'run_id', label: 'Run', render: (r) => { const span = h('span', { text: r.run_id }); if (r.is_fixture) span.appendChild(h('span', { class: 'flag', text: 'Fixture' })); if (r.synthetic) span.appendChild(h('span', { class: 'flag', text: 'synthetisch' })); return span; } },
    { key: 'guidance_level', label: 'Level' },
    { key: 'model', label: 'Modell' },
    { key: 'prompting', label: 'Prompting', render: (r) => promptLabel(r.prompting) },
    { key: 'repetition', label: 'Rep.', num: true },
    { key: 'status', label: 'Status', render: (r) => statusBadge(r.status) },
    { key: 'failure_class', label: 'Fehlerklasse', render: (r) => (r.failure_class && r.failure_class !== 'none' ? h('span', { class: 'flag', title: r.failure_detail || '', text: r.failure_class }) : '–') },
    { key: 'syntax_ok', label: 'Syntax', render: (r) => h('span', { class: r.syntax_ok ? 'ok' : 'fail', text: r.syntax_ok ? '✓' : '✕' }) },
    { key: 'executed_ok', label: 'Ausgef.', render: (r) => h('span', { class: r.executed_ok ? 'ok' : 'fail', text: r.executed_ok ? '✓' : '✕' }) },
    { key: 'semantics_ok', label: 'Semantik', render: (r) => h('span', { class: r.semantics_ok ? 'ok' : 'fail', text: r.semantics_ok ? '✓' : '✕' }) },
    { key: 'coverage_ratio', label: 'Coverage', num: true, render: (r) => pct(r.coverage_ratio) },
    { key: '_throughput', label: 'ops/s', num: true, render: (r) => fmt(stageB(r, state.source, state.operation, 'throughput_ops_s'), 1) },
    { key: '_p95', label: 'p95 ms', num: true, render: (r) => fmt(stageB(r, state.source, state.operation, 'latency_p95_ms'), 1) },
    { key: '_requests', label: 'Req.', num: true, render: (r) => { const st = r.measurements && r.measurements[state.source] && r.measurements[state.source][state.operation]; return st ? String(st.n_requests) : '–'; } },
    { key: '_errors', label: 'Fehler', num: true, render: (r) => { const st = r.measurements && r.measurements[state.source] && r.measurements[state.source][state.operation]; return st ? String(st.error_count) : '–'; } },
    { key: 'feedback_iterations', label: 'Feedback', num: true },
    { key: 'evaluated_at', label: 'ausgewertet', render: (r) => fmtTime(r.evaluated_at || r.generated_at) },
    { key: 'error', label: 'Fehler / Grund', render: (r) => (r.error || r.skip_reason || '').slice(0, 60) },
  ];
  function sortValue(run, key) {
    if (key === '_throughput') return stageB(run, state.source, state.operation, 'throughput_ops_s');
    if (key === '_p95') return stageB(run, state.source, state.operation, 'latency_p95_ms');
    if (key === '_requests' || key === '_errors') { const st = run.measurements && run.measurements[state.source] && run.measurements[state.source][state.operation]; return st ? (key === '_requests' ? st.n_requests : st.error_count) : null; }
    if (key === 'evaluated_at') return run.evaluated_at || run.generated_at || '';
    return run[key];
  }
  function renderRuns(runs) {
    const root = document.getElementById('runs');
    clear(root);
    root.appendChild(h('div', { class: 'section-head' }, [h('h2', { text: 'Alle Runs (' + runs.length + ')' }), h('span', { class: 'sub', text: 'Klick auf eine Zeile öffnet die Details · Spaltenkopf sortiert' })]));
    const sorted = runs.slice().sort((a, b) => {
      const va = sortValue(a, state.sort.key), vb = sortValue(b, state.sort.key);
      if (va === vb) return a.run_id.localeCompare(b.run_id);
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      const cmp = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb));
      return state.sort.dir === 'asc' ? cmp : -cmp;
    });
    renderTable(root, RUN_COLUMNS, sorted, {
      sortable: true,
      sort: state.sort,
      onSort: (key) => { state.sort = { key, dir: state.sort.key === key && state.sort.dir === 'desc' ? 'asc' : 'desc' }; renderRuns(filteredRuns()); },
      onRow: (run) => openDrawer(run),
      rowClass: (run) => 'clickable' + (state.selectedRun === run.run_id ? ' selected' : ''),
    });
  }

  function openDrawer(run) {
    state.selectedRun = run.run_id;
    const drawer = document.getElementById('drawer');
    clear(drawer);
    const close = h('button', { type: 'button', class: 'btn', text: 'Schließen' });
    close.addEventListener('click', closeDrawer);
    drawer.appendChild(h('div', { class: 'drawer-head' }, [h('h2', { text: run.run_id }), close]));
    drawer.appendChild(statusBadge(run.status));
    const kv = h('dl', { class: 'kv' });
    const add = (k, v) => { kv.appendChild(h('dt', { text: k })); const dd = h('dd'); if (v instanceof Node) dd.appendChild(v); else dd.textContent = v === null || v === undefined || v === '' ? '–' : String(v); kv.appendChild(dd); };
    add('Kondition', run.condition_name + (run.condition_id !== null && run.condition_id !== undefined ? ' (id ' + run.condition_id + ')' : ''));
    add('Level / Modell / Prompting', (run.guidance_level || '?') + ' · ' + (run.model || '?') + ' · ' + promptLabel(run.prompting));
    add('Wiederholung', run.repetition);
    add('Modellversion', run.model_version);
    add('Seed', run.seed);
    add('Generierungsstatus', run.generation_status);
    add('Syntax / Ausgeführt / Semantik', (run.syntax_ok ? '✓' : '✕') + ' / ' + (run.executed_ok ? '✓' : '✕') + ' / ' + (run.semantics_ok ? '✓' : '✕'));
    add('Erfolg (Stufe A)', run.success ? 'ja' : 'nein');
    if (run.failure_class && run.failure_class !== 'none') add('Fehlerklasse', run.failure_class + (run.failure_detail ? ': ' + run.failure_detail : ''));
    if (run.execution_feedback_rounds) add('Ausführungs-Feedback (Runden)', run.execution_feedback_rounds);
    if (run.spec_coverage !== null && run.spec_coverage !== undefined) add('Spec-Coverage (spec-exklusive Prüfungen)', pct(run.spec_coverage));
    add('Workload-Konformität (Kriterien v' + (run.criteria_version || 1) + ')', run.conformance_ok === null || run.conformance_ok === undefined ? '–' : (run.conformance_ok ? '✓' : '✕'));
    if (run.sut_api) add('API / Spezifikationstreue', run.sut_api + ' / ' + (run.adherence_ok === null || run.adherence_ok === undefined ? '–' : (run.adherence_ok ? '✓' : '✕ (' + ((run.api_rejections || []).join(', ') || 'legacy ' + run.api_legacy_requests) + ')')));
    if (run.vocab_leak !== null && run.vocab_leak !== undefined) add('InfluxDB-Vokabular im Artefakt', run.vocab_leak === 0 ? '0' : run.vocab_leak + ' (' + Object.entries(run.vocab_hits || {}).map(([k, v]) => k + '×' + v).join(', ') + ')');
    add('Epoche / Temperatur / Backend', (run.epoch === null || run.epoch === undefined ? '1' : run.epoch) + ' / ' + (run.temperature === null || run.temperature === undefined ? '–' : run.temperature) + ' / ' + (run.llm_backend || '–'));
    add('Coverage', pct(run.coverage_ratio) + (run.covered_operations && run.covered_operations.length ? ' (' + run.covered_operations.join(', ') + ')' : ''));
    add('Feedback-Runden / Versuche / Probes', (run.feedback_iterations || 0) + ' / ' + (run.attempts === null || run.attempts === undefined ? '–' : run.attempts) + ' / ' + (run.probe_calls === null || run.probe_calls === undefined ? '–' : run.probe_calls));
    add('behave', run.behave_exit_code === null || run.behave_exit_code === undefined ? '–' : 'exit ' + run.behave_exit_code + ' · passed ' + (run.scenarios_passed === null ? '–' : run.scenarios_passed) + ' · failed ' + (run.scenarios_failed === null ? '–' : run.scenarios_failed) + ' · error ' + (run.scenarios_error === null ? '–' : run.scenarios_error) + (run.timed_out ? ' · TIMEOUT' : ''));
    add('generiert', fmtTime(run.generated_at));
    add('ausgewertet', fmtTime(run.evaluated_at));
    add('Verzeichnis', run.run_directory);
    if (run.usage && Object.keys(run.usage).length) add('Token-Usage', Object.entries(run.usage).map(([k, v]) => k + '=' + v).join(', '));
    drawer.appendChild(kv);
    if (run.error || run.skip_reason) { drawer.appendChild(h('h3', { text: 'Fehler / Grund' })); drawer.appendChild(h('pre', { text: run.error || run.skip_reason })); }
    if (run.validation_issues && run.validation_issues.length) { drawer.appendChild(h('h3', { text: 'Validierungsbefunde' })); drawer.appendChild(h('pre', { text: run.validation_issues.join('\n') })); }
    const conformanceLines = (run.conformance_issues || []).concat((run.conformance_deviations || []).map((d) => 'Abweichung (nur vermerkt): ' + d));
    if (conformanceLines.length) { drawer.appendChild(h('h3', { text: 'Workload-Konformität' })); drawer.appendChild(h('pre', { text: conformanceLines.join('\n') })); }
    drawer.appendChild(h('h3', { text: 'Messung je Quelle und Operation' }));
    const rows = [];
    for (const source of DATA.measured_sources) for (const op of OPS) {
      const st = run.measurements && run.measurements[source] && run.measurements[source][op];
      if (!st) continue;
      rows.push({ source, op, n: st.n_requests, errors: st.error_count, points: st.n_points, thr: fmt(st.throughput_ops_s, 2), p50: fmt(st.latency_p50_ms, 1), p95: fmt(st.latency_p95_ms, 1), p99: fmt(st.latency_p99_ms, 1), covered: st.covered ? '✓' : '✕' });
    }
    renderTable(drawer, [{ key: 'source', label: 'Quelle' }, { key: 'op', label: 'Op' }, { key: 'n', label: 'Req.', num: true }, { key: 'errors', label: 'Fehler', num: true }, { key: 'points', label: 'Punkte', num: true }, { key: 'thr', label: 'ops/s', num: true }, { key: 'p50', label: 'p50', num: true }, { key: 'p95', label: 'p95', num: true }, { key: 'p99', label: 'p99', num: true }, { key: 'covered', label: 'cov.' }], rows);
    drawer.hidden = false;
    renderRuns(filteredRuns());
  }
  function closeDrawer() { state.selectedRun = null; document.getElementById('drawer').hidden = true; renderRuns(filteredRuns()); }
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { closeDrawer(); hideTip(); } });

  function renderMatrices() {
    const root = document.getElementById('matrices');
    clear(root);
    if (!DATA.matrices || !DATA.matrices.length) return;
    root.appendChild(h('div', { class: 'section-head' }, [h('h2', { text: 'Matrix-Läufe (benchmark-agent run)' }), h('span', { class: 'sub', text: 'results/raw/matrix_*.json, neueste zuerst' })]));
    renderTable(root, [
      { key: 'timestamp', label: 'Zeit', render: (m) => fmtTime(m.timestamp) }, { key: 'file', label: 'Datei' }, { key: 'n_runs', label: 'Runs', num: true }, { key: 'n_failed', label: 'nicht erfolgreich', num: true },
      { key: 'conditions', label: 'Konditionen', render: (m) => m.conditions.join(', ') }, { key: 'errors', label: 'Fehler', render: (m) => m.errors.join(' | ').slice(0, 120) },
    ], DATA.matrices);
  }
  function renderHeader() {
    const subtitle = document.getElementById('subtitle');
    clear(subtitle);
    subtitle.appendChild(document.createTextNode('Stand ' + fmtTime(DATA.generated_at) + ' · ' + DATA.runs.length + ' Runs · Quelle: '));
    subtitle.appendChild(h('code', { text: DATA.sources.csv_path || '–' }));
    if (DATA.sources.artifacts_dir) { subtitle.appendChild(document.createTextNode(' + ')); subtitle.appendChild(h('code', { text: DATA.sources.artifacts_dir })); }
    subtitle.appendChild(document.createTextNode(' · ' + DATA.sources.csv_row_count + ' CSV-Zeilen'));
    const quarantined = DATA.sources.quarantined || [];
    if (quarantined.length) {
      const note = h('span', { class: 'quarantine-note', text: ' · ' + quarantined.length + ' Runs in Quarantäne (harness-bedingt, nicht gewertet)' });
      note.title = quarantined.map((q) => q.run_id + ': ' + q.reason).join('\n');
      subtitle.appendChild(note);
    }
    const issues = document.getElementById('issues');
    clear(issues);
    if (DATA.sources.issues && DATA.sources.issues.length) {
      issues.hidden = false;
      issues.appendChild(h('strong', { text: 'Datenprobleme beim Einlesen:' }));
      issues.appendChild(h('ul', {}, DATA.sources.issues.map((i) => h('li', { text: i }))));
    } else issues.hidden = true;
    const empty = document.getElementById('empty');
    clear(empty);
    if (!DATA.runs.length) {
      empty.hidden = false;
      empty.appendChild(h('p', {}, [h('strong', { text: 'Noch keine Runs.' }), ' Erzeuge Daten mit ', h('code', { text: 'uv run benchmark-agent run --condition L2_gemini_zeroshot' }), ' oder miss die Gold-Fixture mit ', h('code', { text: 'uv run benchmark-agent evaluate --run tests/fixtures/gold_live' }), '. Layout-Vorschau: ', h('code', { text: 'uv run benchmark-agent dashboard --seed-demo' })]));
    } else empty.hidden = true;
  }
  function renderDefinitions() {
    const root = document.getElementById('definitions');
    clear(root);
    root.appendChild(h('h2', { text: 'Definitionen (identisch mit dem Evaluations-Code)' }));
    const dl = h('dl');
    for (const [key, text] of Object.entries(DATA.definitions || {})) { dl.appendChild(h('dt', { text: key })); dl.appendChild(h('dd', { text })); }
    root.appendChild(dl);
  }

  let resizeTimer = null;
  function renderAll() {
    renderFilters();
    const runs = filteredRuns();
    renderKpis(runs);
    renderCards(runs);
    renderStatistics(runs);
    renderRuns(runs);
  }
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => renderCards(filteredRuns()), 150); });
  renderHeader();
  renderMatrices();
  renderDefinitions();
  renderAll();
})();
