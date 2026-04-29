// dashboard/static/app.js — Trading Agent Dashboard

const POLL_MS = 15_000;
let equityChart = null;

// ── Formatters ────────────────────────────────────────────────────────────────

const fmt$ = n =>
  n == null ? '—' : '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtPct = n =>
  n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(2) + '%';
const fmtR = n =>
  n == null ? '—' : (n >= 0 ? '+' : '') + Number(n).toFixed(2) + 'R';
const fmtScore = n =>
  (!n && n !== 0) ? '—' : Number(n).toFixed(3);
const fmtTime = s => {
  if (!s) return '—';
  try {
    return new Date(s).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  } catch { return s.slice(11, 16) || '—'; }
};
const colorPnl = n => (n > 0 ? 'pos' : n < 0 ? 'neg' : '');
const colorR   = n => (n > 0 ? 'pos' : n < 0 ? 'neg' : '');
const esc = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const el = id => document.getElementById(id);

function setText(id, text, extraClass) {
  const e = el(id);
  if (!e) return;
  e.textContent = text;
  if (extraClass !== undefined) {
    e.className = e.className.replace(/\b(pos|neg|warn|acc|pos-glow|neg-glow)\b/g, '').trim();
    if (extraClass) e.className += ' ' + extraClass;
  }
}

function setRows(tbodyId, html) {
  const b = el(tbodyId);
  if (b) b.innerHTML = html || `<tr><td colspan="99" class="empty-row">No data</td></tr>`;
}

// ── Live Clock & Market Hours ─────────────────────────────────────────────────

function updateClock() {
  const now = new Date();
  const et  = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const hh  = String(et.getHours()).padStart(2, '0');
  const mm  = String(et.getMinutes()).padStart(2, '0');
  const ss  = String(et.getSeconds()).padStart(2, '0');
  const clockEl = el('clock');
  if (clockEl) clockEl.textContent = `${hh}:${mm}:${ss} ET`;

  // Market hours: Mon–Fri 09:30–16:00 ET
  const dow   = et.getDay(); // 0=Sun, 6=Sat
  const mins  = et.getHours() * 60 + et.getMinutes();
  const open  = 9  * 60 + 30;
  const close = 16 * 60;

  const pill   = el('market-pill');
  const label  = el('mkt-label');
  const cntdn  = el('mkt-countdown');
  if (!pill || !label || !cntdn) return;

  const isWeekday = dow >= 1 && dow <= 5;
  const isOpen    = isWeekday && mins >= open && mins < close;

  pill.className = 'market-pill ' + (isOpen ? 'open' : 'closed');
  label.textContent = isOpen ? 'MARKET OPEN' : 'MARKET CLOSED';

  if (isOpen) {
    const left = close - mins;
    cntdn.textContent = `closes in ${Math.floor(left/60)}h ${left%60}m`;
  } else if (isWeekday && mins < open) {
    const left = open - mins;
    cntdn.textContent = `opens in ${Math.floor(left/60)}h ${left%60}m`;
  } else {
    // Weekend / after close — next open is Monday (or tomorrow if Mon–Thu after close)
    cntdn.textContent = 'reopens Mon 9:30';
  }
}

setInterval(updateClock, 1000);
updateClock();

// ── Portfolio ─────────────────────────────────────────────────────────────────

async function loadPortfolio() {
  const d = await fetch('/api/portfolio').then(r => r.json()).catch(() => ({}));

  setText('h-portfolio', fmt$(d.portfolio_value));
  const pnl = d.pnl_today || 0;
  setText('h-pnl', fmt$(pnl), pnl > 0 ? 'pos' : pnl < 0 ? 'neg' : '');
  setText('h-cash', fmt$(d.cash));
  setText('h-bp', fmt$(d.buying_power));

  const badge = el('mode-badge');
  if (badge) {
    badge.textContent = d.mode || 'PAPER';
    badge.className   = `badge${d.mode === 'LIVE' ? ' live' : ''}`;
  }

  const positions = d.positions || [];
  const posCount  = el('positions-count');
  if (posCount) posCount.textContent = String(positions.length);

  // Heat gauge
  const totalVal  = d.portfolio_value || 0;
  const totalRisk = positions.reduce((sum, p) => sum + Math.abs(p.value || 0), 0);
  const heatPct   = totalVal > 0 ? Math.min((totalRisk / totalVal) * 100, 100) : 0;
  const heatFill  = el('heat-fill');
  const heatPctEl = el('heat-pct');
  if (heatFill)  heatFill.style.width = heatPct.toFixed(1) + '%';
  if (heatPctEl) {
    heatPctEl.textContent = heatPct.toFixed(1) + '%';
    heatPctEl.className   = 'heat-pct ' + (heatPct > 4 ? 'neg' : heatPct > 2 ? 'warn' : '');
  }

  const rows = positions.map(p => {
    const pc = colorPnl(p.pnl);
    const sideClass = p.side === 'long' ? 'dir-long' : 'dir-short';
    return `<tr>
      <td class="acc" style="font-weight:700">${esc(p.symbol)}</td>
      <td class="${sideClass}">${esc((p.side || '').toUpperCase())}</td>
      <td>${fmt$(p.value)}</td>
      <td class="${pc}">${fmt$(p.pnl)}</td>
      <td class="${pc}">${fmtPct(p.pnl_pct)}</td>
    </tr>`;
  }).join('');

  setRows('positions-body', rows || `<tr><td colspan="5" class="empty-row">No open positions</td></tr>`);
}

// ── Stats ─────────────────────────────────────────────────────────────────────

async function loadStats() {
  const s = await fetch('/api/stats?days=30').then(r => r.json()).catch(() => ({}));

  const wr = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';
  const wrClass = s.win_rate > 0.55 ? 'pos' : s.win_rate < 0.4 ? 'neg' : '';
  setText('s-winrate', wr, wrClass);
  if (s.win_rate != null) {
    const sub = el('s-winrate-sub');
    if (sub) sub.textContent = s.win_rate > 0.5 ? 'above breakeven' : 'below breakeven';
  }

  const avgR = s.avg_r != null ? fmtR(s.avg_r) : '—';
  setText('s-avgr', avgR, colorR(s.avg_r));

  const pf = s.profit_factor != null ? s.profit_factor.toFixed(2) + 'x' : '—';
  setText('s-pf', pf, s.profit_factor > 1.5 ? 'pos' : s.profit_factor < 1 ? 'neg' : '');
  if (s.profit_factor != null) {
    const sub = el('s-pf-sub');
    if (sub) sub.textContent = s.profit_factor >= 2 ? 'excellent' : s.profit_factor >= 1.5 ? 'good' : s.profit_factor >= 1 ? 'marginal' : 'unprofitable';
  }

  const totalPnlClass = s.total_pnl > 0 ? 'pos-glow' : s.total_pnl < 0 ? 'neg-glow' : '';
  setText('s-pnl', fmt$(s.total_pnl), totalPnlClass);
  setText('s-closed', String(s.closed_trades || 0));
  setText('s-pending', String(s.pending_approvals || 0), s.pending_approvals > 0 ? 'warn' : '');

  renderEquityCurve(s.equity_curve || []);
}

// ── Trades ────────────────────────────────────────────────────────────────────

async function loadTrades() {
  const rows = await fetch('/api/trades?days=30').then(r => r.json()).catch(() => []);

  const executed = rows.filter(r =>
    r.direction !== 'SKIP' &&
    !['PENDING_APPROVAL', 'FAILED', 'SKIPPED'].includes(r.approval_status)
  );
  const pending = rows.filter(r => r.approval_status === 'PENDING_APPROVAL');
  const skips   = rows.filter(r => r.direction === 'SKIP');

  renderExecuted(executed.slice(0, 100));
  renderPending(pending);
  renderSkips(skips.slice(0, 100));
  renderSignalFeed(rows.slice(0, 30));
  renderSkipBreakdown(skips);

  const sigCount = el('signals-count');
  if (sigCount) sigCount.textContent = String(rows.length);
}

function renderExecuted(rows) {
  if (!rows.length) { setRows('executed-body', ''); return; }
  const html = rows.map(r => {
    const dc = r.direction === 'LONG' ? 'dir-long' : r.direction === 'SHORT' ? 'dir-short' : '';
    const oc = r.outcome === 'WIN' ? 'out-win' : r.outcome === 'LOSS' ? 'out-loss' : 'out-open';
    return `<tr>
      <td>${fmtTime(r.timestamp)}</td>
      <td class="acc" style="font-weight:700">${esc(r.ticker)}</td>
      <td class="${dc}">${esc(r.direction)}</td>
      <td>${fmt$(r.entry_price)}</td>
      <td>${fmt$(r.stop_loss)}</td>
      <td>${fmt$(r.take_profit)}</td>
      <td>${r.shares}</td>
      <td>${fmtScore(r.signal_score)}</td>
      <td>${esc(r.macro_regime || '—')}</td>
      <td>${esc(r.order_status || '—')}</td>
      <td class="${oc}">${esc(r.outcome || 'OPEN')}</td>
      <td class="${colorPnl(r.pnl_dollars)}">${r.closed ? fmt$(r.pnl_dollars) : '—'}</td>
      <td class="${colorR(r.pnl_r)}">${r.closed ? fmtR(r.pnl_r) : '—'}</td>
    </tr>`;
  }).join('');
  setRows('executed-body', html);
}

function renderPending(rows) {
  if (!rows.length) { setRows('pending-body', ''); return; }
  const html = rows.map(r => {
    const dc = r.direction === 'LONG' ? 'dir-long' : 'dir-short';
    const reason = (r.signal_reasoning || '').substring(0, 90);
    return `<tr>
      <td>${fmtTime(r.timestamp)}</td>
      <td class="acc" style="font-weight:700">${esc(r.ticker)}</td>
      <td class="${dc}">${esc(r.direction)}</td>
      <td>${fmt$(r.entry_price)}</td>
      <td>${fmt$(r.stop_loss)}</td>
      <td>${fmt$(r.take_profit)}</td>
      <td>${r.shares}</td>
      <td>${fmtScore(r.signal_score)}</td>
      <td>${esc(reason)}${(r.signal_reasoning||'').length > 90 ? '…' : ''}</td>
    </tr>`;
  }).join('');
  setRows('pending-body', html);
}

function renderSkips(rows) {
  if (!rows.length) { setRows('skips-body', ''); return; }
  const html = rows.map(r => `<tr>
    <td>${fmtTime(r.timestamp)}</td>
    <td class="acc" style="font-weight:700">${esc(r.ticker)}</td>
    <td>${fmtScore(r.signal_score)}</td>
    <td>${esc(r.macro_regime || '—')}</td>
    <td>${esc(r.skip_reason || '—')}</td>
  </tr>`).join('');
  setRows('skips-body', html);
}

function renderSignalFeed(rows) {
  const feed = el('signal-feed');
  if (!feed) return;
  if (!rows.length) {
    feed.innerHTML = '<div class="empty-row">No signals yet</div>';
    return;
  }
  feed.innerHTML = rows.map(r => {
    const isSkip    = r.direction === 'SKIP';
    const isPending = r.approval_status === 'PENDING_APPROVAL';
    const isLong    = r.direction === 'LONG';
    const isShort   = r.direction === 'SHORT';

    let bCls = 'skip', bTxt = 'SKIP';
    if (isPending)    { bCls = 'pending'; bTxt = 'PEND'; }
    else if (isLong)  { bCls = 'long';   bTxt = 'LONG'; }
    else if (isShort) { bCls = 'short';  bTxt = 'SHORT'; }

    const detail = isSkip
      ? esc((r.skip_reason || '').substring(0, 52))
      : `${fmt$(r.entry_price)} · sl ${fmt$(r.stop_loss)} · ${r.shares || '?'}sh`;

    const scoreStr = (!isSkip && r.signal_score) ? fmtScore(r.signal_score) : '';

    return `<div class="signal-item">
      <span class="sig-badge ${bCls}">${bTxt}</span>
      <div class="sig-info">
        <div class="sig-ticker">${esc(r.ticker)}</div>
        <div class="sig-detail">${detail}</div>
      </div>
      <div class="sig-right">
        <span class="sig-score">${scoreStr}</span>
        <span class="sig-time">${fmtTime(r.timestamp)}</span>
      </div>
    </div>`;
  }).join('');
}

// ── Skip Breakdown ────────────────────────────────────────────────────────────

function bucketReason(reason) {
  const r = (reason || '').toLowerCase();
  if (r.includes('rvol'))           return 'Low RVOL';
  if (r.includes('ema'))            return 'EMA not aligned';
  if (r.includes('adx'))            return 'ADX too low';
  if (r.includes('vwap'))           return 'VWAP position';
  if (r.includes('consensus'))      return 'No direction consensus';
  if (r.includes('smc'))            return 'SMC structure missing';
  if (r.includes('momentum'))       return 'No momentum confirm';
  if (r.includes('bars'))           return 'Insufficient bars';
  if (r.includes('score') || r.includes('threshold') || r.includes('confidence')) return 'Below score gate';
  if (r.includes('risk') || r.includes('heat'))   return 'Risk rejected';
  if (r.includes('veto'))           return 'News/macro veto';
  if (r.includes('cooldown'))       return 'Cooldown active';
  if (r.includes('exposure') || r.includes('duplicate')) return 'Duplicate exposure';
  if (r.includes('insufficient') || r.includes('data'))  return 'No data';
  return 'Other';
}

function renderSkipBreakdown(skips) {
  const container = el('skip-breakdown');
  if (!container) return;
  if (!skips.length) {
    container.innerHTML = '<div class="empty-row">Skip data appears during market hours</div>';
    return;
  }

  const counts = {};
  for (const s of skips) {
    const bucket = bucketReason(s.skip_reason);
    counts[bucket] = (counts[bucket] || 0) + 1;
  }

  const sorted  = Object.entries(counts).sort((a,b) => b[1] - a[1]);
  const maxCount = sorted[0]?.[1] || 1;

  container.innerHTML = sorted.map(([label, count]) => {
    const pct = Math.round((count / maxCount) * 100);
    return `<div class="skip-row">
      <span class="skip-label">${esc(label)}</span>
      <div class="skip-track"><div class="skip-bar" style="width:${pct}%"></div></div>
      <span class="skip-count">${count}</span>
    </div>`;
  }).join('');
}

// ── Equity Curve ──────────────────────────────────────────────────────────────

function renderEquityCurve(data) {
  const canvas = el('equity-chart');
  const empty  = el('chart-empty');
  if (!canvas) return;

  if (!data.length) {
    if (equityChart) { equityChart.data.labels = []; equityChart.data.datasets[0].data = []; equityChart.update('none'); }
    if (empty) empty.classList.remove('hidden');
    return;
  }
  if (empty) empty.classList.add('hidden');

  const labels  = data.map(d => d.date);
  const values  = data.map(d => d.cumulative_pnl);
  const lastVal = values[values.length - 1] || 0;
  const lineCol = lastVal >= 0 ? '#00e87a' : '#ff3a5c';
  const fillCol = lastVal >= 0 ? 'rgba(0,232,122,0.06)' : 'rgba(255,58,92,0.06)';

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data            = values;
    equityChart.data.datasets[0].borderColor     = lineCol;
    equityChart.data.datasets[0].backgroundColor = fillCol;
    equityChart.update('none');
    return;
  }

  equityChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: lineCol,
        backgroundColor: fillCol,
        borderWidth: 1.5,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHoverRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0b0d1a',
          borderColor: '#2a2e52',
          borderWidth: 1,
          titleColor: '#7e85b0',
          bodyColor: '#dde1f5',
          padding: 10,
          callbacks: { label: ctx => '  ' + fmt$(ctx.parsed.y) }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(28,31,53,0.6)' },
          ticks: { color: '#3e4268', font: { size: 9, family: "'JetBrains Mono', monospace" }, maxTicksLimit: 8 },
          border: { color: '#1c1f35' },
        },
        y: {
          grid: { color: 'rgba(28,31,53,0.6)' },
          ticks: {
            color: '#3e4268',
            font: { size: 9, family: "'JetBrains Mono', monospace" },
            callback: v => '$' + Number(v).toLocaleString(),
          },
          border: { color: '#1c1f35' },
        }
      }
    }
  });
}

// ── Positions ─────────────────────────────────────────────────────────────────

async function loadPositions() {
  try {
    const res = await fetch('/api/positions?days=7');
    const trades = await res.json();
    const tbody = document.getElementById('positions-all-body');
    if (!tbody) return;

    if (!trades.length) {
      tbody.innerHTML = '<tr><td colspan="13" class="empty-row">No positions yet</td></tr>';
      return;
    }

    tbody.innerHTML = trades.map(t => {
      const time = t.timestamp ? t.timestamp.substring(5, 16).replace('T', ' ') : '';
      const status = t.closed ? (t.outcome || 'CLOSED') : 'OPEN';
      const statusClass = t.closed ? (t.outcome === 'WIN' ? 'clr-green' : t.outcome === 'LOSS' ? 'clr-red' : '') : 'clr-blue';
      const pnl = t.closed ? (t.pnl_dollars > 0 ? '+' : '') + t.pnl_dollars.toFixed(2) : '—';
      const pnlClass = t.pnl_dollars > 0 ? 'clr-green' : t.pnl_dollars < 0 ? 'clr-red' : '';
      const rVal = t.closed ? (t.pnl_r > 0 ? '+' : '') + t.pnl_r.toFixed(2) + 'R' : '—';
      const tierBadge = t.confidence_tier === 'SWING' ? 'tier-swing' : t.confidence_tier === 'STANDARD' ? 'tier-std' : 'tier-scalp';

      return '<tr>' +
        '<td>' + time + '</td>' +
        '<td><strong>' + t.ticker + '</strong></td>' +
        '<td>' + t.direction + '</td>' +
        '<td><span class="badge-sm ' + tierBadge + '">' + t.confidence_tier + '</span></td>' +
        '<td>$' + (t.entry_price || 0).toFixed(2) + '</td>' +
        '<td>$' + (t.stop_loss || 0).toFixed(2) + '</td>' +
        '<td>$' + (t.take_profit || 0).toFixed(2) + '</td>' +
        '<td>' + t.shares + '</td>' +
        '<td>' + t.signal_score.toFixed(3) + '</td>' +
        '<td class="' + statusClass + '">' + status + '</td>' +
        '<td class="' + pnlClass + '">' + pnl + '</td>' +
        '<td class="' + pnlClass + '">' + rVal + '</td>' +
        '<td>' + (t.exit_reason || '—') + '</td>' +
        '</tr>';
    }).join('');
  } catch (e) {
    console.error('positions fetch error', e);
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const name = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const content = el(`tab-${name}`);
    if (content) content.classList.add('active');
  });
});

// ── Poll Loop ─────────────────────────────────────────────────────────────────

async function refresh() {
  const pulse = el('scan-pulse');
  if (pulse) pulse.classList.add('active');

  try {
    await Promise.all([loadPortfolio(), loadStats(), loadTrades(), loadPositions()]);

    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
      timeZone: 'America/New_York',
    });
    setText('last-update', `synced ${timeStr}`);

    const dot = el('status-dot');
    if (dot) {
      dot.classList.add('live');
      setTimeout(() => dot.classList.remove('live'), 1200);
    }
  } catch (err) {
    setText('last-update', 'Error — ' + err.message);
  } finally {
    if (pulse) {
      setTimeout(() => pulse.classList.remove('active'), 800);
    }
  }
}

refresh();
setInterval(refresh, POLL_MS);
