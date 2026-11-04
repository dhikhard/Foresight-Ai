/**
 * ForeSightAI - Quantitative Forecasting Dashboard
 * Client Application Logic
 */

// Application State
const state = {
  currentTab: 'market-overview',
  featureSet: 'v2',
  marketData: null,
  forecastData: null,
  performanceData: null,
  riskData: null,
  portfolioValue: 100000,
  currentShockPct: -5.0,
  charts: {},
  chartRange: 'ALL',
};

// Utility formatters
const fmt = {
  currency: (val) => '$' + Number(val).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  pct: (val, digits = 2) => (val >= 0 ? '+' : '') + (Number(val) * 100).toFixed(digits) + '%',
  rawPct: (val, digits = 2) => (val >= 0 ? '+' : '') + Number(val).toFixed(digits) + '%',
  num: (val, digits = 2) => Number(val).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }),
  compactNum: (val) => {
    if (!val) return 'N/A';
    if (val >= 1e9) return (val / 1e9).toFixed(2) + 'B';
    if (val >= 1e6) return (val / 1e6).toFixed(2) + 'M';
    return Number(val).toLocaleString();
  }
};

// Document Ready
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  setupNavigation();
  setupEventListeners();

  // Check URL hash or query param for initial tab
  const params = new URLSearchParams(window.location.search);
  const queryTab = params.get('tab');
  const hashTab = window.location.hash ? window.location.hash.substring(1) : null;
  const initialTab = queryTab || hashTab || 'market-overview';

  loadInitialData().then(() => {
    if (initialTab && initialTab !== 'market-overview') {
      switchTab(initialTab);
    }
  });

  window.addEventListener('hashchange', () => {
    const newTab = window.location.hash.substring(1);
    if (newTab && newTab !== state.currentTab) {
      switchTab(newTab);
    }
  });
});

// Navigation Setup
function setupNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');
      window.location.hash = targetTab;
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  state.currentTab = tabId;

  // Update nav buttons
  document.querySelectorAll('.nav-tab').forEach(tab => {
    if (tab.getAttribute('data-tab') === tabId) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Update tab content panels
  document.querySelectorAll('.tab-content').forEach(panel => {
    panel.classList.remove('active');
  });

  const activePanel = document.getElementById(`tab-${tabId}`);
  if (activePanel) {
    activePanel.classList.add('active');
  }

  // Trigger chart resize on tab reveal
  setTimeout(() => {
    Object.values(state.charts).forEach(c => c && c.resize && c.resize());
  }, 50);

  // Lazy load tab data if not already fetched
  if (tabId === 'forecasting-engine' && !state.forecastData) {
    fetchForecastData();
  } else if (tabId === 'model-performance' && !state.performanceData) {
    fetchPerformanceData();
  } else if (tabId === 'risk-analysis' && !state.riskData) {
    fetchRiskData();
  }
}

// Event Listeners Setup
function setupEventListeners() {
  // Refresh button
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      state.marketData = null;
      state.forecastData = null;
      state.performanceData = null;
      state.riskData = null;
      loadInitialData();
    });
  }

  // Feature set selector
  const featSelector = document.getElementById('feature-set-selector');
  if (featSelector) {
    featSelector.addEventListener('change', (e) => {
      state.featureSet = e.target.value;
      state.forecastData = null;
      state.performanceData = null;
      if (state.currentTab === 'forecasting-engine') {
        fetchForecastData();
      } else if (state.currentTab === 'model-performance') {
        fetchPerformanceData();
      }
    });
  }

  // Feature category filter
  const catFilter = document.getElementById('feature-category-filter');
  if (catFilter) {
    catFilter.addEventListener('change', (e) => {
      renderFeaturesTable(e.target.value);
    });
  }

  // Chart range buttons
  document.querySelectorAll('.chart-range-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.chart-range-btn').forEach(b => {
        b.classList.remove('active', 'bg-cyan-500/20', 'text-cyan-300', 'font-medium');
      });
      btn.classList.add('active', 'bg-cyan-500/20', 'text-cyan-300', 'font-medium');
      state.chartRange = btn.getAttribute('data-range');
      updateMarketOverviewChart();
    });
  });

  // Chart SMA toggle checkboxes
  ['sma20', 'sma50', 'sma200'].forEach(id => {
    const el = document.getElementById(`toggle-${id}`);
    if (el) {
      el.addEventListener('change', () => updateMarketOverviewChart());
    }
  });

  // Portfolio input
  const portfolioInput = document.getElementById('portfolio-input');
  if (portfolioInput) {
    portfolioInput.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value) || 1000;
      state.portfolioValue = val;
      updateRiskCalculations();
    });
  }

  // Capital quick select chips
  document.querySelectorAll('.cap-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.cap-chip').forEach(c => c.classList.remove('active', 'bg-cyan-500/20', 'text-cyan-300', 'font-medium'));
      chip.classList.add('active', 'bg-cyan-500/20', 'text-cyan-300', 'font-medium');
      const cap = parseFloat(chip.getAttribute('data-cap'));
      state.portfolioValue = cap;
      portfolioInput.value = cap;
      updateRiskCalculations();
    });
  });

  // Market shock buttons
  document.querySelectorAll('.shock-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.shock-btn').forEach(b => {
        b.classList.remove('border-2', 'border-rose-600', 'border-emerald-500', 'shadow-lg');
      });
      const shock = parseFloat(btn.getAttribute('data-shock'));
      btn.classList.add('border-2', shock < 0 ? 'border-rose-600' : 'border-emerald-500', 'shadow-lg');
      state.currentShockPct = shock;
      
      const slider = document.getElementById('shock-slider');
      if (slider) slider.value = shock;
      const sliderVal = document.getElementById('slider-shock-value');
      if (sliderVal) sliderVal.textContent = fmt.rawPct(shock);

      updateRiskCalculations();
    });
  });

  // Shock slider
  const shockSlider = document.getElementById('shock-slider');
  const sliderVal = document.getElementById('slider-shock-value');
  if (shockSlider) {
    shockSlider.addEventListener('input', (e) => {
      const shock = parseFloat(e.target.value);
      state.currentShockPct = shock;
      if (sliderVal) sliderVal.textContent = fmt.rawPct(shock);

      // Deselect fixed shock buttons if slider differs
      document.querySelectorAll('.shock-btn').forEach(b => {
        b.classList.remove('border-2', 'border-rose-600', 'border-emerald-500', 'shadow-lg');
      });

      updateRiskCalculations();
    });
  }
}

// Data Fetchers
async function loadInitialData() {
  await fetchMarketOverview();
  // Preload forecasting and performance in background
  fetchForecastData();
  fetchPerformanceData();
  fetchRiskData();
}

async function fetchMarketOverview() {
  try {
    const res = await fetch('/api/market/overview');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.marketData = data;
    renderMarketOverview(data);
  } catch (err) {
    console.error('Failed to load market overview:', err);
  }
}

async function fetchForecastData() {
  try {
    const res = await fetch(`/api/models/forecast?feature_set=${state.featureSet}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.forecastData = data;
    renderForecastingEngine(data);
  } catch (err) {
    console.error('Failed to load forecast data:', err);
  }
}

async function fetchPerformanceData() {
  try {
    const res = await fetch(`/api/models/performance?feature_set=${state.featureSet}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.performanceData = data;
    renderPerformance(data);
  } catch (err) {
    console.error('Failed to load performance data:', err);
  }
}

async function fetchRiskData() {
  try {
    const res = await fetch(`/api/risk/analysis?portfolio_value=${state.portfolioValue}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.riskData = data;
    renderRiskAnalysis(data);
  } catch (err) {
    console.error('Failed to load risk data:', err);
  }
}

// ---------------------------------------------------------------------------
// Render Tab 1: Market Overview
// ---------------------------------------------------------------------------
function renderMarketOverview(data) {
  const obs = data.observation;
  const ind = data.indicators;
  const fresh = data.freshness;

  // Update header indicators
  const hDate = document.getElementById('header-data-date');
  const hPrice = document.getElementById('header-sp500-price');
  const hDelta = document.getElementById('header-sp500-delta');
  if (hDate) hDate.textContent = `S&P 500: ${obs.date}`;
  if (hPrice) hPrice.textContent = fmt.currency(obs.close);
  if (hDelta) {
    hDelta.textContent = `(${fmt.pct(obs.daily_return)})`;
    hDelta.className = obs.daily_return >= 0 ? 'text-emerald-400 font-mono text-[11px]' : 'text-rose-400 font-mono text-[11px]';
  }

  // Update KPI cards
  const kpiDate = document.getElementById('kpi-obs-date');
  const kpiClose = document.getElementById('kpi-close-price');
  const kpiReturn = document.getElementById('kpi-return-badge');
  const kpiRange = document.getElementById('kpi-day-range');

  if (kpiDate) kpiDate.textContent = obs.date;
  if (kpiClose) kpiClose.textContent = fmt.currency(obs.close);
  if (kpiReturn) {
    kpiReturn.textContent = fmt.pct(obs.daily_return);
    kpiReturn.className = obs.daily_return >= 0 
      ? 'text-xs font-mono font-medium px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800' 
      : 'text-xs font-mono font-medium px-1.5 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800';
  }
  if (kpiRange) kpiRange.textContent = `${fmt.currency(obs.low)} - ${fmt.currency(obs.high)}`;

  // Volatility KPI
  const kpiVol20 = document.getElementById('kpi-vol-20');
  const kpiVol60 = document.getElementById('kpi-vol-60');
  const kpiVolRatio = document.getElementById('kpi-vol-ratio');
  if (kpiVol20) kpiVol20.textContent = (ind.vol_20_annualized * 100).toFixed(1) + '%';
  if (kpiVol60) kpiVol60.textContent = (ind.vol_60_annualized * 100).toFixed(1) + '%';
  if (kpiVolRatio) {
    const ratio = ind.vol_ratio_20_60;
    const regime = ratio > 1.25 ? 'High Stress' : ratio < 0.85 ? 'Compressed' : 'Normal';
    kpiVolRatio.textContent = `${ratio.toFixed(2)} (${regime})`;
  }

  // Drawdown KPI
  const kpiAth = document.getElementById('kpi-ath-level');
  const kpiDd = document.getElementById('kpi-drawdown');
  const kpiMaxDd = document.getElementById('kpi-max-drawdown');
  if (kpiAth) kpiAth.textContent = `Peak: ${fmt.currency(ind.all_time_high)}`;
  if (kpiDd) kpiDd.textContent = (ind.current_drawdown * 100).toFixed(2) + '%';
  if (kpiMaxDd) kpiMaxDd.textContent = `${(ind.max_drawdown * 100).toFixed(1)}% (2008 GFC)`;

  // 52-Week Range KPI
  const kpi52High = document.getElementById('kpi-52w-high');
  const kpi52Low = document.getElementById('kpi-52w-low');
  const kpiVol = document.getElementById('kpi-volume');
  const kpiBar = document.getElementById('kpi-52w-bar');
  if (kpi52High) kpi52High.textContent = fmt.currency(ind.high_52w);
  if (kpi52Low) kpi52Low.textContent = fmt.currency(ind.low_52w);
  if (kpiVol) kpiVol.textContent = `Vol: ${fmt.compactNum(obs.volume)}`;
  if (kpiBar) {
    const pct = ((obs.close - ind.low_52w) / (ind.high_52w - ind.low_52w)) * 100;
    kpiBar.style.width = `${Math.max(5, Math.min(100, pct))}%`;
  }

  // Freshness banner text
  const freshText = document.getElementById('overview-freshness-text');
  if (freshText) freshText.textContent = `${fresh.total_records.toLocaleString()} days through ${fresh.latest_date}`;

  // Render Table & Chart
  renderFeaturesTable('all');
  initMarketOverviewChart(data.chart_series);
}

function renderFeaturesTable(categoryFilter = 'all') {
  if (!state.marketData) return;
  const tbody = document.getElementById('features-table-body');
  if (!tbody) return;

  const features = state.marketData.feature_breakdown;
  const filtered = categoryFilter === 'all' 
    ? features 
    : features.filter(f => f.category === categoryFilter);

  tbody.innerHTML = filtered.map(f => {
    let formattedVal = 'N/A';
    if (f.value !== null && f.value !== undefined) {
      if (f.key.includes('ret') || f.key.includes('mom') || f.key.includes('ma_dist') || f.key.includes('change')) {
        formattedVal = (f.value >= 0 ? '+' : '') + (f.value * 100).toFixed(2) + '%';
      } else if (f.key.includes('vol_60')) {
        formattedVal = (f.value * 100).toFixed(2) + '%';
      } else {
        formattedVal = f.value.toFixed(2);
      }
    }

    let statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300">Normal</span>';
    if (f.key === 'rsi_14') {
      if (f.value > 70) statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-rose-950 text-rose-400 border border-rose-800">Overbought (>70)</span>';
      else if (f.value < 30) statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800">Oversold (<30)</span>';
      else statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-blue-950 text-blue-300 border border-blue-800">Neutral (30-70)</span>';
    } else if (f.key === 'bollinger_pctb_20') {
      if (f.value > 1.0) statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-amber-950 text-amber-400 border border-amber-800">Above Upper Band</span>';
      else if (f.value < 0.0) statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-amber-950 text-amber-400 border border-amber-800">Below Lower Band</span>';
      else statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] bg-navy-800 text-slate-300">Within Bands</span>';
    }

    return `
      <tr class="hover:bg-navy-900/40">
        <td class="py-2.5 px-3 font-semibold text-slate-200">
          <div>${f.label}</div>
          <div class="text-[10px] font-mono text-cyan-400">${f.key}</div>
        </td>
        <td class="py-2.5 px-3 text-slate-400 text-xs">${f.category}</td>
        <td class="py-2.5 px-3 font-mono font-bold text-right ${f.value >= 0 ? 'text-slate-100' : 'text-rose-400'}">
          ${formattedVal}
        </td>
        <td class="py-2.5 px-3 text-slate-400 text-xs">${f.description}</td>
        <td class="py-2.5 px-3 text-center">${statusBadge}</td>
      </tr>
    `;
  }).join('');
}

function initMarketOverviewChart(rawSeries) {
  const canvas = document.getElementById('marketOverviewChart');
  if (!canvas) return;

  state.fullChartSeries = rawSeries;
  updateMarketOverviewChart();
}

function updateMarketOverviewChart() {
  if (!state.fullChartSeries) return;
  const canvas = document.getElementById('marketOverviewChart');
  if (!canvas) return;

  // Filter series based on range (1Y, 3Y, 5Y, All)
  let series = [...state.fullChartSeries];
  const total = series.length;
  if (state.chartRange === '1Y') {
    series = series.slice(-Math.min(total, 252));
  } else if (state.chartRange === '3Y') {
    series = series.slice(-Math.min(total, 756));
  } else if (state.chartRange === '5Y') {
    series = series.slice(-Math.min(total, 1260));
  }

  const showSma20 = document.getElementById('toggle-sma20')?.checked;
  const showSma50 = document.getElementById('toggle-sma50')?.checked;
  const showSma200 = document.getElementById('toggle-sma200')?.checked;

  const labels = series.map(r => r.date);
  const closes = series.map(r => r.close);
  const sma20s = series.map(r => r.ma_20);
  const sma50s = series.map(r => r.ma_50);
  const sma200s = series.map(r => r.ma_200);

  const datasets = [
    {
      label: 'S&P 500 Close',
      data: closes,
      borderColor: '#38bdf8',
      backgroundColor: 'rgba(56, 189, 248, 0.08)',
      fill: true,
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.1,
    }
  ];

  if (showSma20) {
    datasets.push({
      label: 'SMA 20',
      data: sma20s,
      borderColor: '#06b6d4',
      borderWidth: 1.2,
      pointRadius: 0,
      borderDash: [2, 2],
      tension: 0.1,
    });
  }

  if (showSma50) {
    datasets.push({
      label: 'SMA 50',
      data: sma50s,
      borderColor: '#f59e0b',
      borderWidth: 1.2,
      pointRadius: 0,
      tension: 0.1,
    });
  }

  if (showSma200) {
    datasets.push({
      label: 'SMA 200',
      data: sma200s,
      borderColor: '#a855f7',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1,
    });
  }

  if (state.charts.marketOverview) {
    state.charts.marketOverview.destroy();
  }

  state.charts.marketOverview = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 }, boxWidth: 12 }
        },
        tooltip: {
          backgroundColor: '#0d1527',
          borderColor: '#1e2e4f',
          borderWidth: 1,
          titleFont: { family: 'JetBrains Mono' },
          bodyFont: { family: 'JetBrains Mono' },
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: $${Number(ctx.raw).toLocaleString('en-US', { minimumFractionDigits: 2 })}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 46, 79, 0.4)' },
          ticks: { color: '#64748b', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(30, 46, 79, 0.4)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (val) => '$' + val.toLocaleString()
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Render Tab 2: Forecasting Engine
// ---------------------------------------------------------------------------
function renderForecastingEngine(data) {
  const models = data.models;
  const ridge = models.find(m => m.id === 'ridge');
  const mean = models.find(m => m.id === 'train_mean');
  const enet = models.find(m => m.id === 'elastic_net');
  const zero = models.find(m => m.id === 'zero_return');

  // Fill KPI cards
  const rRet = document.getElementById('forecast-ridge-return');
  const rPrice = document.getElementById('forecast-ridge-price');
  if (rRet) rRet.textContent = fmt.pct(ridge.predicted_return, 3);
  if (rPrice) rPrice.textContent = fmt.currency(ridge.implied_index_price);

  const mRet = document.getElementById('forecast-mean-return');
  const mPrice = document.getElementById('forecast-mean-price');
  if (mRet) mRet.textContent = fmt.pct(mean.predicted_return, 3);
  if (mPrice) mPrice.textContent = fmt.currency(mean.implied_index_price);

  const eRet = document.getElementById('forecast-enet-return');
  const ePrice = document.getElementById('forecast-enet-price');
  if (eRet) eRet.textContent = fmt.pct(enet.predicted_return, 3);
  if (ePrice) ePrice.textContent = fmt.currency(enet.implied_index_price);

  const zPrice = document.getElementById('forecast-zero-price');
  if (zPrice) zPrice.textContent = fmt.currency(zero.implied_index_price);

  // Sub-metrics
  const basePrice = document.getElementById('forecast-base-price');
  const ridgeDiff = document.getElementById('forecast-ridge-diff');
  if (basePrice) basePrice.textContent = fmt.currency(data.current_index_close);
  if (ridgeDiff) {
    const diffPct = (ridge.predicted_return - mean.predicted_return) * 100;
    const diffDollars = ridge.implied_index_price - mean.implied_index_price;
    ridgeDiff.textContent = `${diffPct >= 0 ? '+' : ''}${diffPct.toFixed(3)}% (${diffDollars >= 0 ? '+' : ''}$${diffDollars.toFixed(2)})`;
  }

  // Render Forecast Comparison Bar Chart
  renderForecastComparisonChart(models, data.current_index_close);

  // Render Feature Attributions Chart
  renderRidgeAttributionChart(data.ridge_feature_attributions);
}

function renderForecastComparisonChart(models, basePrice) {
  const canvas = document.getElementById('forecastComparisonChart');
  if (!canvas) return;

  const labels = models.map(m => m.name.split(' (')[0]);
  const returnPcts = models.map(m => m.predicted_return * 100);
  const colors = models.map(m => {
    if (m.id === 'ridge') return '#06b6d4';
    if (m.id === 'train_mean') return '#38bdf8';
    if (m.id === 'elastic_net') return '#818cf8';
    return '#64748b';
  });

  if (state.charts.forecastComp) {
    state.charts.forecastComp.destroy();
  }

  state.charts.forecastComp = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Predicted Next-Day Return (%)',
        data: returnPcts,
        backgroundColor: colors,
        borderRadius: 4,
        borderWidth: 1,
        borderColor: colors.map(c => c),
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1527',
          borderColor: '#1e2e4f',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => {
              const m = models[ctx.dataIndex];
              return [
                `Return: ${fmt.rawPct(ctx.raw, 4)}`,
                `Implied Index: ${fmt.currency(m.implied_index_price)}`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(30, 46, 79, 0.4)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (val) => val.toFixed(3) + '%'
          }
        }
      }
    }
  });
}

function renderRidgeAttributionChart(attributions) {
  const canvas = document.getElementById('ridgeAttributionChart');
  if (!canvas || !attributions) return;

  // Take top 8 contributors by absolute magnitude
  const topAttribs = attributions.slice(0, 8);
  const labels = topAttribs.map(a => a.feature);
  const values = topAttribs.map(a => a.contribution * 100); // percentage points
  const colors = values.map(v => v >= 0 ? '#10b981' : '#f43f5e');

  if (state.charts.ridgeAttribution) {
    state.charts.ridgeAttribution.destroy();
  }

  state.charts.ridgeAttribution = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Contribution to Return (w_j * x_j %)',
        data: values,
        backgroundColor: colors,
        borderRadius: 3,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1527',
          borderColor: '#1e2e4f',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => `Contribution: ${fmt.rawPct(ctx.raw, 4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 46, 79, 0.4)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (val) => val.toFixed(3) + '%'
          }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11 } }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Render Tab 3: Model Performance
// ---------------------------------------------------------------------------
function renderPerformance(data) {
  const tbody = document.getElementById('performance-table-body');
  if (tbody) {
    tbody.innerHTML = data.models_comparison.map(m => {
      const isWinner = m.model_key === 'train_mean';
      const outcomeBadge = isWinner 
        ? '<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold">Baseline Winner</span>'
        : m.model_key === 'elastic_net'
        ? '<span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300">Degenerate to Mean</span>'
        : '<span class="px-2 py-0.5 rounded text-[10px] bg-navy-800 text-slate-400">Below Baseline</span>';

      const hpText = m.model_key === 'ridge' ? 'α = 0.01 (Val Selected)'
        : m.model_key === 'elastic_net' ? 'α = 0.01, ρ = 0.1'
        : 'Fixed Rule';

      return `
        <tr class="hover:bg-navy-900/40 ${isWinner ? 'bg-cyan-950/20' : ''}">
          <td class="py-2.5 px-3 font-semibold text-slate-200">
            <div>${m.name}</div>
            <div class="text-[10px] text-slate-400 font-sans">${m.model_key}</div>
          </td>
          <td class="py-2.5 px-3 text-slate-400 font-mono text-[11px]">${hpText}</td>
          <td class="py-2.5 px-3 text-right text-slate-300">${m.validation.mae ? m.validation.mae.toFixed(6) : 'N/A'}</td>
          <td class="py-2.5 px-3 text-right font-bold bg-navy-800/40 text-cyan-300">${m.test.mae ? m.test.mae.toFixed(6) : 'N/A'}</td>
          <td class="py-2.5 px-3 text-right text-slate-300">${m.test.rmse ? m.test.rmse.toFixed(6) : 'N/A'}</td>
          <td class="py-2.5 px-3 text-right ${m.test.r2 < 0 ? 'text-rose-400' : 'text-emerald-400'}">
            ${m.test.r2 !== null && m.test.r2 !== undefined ? m.test.r2.toFixed(4) : 'N/A'}
          </td>
          <td class="py-2.5 px-3 text-right text-slate-300">
            ${m.test.directional_accuracy ? (m.test.directional_accuracy * 100).toFixed(1) + '%' : 'N/A'}
          </td>
          <td class="py-2.5 px-3 text-center">${outcomeBadge}</td>
        </tr>
      `;
    }).join('');
  }

  // Render Actual vs. Predicted Chart
  if (data.test_evaluation && data.test_evaluation.available) {
    renderActualVsPredictedChart(data.test_evaluation.series);
    renderResidualsChart(data.test_evaluation.series);
    renderResidualHistogram(data.test_evaluation.residual_histogram);
  }
}

function renderActualVsPredictedChart(testSeries) {
  const canvas = document.getElementById('actualVsPredictedChart');
  if (!canvas || !testSeries) return;

  const labels = testSeries.map(s => s.date);
  const actuals = testSeries.map(s => s.actual_return * 100);
  const ridgePreds = testSeries.map(s => s.predicted_ridge * 100);
  const trainMeans = testSeries.map(s => s.predicted_train_mean * 100);

  if (state.charts.actVsPred) {
    state.charts.actVsPred.destroy();
  }

  state.charts.actVsPred = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Actual S&P 500 Daily Return',
          data: actuals,
          borderColor: 'rgba(148, 163, 184, 0.4)',
          backgroundColor: 'transparent',
          borderWidth: 1,
          pointRadius: 1.5,
          pointHoverRadius: 4,
          pointBackgroundColor: actuals.map(v => v >= 0 ? '#10b981' : '#f43f5e'),
        },
        {
          label: 'Ridge Predicted Return',
          data: ridgePreds,
          borderColor: '#06b6d4',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: 'Train-Mean Constant (+0.020%)',
          data: trainMeans,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
        },
        tooltip: {
          backgroundColor: '#0d1527',
          borderColor: '#1e2e4f',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${fmt.rawPct(ctx.raw, 3)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 46, 79, 0.3)' },
          ticks: { color: '#64748b', maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(30, 46, 79, 0.3)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => v.toFixed(1) + '%'
          }
        }
      }
    }
  });
}

function renderResidualsChart(testSeries) {
  const canvas = document.getElementById('residualsChart');
  if (!canvas || !testSeries) return;

  const labels = testSeries.map(s => s.date);
  const residuals = testSeries.map(s => s.residual_ridge * 100);

  if (state.charts.residuals) {
    state.charts.residuals.destroy();
  }

  state.charts.residuals = new Chart(canvas, {
    type: 'scatter',
    data: {
      labels,
      datasets: [{
        label: 'Ridge Residual Error (y - y_hat)',
        data: testSeries.map((s, idx) => ({ x: idx, y: s.residual_ridge * 100 })),
        backgroundColor: 'rgba(56, 189, 248, 0.5)',
        pointRadius: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1527',
          callbacks: {
            label: (ctx) => `Error: ${fmt.rawPct(ctx.raw.y, 3)}`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { display: false }
        },
        y: {
          grid: { color: 'rgba(30, 46, 79, 0.4)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => v.toFixed(1) + '%'
          }
        }
      }
    }
  });
}

function renderResidualHistogram(histData) {
  const canvas = document.getElementById('residualHistogramChart');
  if (!canvas || !histData) return;

  const labels = histData.map(h => `${(h.bin_start * 100).toFixed(1)}%`);
  const counts = histData.map(h => h.count);

  if (state.charts.residualHist) {
    state.charts.residualHist.destroy();
  }

  state.charts.residualHist = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Frequency',
        data: counts,
        backgroundColor: 'rgba(6, 182, 212, 0.65)',
        borderRadius: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#64748b', maxTicksLimit: 8, font: { family: 'JetBrains Mono', size: 9 } }
        },
        y: {
          grid: { color: 'rgba(30, 46, 79, 0.4)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Render Tab 4: Risk Analysis
// ---------------------------------------------------------------------------
function renderRiskAnalysis(data) {
  const rm = data.risk_metrics;

  // Fill VaR / CVaR cards
  const var95Pct = document.getElementById('var-95-pct');
  const var95Dol = document.getElementById('var-95-dollar');
  const cvar95Pct = document.getElementById('cvar-95-pct');
  const cvar95Dol = document.getElementById('cvar-95-dollar');

  const var99Pct = document.getElementById('var-99-pct');
  const var99Dol = document.getElementById('var-99-dollar');
  const cvar99Pct = document.getElementById('cvar-99-pct');
  const cvar99Dol = document.getElementById('cvar-99-dollar');

  if (var95Pct) var95Pct.textContent = `-${rm.var_95_pct.toFixed(2)}%`;
  if (var95Dol) var95Dol.textContent = `-${fmt.currency(rm.var_95_dollar)} on ${fmt.currency(state.portfolioValue)}`;
  if (cvar95Pct) cvar95Pct.textContent = `-${rm.cvar_95_pct.toFixed(2)}%`;
  if (cvar95Dol) cvar95Dol.textContent = `-${fmt.currency(rm.cvar_95_dollar)} on ${fmt.currency(state.portfolioValue)}`;

  if (var99Pct) var99Pct.textContent = `-${rm.var_99_pct.toFixed(2)}%`;
  if (var99Dol) var99Dol.textContent = `-${fmt.currency(rm.var_99_dollar)} on ${fmt.currency(state.portfolioValue)}`;
  if (cvar99Pct) cvar99Pct.textContent = `-${rm.cvar_99_pct.toFixed(2)}%`;
  if (cvar99Dol) cvar99Dol.textContent = `-${fmt.currency(rm.cvar_99_dollar)} on ${fmt.currency(state.portfolioValue)}`;

  // Historical Crisis Benchmark Table
  renderHistoricalStressTable(data.historical_stress_events);

  // Render Underwater Drawdown Chart
  renderUnderwaterDrawdownChart(data.drawdown_series);

  // Initial Calculation of Reactive Shock Outcomes
  updateRiskCalculations();
}

function updateRiskCalculations() {
  const cap = state.portfolioValue;
  const shockPct = state.currentShockPct;
  const shockDec = shockPct / 100.0;
  const pnl = cap * shockDec;
  const newCap = cap + pnl;

  const baseIndex = state.marketData ? state.marketData.observation.close : 7641.16;
  const impliedIndex = baseIndex * (1.0 + shockDec);

  const simPnl = document.getElementById('sim-pnl');
  const simPnlPct = document.getElementById('sim-pnl-pct');
  const simEquity = document.getElementById('sim-portfolio-equity');
  const simIndex = document.getElementById('sim-index-level');

  if (simPnl) {
    simPnl.textContent = (pnl >= 0 ? '+' : '') + fmt.currency(pnl);
    simPnl.className = pnl >= 0 ? 'text-2xl font-bold font-mono-num mt-2 text-emerald-400' : 'text-2xl font-bold font-mono-num mt-2 text-rose-400';
  }
  if (simPnlPct) {
    simPnlPct.textContent = `Under ${fmt.rawPct(shockPct)} Market Shock on ${fmt.currency(cap)}`;
  }
  if (simEquity) {
    simEquity.textContent = fmt.currency(newCap);
  }
  if (simIndex) {
    simIndex.textContent = impliedIndex.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  // Also update dollar losses in the crisis table
  if (state.riskData) {
    renderHistoricalStressTable(state.riskData.historical_stress_events);
  }
}

function renderHistoricalStressTable(events) {
  const tbody = document.getElementById('stress-table-body');
  if (!tbody || !events) return;

  const cap = state.portfolioValue;
  tbody.innerHTML = events.map(ev => {
    const dropDec = ev.peak_to_trough_drop_pct / 100.0;
    const dollarLoss = cap * dropDec;
    const finalVal = cap + dollarLoss;

    return `
      <tr class="hover:bg-navy-900/40">
        <td class="py-2.5 px-3 font-semibold text-slate-200">${ev.event}</td>
        <td class="py-2.5 px-3 text-slate-400 font-mono text-[11px]">${ev.period}</td>
        <td class="py-2.5 px-3 text-right font-mono text-rose-400 font-bold">${ev.peak_to_trough_drop_pct.toFixed(1)}%</td>
        <td class="py-2.5 px-3 text-right font-mono text-rose-400">-${fmt.currency(Math.abs(dollarLoss))}</td>
        <td class="py-2.5 px-3 text-right font-mono font-medium text-slate-200">${fmt.currency(finalVal)}</td>
        <td class="py-2.5 px-3 text-slate-400 text-xs">${ev.notes}</td>
      </tr>
    `;
  }).join('');
}

function renderUnderwaterDrawdownChart(drawdownSeries) {
  const canvas = document.getElementById('historicalDrawdownChart');
  if (!canvas || !drawdownSeries) return;

  const labels = drawdownSeries.map(d => d.date);
  const dataVals = drawdownSeries.map(d => d.drawdown_pct);

  if (state.charts.underwater) {
    state.charts.underwater.destroy();
  }

  state.charts.underwater = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Underwater Drawdown (%)',
        data: dataVals,
        borderColor: '#f43f5e',
        backgroundColor: 'rgba(244, 63, 94, 0.15)',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1527',
          borderColor: '#1e2e4f',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => `Drawdown: ${ctx.raw.toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 46, 79, 0.3)' },
          ticks: { color: '#64748b', maxTicksLimit: 12, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(30, 46, 79, 0.3)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => v.toFixed(0) + '%'
          }
        }
      }
    }
  });
}
