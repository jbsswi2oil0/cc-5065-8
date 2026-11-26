"use strict";

const $ = (id) => document.getElementById(id);
let STATE = null;
let playTimer = null;

const params = () => ({
  scene: $("scene").value,
  samples: +$("samples").value,
  bounces: +$("bounces").value,
  width: +$("width").value,
  height: +$("height").value,
  dt: +$("dt").value,
  t_min: 0,
  t_max: +$("tmax").value,
  kernel: $("kernel").value,
  kernel_width: +$("kwidth").value,
  historical: $("historical").checked,
  coarse_dt: +$("coarseDt").value,
  seed: 12345,
});

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// viridis-ish heat colour from normalized intensity 0..1
function heatColor(x) {
  x = Math.max(0, Math.min(1, x));
  const r = Math.round(40 + 215 * Math.pow(x, 0.8));
  const g = Math.round(30 + 200 * x);
  const b = Math.round(80 + 160 * (1 - x) * (1 - x));
  return `rgb(${r},${g},${b})`;
}

function drawFrame(idx) {
  if (!STATE) return;
  const [w, h] = STATE.resolution;
  const cv = $("sliceCanvas");
  cv.width = w * 6;
  cv.height = h * 6;
  const ctx = cv.getContext("2d");
  const frame = STATE.frames[idx] || [];
  let peak = 0;
  for (const v of STATE.frames) for (const e of v) if (e > peak) peak = e;
  peak = peak || 1;
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const e = frame[py * w + px] || 0;
      ctx.fillStyle = heatColor(e / peak);
      ctx.fillRect(px * 6, py * 6, 6, 6);
    }
  }
  const g = STATE.time_grid;
  $("sliceIdx").textContent = idx;
  $("sliceTime").textContent = (g.t_min + (idx + 0.5) * g.dt).toFixed(2);
  drawSteady();
}

function drawSteady() {
  if (!STATE) return;
  const h = STATE.steady.length;
  const w = STATE.steady[0].length;
  const cv = $("steadyCanvas");
  cv.width = w * 4;
  cv.height = h * 4;
  const ctx = cv.getContext("2d");
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const c = STATE.steady[py][px];
      const v = (c[0] + c[1] + c[2]) / 3;
      ctx.fillStyle = heatColor(v);
      ctx.fillRect(px * 4, py * 4, 4, 4);
    }
  }
}

function drawLineChart(canvas, series, labels, colors, yMax, xLabel) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0a0f16";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#243040";
  ctx.strokeRect(0, 0, canvas.width, canvas.height);
  const pad = 38;
  const plotW = canvas.width - pad - 10;
  const plotH = canvas.height - 30;
  series.forEach((data, si) => {
    ctx.strokeStyle = colors[si];
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad + (i / Math.max(1, data.length - 1)) * plotW;
      const y = plotH - (v / yMax) * (plotH - 10) + 8;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#8fa3b8";
  ctx.font = "11px sans-serif";
  ctx.fillText(labels[0], pad, canvas.height - 8);
  ctx.fillText(labels[1], 4, 14);
  if (xLabel) ctx.fillText(xLabel, pad, 14);
}

function renderAnalysis() {
  const d = STATE.readout_decision;
  const el = $("decision");
  el.className = "decision " + d.action;
  el.textContent =
    `决策：${d.action === "resolved" ? "可分辨（无需重追踪）" : "需要重新追踪 / 只能给可辨范围"}\n` +
    `可分辨半宽：±${d.resolvable_half_width_ns.toFixed(3)} ns\n` +
    `历史粗箱模式：${d.historical ? "是" : "否"}\n说明：${d.detail}\n` +
    (d.reasons.length ? "依据：\n- " + d.reasons.join("\n- ") : "依据：两门内能量定位清晰，泄漏低于阈值。");

  const gt = $("gateTable").querySelector("tbody");
  gt.innerHTML = "";
  for (const g of STATE.gate_readout.gates) {
    const routes = Object.entries(g.per_route)
      .map(([k, v]) => `${k}: ${v.toFixed(2)}`).join("<br>");
    gt.insertAdjacentHTML("beforeend",
      `<tr><td>${g.name}</td><td>${g.start_ns.toFixed(2)}–${g.end_ns.toFixed(2)}</td>
       <td>${g.energy.toFixed(3)}</td><td>${(g.fraction_of_total * 100).toFixed(1)}%</td><td>${routes}</td></tr>`);
  }
  gt.insertAdjacentHTML("beforeend",
    `<tr><td>门外泄漏</td><td>–</td><td>${STATE.gate_readout.outside_gate_leakage.toFixed(3)}</td>
     <td>${(STATE.gate_readout.leakage_fraction * 100).toFixed(1)}%</td><td>–</td></tr>`);

  const rt = $("routeTable").querySelector("tbody");
  rt.innerHTML = "";
  const total = STATE.energy_total || 1;
  for (const [name, e] of Object.entries(STATE.route_energy)) {
    rt.insertAdjacentHTML("beforeend",
      `<tr><td>${name}</td><td>${e.toFixed(3)}</td><td>${(e / total * 100).toFixed(1)}%</td></tr>`);
  }
  $("analytic").textContent = JSON.stringify(STATE.expected_analytic, null, 2);
}

function syncGateInputs() {
  const [g1, g2] = STATE.gates;
  $("g1start").value = g1.start.toFixed(2);
  $("gboundary").value = g1.end.toFixed(2);
  $("g2end").value = g2.end.toFixed(2);
  const span = g2.end - g1.start;
  const bFrac = (g1.end - g1.start) / span;
  $("gateRange").value = Math.round(bFrac * 1000);
  $("g1fill").style.width = `${bFrac * 100}%`;
  $("g2fill").style.width = `${(1 - bFrac) * 100}%`;
}

async function applyGates() {
  if (!STATE) return;
  const gates = [
    { name: "G1", start: +$("g1start").value, end: +$("gboundary").value },
    { name: "G2", start: +$("gboundary").value, end: +$("g2end").value },
  ];
  $("status").textContent = "按新门重新读出（复用已追踪光路，不重新追踪）…";
  const p = params();
  p.gates = gates;
  STATE = await postJSON("/api/regate", p);
  $("status").textContent = "门读出已更新（仅重新积分时间箱）。";
  renderAnalysis();
  drawFrame(+$("timeSlider").value || 0);
}

async function doRender() {
  const btn = $("renderBtn");
  btn.disabled = true;
  $("status").textContent = "瞬态追踪进行中…";
  try {
    const p = params();
    STATE = await postJSON("/api/render", p);
    $("timeSlider").max = STATE.time_grid.n_bins - 1;
    $("timeSlider").disabled = false;
    $("timeSlider").value = Math.min(
      STATE.time_grid.n_bins - 1,
      Math.round(30 / STATE.time_grid.dt));
    syncGateInputs();
    renderAnalysis();
    drawFrame(+$("timeSlider").value);
    $("status").textContent =
      `渲染完成：${STATE.frames.length} 个时间箱，边缘损失 ${
        (STATE.energy_lost_edge / Math.max(1e-9, STATE.energy_total) * 100).toFixed(3)}%`;
  } catch (e) {
    $("status").textContent = "渲染失败：" + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function selectPixel(px, py) {
  if (!STATE) return;
  $("profileLabel").textContent = `（像素 ${px}, ${py}）`;
  const p = params();
  p.px = px;
  p.py = py;
  const prof = await postJSON("/api/pixel", p);
  const g = STATE.time_grid;
  const tMax = Math.max(...prof.time_luminance, 1e-12);
  drawLineChart($("timeChart"), [prof.time_luminance],
    [`时间 (ns), 箱宽 ${g.dt}`, "辐亮度"],
    ["#7fd0ff"], tMax,
    prof.centroid_ns != null ? `到达重心 ${prof.centroid_ns.toFixed(2)} ns` : "");
  const oMax = Math.max(...prof.opl_bins, 1e-12);
  drawLineChart($("oplChart"), [prof.opl_bins],
    ["光程 OPL (m)", "能量"], ["#ffd27a"], oMax);
  if (STATE.coarse) {
    const c = STATE.coarse;
    const cMax = Math.max(...c.bins, 1e-12);
    drawLineChart($("coarseChart"), [c.bins],
      [`粗箱 (ns), 宽 ${c.grid.dt}`, "整箱总量"], ["#ff9b9b"], cMax,
      "只有整箱总量，亚箱顺序无法恢复");
  } else {
    const ctx = $("coarseChart").getContext("2d");
    ctx.fillStyle = "#0a0f16";
    ctx.fillRect(0, 0, $("coarseChart").width, $("coarseChart").height);
    ctx.fillStyle = "#7f93a8";
    ctx.font = "12px sans-serif";
    ctx.fillText("未启用历史粗箱模式。", 40, 70);
  }
}

// events
$("renderBtn").addEventListener("click", doRender);
$("timeSlider").addEventListener("input", (e) => drawFrame(+e.target.value));
$("applyGate").addEventListener("click", applyGates);
$("gateRange").addEventListener("input", () => {
  if (!STATE) return;
  const [g1, g2] = STATE.gates;
  const start = g1.start;
  const end = g2.end;
  const boundary = start + ($("gateRange").value / 1000) * (end - start);
  g1.end = boundary;
  g2.start = boundary;
  syncGateInputs();
});
["g1start", "gboundary", "g2end"].forEach((id) =>
  $(id).addEventListener("change", () => {
    if (!STATE) return;
    STATE.gates = [
      { name: "G1", start: +$("g1start").value, end: +$("gboundary").value },
      { name: "G2", start: +$("gboundary").value, end: +$("g2end").value },
    ];
    syncGateInputs();
  }));
$("sliceCanvas").addEventListener("click", (e) => {
  if (!STATE) return;
  const [w, h] = STATE.resolution;
  const rect = e.target.getBoundingClientRect();
  const px = Math.floor((e.clientX - rect.left) / rect.width * w);
  const py = Math.floor((e.clientY - rect.top) / rect.height * h);
  selectPixel(Math.max(0, Math.min(w - 1, px)),
              Math.max(0, Math.min(h - 1, py)));
});
$("playBtn").addEventListener("click", () => {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    $("playBtn").textContent = "播放波前";
    return;
  }
  $("playBtn").textContent = "暂停";
  let i = +$("timeSlider").value || 0;
  playTimer = setInterval(() => {
    if (!STATE) return;
    i = (i + 1) % STATE.time_grid.n_bins;
    $("timeSlider").value = i;
    drawFrame(i);
  }, 120);
});
$("historical").addEventListener("change", () => {
  $("coarseDt").disabled = !$("historical").checked;
});
