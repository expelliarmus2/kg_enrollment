/* Patient-Centered Graph Retrieval - frontend logic */
"use strict";

const NODE_STYLE = {
  patient:     { color: "#5fd36a", shape: "dot",      size: 18, font: { color: "#fff", size: 13 } },
  disease:     { color: "#e06a6a", shape: "ellipse",   size: 16, font: { color: "#fff" } },
  indicator:   { color: "#e0a04a", shape: "diamond",   size: 14, font: { color: "#fff" } },
  drug:        { color: "#4a90d9", shape: "box",       size: 14, font: { color: "#fff" } },
  intermediate:{ color: "#5a6a7a", shape: "dot",       size: 8,  font: { color: "#b9c8d8", size: 10 } },
};

const API = {
  graph: "/api/graph",
  conditions: "/api/conditions",
  groundTruth: "/api/ground_truth",
  retrieve: "/api/retrieve",
  compare: "/api/compare",
  filterReport: "/api/filter_report",
  animate: "/api/animate_search",
};

let fullNodes = null, fullEdges = null, fullNetwork = null;
let subNetwork = null, subNodes = null, subEdges = null;
let currentConditions = [];
let currentResults = [];
let lastGraphData = null;
let compareData = null, comparePatient = null, compareAlgo = "bfs";

/* ---------- helpers ---------- */
function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 1800);
}
async function getJSON(url) {
  const r = await fetch(url);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}
function fmt(n, d = 2) { return Number(n).toFixed(d); }

/* ---------- full graph ---------- */
function nodeVis(n) {
  const st = NODE_STYLE[n.type] || NODE_STYLE.intermediate;
  return {
    id: n.id,
    label: n.id,
    group: n.type,
    ...st,
    title: `id: ${n.id}\ntype: ${n.type}\ndegree: ${n.degree ?? "?"}`,
  };
}
function edgeVis(e, opts = {}) {
  let label = e.relation_type;
  if (e.value !== undefined) label += `\n${e.value}${e.unit ? " " + e.unit : ""}`;
  return {
    id: `${e.source}->${e.target}:${e.relation_type}`,
    from: e.source, to: e.target,
    label: opts.showLabels ? label : undefined,
    title: `relation: ${e.relation_type}` +
      (e.value !== undefined ? `\nvalue: ${e.value} ${e.unit || ""}` : "") +
      (e.time ? `\ntime: ${e.time}` : ""),
    arrows: "to",
    color: { color: "#3a5070", highlight: "#6fa8dc", hover: "#6fa8dc" },
    font: { size: 9, color: "#8aa6c1" },
  };
}

async function initFullGraph() {
  lastGraphData = await getJSON(API.graph);
  fullNodes = new vis.DataSet(lastGraphData.nodes.map(nodeVis));
  fullEdges = new vis.DataSet(lastGraphData.edges.map(e => edgeVis(e)));
  const container = document.getElementById("full-graph");
  const data = { nodes: fullNodes, edges: fullEdges };
  const options = {
    nodes: { borderWidth: 1, borderWidthSelected: 2 },
    edges: { smooth: { type: "continuous", roundness: 0.15 }, selectionWidth: 2 },
    physics: { solver: "forceAtlas2Based", stabilization: { iterations: 120 } },
    interaction: { hover: true, tooltipDelay: 100 },
    groups: {},
  };
  fullNetwork = new vis.Network(container, data, options);
  document.getElementById("full-info").addEventListener !== undefined;
  fullNetwork.on("selectNode", (p) => {
    if (p.nodes.length) toast(`Node: ${p.nodes[0]}`);
  });
}

/* ---------- query panel ---------- */
async function initQueryPanel() {
  const c = await getJSON(API.conditions);
  currentConditions = c.conditions;
  const gt = await getJSON(API.ground_truth);
  const panel = document.getElementById("query-panel");
  panel.innerHTML = "";
  currentConditions.forEach(cond => {
    const row = document.createElement("div");
    row.className = "cond-row";
    const checked = "checked";
    if (cond.kind === "numeric") {
      row.innerHTML = `
        <input type="checkbox" data-id="${cond.id}" ${checked}>
        <span class="anchor">${cond.anchor}</span>
        <span class="num-input">
          <select data-id="${cond.id}" data-field="operator">
            <option ${cond.operator === ">" ? "selected" : ""}>&gt;</option>
            <option ${cond.operator === ">=" ? "selected" : ""}>&gt;=</option>
            <option ${cond.operator === "<" ? "selected" : ""}>&lt;</option>
            <option ${cond.operator === "<=" ? "selected" : ""}>&lt;=</option>
          </select>
          <input type="number" step="0.1" value="${cond.value}" data-id="${cond.id}" data-field="value">
        </span>`;
    } else {
      row.innerHTML = `
        <input type="checkbox" data-id="${cond.id}" ${checked}>
        <span class="anchor">${cond.anchor}</span>
        <span style="color:#7a8da0;font-size:11px">categorical</span>`;
    }
    panel.appendChild(row);
  });
  // legend
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = Object.entries(NODE_STYLE).map(([k, v]) =>
    `<span class="item"><span class="swatch" style="background:${v.color}"></span>${k}</span>`
  ).join("");
  panel.appendChild(legend);
}

function collectConditions() {
  return currentConditions.map(cond => {
    const cb = document.querySelector(`input[type=checkbox][data-id="${cond.id}"]`);
    if (cb && !cb.checked) return { ...cond, _disabled: true };
    if (cond.kind === "numeric") {
      const sel = document.querySelector(`select[data-id="${cond.id}"]`);
      const inp = document.querySelector(`input[type=number][data-id="${cond.id}"]`);
      const op = sel ? sel.value.replace("&gt;", ">").replace("&lt;", "<") : cond.operator;
      // text may contain entity; read directly
      const opClean = (sel ? sel.options[sel.selectedIndex].text : cond.operator)
        .replace("&gt;", ">").replace("&lt;", "<");
      return {
        ...cond,
        operator: opClean,
        value: inp ? parseFloat(inp.value) : cond.value,
      };
    }
    return cond;
  }).filter(c => !c._disabled);
}

function getParams() {
  const mode = document.getElementById("mode-select").value;
  return {
    algorithm: document.getElementById("algo-select").value,
    max_hops: parseInt(document.getElementById("max-hops").value),
    strict: mode === "strict",
    soft_threshold: 0.8,
    rho: document.getElementById("hub-penalty").checked ? 1.3 : 0.0,
    condition_prize: 10.0,
  };
}

/* ---------- run retrieval ---------- */
async function runRetrieval() {
  const params = getParams();
  const conds = collectConditions();
  toast(`Running ${params.algorithm} (hops=${params.max_hops})...`);
  document.getElementById("compare-tabs").style.display = "none";
  document.getElementById("compare-metrics").style.display = "none";
  const data = await postJSON(API.retrieve, { ...params, conditions: conds });
  currentResults = data.results;
  renderPatientTable(data);
  renderFilterReport(data.filter_report);
  renderMetrics(data.evaluation, params.algorithm);
  // auto-select top patient
  if (data.results.length) selectPatient(data.results[0].patient_id);
  toast(`Done: ${data.evaluation.num_returned} patients returned`);
}

/* ---------- patient table ---------- */
function renderPatientTable(data) {
  const tb = document.querySelector("#patient-table tbody");
  tb.innerHTML = "";
  const m = 5;
  data.results.forEach(r => {
    const tr = document.createElement("tr");
    tr.className = "patient-row";
    tr.dataset.patient = r.patient_id;
    const dists = Object.values(r.condition_paths).filter(c => c.matched).map(c => c.distance);
    const avg = dists.length ? (dists.reduce((a, b) => a + b, 0) / dists.length) : 0;
    const cls = r.coverage_count === m ? "match" : (r.coverage_count >= 4 ? "partial" : "nomatch");
    tr.innerHTML = `
      <td>${r.patient_id}</td>
      <td class="${cls}">${r.coverage_count}/${m}</td>
      <td>${fmt(r.score)}</td>
      <td>${fmt(avg, 1)}</td>
      <td>${fmt(r.graph_cost)}</td>
      <td class="${cls}">${r.strict_match ? "Match" : "Partial"}</td>`;
    tr.onclick = () => selectPatient(r.patient_id);
    tb.appendChild(tr);
  });
}

/* ---------- select patient + subgraph ---------- */
function selectPatient(pid) {
  document.querySelectorAll(".patient-row").forEach(r => {
    r.classList.toggle("active", r.dataset.patient === pid);
  });
  document.getElementById("sub-patient-label").textContent = pid;
  const r = currentResults.find(x => x.patient_id === pid);
  if (!r) return;
  renderSubgraph(r);
  renderPathExplanation(r);
}

function renderSubgraph(r) {
  const anchorSet = new Set(currentConditions.map(c => c.anchor));
  const nodes = r.nodes.map(id => {
    const nd = lastGraphData.nodes.find(n => n.id === id) || { id, type: "intermediate" };
    const st = NODE_STYLE[nd.type] || NODE_STYLE.intermediate;
    let style = { ...st };
    if (id === r.patient_id) {
      style = { ...st, size: 28, color: { background: "#5fd36a", border: "#ffffff", highlight: { background: "#5fd36a", border: "#ffffff" } }, borderWidth: 3 };
    } else if (anchorSet.has(id)) {
      style = { ...st, size: st.size + 6, color: { background: st.color, border: "#ffe14a", highlight: { background: st.color, border: "#ffe14a" } }, borderWidth: 3 };
    } else {
      style = { ...st, color: { background: "#3a4a5a", border: "#2a3a4a" }, font: { color: "#8aa6c1", size: 9 } };
    }
    return { id, label: id, ...style, title: `${id} (${nd.type})` };
  });
  const edges = r.edges.map((e, i) => ({
    id: `sub-${i}`,
    from: e.source, to: e.target,
    label: e.value !== undefined ? `${e.value}` : undefined,
    arrows: "to",
    color: { color: "#4a90d9", highlight: "#6fa8dc" },
    font: { size: 10, color: "#e0c34a" },
    title: `${e.relation_type}` + (e.value !== undefined ? ` | ${e.value} ${e.unit || ""}` : ""),
  }));
  if (subNetwork) subNetwork.destroy();
  subNodes = new vis.DataSet(nodes);
  subEdges = new vis.DataSet(edges);
  subNetwork = new vis.Network(
    document.getElementById("sub-graph"),
    { nodes: subNodes, edges: subEdges },
    {
      layout: { hierarchical: { enabled: true, direction: "LR", sortMethod: "directed", levelSeparation: 120, nodeSpacing: 80 } },
      edges: { smooth: true, arrows: { to: { enabled: true, scaleFactor: 0.6 } } },
      physics: { enabled: false },
      interaction: { hover: true, tooltipDelay: 80 },
    }
  );
}

/* ---------- path explanation ---------- */
function renderPathExplanation(r) {
  const box = document.getElementById("path-explanation");
  box.innerHTML = `<div class="path-block"><div class="head">Patient: ${r.patient_id} — coverage ${r.coverage_count}/5 — score ${fmt(r.score)}</div></div>`;
  currentConditions.forEach(cond => {
    const cp = r.condition_paths[cond.id];
    if (!cp) return;
    const block = document.createElement("div");
    const cls = cp.matched ? "" : (cp.pruned ? "pruned" : "fail");
    block.className = "path-block " + cls;
    const chain = cp.path && cp.path.length ? cp.path.join(" → ") : "—";
    let meta = "";
    if (cond.kind === "numeric") {
      meta = `constraint: ${cond.operator} ${cond.value}`;
      if (cp.matched) meta += " → MATCH";
    }
    if (cp.distance) meta += (meta ? " | " : "") + `${cp.distance} hop(s)`;
    if (cp.reason) meta += (meta ? " | " : "") + cp.reason;
    block.innerHTML = `
      <div class="head">${cond.id} · ${cond.anchor}${cond.kind === "numeric" ? ` ${cond.operator} ${cond.value}` : ""} — ${cp.matched ? "MATCH" : (cp.pruned ? "PRUNED" : "NOT MATCH")}</div>
      <div class="chain">${chain}</div>
      <div class="meta">${meta}</div>`;
    box.appendChild(block);
  });
}

/* ---------- filter report ---------- */
function renderFilterReport(report) {
  const wrap = document.getElementById("filter-report");
  if (!report || !report.length) { wrap.innerHTML = "<span style='color:#7a8da0'>No numeric edges.</span>"; return; }
  wrap.innerHTML = "";
  report.forEach(f => {
    const card = document.createElement("div");
    card.className = "filter-card" + (f.satisfied ? "" : " fail");
    card.innerHTML = `
      <span class="anchor">${f.anchor}</span> → ${f.target}
      <br>value: <b>${f.value}</b> ${f.satisfied ? "" : f.unit || ""}
      <br>${f.reason}`;
    wrap.appendChild(card);
  });
}

/* ---------- metrics ---------- */
function renderMetrics(ev, algo) {
  let area = document.getElementById("metrics-row");
  if (!area) {
    area = document.createElement("div");
    area.id = "metrics-row";
    area.className = "metrics-row";
    document.getElementById("comparison-section").prepend(area);
  }
  const cards = [
    ["Algorithm", algo.toUpperCase()],
    ["Precision", ev.precision],
    ["Recall", ev.recall],
    ["F1", ev.f1],
    ["Returned", ev.num_returned],
    ["Visited nodes", ev.total_visited_nodes],
    ["Visited edges", ev.total_visited_edges],
    ["Runtime (ms)", ev.elapsed_ms],
  ];
  area.innerHTML = cards.map(([l, v]) =>
    `<div class="metric"><div class="label">${l}</div><div class="value">${v}</div></div>`
  ).join("");
}

/* ---------- compare algorithms ---------- */
async function compareAlgorithms() {
  const params = getParams();
  const conds = collectConditions();
  toast("Running all 3 algorithms...");
  const data = await postJSON(API.compare, { ...params, conditions: conds });
  compareData = data;
  const tb = document.querySelector("#comparison-table tbody");
  tb.innerHTML = "";
  function cell(v, key) {
    if (!v) return "<td>-</td>";
    const cls = v.strict ? "match" : "partial";
    return `<td class="${cls}">${v[key]}</td>`;
  }
  function rankCell(v) {
    if (!v) return "<td>-</td>";
    const cls = v.strict ? "match" : "partial";
    return `<td class="${cls}">#${v.rank} (${fmt(v.score, 1)})</td>`;
  }
  function sizeCell(v) {
    if (!v) return "<td>-</td>";
    return `<td>${v.visited_nodes} / ${v.visited_edges}</td>`;
  }
  data.comparison_table.forEach(row => {
    const tr = document.createElement("tr");
    tr.className = "compare-row";
    tr.dataset.patient = row.patient;
    tr.innerHTML = `
      <td><b>${row.patient}</b></td>
      <td>${row.ground_truth ? "✓" : ""}</td>
      ${cell(row.bfs, "coverage")}${rankCell(row.bfs)}${sizeCell(row.bfs)}
      ${cell(row.pcst, "coverage")}${rankCell(row.pcst)}${sizeCell(row.pcst)}
      ${cell(row.steiner, "coverage")}${rankCell(row.steiner)}${sizeCell(row.steiner)}`;
    tr.onclick = () => selectComparePatient(row.patient);
    tb.appendChild(tr);
  });
  // metrics for each
  const area = document.getElementById("metrics-row");
  const cards = [];
  for (const algo of ["bfs", "pcst", "steiner"]) {
    const ev = data.per_algorithm[algo].evaluation;
    cards.push([`${algo.toUpperCase()} P`, ev.precision]);
    cards.push([`${algo.toUpperCase()} R`, ev.recall]);
    cards.push([`${algo.toUpperCase()} t(ms)`, ev.elapsed_ms]);
  }
  area.innerHTML = cards.map(([l, v]) =>
    `<div class="metric"><div class="label">${l}</div><div class="value">${v}</div></div>`
  ).join("");
  // show compare-mode tabs + select first patient
  document.getElementById("compare-tabs").style.display = "";
  document.getElementById("compare-metrics").style.display = "";
  if (data.comparison_table.length) {
    selectComparePatient(data.comparison_table[0].patient);
  }
  toast("Comparison ready");
}

function selectComparePatient(pid) {
  comparePatient = pid;
  document.querySelectorAll(".compare-row").forEach(r => {
    r.classList.toggle("active", r.dataset.patient === pid);
  });
  document.getElementById("sub-patient-label").textContent = `${pid} (compare)`;
  document.getElementById("path-explanation").innerHTML =
    `<div class="path-block"><div class="head">Compare mode: ${pid} · switch tabs to compare subgraphs.</div>` +
    `<div class="meta">Run single retrieval for full path explanation.</div></div>`;
  renderCompareTabs();
  renderCompareSubgraph();
}

function renderCompareTabs() {
  const bar = document.getElementById("compare-tabs");
  bar.innerHTML = "";
  ["bfs", "pcst", "steiner"].forEach(algo => {
    const btn = document.createElement("button");
    btn.className = "tab" + (algo === compareAlgo ? " active" : "");
    btn.textContent = algo.toUpperCase();
    btn.onclick = () => {
      compareAlgo = algo;
      renderCompareTabs();
      renderCompareSubgraph();
    };
    bar.appendChild(btn);
  });
}

function renderCompareSubgraph() {
  if (!compareData || !comparePatient) return;
  const results = compareData.per_algorithm[compareAlgo].results;
  const r = results.find(x => x.patient_id === comparePatient);
  if (!r) return;
  renderSubgraph(r);
  const strip = document.getElementById("compare-metrics");
  const m = 5;
  const cards = [
    ["Algorithm", compareAlgo.toUpperCase()],
    ["Nodes", r.visited_nodes],
    ["Edges", r.visited_edges],
    ["Coverage", `${r.coverage_count}/${m}`],
    ["Score", fmt(r.score)],
    ["Runtime (ms)", fmt(r.elapsed_ms, 3)],
  ];
  strip.innerHTML = cards.map(([l, v]) =>
    `<div class="metric"><div class="label">${l}</div><div class="value">${v}</div></div>`
  ).join("");
}

/* ---------- animate search ---------- */
let animTimer = null;
async function animateSearch() {
  if (animTimer) { clearInterval(animTimer); animTimer = null; toast("Animation stopped"); return; }
  const params = getParams();
  const data = await getJSON(`${API.animate}?algorithm=${params.algorithm}&max_hops=${params.max_hops}&rho=${params.rho}`);
  if (!fullNetwork) return;
  // reset colors
  fullNodes.update(lastGraphData.nodes.map(n => ({ id: n.id, color: null })));
  const colorMap = { C1: "#e06a6a", C2: "#e0a04a", C3: "#4a90d9", C4: "#9a6ae0", C5: "#5fd36a" };
  const allSteps = [];
  data.frames.forEach(fr => {
    fr.steps.forEach(s => allSteps.push({ ...s, color: colorMap[fr.condition] || "#6fa8dc" }));
  });
  let i = 0;
  animTimer = setInterval(() => {
    if (i >= allSteps.length) { clearInterval(animTimer); animTimer = null; toast("Search complete"); return; }
    const s = allSteps[i];
    s.frontier.forEach(n => {
      fullNodes.update({ id: n, color: { background: s.color, border: s.color } });
    });
    s.patients.forEach(p => {
      fullNodes.update({ id: p, color: { background: "#5fd36a", border: "#ffffff" } });
    });
    i++;
  }, 700);
  toast("Animating search...");
}

/* ---------- init ---------- */
window.addEventListener("DOMContentLoaded", async () => {
  await initFullGraph();
  await initQueryPanel();
  document.getElementById("max-hops").oninput = e => {
    document.getElementById("max-hops-val").textContent = e.target.value;
  };
  document.getElementById("btn-run").onclick = runRetrieval;
  document.getElementById("btn-compare").onclick = compareAlgorithms;
  document.getElementById("btn-animate").onclick = animateSearch;
  document.getElementById("btn-fit-full").onclick = () => fullNetwork && fullNetwork.fit({ animation: true });
  // initial run
  runRetrieval();
});
