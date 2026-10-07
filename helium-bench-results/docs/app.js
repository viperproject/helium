// Helium benchmark site: plain JavaScript, no build step.
//
// Reads benchmarks/results/index.json (one summary entry per run) and fetches
// a full run file only when a page needs it. Every selection lives in the URL
// hash (#page?key=value&...), so any view can be shared as a link.
//
// Data location: ../benchmarks/results/ relative to this page (the layout of
// the `benchmarks` branch), or ?data=<url> to point elsewhere.

"use strict";

const DATA = (new URLSearchParams(location.search).get("data") || "../benchmarks/results/").replace(/\/?$/, "/");
const state = { index: null, runs: new Map() };
const app = document.getElementById("app");

// ── Utilities ──

const $ = (html) => {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
};
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const short = (sha) => (sha || "").slice(0, 8);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"];

function fmtSecs(x) {
  if (x == null || !isFinite(x)) return "–";
  if (x >= 100) return x.toFixed(0) + " s";
  if (x >= 1) return x.toFixed(2) + " s";
  if (x >= 0.001) return (x * 1000).toFixed(1) + " ms";
  return (x * 1e6).toFixed(0) + " µs";
}
function fmtNum(x, digits = 2) {
  if (x == null || !isFinite(x)) return "–";
  if (Number.isInteger(x)) return x.toLocaleString("en-US");
  return Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(digits);
}
function fmtRatio(r) {
  if (r == null || !isFinite(r)) return "–";
  return "×" + r.toFixed(r >= 10 ? 1 : 2);
}
function fmtChange(r) {
  if (r == null || !isFinite(r)) return "–";
  const pct = (r - 1) * 100;
  return (pct >= 0 ? "+" : "−") + Math.abs(pct).toFixed(1) + "%";
}
function median(xs) {
  const v = xs.filter((x) => x != null && isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
function geomean(xs) {
  const v = xs.filter((x) => x > 0 && isFinite(x));
  return v.length ? Math.exp(v.reduce((a, x) => a + Math.log(x), 0) / v.length) : null;
}
function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
// The median of a timing column, only when it finished.
function med(t) {
  if (!t || (t.status && t.status !== "ok")) return null;
  return t.median ?? null;
}

// Spearman rank correlation.
function spearman(xs, ys) {
  const rank = (v) => {
    const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
    const r = new Array(v.length);
    for (let i = 0; i < idx.length; ) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return r;
  };
  const n = xs.length;
  if (n < 3) return null;
  const rx = rank(xs), ry = rank(ys);
  const mx = rx.reduce((a, b) => a + b) / n, my = ry.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (rx[i] - mx) * (ry[i] - my);
    sxx += (rx[i] - mx) ** 2;
    syy += (ry[i] - my) ** 2;
  }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
}

// ── Data ──

async function loadIndex() {
  const r = await fetch(DATA + "index.json", { cache: "no-cache" });
  if (!r.ok) throw new Error(`index.json: HTTP ${r.status}`);
  state.index = await r.json();
  state.index.runs = state.index.runs || [];
}
async function loadRun(commit) {
  if (state.runs.has(commit)) return state.runs.get(commit);
  const entry = runEntry(commit);
  if (!entry) return null;
  const p = fetch(DATA + entry.file).then((r) => {
    if (!r.ok) throw new Error(`${entry.file}: HTTP ${r.status}`);
    return r.json();
  });
  state.runs.set(commit, p);
  return p;
}
const runs = () => state.index.runs;
const runEntry = (commit) => runs().find((e) => e.commit === commit || e.commit.startsWith(commit || "\u0000"));
const latest = () => runs()[runs().length - 1];
const previousOf = (commit) => {
  const i = runs().findIndex((e) => e.commit === commit);
  return i > 0 ? runs()[i - 1] : null;
};
const commitLabel = (e) => `${short(e.commit)} ${e.subject || ""}`.trim() + (e.dirty ? " (dirty)" : "");
const commitUrl = (sha) => (state.index.repo_url ? `${state.index.repo_url.replace(/\/$/, "")}/commit/${sha}` : null);
const fileKey = (f) => `${f.suite}/${f.stem}`;
const fingerprint = (f) => `${f.sha256?.rs || ""}|${f.sha256?.vpr || ""}`;

// Per-file comparison over files both runs share with identical inputs.
function compareRuns(a, b, metric = "times.helium_verify") {
  const fa = new Map(a.files.map((f) => [fileKey(f), f]));
  const fb = new Map(b.files.map((f) => [fileKey(f), f]));
  const rows = [], changedInput = [], added = [], removed = [];
  for (const [k, f] of fb) {
    const g = fa.get(k);
    if (!g) { added.push(k); continue; }
    if (fingerprint(f) !== fingerprint(g)) { changedInput.push(k); continue; }
    const x = fileValue(g, metric), y = fileValue(f, metric);
    rows.push({ key: k, suite: f.suite, a: x, b: y, ratio: x > 0 && y != null ? y / x : null, fa: g, fb: f });
  }
  for (const k of fa.keys()) if (!fb.has(k)) removed.push(k);
  return { rows, changedInput, added, removed };
}

// Numeric value of a file-level metric path. `times.X` means its median.
function fileValue(f, path) {
  if (path.startsWith("times.")) return med(f.times?.[path.slice(6)]);
  const v = get(f, path);
  return typeof v === "number" ? v : null;
}
function memberValue(m, path) {
  const v = get(m, path);
  return typeof v === "number" ? v : null;
}

// Numeric paths found in a set of objects (for metric pickers).
function numericPaths(objs, prefix, skip = new Set()) {
  const out = new Set();
  const walk = (o, p, depth) => {
    if (o == null || depth > 3) return;
    for (const [k, v] of Object.entries(o)) {
      const q = p ? `${p}.${k}` : k;
      if (skip.has(q) || k === "runs" || k === "rule_timing") continue;
      if (typeof v === "number") out.add(q);
      else if (v && typeof v === "object" && !Array.isArray(v)) walk(v, q, depth + 1);
    }
  };
  for (const o of objs) walk(prefix ? get(o, prefix) : o, prefix || "", 0);
  return [...out].sort();
}
// Only the times each tool reports itself, so no column counts process or JVM
// startup; the wall-clock columns stay in the run files.
const TIME_COLUMNS = ["rustc_self", "helium_verify", "silicon_verify"];
function fileMetricPaths(run) {
  const times = TIME_COLUMNS.filter((c) => run.files.some((f) => med(f.times?.[c]) != null)).map((c) => "times." + c);
  const rest = numericPaths(run.files, "").filter((p) => !p.startsWith("times.") && !p.startsWith("knobs."));
  return [...times, ...rest.filter((p) => !p.startsWith("stats.per_rule"))];
}

// ── URL state ──

function parseHash() {
  const [route, query] = location.hash.replace(/^#/, "").split("?");
  return { route: route || "overview", params: new URLSearchParams(query || "") };
}
function setParams(patch) {
  const { route, params } = parseHash();
  for (const [k, v] of Object.entries(patch)) {
    params.delete(k);
    if (Array.isArray(v)) v.forEach((x) => params.append(k, x));
    else if (v != null && v !== "") params.set(k, v);
  }
  const q = params.toString();
  history.replaceState(null, "", `#${route}${q ? "?" + q : ""}`);
  render();
}

// ── Plotly ──

function baseLayout(extra = {}) {
  const font = { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: css("--text-2") };
  const axis = { gridcolor: css("--grid"), linecolor: css("--axis"), zerolinecolor: css("--axis"), tickfont: { color: css("--muted") }, automargin: true };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font,
    margin: { l: 60, r: 16, t: 16, b: 48 },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--border"), font: { color: css("--text") } },
    legend: { orientation: "h", y: -0.2, font: { color: css("--text-2") } },
    ...extra,
    xaxis: { ...axis, ...(extra.xaxis || {}) },
    yaxis: { ...axis, ...(extra.yaxis || {}) },
  };
}
function plot(el, traces, layout, onClick) {
  Plotly.react(el, traces, baseLayout(layout), { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] });
  if (onClick) {
    el.removeAllListeners?.("plotly_click");
    el.on("plotly_click", onClick);
  }
}

// ── Components ──

function tile(label, value, delta, cls = "") {
  return `<div class="card tile"><div class="label">${esc(label)}</div><div class="value">${value}</div><div class="delta ${cls}">${delta || "&nbsp;"}</div></div>`;
}
function table(cols, rows) {
  const head = cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("");
  const body = rows
    .map((r) => "<tr>" + cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.html ? c.html(r) : esc(c.get(r))}</td>`).join("") + "</tr>")
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${cols.length}" class="muted">none</td></tr>`}</tbody></table></div>`;
}
function select(name, options, value, { multiple = false, label = name } = {}) {
  const opts = options
    .map((o) => {
      const [v, t] = Array.isArray(o) ? o : [o, o];
      const sel = multiple ? (value || []).includes(v) : v === value;
      return `<option value="${esc(v)}"${sel ? " selected" : ""}>${esc(t)}</option>`;
    })
    .join("");
  return `<label>${esc(label)}<select data-param="${esc(name)}"${multiple ? " multiple" : ""}>${opts}</select></label>`;
}
function wireControls(root) {
  root.querySelectorAll("select[data-param]").forEach((s) =>
    s.addEventListener("change", () =>
      setParams({ [s.dataset.param]: s.multiple ? [...s.selectedOptions].map((o) => o.value) : s.value })
    )
  );
}
const commitOptions = () => [...runs()].reverse().map((e) => [e.commit, commitLabel(e)]);
const coverageLabel = (c) =>
  ["OK", "FAIL", "UNSUPPORTED", "ERROR", "SKIP"].filter((k) => c?.[k]).map((k) => `${k} ${c[k]}`).join(" · ") || "–";

// ── Pages ──

async function pageOverview() {
  const cur = latest();
  const prev = previousOf(cur.commit);
  const [run, prevRun] = await Promise.all([loadRun(cur.commit), prev ? loadRun(prev.commit) : null]);
  const s = cur.summary, ps = prev?.summary;
  const cmp = prevRun ? compareRuns(prevRun, run) : null;
  const fair = cmp ? cmp.rows.reduce((a, r) => [a[0] + (r.a || 0), a[1] + (r.b || 0)], [0, 0]) : null;
  const fairRatio = fair && fair[0] > 0 ? fair[1] / fair[0] : null;
  const dRatio = (a, b) => (a && b ? fmtChange(b / a) + " vs previous" : "");

  const link = commitUrl(cur.commit);
  const el = $(`<div>
    <h1>Latest run: <code>${esc(short(cur.commit))}</code> ${esc(cur.subject || "")}</h1>
    <p class="muted">${esc(cur.commit_date || cur.date)} · host ${esc(cur.host)} · ${s.files} files, ${s.members} members
      ${link ? `· <a href="${link}" target="_blank" rel="noopener">commit</a>` : ""}
      ${prev ? `· previous <code>${esc(short(prev.commit))}</code>` : ""}</p>
    <div class="tiles">
      ${tile("Helium verify, total", fmtSecs(s.totals.helium_verify),
        fairRatio ? `${fmtChange(fairRatio)} over ${cmp.rows.length} shared files` : "", fairRatio > 1.15 ? "bad" : fairRatio && fairRatio < 0.95 ? "good" : "")}
      ${tile("Overhead vs rustc", fmtRatio(s.geomean_overhead), ps ? dRatio(ps.geomean_overhead, s.geomean_overhead) : "geometric mean")}
      ${tile("Speedup vs Silicon", fmtRatio(s.geomean_speedup), s.geomean_speedup ? (ps ? dRatio(ps.geomean_speedup, s.geomean_speedup) : "geometric mean") : "no Silicon data")}
      ${tile("Members verified", fmtNum(s.coverage.OK || 0), `of ${fmtNum(s.members)}${ps ? ` · was ${fmtNum(ps.coverage.OK || 0)}` : ""}`)}
      ${tile("Not verified", fmtNum(s.members - (s.coverage.OK || 0)), coverageLabel({ ...s.coverage, OK: 0 }))}
      ${tile("Helium/Silicon disagree", fmtNum(s.disagreements), s.timeouts ? `${s.timeouts} timeouts` : "")}
    </div>
    <div class="two">
      <div><h2>Biggest slowdowns</h2><div id="up"></div></div>
      <div><h2>Biggest speedups</h2><div id="down"></div></div>
    </div>
    <h2>Coverage by suite</h2><div id="suites"></div>
    <h2>The 20 slowest members</h2><div id="slowest"></div>
    <div id="notes"></div>
  </div>`);
  app.replaceChildren(el);

  const changeCols = [
    { label: "file", html: (r) => `<a href="#compare?a=${prev?.commit}&b=${cur.commit}&suite=${encodeURIComponent(r.suite)}">${esc(r.key)}</a>` },
    { label: "before", num: true, get: (r) => fmtSecs(r.a) },
    { label: "now", num: true, get: (r) => fmtSecs(r.b) },
    { label: "change", num: true, html: (r) => `<span class="${r.ratio > 1 ? "bad" : "good"}">${fmtChange(r.ratio)}</span>` },
  ];
  const moved = (cmp?.rows || []).filter((r) => r.ratio && Math.max(r.a, r.b) > 0.01);
  el.querySelector("#up").innerHTML = cmp ? table(changeCols, moved.filter((r) => r.ratio > 1).sort((a, b) => b.ratio - a.ratio).slice(0, 8)) : `<p class="muted">No previous run.</p>`;
  el.querySelector("#down").innerHTML = cmp ? table(changeCols, moved.filter((r) => r.ratio < 1).sort((a, b) => a.ratio - b.ratio).slice(0, 8)) : "";

  el.querySelector("#suites").innerHTML = table(
    [
      { label: "suite", get: (r) => r[0] },
      { label: "files", num: true, get: (r) => r[1].files },
      { label: "helium verify", num: true, get: (r) => fmtSecs(r[1].helium_verify) },
      { label: "coverage", get: (r) => coverageLabel(r[1].coverage) },
    ],
    Object.entries(s.suites || {})
  );

  const members = run.files.flatMap((f) => (f.members || []).filter((m) => m.time != null).map((m) => ({ f, m })));
  members.sort((a, b) => b.m.time - a.m.time);
  el.querySelector("#slowest").innerHTML = table(
    [
      { label: "member", html: (r) => `${esc(r.m.name)} <span class="muted">${esc(fileKey(r.f))}</span>` },
      { label: "time", num: true, get: (r) => fmtSecs(r.m.time) },
      { label: "status", get: (r) => r.m.helium + (r.m.silicon ? ` / Si ${r.m.silicon}` : "") },
      { label: "Rust fn", get: (r) => r.m.rust_fn || "" },
      { label: "loc", num: true, get: (r) => fmtNum(r.m.rust_metrics?.loc) },
      { label: "paths", num: true, get: (r) => fmtNum(r.m.rust_metrics?.paths) },
      { label: "loops", num: true, get: (r) => fmtNum(r.m.rust_metrics?.loops) },
      { label: "&mut args", num: true, get: (r) => fmtNum(r.m.rust_metrics?.args_by_mut_ref) },
      { label: "Viper stmts", num: true, get: (r) => fmtNum(r.m.viper_metrics?.stmts) },
      { label: "fold+unfold", num: true, get: (r) => (r.m.viper_metrics ? fmtNum(r.m.viper_metrics.folds + r.m.viper_metrics.unfolds) : "–") },
    ],
    members.slice(0, 20)
  );

  const dis = run.files.flatMap((f) => (f.members || []).filter((m) => m.disagreement).map((m) => ({ f, m })));
  const warn = run.warnings || [];
  el.querySelector("#notes").innerHTML =
    (dis.length
      ? `<h2>Helium and Silicon disagree</h2>` +
        table(
          [
            { label: "member", get: (r) => `${fileKey(r.f)} ${r.m.name}` },
            { label: "Helium", get: (r) => r.m.helium },
            { label: "Silicon", get: (r) => r.m.silicon },
            { label: "kind", get: (r) => (r.m.disagreement === "soundness" ? "soundness: needs a look" : "incompleteness") },
          ],
          dis
        )
      : "") +
    (warn.length ? `<h2>Warnings</h2><ul class="warn-list">${warn.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : "");
}

async function pageTrends(params) {
  const summaryMetrics = [
    ["summary.totals.helium_verify", "Helium verify, total (s)"],
    ["summary.totals.rustc_self", "rustc check, total (s)"],
    ["summary.totals.silicon_verify", "Silicon verify, total (s)"],
    ["summary.geomean_overhead", "Overhead vs rustc (geomean ×)"],
    ["summary.geomean_speedup", "Speedup vs Silicon (geomean ×)"],
    ["summary.coverage.OK", "Members OK"],
    ["summary.coverage.FAIL", "Members FAIL"],
    ["summary.coverage.UNSUPPORTED", "Members UNSUPPORTED"],
    ["summary.coverage.ERROR", "Members ERROR"],
    ["summary.members", "Members"],
    ["summary.files", "Files"],
    ["summary.disagreements", "Helium/Silicon disagreements"],
  ];
  const suiteNames = [...new Set(runs().flatMap((e) => Object.keys(e.summary.suites || {})))].sort();
  for (const s of suiteNames) summaryMetrics.push([`summary.suites.${s}.helium_verify`, `Suite ${s}: Helium verify (s)`]);
  const scalingKeys = [...new Set(runs().flatMap((e) => Object.keys(e.scaling || {})))].sort();
  for (const k of scalingKeys) summaryMetrics.push([`scaling.${k}.exponent`, `Scaling exponent: ${k}`]);

  const scope = params.get("scope") || "summary";
  const metric = params.get("metric") || summaryMetrics[0][0];
  const el = $(`<div><h1>Trends</h1><div class="controls"></div><div class="card"><div class="chart" id="chart"></div></div>
    <p class="muted">Click a point to open that commit${state.index.repo_url ? " on GitHub" : " (set repo_url in tools/bench/config.json)"}.
    A ◆ marks a run where the file's inputs changed (re-encoded): the jump is not a Helium change.</p>
    <details><summary>Data table</summary><div id="tbl"></div></details></div>`);
  app.replaceChildren(el);
  const controls = el.querySelector(".controls");
  controls.innerHTML = select("scope", [["summary", "Whole corpus"], ["file", "One file"]], scope, { label: "scope" });

  let xs = [], ys = [], texts = [], breaks = [], label = "";
  if (scope === "summary") {
    controls.innerHTML += select("metric", summaryMetrics, metric, { label: "metric" });
    label = (summaryMetrics.find((m) => m[0] === metric) || [, metric])[1];
    for (const e of runs()) {
      const v = get(e, metric);
      if (typeof v === "number") { xs.push(e.commit_date || e.date); ys.push(v); texts.push(e); }
    }
  } else {
    // Per-file history needs every run file.
    const all = (await Promise.all(runs().map((e) => loadRun(e.commit)))).map((r, i) => ({ r, e: runs()[i] }));
    const files = [...new Set(all.flatMap(({ r }) => r.files.map(fileKey)))].sort();
    const file = params.get("file") || files[0];
    const paths = fileMetricPaths(all[all.length - 1].r);
    const fm = paths.includes(params.get("fmetric")) ? params.get("fmetric") : "times.helium_verify";
    controls.innerHTML += select("file", files, file, { label: "file" }) + select("fmetric", paths, fm, { label: "metric" });
    label = `${file}: ${fm}`;
    let lastFp = null;
    for (const { r, e } of all) {
      const f = r.files.find((g) => fileKey(g) === file);
      if (!f) continue;
      const v = fileValue(f, fm);
      if (v == null) continue;
      if (lastFp && fingerprint(f) !== lastFp) breaks.push(xs.length);
      lastFp = fingerprint(f);
      xs.push(e.commit_date || e.date); ys.push(v); texts.push(e);
    }
  }
  wireControls(controls);

  const color = css("--s1");
  const traces = [{
    x: xs, y: ys, type: "scatter", mode: "lines+markers", name: label,
    line: { color, width: 2 }, marker: { color, size: 8, line: { color: css("--surface"), width: 2 } },
    customdata: texts.map((e) => e.commit),
    text: texts.map((e) => esc(commitLabel(e))),
    hovertemplate: "%{text}<br>%{y:.4g}<extra></extra>",
  }];
  if (breaks.length) {
    traces.push({
      x: breaks.map((i) => xs[i]), y: breaks.map((i) => ys[i]), type: "scatter", mode: "markers", name: "inputs changed",
      marker: { symbol: "diamond", size: 12, color: css("--surface"), line: { color: css("--text"), width: 2 } },
      hovertemplate: "inputs changed (re-encoded)<extra></extra>",
    });
  }
  plot(el.querySelector("#chart"), traces, { showlegend: breaks.length > 0, yaxis: { title: { text: label }, rangemode: "tozero" }, xaxis: { type: "date" } }, (ev) => {
    const sha = ev.points[0]?.customdata;
    const url = sha && commitUrl(sha);
    if (url) window.open(url, "_blank", "noopener");
  });
  el.querySelector("#tbl").innerHTML = table(
    [
      { label: "commit", get: (i) => commitLabel(texts[i]) },
      { label: "date", get: (i) => xs[i] },
      { label: "value", num: true, get: (i) => fmtNum(ys[i], 4) },
    ],
    xs.map((_, i) => i).reverse()
  );
}

async function pageCompare(params) {
  const opts = commitOptions();
  const b = params.get("b") || latest().commit;
  const a = params.get("a") || previousOf(b)?.commit || b;
  const [ra, rb] = await Promise.all([loadRun(a), loadRun(b)]);
  const suites = ["", ...new Set([...ra.files, ...rb.files].map((f) => f.suite))].sort();
  const suite = params.get("suite") || "";
  const metrics = fileMetricPaths(rb);
  const metric = metrics.includes(params.get("metric")) ? params.get("metric") : "times.helium_verify";

  const el = $(`<div><h1>Compare commits</h1><div class="controls"></div>
    <div class="tiles" id="tiles"></div>
    <div class="card"><div class="chart" id="chart"></div></div>
    <p class="muted">One point per file both commits share with identical inputs; above the line is slower in B.</p>
    <h2>Members, by change</h2><div class="controls" id="mctl"></div><div id="members"></div>
    <h2>Verdicts that changed</h2><div id="verdicts"></div>
    <h2>Files added, removed or re-encoded</h2><div id="files"></div></div>`);
  app.replaceChildren(el);
  const controls = el.querySelector(".controls");
  controls.innerHTML =
    select("a", opts, a, { label: "commit A (before)" }) +
    select("b", opts, b, { label: "commit B (after)" }) +
    select("suite", suites.map((s) => [s, s || "all suites"]), suite, { label: "suite" }) +
    select("metric", metrics, metric, { label: "metric" });
  wireControls(controls);

  const cmp = compareRuns(ra, rb, metric);
  const rows = cmp.rows.filter((r) => (!suite || r.suite === suite) && r.a != null && r.b != null);
  const ta = rows.reduce((s, r) => s + r.a, 0), tb = rows.reduce((s, r) => s + r.b, 0);
  el.querySelector("#tiles").innerHTML =
    tile(`Total ${metric}, A`, fmtNum(ta, 3), `${rows.length} shared files`) +
    tile(`Total ${metric}, B`, fmtNum(tb, 3), fmtChange(ta > 0 ? tb / ta : null), tb > ta * 1.05 ? "bad" : tb < ta * 0.95 ? "good" : "") +
    tile("Geomean per-file ratio", fmtRatio(geomean(rows.map((r) => r.ratio))), "B / A");

  const pos = rows.filter((r) => r.a > 0 && r.b > 0);
  const lo = Math.min(...pos.map((r) => Math.min(r.a, r.b))), hi = Math.max(...pos.map((r) => Math.max(r.a, r.b)));
  const allSuites = [...new Set(rows.map((r) => r.suite))].sort();
  // Color by suite only when few enough to tell apart (≤3, the all-pairs
  // limit of the palette); otherwise one color, suite in the tooltip.
  const bySuite = allSuites.length <= 3;
  const groups = bySuite ? allSuites.map((s, i) => [s, css(SERIES[i]), pos.filter((r) => r.suite === s)]) : [["files", css("--s1"), pos]];
  const traces = groups.map(([name, color, rs]) => ({
    x: rs.map((r) => r.a), y: rs.map((r) => r.b), text: rs.map((r) => `${r.key}<br>${fmtChange(r.ratio)}`),
    type: "scatter", mode: "markers", name,
    marker: { color, size: 9, line: { color: css("--surface"), width: 2 } },
    hovertemplate: "%{text}<br>A %{x:.4g} · B %{y:.4g}<extra></extra>",
  }));
  if (pos.length) {
    traces.push({ x: [lo, hi], y: [lo, hi], type: "scatter", mode: "lines", name: "y = x", line: { color: css("--muted"), width: 1, dash: "dot" }, hoverinfo: "skip" });
  }
  plot(el.querySelector("#chart"), traces, {
    xaxis: { type: "log", title: { text: `A ${short(a)}` } },
    yaxis: { type: "log", title: { text: `B ${short(b)}` } },
    showlegend: true,
  });

  // Members, matched by (suite, stem, member) within shared, unchanged files.
  const mrows = [];
  const verdicts = [];
  for (const r of cmp.rows.filter((r) => !suite || r.suite === suite)) {
    const ma = new Map((r.fa.members || []).map((m) => [m.name, m]));
    for (const m of r.fb.members || []) {
      const o = ma.get(m.name);
      if (!o) continue;
      if (o.helium !== m.helium) verdicts.push({ key: r.key, name: m.name, a: o.helium, b: m.helium });
      if (o.time > 0 && m.time > 0) mrows.push({ key: r.key, name: m.name, a: o.time, b: m.time, ratio: m.time / o.time });
    }
  }
  const minTime = parseFloat(params.get("min") || "0.001");
  el.querySelector("#mctl").innerHTML = select("min", [["0", "all"], ["0.001", "≥ 1 ms"], ["0.01", "≥ 10 ms"], ["0.1", "≥ 100 ms"]], String(params.get("min") || "0.001"), { label: "members slower than" });
  wireControls(el.querySelector("#mctl"));
  const shown = mrows.filter((r) => Math.max(r.a, r.b) >= minTime).sort((x, y) => Math.abs(Math.log(y.ratio)) - Math.abs(Math.log(x.ratio)));
  el.querySelector("#members").innerHTML =
    table(
      [
        { label: "file", get: (r) => r.key },
        { label: "member", get: (r) => r.name },
        { label: "A", num: true, get: (r) => fmtSecs(r.a) },
        { label: "B", num: true, get: (r) => fmtSecs(r.b) },
        { label: "change", num: true, html: (r) => `<span class="${r.ratio > 1 ? "bad" : "good"}">${fmtChange(r.ratio)}</span>` },
      ],
      shown.slice(0, 100)
    ) + (shown.length > 100 ? `<p class="muted">${shown.length - 100} more not shown.</p>` : "");
  el.querySelector("#verdicts").innerHTML = table(
    [
      { label: "file", get: (r) => r.key },
      { label: "member", get: (r) => r.name },
      { label: "A", get: (r) => r.a },
      { label: "B", html: (r) => `<span class="${r.b === "OK" ? "good" : "bad"}">${esc(r.b)}</span>` },
    ],
    verdicts
  );
  const inSuite = (k) => !suite || k.startsWith(suite + "/");
  el.querySelector("#files").innerHTML = table(
    [{ label: "file", get: (r) => r[0] }, { label: "", get: (r) => r[1] }],
    [
      ...cmp.added.filter(inSuite).map((k) => [k, "added in B"]),
      ...cmp.removed.filter(inSuite).map((k) => [k, "removed in B"]),
      ...cmp.changedInput.filter(inSuite).map((k) => [k, "inputs changed (not compared)"]),
    ]
  );
}

async function pageScaling(params) {
  const cur = await loadRun(latest().commit);
  const families = [];
  for (const [suite, info] of Object.entries(cur.suites || {})) for (const f of info.families || []) families.push({ suite, ...f, key: `${suite}/${f.name}` });
  const el = $(`<div><h1>Scaling</h1><div class="controls"></div><div class="card"><div class="chart" id="chart"></div><p class="muted" id="note"></p></div>
    <p class="muted">Time against one knob with the others held fixed, log–log. One line per commit, or, with <em>lines: tools</em>, one line per tool (rustc, Helium, Silicon) at one commit. Every time is the one the tool reports itself (rustc's <code>-Z time-passes</code> total, Helium's pipeline total, Silicon's summary line), so process and JVM startup are not in it. The legend gives each line's fitted power-law exponent k (time ∝ knob^k); a straight line on these axes is polynomial, an upward bend exponential.
    With <em>baseline: on</em>, each line's value at the smallest knob is subtracted from all its points, so a fixed cost (Prusti's prelude) drops out and only the growth is left; the y-axis is then linear, since the first point is zero, and k is fitted to that growth.</p>
    <h2>Fitted exponents</h2><div id="fits"></div></div>`);
  app.replaceChildren(el);
  if (!families.length) {
    el.querySelector(".card").innerHTML = `<p class="muted">No generated families in the latest run. Families come from a suite's <code>suite.json</code>.</p>`;
    return;
  }
  const famKey = families.some((f) => f.key === params.get("family")) ? params.get("family") : families[0].key;
  const fam = families.find((f) => f.key === famKey);
  const knob = fam.knobs.includes(params.get("knob")) ? params.get("knob") : fam.knobs[0];
  const metrics = ["times.helium_verify", "times.silicon_verify", "times.rustc_self", "stats.prove_probe", "stats.sat_iterations", "stats.egraph_nodes_peak", "viper_metrics.loc"];
  const metric = metrics.includes(params.get("metric")) ? params.get("metric") : metrics[0];
  const byTool = params.get("lines") === "tools";
  const baseline = params.get("baseline") === "on";
  let chosen = params.getAll("commit");
  if (!chosen.length) chosen = [latest().commit, previousOf(latest().commit)?.commit].filter(Boolean);
  // One commit when comparing tools, so every line comes from the same build.
  chosen = chosen.slice(0, byTool ? 1 : 8);
  const loaded = await Promise.all(chosen.map((c) => loadRun(c)));

  // A run's files in this family, with their knob values. The latest run's
  // pattern is applied to every run's stems, so runs measured before the
  // family was declared in suite.json chart too.
  let re = null;
  try {
    if (fam.pattern) re = new RegExp(fam.pattern.replace(/\(\?P</g, "(?<"));
  } catch (_) {}
  const famFiles = (r) =>
    r.files
      .filter((f) => f.suite === fam.suite)
      .map((f) => {
        if (!re) return f.family === fam.name && f.knobs ? f : null;
        const m = re.exec(f.stem);
        if (!m) return null;
        const knobs = Object.fromEntries(fam.knobs.map((k) => [k, Number(m.groups?.[k])]));
        return { ...f, knobs };
      })
      .filter(Boolean);
  // Other knobs: pick one fixed value for each.
  const others = fam.knobs.filter((k) => k !== knob);
  const fixed = {};
  let otherControls = "";
  for (const k of others) {
    const vals = [...new Set(famFiles(cur).map((f) => f.knobs[k]))].sort((x, y) => x - y);
    const v = vals.map(String).includes(params.get("fix_" + k)) ? params.get("fix_" + k) : String(vals[0]);
    fixed[k] = Number(v);
    otherControls += select("fix_" + k, vals.map(String), v, { label: `${k} fixed at` });
  }
  const controls = el.querySelector(".controls");
  controls.innerHTML =
    select("family", families.map((f) => [f.key, f.key]), famKey, { label: "family" }) +
    select("knob", fam.knobs, knob, { label: "x: knob" }) +
    otherControls +
    select("lines", [["commits", "commits"], ["tools", "tools (rustc, Helium, Silicon)"]], byTool ? "tools" : "commits", { label: "lines" }) +
    (byTool ? "" : select("metric", metrics, metric, { label: "y: metric" })) +
    select("baseline", [["off", "off"], ["on", "on: subtract smallest-knob value"]], baseline ? "on" : "off", { label: "baseline" }) +
    (byTool
      ? select("commit", commitOptions(), chosen[0], { label: "commit" })
      : select("commit", commitOptions(), chosen, { multiple: true, label: "commits (up to 8)" }));
  wireControls(controls);
  controls.querySelector('select[data-param="family"]').addEventListener("change", () => setParams({ knob: null }));

  // One line's points, sorted by knob; with a baseline, the value at the
  // smallest knob is subtracted (`abs` keeps the measured value for hover).
  const points = (r, path) => {
    const pts = famFiles(r)
      .filter((f) => others.every((k) => f.knobs[k] === fixed[k]))
      .map((f) => ({ x: f.knobs[knob], y: fileValue(f, path), f }))
      .filter((p) => p.y != null && p.y > 0 && p.x > 0)
      .sort((p, q) => p.x - q.x)
      .map((p) => ({ ...p, abs: p.y }));
    if (baseline && pts.length) {
      const y0 = pts[0].abs;
      for (const p of pts) p.y = p.abs - y0;
    }
    return pts;
  };
  // Power-law exponent: slope of log y against log x, over the positive
  // points (with a baseline, the growth above the first point).
  const fitK = (pts) => {
    const ps = pts.filter((p) => p.y > 0);
    if (new Set(ps.map((p) => p.x)).size < 2) return null;
    const lx = ps.map((p) => Math.log(p.x)), ly = ps.map((p) => Math.log(p.y));
    const mx = lx.reduce((a, b) => a + b) / lx.length, my = ly.reduce((a, b) => a + b) / ly.length;
    const sxx = lx.reduce((a, x) => a + (x - mx) ** 2, 0);
    return sxx ? lx.reduce((a, x, j) => a + (x - mx) * (ly[j] - my), 0) / sxx : null;
  };
  const trace = (pts, label, i) => {
    const k = fitK(pts);
    const color = css(SERIES[i % SERIES.length]);
    return {
      x: pts.map((p) => p.x), y: pts.map((p) => p.y), text: pts.map((p) => p.f.stem), customdata: pts.map((p) => p.abs),
      type: "scatter", mode: "lines+markers",
      name: `${label}${k != null ? ` · k = ${k.toFixed(2)}` : ""}`,
      line: { color, width: 2 }, marker: { color, size: 8, line: { color: css("--surface"), width: 2 } },
      hovertemplate:
        "%{text}<br>" + knob + " %{x}<br>" + (baseline ? "+%{y:.4g} over the first point (measured %{customdata:.4g})" : "%{y:.4g}") + "<extra>" + esc(label) + "</extra>",
    };
  };

  const TOOLS = [
    ["times.rustc_self", "rustc"],
    ["times.helium_verify", "Helium"],
    ["times.silicon_verify", "Silicon"],
  ];
  let traces;
  const missing = [];
  if (byTool) {
    traces = [];
    TOOLS.forEach(([path, label], i) => {
      const pts = points(loaded[0], path);
      if (pts.length) traces.push(trace(pts, label, i));
      else missing.push(label);
    });
  } else {
    traces = loaded.map((r, i) => trace(points(r, metric), short(chosen[i]), i));
  }
  el.querySelector("#note").textContent = missing.length
    ? `No ${missing.join(" or ")} times in this run for this family${missing.includes("Silicon") ? " (Silicon runs only when a jar is configured)" : ""}.`
    : "";
  const yName = byTool ? "time (s)" : metric;
  plot(el.querySelector("#chart"), traces, {
    xaxis: { type: "log", title: { text: knob } },
    yaxis: baseline ? { type: "linear", title: { text: `${yName}, minus its value at the smallest ${knob}` } } : { type: "log", title: { text: yName } },
    showlegend: true,
  });

  // Recorded fits (from run.py) for this family, across all commits.
  const keyPrefix = `${fam.suite}/${fam.name}/`;
  el.querySelector("#fits").innerHTML = table(
    [
      { label: "commit", get: (e) => commitLabel(e) },
      ...fam.knobs.map((kn) => ({ label: `${kn}: exponent`, num: true, get: (e) => fmtNum(e.scaling?.[keyPrefix + kn]?.exponent) })),
      ...fam.knobs.map((kn) => ({ label: `${kn}: best fit`, get: (e) => { const s = e.scaling?.[keyPrefix + kn]; return s ? (s.best === "exp" ? `exponential ×${s.factor.toFixed(2)}/step` : "power law") : "–"; } })),
    ],
    [...runs()].reverse()
  );
}

async function pageExplore(params) {
  const commit = params.get("commit") || latest().commit;
  const run = await loadRun(commit);
  const level = params.get("level") === "file" ? "file" : "member";
  let items, xPaths, yPaths;
  if (level === "member") {
    items = run.files.flatMap((f) => (f.members || []).filter((m) => m.rust_metrics || m.viper_metrics).map((m) => ({ f, m })));
    const ms = items.map((i) => i.m);
    xPaths = [...numericPaths(ms, "rust_metrics").map((p) => p), ...numericPaths(ms, "viper_metrics")];
    yPaths = ["time", ...xPaths];
  } else {
    items = run.files.map((f) => ({ f }));
    xPaths = fileMetricPaths(run).filter((p) => !p.startsWith("times."));
    yPaths = fileMetricPaths(run);
  }
  const x = xPaths.includes(params.get("x")) ? params.get("x") : xPaths.includes("rust_metrics.loc") ? "rust_metrics.loc" : xPaths[0];
  const y = yPaths.includes(params.get("y")) ? params.get("y") : level === "member" ? "time" : "times.helium_verify";
  const suites = [...new Set(items.map((i) => i.f.suite))].sort();
  const hl = params.get("highlight") || "";

  const el = $(`<div><h1>Explore</h1><div class="controls"></div>
    <div class="tiles" id="tiles"></div>
    <div class="card"><div class="chart" id="chart"></div></div>
    <p class="muted">${level === "member" ? "One point per Rust function (its Viper method), joined by name" : "One point per file"}.
    Spearman's ρ measures monotone association; it does not say which metric drives time: compare with the verifier counters (stats.*) at file level.</p></div>`);
  app.replaceChildren(el);
  const controls = el.querySelector(".controls");
  controls.innerHTML =
    select("commit", commitOptions(), commit, { label: "commit" }) +
    select("level", [["member", "per member"], ["file", "per file"]], level, { label: "level" }) +
    select("x", xPaths, x, { label: "x" }) +
    select("y", yPaths, y, { label: "y" }) +
    select("highlight", [["", "all suites alike"], ...suites.map((s) => [s, s])], hl, { label: "highlight suite" }) +
    `<label>axes<select data-param="log">${["loglog", "linear", "logy"].map((v) => `<option${(params.get("log") || "loglog") === v ? " selected" : ""}>${v}</option>`).join("")}</select></label>`;
  wireControls(controls);

  const value = (i, p) => (level === "member" ? (p === "time" ? i.m.time : memberValue(i.m, p)) : fileValue(i.f, p));
  const pts = items.map((i) => ({ i, x: value(i, x), y: value(i, y) })).filter((p) => p.x != null && p.y != null);
  const rho = spearman(pts.map((p) => p.x), pts.map((p) => p.y));
  el.querySelector("#tiles").innerHTML = tile("Spearman ρ", rho == null ? "–" : rho.toFixed(2), `${pts.length} points`) + tile("Commit", `<code>${esc(short(commit))}</code>`, esc(runEntry(commit)?.subject || ""));

  const name = (p) => (level === "member" ? `${fileKey(p.i.f)} · ${p.i.m.rust_fn || p.i.m.name}` : fileKey(p.i.f));
  const mk = (ps, nm, color, size) => ({
    x: ps.map((p) => p.x), y: ps.map((p) => p.y), text: ps.map(name), name: nm,
    type: "scatter", mode: "markers",
    marker: { color, size, line: { color: css("--surface"), width: 1 } },
    hovertemplate: "%{text}<br>x %{x:.4g} · y %{y:.4g}<extra></extra>",
  });
  const traces = hl
    ? [mk(pts.filter((p) => p.i.f.suite !== hl), "other suites", css("--other"), 7), mk(pts.filter((p) => p.i.f.suite === hl), hl, css("--s1"), 9)]
    : [mk(pts, "members", css("--s1"), 8)];
  const log = params.get("log") || "loglog";
  const pos = (axis) => pts.every((p) => p[axis] > 0);
  plot(el.querySelector("#chart"), traces, {
    showlegend: !!hl,
    xaxis: { type: log === "loglog" && pos("x") ? "log" : "linear", title: { text: x } },
    yaxis: { type: log !== "linear" && pos("y") ? "log" : "linear", title: { text: y } },
  });
}

// ── Router ──

const PAGES = { overview: pageOverview, trends: pageTrends, compare: pageCompare, scaling: pageScaling, explore: pageExplore };
let renderSeq = 0;
async function render() {
  const seq = ++renderSeq;
  const { route, params } = parseHash();
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
  if (!state.index.runs.length) {
    app.innerHTML = `<h1>No runs yet</h1><p class="muted">Run <code>python3 tools/bench/run.py</code> on the benchmark host.</p>`;
    return;
  }
  try {
    await (PAGES[route] || pageOverview)(params);
  } catch (e) {
    if (seq === renderSeq) app.innerHTML = `<h1>Error</h1><pre>${esc(e.stack || e)}</pre>`;
  }
}

document.getElementById("theme").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (_) {}
  render();
});
try {
  const t = localStorage.getItem("theme");
  if (t) document.documentElement.dataset.theme = t;
} catch (_) {}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
window.addEventListener("hashchange", render);

loadIndex()
  .then(render)
  .catch((e) => {
    app.innerHTML = `<h1>No data</h1><p>Could not load <code>${esc(DATA)}index.json</code>: ${esc(e.message)}</p>`;
  });
