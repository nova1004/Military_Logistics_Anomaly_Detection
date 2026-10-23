/* =========================================================
   Logistics Anomaly Detection Dashboard - vanilla JavaScript
   Reads shipment_anomaly_results.csv (auto-load or upload).
   No data is invented: every number comes from the loaded CSV.
   ========================================================= */

"use strict";

// ---------- Settings ----------
const DATA_FILE = "shipment_anomaly_results.csv";

// Table columns: key must match the CSV header written by train_model.py
const COLUMNS = [
  { key: "shipment_id",            label: "Shipment ID",   type: "text" },
  { key: "category",               label: "Category",      type: "text" },
  { key: "shipment_quantity",      label: "Qty",           type: "num", dec: 0 },
  { key: "route_distance_km",      label: "Distance (km)", type: "num", dec: 1 },
  { key: "expected_delivery_days", label: "Exp. days",     type: "num", dec: 2 },
  { key: "actual_delivery_days",   label: "Actual days",   type: "num", dec: 2 },
  { key: "delivery_delay_days",    label: "Delay (days)",  type: "num", dec: 2 },
  { key: "expected_inventory",     label: "Exp. inv.",     type: "num", dec: 0 },
  { key: "recorded_inventory",     label: "Rec. inv.",     type: "num", dec: 0 },
  { key: "inventory_discrepancy",  label: "Discrepancy",   type: "num", dec: 0 },
  { key: "prediction",             label: "Status",        type: "status" },
  { key: "anomaly_score",          label: "Score",         type: "num", dec: 4 },
];

// Column order used in the downloaded risk report (same as logistics_risk_report.csv)
const RISK_COLUMNS = [
  "shipment_id", "category", "shipment_quantity", "route_distance_km",
  "expected_delivery_days", "actual_delivery_days", "delivery_delay_days",
  "expected_inventory", "recorded_inventory", "inventory_discrepancy",
  "prediction", "anomaly_score",
];

const TITLES = {
  dashboard: "Dashboard",
  explorer: "Shipment Explorer",
  analysis: "Anomaly Analysis",
  reports: "Reports",
};

// ---------- State ----------
let allRows = [];          // {values: {...numbers/text}, raw: {...original strings}}
let headers = [];          // headers of the loaded CSV
let sourceName = "";
let sourceKind = "";       // "auto" or "upload"
const state = {
  search: "",
  category: "all",
  status: "all",
  sortKey: "anomaly_score",
  sortDir: "asc",
  page: 1,
  pageSize: 25,
};

// ---------- Small helpers ----------
const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Rounds with toFixed (same result as the Python report) and adds thousands commas
function fmt(value, dec) {
  const [whole, fraction] = Number(value).toFixed(dec).split(".");
  const negative = whole.startsWith("-");
  const digits = negative ? whole.slice(1) : whole;
  const grouped = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return (negative ? "-" : "") + grouped + (fraction ? "." + fraction : "");
}

function showMessage(text, kind) {
  const box = $("message");
  box.textContent = text;
  box.className = "message" + (kind ? " " + kind : "");
  box.hidden = false;
}

function hideMessage() {
  $("message").hidden = true;
}

// ---------- CSV parsing (handles quotes, commas in fields, CRLF, BOM) ----------
function parseCSV(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field); field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else {
      field += ch;
    }
  }
  row.push(field);
  if (row.length > 1 || row[0] !== "") rows.push(row);
  return rows;
}

// ---------- Load + validate ----------
function loadFromText(text, name, kind) {
  const table = parseCSV(text);
  if (table.length < 2) {
    throw new Error("The file is empty or has no data rows.");
  }

  const head = table[0].map((h) => h.trim());
  const missing = RISK_COLUMNS.filter((c) => !head.includes(c));
  if (missing.length) {
    throw new Error(
      "This does not look like shipment_anomaly_results.csv. Missing column(s): " + missing.join(", ") + "."
    );
  }

  const index = {};
  head.forEach((h, i) => { index[h] = i; });

  const parsed = [];
  for (let r = 1; r < table.length; r++) {
    const cells = table[r];
    const raw = {};
    head.forEach((h, i) => { raw[h] = (cells[i] ?? "").trim(); });

    const values = {};
    for (const col of COLUMNS) {
      const v = raw[col.key];
      if (col.type === "num") {
        const n = Number(v);
        if (v === "" || !Number.isFinite(n)) {
          throw new Error("Row " + (r + 1) + ": '" + col.key + "' is not a valid number (" + (v || "empty") + ").");
        }
        values[col.key] = n;
      } else {
        values[col.key] = v;
      }
    }
    if (values.prediction !== "Anomaly" && values.prediction !== "Normal") {
      throw new Error("Row " + (r + 1) + ": 'prediction' must be Anomaly or Normal (found '" + values.prediction + "').");
    }
    parsed.push({ values, raw });
  }

  allRows = parsed;
  headers = head;
  sourceName = name;
  sourceKind = kind;
}

async function autoLoad() {
  try {
    const response = await fetch(DATA_FILE, { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const text = await response.text();
    loadFromText(text, DATA_FILE, "auto");
    hideMessage();
    onDataLoaded();
  } catch (err) {
    // Common reasons: page opened with file:// (browser blocks fetch) or file not found.
    const isFile = location.protocol === "file:";
    $("emptyText").textContent = isFile
      ? "Chrome does not let a page opened directly from your disk read local files automatically. Choose shipment_anomaly_results.csv below to load it, or start the page with VS Code Live Server for automatic loading."
      : "Could not load " + DATA_FILE + " from the project folder (" + err.message + "). Make sure index.html is saved in the same folder as that file, or choose the file below.";
    $("emptyState").hidden = false;
    $("sourceChip").textContent = "waiting for file";
    $("sourceChip").className = "chip warn";
    renderAll();
  }
}

function handleFile(file) {
  if (!file) return;
  if (!/\.csv$/i.test(file.name)) {
    showMessage("Please choose a .csv file (for example shipment_anomaly_results.csv).", "error");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    try {
      loadFromText(String(reader.result), file.name, "upload");
      showMessage("Loaded " + allRows.length + " shipments from " + file.name + ".", "success");
      onDataLoaded();
    } catch (err) {
      showMessage(err.message, "error");
    }
  };
  reader.onerror = () => showMessage("The file could not be read. Please try again.", "error");
  reader.readAsText(file);
}

function onDataLoaded() {
  $("emptyState").hidden = true;
  const chip = $("sourceChip");
  chip.textContent = sourceName + " (" + allRows.length + " rows)";
  chip.className = "chip ok";
  $("graphNote").hidden = sourceKind !== "upload";

  // Fill the category dropdown from the data
  const select = $("categorySelect");
  const categories = [...new Set(allRows.map((r) => r.values.category))].sort();
  select.textContent = "";
  const optAll = el("option", "", "All categories");
  optAll.value = "all";
  select.appendChild(optAll);
  categories.forEach((c) => {
    const o = el("option", "", c);
    o.value = c;
    select.appendChild(o);
  });
  state.category = "all";
  state.search = "";
  state.status = "all";
  state.page = 1;
  $("searchInput").value = "";
  document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b.dataset.status === "all"));

  renderAll();
}

// ---------- Rendering ----------
function renderAll() {
  renderMetrics();
  renderTopFlagged();
  renderCategoryBars();
  renderExplorer();
  renderCompare();
  renderReports();
}

function flaggedRows() {
  return allRows.filter((r) => r.values.prediction === "Anomaly");
}

function renderMetrics() {
  const total = allRows.length;
  const flagged = flaggedRows().length;
  const has = total > 0;
  $("mTotal").textContent = has ? total.toLocaleString("en-US") : "—";
  $("mFlagged").textContent = has ? flagged.toLocaleString("en-US") : "—";
  $("mRate").textContent = has ? ((flagged / total) * 100).toFixed(1) + "%" : "—";
  $("mNormal").textContent = has ? (total - flagged).toLocaleString("en-US") : "—";
}

function fillRowCells(tr, cells) {
  cells.forEach(([text, isNum]) => tr.appendChild(el("td", isNum ? "num" : "", text)));
}

function emptyRow(tbody, colSpan, text) {
  const tr = el("tr", "empty-row");
  const td = el("td", "", text);
  td.colSpan = colSpan;
  tr.appendChild(td);
  tbody.appendChild(tr);
}

function renderTopFlagged() {
  const tbody = document.querySelector("#topTable tbody");
  tbody.textContent = "";
  const top = flaggedRows()
    .sort((a, b) => a.values.anomaly_score - b.values.anomaly_score)
    .slice(0, 5);
  if (!top.length) {
    emptyRow(tbody, 5, allRows.length ? "No flagged shipments in this file." : "No data loaded.");
    return;
  }
  top.forEach((r) => {
    const v = r.values;
    const tr = el("tr");
    fillRowCells(tr, [
      [v.shipment_id, false], [v.category, false],
      [fmt(v.delivery_delay_days, 2), true], [fmt(v.route_distance_km, 1), true],
      [fmt(v.anomaly_score, 4), true],
    ]);
    tbody.appendChild(tr);
  });
}

function renderCategoryBars() {
  const box = $("categoryBars");
  box.textContent = "";
  if (!allRows.length) {
    box.appendChild(el("p", "muted small", "No data loaded."));
    return;
  }
  const stats = {};
  allRows.forEach((r) => {
    const c = r.values.category;
    stats[c] = stats[c] || { total: 0, flagged: 0 };
    stats[c].total++;
    if (r.values.prediction === "Anomaly") stats[c].flagged++;
  });
  const entries = Object.entries(stats).sort((a, b) => b[1].flagged - a[1].flagged);
  const maxFlagged = Math.max(1, ...entries.map((e) => e[1].flagged));

  entries.forEach(([name, s]) => {
    const row = el("div", "bar-row");
    const top = el("div", "bar-row-top");
    top.appendChild(el("span", "", name));
    const pct = ((s.flagged / s.total) * 100).toFixed(1);
    top.appendChild(el("span", "", s.flagged + " flagged of " + s.total + " (" + pct + "%)"));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = (s.flagged / maxFlagged) * 100 + "%";
    track.appendChild(fill);
    row.appendChild(top);
    row.appendChild(track);
    box.appendChild(row);
  });
}

// ----- Explorer -----
function buildHeader() {
  const tr = $("headRow");
  tr.textContent = "";
  COLUMNS.forEach((col) => {
    const th = el("th", col.type === "num" ? "num" : "");
    th.scope = "col";
    const btn = el("button", "sort-btn");
    btn.type = "button";
    btn.appendChild(document.createTextNode(col.label));
    const arrow = el("span", "arrow");
    if (state.sortKey === col.key) {
      arrow.textContent = state.sortDir === "asc" ? "▲" : "▼";
      th.setAttribute("aria-sort", state.sortDir === "asc" ? "ascending" : "descending");
    }
    btn.appendChild(arrow);
    btn.addEventListener("click", () => {
      if (state.sortKey === col.key) {
        state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = col.key;
        state.sortDir = "asc";
      }
      state.page = 1;
      renderExplorer();
    });
    th.appendChild(btn);
    tr.appendChild(th);
  });
}

function filteredRows() {
  const q = state.search.trim().toLowerCase();
  let rows = allRows.filter((r) => {
    const v = r.values;
    if (state.status !== "all" && v.prediction !== state.status) return false;
    if (state.category !== "all" && v.category !== state.category) return false;
    if (q && !(v.shipment_id.toLowerCase().includes(q) || v.category.toLowerCase().includes(q))) return false;
    return true;
  });

  const col = COLUMNS.find((c) => c.key === state.sortKey) || COLUMNS[0];
  const dir = state.sortDir === "asc" ? 1 : -1;
  rows = rows.slice().sort((a, b) => {
    const x = a.values[col.key];
    const y = b.values[col.key];
    const cmp = col.type === "num" ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true });
    return cmp * dir;
  });
  return rows;
}

function renderExplorer() {
  buildHeader();
  const body = $("dataBody");
  body.textContent = "";

  if (!allRows.length) {
    emptyRow(body, COLUMNS.length, "No data loaded. Upload shipment_anomaly_results.csv to begin.");
    $("resultLine").textContent = "No data loaded.";
    $("pageInfo").textContent = "Page 1 of 1";
    $("prevBtn").disabled = true;
    $("nextBtn").disabled = true;
    return;
  }

  const rows = filteredRows();
  const pages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  if (state.page > pages) state.page = pages;
  const start = (state.page - 1) * state.pageSize;
  const slice = rows.slice(start, start + state.pageSize);

  if (!slice.length) {
    emptyRow(body, COLUMNS.length, "No shipments match your search and filters.");
  }

  slice.forEach((r) => {
    const v = r.values;
    const tr = el("tr", v.prediction === "Anomaly" ? "flagged" : "");
    COLUMNS.forEach((col) => {
      const td = el("td", col.type === "num" ? "num" : "");
      if (col.type === "status") {
        td.appendChild(el("span", "badge " + (v.prediction === "Anomaly" ? "anomaly" : "normal"), v.prediction));
      } else if (col.type === "num") {
        td.textContent = fmt(v[col.key], col.dec);
      } else {
        td.textContent = v[col.key];
      }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });

  const shownFrom = rows.length ? start + 1 : 0;
  const shownTo = Math.min(start + state.pageSize, rows.length);
  $("resultLine").textContent =
    "Showing " + shownFrom + "–" + shownTo + " of " + rows.length +
    " matching shipments (" + allRows.length + " total).";
  $("pageInfo").textContent = "Page " + state.page + " of " + pages;
  $("prevBtn").disabled = state.page <= 1;
  $("nextBtn").disabled = state.page >= pages;
}

// ----- Analysis: means of normal vs flagged -----
function renderCompare() {
  const tbody = document.querySelector("#compareTable tbody");
  tbody.textContent = "";
  if (!allRows.length) { emptyRow(tbody, 3, "No data loaded."); return; }

  const features = [
    ["shipment_quantity", "Shipment quantity", 2],
    ["route_distance_km", "Route distance (km)", 2],
    ["delivery_delay_days", "Delivery delay (days)", 2],
    ["inventory_discrepancy", "Inventory discrepancy", 2],
  ];
  const flagged = allRows.filter((r) => r.values.prediction === "Anomaly");
  const normal = allRows.filter((r) => r.values.prediction === "Normal");
  const mean = (rows, key) => rows.length ? rows.reduce((s, r) => s + r.values[key], 0) / rows.length : null;

  features.forEach(([key, label, dec]) => {
    const n = mean(normal, key);
    const f = mean(flagged, key);
    const tr = el("tr");
    fillRowCells(tr, [
      [label, false],
      [n === null ? "n/a" : fmt(n, dec), true],
      [f === null ? "n/a" : fmt(f, dec), true],
    ]);
    tbody.appendChild(tr);
  });
}

// ----- Reports -----
function sortedFlagged() {
  return flaggedRows().sort((a, b) => a.values.anomaly_score - b.values.anomaly_score);
}

function renderReports() {
  const flagged = sortedFlagged();
  const has = allRows.length > 0;
  $("reportCount").textContent = has ? flagged.length : "—";
  $("fullCount").textContent = has ? allRows.length : "—";
  $("riskDownloadBtn").disabled = !has;
  $("topDownloadBtn").disabled = !has;
  $("fullDownloadBtn").disabled = !has;

  const tbody = document.querySelector("#previewTable tbody");
  tbody.textContent = "";
  if (!has) { emptyRow(tbody, 7, "No data loaded."); return; }
  if (!flagged.length) { emptyRow(tbody, 7, "No flagged shipments in this file."); return; }

  const preview = flagged.slice(0, 10);
  $("previewSub").textContent = "Showing " + preview.length + " of " + flagged.length + " flagged shipments (the download contains all of them).";
  preview.forEach((r) => {
    const v = r.values;
    const tr = el("tr");
    fillRowCells(tr, [
      [v.shipment_id, false], [v.category, false],
      [fmt(v.shipment_quantity, 0), true], [fmt(v.route_distance_km, 1), true],
      [fmt(v.delivery_delay_days, 2), true], [fmt(v.inventory_discrepancy, 0), true],
      [fmt(v.anomaly_score, 4), true],
    ]);
    tbody.appendChild(tr);
  });
}

// ---------- CSV download ----------
function csvEscape(value) {
  const s = String(value);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function downloadCSV(filename, columns, rows) {
  const lines = [columns.join(",")];
  rows.forEach((r) => lines.push(columns.map((c) => csvEscape(r.raw[c] ?? "")).join(",")));
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadRisk() {
  if (!allRows.length) return;
  downloadCSV("logistics_risk_report.csv", RISK_COLUMNS, sortedFlagged());
}

function downloadFull() {
  if (!allRows.length) return;
  downloadCSV("shipment_anomaly_results.csv", headers, allRows);
}

// ---------- Navigation ----------
function showView(name) {
  if (!TITLES[name]) name = "dashboard";
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== "view-" + name; });
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $("pageTitle").textContent = TITLES[name];
  document.title = TITLES[name] + " | Logistics Anomaly Detection";
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
}

// ---------- Lightbox ----------
function openLightbox(src, title) {
  const dlg = $("lightbox");
  $("lightboxImg").src = src;
  $("lightboxImg").alt = title;
  $("lightboxTitle").textContent = title;
  if (typeof dlg.showModal === "function") dlg.showModal();
}

// ---------- Wire up events ----------
function init() {
  // Navigation
  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.addEventListener("click", () => showView(b.dataset.view)));
  window.addEventListener("hashchange", () => showView(location.hash.slice(1)));

  // Upload
  const fileInput = $("fileInput");
  $("uploadBtn").addEventListener("click", () => fileInput.click());
  $("emptyUploadBtn").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    handleFile(fileInput.files[0]);
    fileInput.value = "";   // lets you choose the same file again
  });

  // Drag and drop
  const drop = $("dropZone");
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

  // Downloads
  $("topDownloadBtn").addEventListener("click", downloadRisk);
  $("riskDownloadBtn").addEventListener("click", downloadRisk);
  $("fullDownloadBtn").addEventListener("click", downloadFull);

  // Explorer controls
  $("searchInput").addEventListener("input", (e) => {
    state.search = e.target.value;
    state.page = 1;
    renderExplorer();
  });
  $("categorySelect").addEventListener("change", (e) => {
    state.category = e.target.value;
    state.page = 1;
    renderExplorer();
  });
  document.querySelectorAll(".seg").forEach((b) =>
    b.addEventListener("click", () => {
      state.status = b.dataset.status;
      state.page = 1;
      document.querySelectorAll(".seg").forEach((x) => x.classList.toggle("active", x === b));
      renderExplorer();
    }));
  $("resetBtn").addEventListener("click", () => {
    state.search = ""; state.category = "all"; state.status = "all";
    state.sortKey = "anomaly_score"; state.sortDir = "asc"; state.page = 1;
    $("searchInput").value = "";
    $("categorySelect").value = "all";
    document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b.dataset.status === "all"));
    renderExplorer();
  });
  $("prevBtn").addEventListener("click", () => { state.page--; renderExplorer(); });
  $("nextBtn").addEventListener("click", () => { state.page++; renderExplorer(); });

  // Graphs: lightbox + friendly message if an image is missing
  document.querySelectorAll(".fig-btn").forEach((b) =>
    b.addEventListener("click", () => openLightbox(b.dataset.src, b.dataset.title)));
  document.querySelectorAll(".figure img").forEach((img) =>
    img.addEventListener("error", () => {
      const msg = el("div", "img-missing",
        "Graph not found. Keep the visualizations folder next to index.html.");
      img.replaceWith(msg);
    }));
  $("lightboxClose").addEventListener("click", () => $("lightbox").close());
  $("lightbox").addEventListener("click", (e) => { if (e.target === $("lightbox")) $("lightbox").close(); });

  showView(location.hash.slice(1) || "dashboard");
  renderAll();
  autoLoad();
}

document.addEventListener("DOMContentLoaded", init);
