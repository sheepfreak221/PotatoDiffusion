/* ============================================================
   PotatoDiffusion - Frontend
   ============================================================ */
"use strict";

const SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
  "dpm++2m_sde", "dpm++2m_sde_bt", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
  "res_multistep", "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp"];

const SCHEDULERS = ["auto", "discrete", "karras", "exponential", "ays", "gits", "smoothstep",
  "sgm_uniform", "simple", "kl_optimal", "lcm", "bong_tangent", "ltx2", "logit_normal",
  "flux2", "flux", "beta"];

const WEIGHT_TYPES = ["", "f32", "f16", "q8_0", "q6_K", "q5_1", "q5_0", "q4_K", "q4_1", "q4_0", "q3_K", "q2_K"];

const SIZE_PRESETS = {
  sd15: [[512, 512], [512, 768], [768, 512], [768, 768], [640, 640]],
  sdxl: [[768, 768], [896, 896], [1024, 1024], [768, 1024], [1024, 768], [512, 512]],
  flux: [[512, 512], [512, 768], [768, 512], [768, 768], [1024, 1024]],
};

const PATH_FIELDS = {
  sd15: [
    { name: "model", label: "Modell (-m)" },
    { name: "clip_l", label: "Text-Encoder clip_l (optional)", hint: "nur noetig, wenn das Modell keinen enthaelt" },
  ],
  sdxl: [
    { name: "model", label: "Modell (-m)" },
    { name: "clip_l", label: "Text-Encoder clip_l (optional)" },
    { name: "clip_g", label: "Text-Encoder clip_g (optional)" },
  ],
  flux: [
    { name: "diffusion_model", label: "Diffusions-Modell (--diffusion-model)" },
    { name: "llm", label: "Text-Encoder / LLM (--llm)", hint: "z.B. Qwen3-4B-Q4_K_M.gguf" },
    { name: "vae", label: "VAE (--vae)" },
    { name: "vae_format", label: "VAE-Format", type: "select", options: ["auto", "flux", "sd3", "flux2", "wan"] },
    { name: "clip_l", label: "clip_l (optional)" },
    { name: "clip_g", label: "clip_g (optional)" },
    { name: "t5xxl", label: "t5xxl (optional)" },
  ],
};

// Reiter, die ihre Checkpoints als Dropdown anbieten (Modell-Wahl nach Stil)
const CKPT_TABS = ["sd15", "sdxl"];

const state = {
  cfg: null,
  tab: "sd15",
  checkpoints: {},
  params: { sd15: null, sdxl: null, flux: null },
  running: false,
  es: null,
  timer: null,
  startedAt: 0,
  picker: null,
  gallery: [],
  // Real-ESRGAN
  upscaleModels: [],
  upscaleSel: { model: "", scale: 4, tile: 0, gpuid: 0 },
  upscaleSrc: null,
  upscaleHistory: [],
  streamKind: "generate",
  streamModel: "sd15",
  emptyHtml: "",
};

const isUpscale = () => state.tab === "upscale";

/* ---------------------------------------------------------------- utils */

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { data = { error: "Antwort nicht lesbar" }; }
  return { ok: res.ok, status: res.status, data };
}

function post(url, body) {
  return api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

function toast(msg, type, items) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = "toast " + (type || "");
  el.innerHTML = `<div>${esc(msg)}</div>` +
    (items && items.length ? `<ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : "");
  if (type === "err") {
    const b = document.createElement("button");
    b.className = "btn ghost small";
    b.textContent = "Schließen";
    b.onclick = () => el.remove();
    el.appendChild(b);
  }
  box.appendChild(el);
  if (type !== "err") setTimeout(() => el.remove(), 4200);
}

function fmtSize(bytes) {
  if (!bytes) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + " " + units[i];
}

/* ------------------------------------------------------- Real-ESRGAN Hilfen */

function scalesFor(modelName) {
  const m = state.upscaleModels.find((x) => x.name === modelName);
  return m && m.scales && m.scales.length ? m.scales : [4];
}

function normalizeUpscaleSel(sel) {
  if (!state.upscaleModels.length) return sel;
  if (!state.upscaleModels.some((m) => m.name === sel.model)) {
    sel.model = state.upscaleModels[0].name;
  }
  const scales = scalesFor(sel.model);
  if (!scales.includes(Number(sel.scale))) sel.scale = scales[scales.length - 1];
  sel.tile = Number(sel.tile) || 0;
  sel.gpuid = Number(sel.gpuid) || 0;
  return sel;
}

/* --------------------------------------------------------- Formular */

function groupSpec(key) {
  const groups = [];

  if (CKPT_TABS.includes(key)) {
    groups.push({ title: "Modell", fields: [{ type: "ckpt" }] });
  }

  groups.push({
    title: "Prompt",
    fields: [
      { type: "textarea", name: "prompt", label: "Prompt", rows: 5, placeholder: "was soll entstehen?" },
      { type: "textarea", name: "negative_prompt", label: "Negativ-Prompt", rows: 2, placeholder: "was nicht sein soll" },
    ],
  });

  const bild = [
    { type: "size" },
    { type: "number", name: "steps", label: "Schritte", min: 1, max: 200,
      hint: key === "flux" ? "4 reichen bei FLUX" : "20 ist Standard" },
    { type: "number", name: "cfg_scale", label: "CFG Scale", min: 0, max: 30, step: 0.5,
      hint: key === "flux" ? "bei FLUX 1.0" : "7.0 Standard" },
  ];
  if (key === "flux") {
    bild.push({ type: "number", name: "guidance", label: "Guidance", min: 0, max: 20, step: 0.1,
      hint: "nur bei gedistillierten Modellen" });
  }
  bild.push(
    { type: "select", name: "sampling_method", label: "Sampler", options: SAMPLERS },
    { type: "select", name: "scheduler", label: "Scheduler", options: SCHEDULERS },
    { type: "seed" },
    { type: "pair",
      fields: [
        { type: "number", name: "batch_count", label: "Batch", min: 1, max: 8 },
        ...(key === "flux" ? [] : [{ type: "number", name: "clip_skip", label: "Clip Skip", min: -1, max: 6,
          hint: "-1 = auto" }]),
      ] },
  );
  groups.push({ title: "Bild & Sampler", fields: bild });

  groups.push({
    title: "Nachlauf · Real-ESRGAN",
    fields: [
      { type: "checkbox", name: "upscale_after", label: "Bild nach dem Generieren hochskalieren" },
      { type: "pair",
        fields: [
          { type: "modelsel", name: "upscale_model", label: "Upscale-Modell" },
          { type: "scalesel", name: "upscale_scale", label: "Faktor" },
        ] },
      { type: "pair",
        fields: [
          { type: "number", name: "upscale_tile", label: "Tile", min: 0, max: 2048, hint: "0 = automatisch" },
          { type: "number", name: "upscale_gpuid", label: "GPU", min: 0, max: 8 },
        ] },
      { type: "hint", text: "Ergebnis landet im upscale-Ordner und wird nicht in der Galerie angezeigt." },
    ],
  });

  groups.push({
    title: "Erweitert",
    adv: true,
    fields: [
      { type: "checkbox", name: "vae_tiling", label: "VAE-Tiling (weniger VRAM)" },
      { type: "checkbox", name: "offload_to_cpu", label: "Gewichte nach CPU auslagern" },
      { type: "checkbox", name: "preview", label: "Live-Vorschau während des Laufs" },
      { type: "text", name: "backend", label: "Backend-Zuweisung",
        hint: "diffusion=vulkan0,te=cpu,vae=vulkan0" },
      { type: "select", name: "weight_type", label: "Weight-Type", options: WEIGHT_TYPES, empty: "(Auto)" },
      { type: "text", name: "max_vram", label: "Max. VRAM (GiB)", hint: "z.B. 3.5 oder vulkan0=3.5" },
      { type: "number", name: "threads", label: "Threads", min: 0, max: 32, hint: "0 = automatisch" },
      { type: "text", name: "extra_args", label: "Zusätzliche Argumente", hint: "frei, z.B. --fa --mmap" },
    ],
  });

  return groups;
}

function fieldHtml(f, val) {
  const v = val == null ? "" : val;
  switch (f.type) {
    case "textarea":
      return `<label class="field"><span>${esc(f.label)}</span>
        <textarea data-name="${f.name}" rows="${f.rows || 3}" placeholder="${esc(f.placeholder || "")}">${esc(v)}</textarea></label>`;

    case "number":
      return `<label class="field"><span>${esc(f.label)}${f.hint ? `<em class="hint">${esc(f.hint)}</em>` : ""}</span>
        <input type="number" data-name="${f.name}" value="${esc(v)}"
          ${f.min != null ? `min="${f.min}"` : ""} ${f.max != null ? `max="${f.max}"` : ""}
          step="${f.step != null ? f.step : 1}"></label>`;

    case "text":
      return `<label class="field"><span>${esc(f.label)}${f.hint ? `<em class="hint">${esc(f.hint)}</em>` : ""}</span>
        <input type="text" data-name="${f.name}" value="${esc(v)}" spellcheck="false"></label>`;

    case "select": {
      const opts = (f.options || []).map((o) =>
        `<option value="${esc(o)}" ${String(v) === String(o) ? "selected" : ""}>${esc(o)}</option>`).join("");
      const empty = f.empty ? `<option value="" ${v === "" ? "selected" : ""}>${esc(f.empty)}</option>` : "";
      return `<label class="field"><span>${esc(f.label)}</span>
        <select data-name="${f.name}">${empty}${opts}</select></label>`;
    }

    case "checkbox":
      return `<label class="check ${v ? "is-on" : ""}">
        <input type="checkbox" data-name="${f.name}" ${v ? "checked" : ""}><span>${esc(f.label)}</span></label>`;

    case "pair":
      return `<div class="row c2">${f.fields.map((x) => fieldHtml(x, valFor(x.name))).join("")}</div>`;

    case "size": {
      const presets = SIZE_PRESETS[state.tab] || [];
      const chips = presets.map(([w, h]) => {
        const active = Number(valFor("width")) === w && Number(valFor("height")) === h;
        return `<button type="button" class="chip ${active ? "is-active" : ""}" data-size="${w}x${h}">${w}&times;${h}</button>`;
      }).join("");
      return `<div class="field"><span>Bildgröße<em class="hint">vielfaches von 8</em></span>
        <div class="row c2">
          <input type="number" data-name="width" value="${esc(valFor("width"))}" min="64" max="4096" step="8">
          <input type="number" data-name="height" value="${esc(valFor("height"))}" min="64" max="4096" step="8">
        </div>
        <div class="chips">${chips}</div></div>`;
    }

    case "seed": {
      const rand = !!valFor("random_seed");
      const seedVal = valFor("seed");
      const shown = rand && !(Number(seedVal) > 0) ? "" : seedVal;
      return `<div class="field"><span>Seed</span>
        <div class="seed-row">
          <input type="number" data-name="seed" value="${esc(shown)}" placeholder="zufällig"
            ${rand ? "disabled" : ""}>
          <label class="check ${rand ? "is-on" : ""}" style="margin:0;padding:7px 10px">
            <input type="checkbox" data-name="random_seed" ${rand ? "checked" : ""}><span>zufällig</span>
          </label>
          <button type="button" class="icon-btn" data-act="dice" title="Neuen Zufalls-Seed ziehen">&#127922;</button>
        </div></div>`;
    }
    case "modelsel": {
      const opts = state.upscaleModels.map((m) =>
        `<option value="${esc(m.name)}" ${String(v) === String(m.name) ? "selected" : ""}>${esc(m.label || m.name)}</option>`).join("");
      return `<label class="field"><span>${esc(f.label)}${f.hint ? `<em class="hint">${esc(f.hint)}</em>` : ""}</span>
        <select data-name="${f.name}">${opts || '<option value="">keine Modelle</option>'}</select></label>`;
    }

    case "scalesel": {
      const scales = scalesFor(valFor("upscale_model"));
      const opts = scales.map((s) =>
        `<option value="${s}" ${Number(v) === Number(s) ? "selected" : ""}>${s}&times;</option>`).join("");
      return `<label class="field"><span>${esc(f.label)}</span>
        <select data-name="${f.name}">${opts}</select></label>`;
    }

    case "ckpt": {
      const info = state.checkpoints[state.tab] || {};
      const cur = info.current || "";
      const opts = (info.models || []).map((m) =>
        `<option value="${esc(m.path)}" ${m.path === cur ? "selected" : ""}>` +
        `${esc(m.name)}${m.size ? ` · ${fmtSize(m.size)}` : ""}</option>`).join("");
      const empty = '<option value="">keine Modelle gefunden</option>';
      const shown = cur || (info.dir ? `Ordner: ${info.dir}` : "keiner eingestellt");
      return `<div class="field"><span>Checkpoint<em class="hint">Klick = Modell für diesen Reiter wechseln</em></span>
        <select id="ckptSelect">${opts || empty}</select>
        <p class="muted field-hint ckpt-path" id="ckptPath" title="${esc(shown)}">${esc(shown)}</p></div>`;
    }

    case "hint":
      return `<p class="muted field-hint">${esc(f.text || "")}</p>`;

    default:
      return "";
  }
}

let draft = null; // aktuell sichtbare Parameter (Referenz auf state.params[tab])

function valFor(name) {
  return draft && draft[name] != null ? draft[name] : "";
}

function setEmptyState(kind) {
  const box = $("#emptyState");
  if (!state.emptyHtml) state.emptyHtml = box.innerHTML;
  if (kind === "upscale") {
    box.innerHTML = '<div class="empty-glyph">&#8685;</div>' +
      '<p>Noch kein Bild.</p>' +
      '<p class="muted">Bild hochladen oder aus der Galerie w&auml;hlen<br>' +
      'und dann <b>Upscalen</b> dr&uuml;cken.</p>';
  } else {
    box.innerHTML = state.emptyHtml;
  }
}

function applyStageChrome(mode) {
  const up = mode === "upscale";
  $("#btnGenerate").innerHTML = up
    ? '<span class="btn-ico">&#8593;</span> Upscalen'
    : '<span class="btn-ico">&#9654;</span> Generieren';
  $("#seedChip").hidden = up;
  $("#galleryPanel").hidden = up;
  $("#upscalePanel").hidden = !up;
  $("#cmdSummary").textContent = up ? "realesrgan-Befehl" : "sd-cli Befehl";
  setEmptyState(up ? "upscale" : "model");
}

function normalizeDraftUpscale(d) {
  if (!d || !state.upscaleModels.length) return;
  if (!state.upscaleModels.some((m) => m.name === d.upscale_model)) {
    d.upscale_model = state.upscaleModels[0].name;
  }
  const scales = scalesFor(d.upscale_model);
  if (!scales.includes(Number(d.upscale_scale))) d.upscale_scale = scales[scales.length - 1];
}

function renderForm() {
  const key = state.tab;
  draft = state.params[key];
  normalizeDraftUpscale(draft);
  const groups = groupSpec(key);
  const html = groups.map((g) => {
    const inner = g.fields.map((f) => fieldHtml(f, valFor(f.name))).join("");
    if (g.adv) {
      return `<details class="adv group" ${draft.__adv ? "open" : ""}>
        <summary>${esc(g.title)}</summary><div class="adv-body">${inner}</div></details>`;
    }
    return `<div class="group"><h3>${esc(g.title)}</h3>${inner}</div>`;
  }).join("");

  $("#paramForm").innerHTML = html;
  $("#modelBadge").textContent = (state.cfg.models[key].label || key);
  $("#paramsTitle").textContent = "Parameter";
  applyStageChrome("model");
  bindForm();
}

function bindForm() {
  const root = $("#paramForm");

  $$("input[data-name], textarea[data-name], select[data-name]", root).forEach((inp) => {
    const name = inp.dataset.name;
    const handler = () => {
      if (inp.type === "checkbox") {
        draft[name] = inp.checked;
        inp.closest(".check").classList.toggle("is-on", inp.checked);
        if (name === "random_seed") {
          const seedInp = $('[data-name="seed"]', root);
          if (seedInp) seedInp.disabled = inp.checked;
        }
      } else if (inp.type === "number") {
        const n = parseFloat(inp.value);
        draft[name] = isNaN(n) ? inp.value : n;
      } else {
        draft[name] = inp.value;
      }
      persist();
    };
    inp.addEventListener("input", handler);
    inp.addEventListener("change", handler);
  });

  // Faktor-Anzeige an das gewaehlte Upscale-Modell anpassen
  const upModel = $('[data-name="upscale_model"]', root);
  if (upModel) {
    upModel.addEventListener("change", () => {
      normalizeDraftUpscale(draft);
      renderForm();
    });
  }

  $$("[data-act='dice']", root).forEach((btn) => {
    btn.addEventListener("click", () => {
      draft.random_seed = true;
      draft.seed = Math.floor(Math.random() * 2147483646) + 1;
      renderForm();
      persist();
    });
  });

  $$("[data-size]", root).forEach((chip) => {
    chip.addEventListener("click", () => {
      const [w, h] = chip.dataset.size.split("x").map(Number);
      draft.width = w; draft.height = h;
      renderForm();
      persist();
    });
  });

  const adv = $("details.adv", root);
  if (adv) adv.addEventListener("toggle", () => { draft.__adv = adv.open; persist(); });

  const ckpt = $("#ckptSelect", root);
  if (ckpt) ckpt.addEventListener("change", () => selectCheckpoint(state.tab, ckpt.value));
}

/* Checkpoint-Dropdown: Pfad in config.json schreiben (Pfad lebt in der Config,
   nicht in den Pro-parametern des Reiters). */
async function selectCheckpoint(tab, path) {
  if (!path) return;
  const res = await post("/api/checkpoints/select", { tab, path });
  if (!res.ok) {
    toast((res.data && res.data.error) || "Modellwechsel fehlgeschlagen", "err");
    renderForm();
    return;
  }
  state.cfg = res.data.config;
  const info = state.checkpoints[tab];
  if (info) info.current = res.data.path;
  const hint = $("#ckptPath");
  if (hint) { hint.textContent = res.data.path; hint.title = res.data.path; }
  toast("Modell gewählt: " + res.data.path.split("/").pop(), "ok");
}

async function loadCheckpoints() {
  await Promise.all(CKPT_TABS.map(async (key) => {
    const res = await api("/api/checkpoints/" + key);
    if (res.ok) state.checkpoints[key] = res.data;
  }));
}

function persist() {
  try { localStorage.setItem("sdwebui.params." + state.tab, JSON.stringify(draft)); } catch (e) { /* ignore */ }
}

function loadParams() {
  ["sd15", "sdxl", "flux"].forEach((key) => {
    const defaults = Object.assign({}, state.cfg.models[key].defaults || {});
    try {
      const saved = JSON.parse(localStorage.getItem("sdwebui.params." + key) || "null");
      if (saved && typeof saved === "object") Object.assign(defaults, saved);
    } catch (e) { /* ignore */ }
    state.params[key] = defaults;
  });
}

function clearStageImage() {
  const img = $("#resultImage");
  img.hidden = true;
  img.removeAttribute("src");
  $("#emptyState").style.display = "";
  $("#upscaleNote").hidden = true;
}

function renderTab() {
  if (state.tab === "upscale") renderUpscaleTab();
  else renderForm();
}

function switchTab(key) {
  if (!state.cfg) return;
  if (key !== "upscale" && !state.cfg.models[key]) return;
  if (state.tab === key) return;
  if (state.tab !== "upscale") persist();
  const wasRunning = state.running;
  state.tab = key;
  $$(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.key === key));
  history.replaceState(null, "", "#" + key);
  if (!wasRunning) clearStageImage();
  renderTab();
}

/* --------------------------------------------------------- Generierung */

function collectParams() {
  const out = Object.assign({}, draft);
  delete out.__adv;
  return out;
}

function setRunning(on) {
  state.running = on;
  $("#btnGenerate").disabled = on;
  $("#btnCancel").disabled = !on;
  $("#scanline").hidden = !on;
  $("#progressArea").hidden = !on;
  const pill = $("#statusPill");
  if (on) {
    pill.dataset.state = "run";
    pill.textContent = "läuft …";
    state.startedAt = Date.now();
    clearInterval(state.timer);
    state.timer = setInterval(() => {
      $("#elapsed").textContent = ((Date.now() - state.startedAt) / 1000).toFixed(1) + " s";
    }, 200);
  } else {
    clearInterval(state.timer);
    state.timer = null;
    pill.dataset.state = "idle";
    pill.textContent = "bereit";
  }
}

function addLog(line, level) {
  const log = $("#log");
  const span = document.createElement("span");
  span.className = "lvl-" + (level || "info");
  span.textContent = line + "\n";
  log.appendChild(span);
  while (log.childNodes.length > 900) log.removeChild(log.firstChild);
  log.scrollTop = log.scrollHeight;
  $("#logCount").textContent = log.childNodes.length + " Zeilen";
  const panel = $("#logPanel");
  if (!panel.open && (level === "error" || level === "warn")) panel.open = true;
}

function showImage(url, isPreview) {
  const img = $("#resultImage");
  img.src = url;
  img.hidden = false;
  img.classList.toggle("is-preview", !!isPreview);
  $("#emptyState").style.display = "none";
}

function resetStage(command) {
  $("#cmdText").textContent = command || "";
  $("#log").innerHTML = "";
  $("#logCount").textContent = "";
  $("#resultImage").hidden = true;
  $("#resultImage").removeAttribute("src");
  $("#emptyState").style.display = "";
  $("#upscaleNote").hidden = true;
  $("#progressFill").style.width = "2%";
  $("#phaseLabel").textContent = "Start";
  $("#stepLabel").textContent = "";
  $("#elapsed").textContent = "0.0 s";
}

async function generate() {
  if (state.running) return;
  const params = collectParams();
  if (!String(params.prompt || "").trim()) {
    toast("Bitte zuerst einen Prompt eingeben.", "warn");
    const inp = $('[data-name="prompt"]');
    if (inp) inp.focus();
    return;
  }

  $("#btnGenerate").disabled = true;
  const res = await post("/api/generate", { model: state.tab, params });
  const data = res.data || {};

  if (!res.ok) {
    $("#btnGenerate").disabled = false;
    const errs = data.errors || [data.error || "Unbekannter Fehler"];
    toast(state.tab.toUpperCase() + ": konnte nicht starten", "err", errs);
    if (errs.some((e) => /Zahnrad|fehlt|nicht gefunden/i.test(e))) openSettings();
    return;
  }

  if (data.warnings && data.warnings.length) toast("Hinweis", "warn", data.warnings);

  resetStage(data.command);
  $("#seedValue").textContent = data.seed;
  state.streamKind = "generate";
  state.streamModel = state.tab;
  setRunning(true);
  startStream(data.job, "generate");
}

async function cancel() {
  const res = await post("/api/cancel");
  if (!res.ok) toast(res.data.error || "Kein Job aktiv.", "warn");
}

function startStream(jobId, kind) {
  if (kind) state.streamKind = kind;
  if (state.es) { state.es.close(); state.es = null; }
  const es = new EventSource("/api/stream?job=" + encodeURIComponent(jobId));
  state.es = es;
  const upRun = () => state.streamKind === "upscale";

  const finish = () => {
    if (state.es === es) { es.close(); state.es = null; }
    if (state.running) setRunning(false);
  };

  es.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    addLog(d.line, d.level);
  });

  es.addEventListener("cmd", (e) => {
    const d = JSON.parse(e.data);
    $("#cmdText").textContent = d.command;
  });

  es.addEventListener("status", (e) => {
    const d = JSON.parse(e.data);
    $("#progressFill").style.width = d.progress + "%";
    let label = d.phase;
    if (d.phase === "Prompt codieren" && state.streamModel === "flux") {
      label += " · Text-Encoder auf CPU (gedulden)";
    }
    $("#phaseLabel").textContent = label;
  });

  es.addEventListener("steps", (e) => {
    const d = JSON.parse(e.data);
    $("#stepLabel").textContent = `${d.step}/${d.total} · ${d.rate}`;
    $("#phaseLabel").textContent = "Sampling";
  });

  es.addEventListener("seed", (e) => {
    const d = JSON.parse(e.data);
    $("#seedValue").textContent = d.seed;
    const inp = $('[data-name="seed"]');
    if (inp) { inp.value = d.seed; draft.seed = d.seed; }
  });

  es.addEventListener("preview", (e) => {
    const d = JSON.parse(e.data);
    showImage(d.url + "&t=" + Date.now(), true);
  });

  es.addEventListener("images", (e) => {
    const d = JSON.parse(e.data);
    if (d.images && d.images.length) {
      showImage(d.images[0] + "?t=" + Date.now(), false);
    }
  });

  // Ergebnis des Real-ESRGAN-Nachlaufs nach einer Generierung
  es.addEventListener("upscaled", (e) => {
    const d = JSON.parse(e.data);
    const url = (d.images && d.images[0]) || "";
    const note = $("#upscaleNote");
    note.innerHTML = `Nachlauf <b>${esc(d.model)} &times;${esc(d.scale)}</b> gespeichert: ` +
      `<a href="${esc(url)}" target="_blank" rel="noopener">ansehen</a> ` +
      `<span class="muted">(${esc(url.split("/").pop())})</span>`;
    note.hidden = false;
    loadUpscaleHistory();
    toast("Bild wurde hochskaliert (upscale-Ordner)", "ok");
  });

  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    finish();
    $("#progressFill").style.width = "100%";
    $("#phaseLabel").textContent = d.cancelled ? "Abgebrochen" : "Fertig";
    if (!d.cancelled) {
      const pill = $("#statusPill");
      pill.dataset.state = "ok";
      pill.textContent = "fertig";
      setTimeout(() => { if (!state.running) pill.dataset.state = "idle"; }, 4000);
      if (upRun()) {
        toast(`Upscale fertig in ${d.seconds}s`, "ok");
        loadUpscaleHistory();
      } else {
        toast(`Fertig in ${d.seconds}s · Seed ${$("#seedValue").textContent}`, "ok");
        loadGallery();
      }
    } else if (upRun()) {
      loadUpscaleHistory();
    }
  });

  es.addEventListener("fail", (e) => {
    const d = JSON.parse(e.data);
    addLog(d.message, "error");
    toast(d.message, "err");
    finish();
    const pill = $("#statusPill");
    pill.dataset.state = "err";
    pill.textContent = "Fehler";
    setTimeout(() => { if (!state.running) pill.dataset.state = "idle"; }, 6000);
    if (upRun()) loadUpscaleHistory(); else loadGallery();
  });

  es.addEventListener("gone", () => finish());

  es.onerror = () => {
    // Verbindung unterbrochen: EventSource versucht automatisch neu (mit Last-Event-ID).
    if (!state.running) finish();
  };
}

/* --------------------------------------------------------- Upscale-Reiter */

function renderUpscaleTab() {
  draft = null;
  normalizeUpscaleSel(state.upscaleSel);
  $("#paramsTitle").textContent = "Real-ESRGAN";
  $("#modelBadge").textContent = "Upscale";
  $("#paramForm").innerHTML = upscaleFormHtml();
  applyStageChrome("upscale");
  bindUpscaleForm();
  loadUpscaleHistory();
}

function upscaleFormHtml() {
  const sel = state.upscaleSel;
  const modelOpts = state.upscaleModels.map((m) =>
    `<option value="${esc(m.name)}" ${sel.model === m.name ? "selected" : ""}>${esc(m.label || m.name)}</option>`).join("");
  const scaleOpts = scalesFor(sel.model).map((s) =>
    `<option value="${s}" ${Number(sel.scale) === s ? "selected" : ""}>${s}&times;</option>`).join("");
  const src = state.upscaleSrc;
  const srcName = src ? decodeURIComponent(src.split("/").pop() || src) : "";

  const thumbs = state.gallery.length
    ? `<div class="up-thumbs" id="upThumbs">` + state.gallery.map((it) => `
        <button type="button" class="up-thumb ${src === it.url ? "is-active" : ""}"
          data-src="${esc(it.url)}" title="${esc(it.file)}">
          <img src="${esc(it.url)}?t=${it.mtime}" loading="lazy" alt=""></button>`).join("") + `</div>`
    : `<p class="muted field-hint">Noch keine erzeugten Bilder - bitte zuerst eine Datei hochladen.</p>`;

  const srcBox = src
    ? `<div class="up-src">
         <img src="${esc(src)}" alt="">
         <div class="up-src-meta">
           <b title="${esc(src)}">${esc(srcName)}</b>
           <button class="icon-btn" id="upClear" type="button" title="Auswahl aufheben">&#10005;</button>
         </div>
       </div>`
    : `<p class="muted field-hint">Kein Bild gew&auml;hlt - oben hochladen oder unten aus der Galerie w&auml;hlen.</p>`;

  return `
  <div class="group up-group">
    <h3>Quelle</h3>
    <div class="row c2 up-src-actions">
      <label class="btn ghost small" for="upFile">Datei hochladen</label>
      <button class="btn ghost small" id="upFromGallery" type="button">aus Galerie w&auml;hlen</button>
    </div>
    ${srcBox}
    <p class="muted field-hint">Tipp: Bild auch einfach per <b>Drag &amp; Drop</b> in diese Leiste ziehen.</p>
    <div class="field"><span>Galerie-Bilder<em class="hint">klicken zum W&auml;hlen</em></span>${thumbs}</div>
  </div>

  <div class="group">
    <h3>Real-ESRGAN</h3>
    <label class="field"><span>Modell</span>
      <select id="upModel">${modelOpts || '<option value="">keine Modelle gefunden</option>'}</select></label>
    <div class="row c2">
      <label class="field"><span>Faktor</span><select id="upScale">${scaleOpts}</select></label>
      <label class="field"><span>GPU</span><input type="number" id="upGpu" min="0" max="8" value="${esc(sel.gpuid)}"></label>
    </div>
    <label class="field"><span>Tile-Gr&ouml;&szlig;e<em class="hint">0 = automatisch</em></span>
      <input type="number" id="upTile" min="0" max="2048" step="32" value="${esc(sel.tile)}"></label>
    <p class="muted field-hint">Ergebnisse landen im <code>upscale/</code>-Ordner und erscheinen
      <b>nicht</b> in der Galerie.</p>
  </div>`;
}

function bindUpscaleForm() {
  const sel = state.upscaleSel;
  const modelEl = $("#upModel");
  if (modelEl) {
    modelEl.addEventListener("change", () => {
      sel.model = modelEl.value;
      normalizeUpscaleSel(sel);
      renderUpscaleTab();
    });
  }
  [["#upScale", "scale"], ["#upTile", "tile"], ["#upGpu", "gpuid"]].forEach(([id, key]) => {
    const el = $(id);
    if (!el) return;
    const handler = () => { sel[key] = Number(el.value) || 0; };
    el.addEventListener("input", handler);
    el.addEventListener("change", handler);
  });

  // Upload-Trigger ist das <label for="upFile"> (oeffnet den Dialog ohne JS-Klick)

  const gal = $("#upFromGallery");
  if (gal) gal.addEventListener("click", () => {
    const box = $("#upThumbs");
    if (!box) { toast("Noch keine Bilder in der Galerie.", "warn"); return; }
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
    box.classList.add("is-flash");
    setTimeout(() => box.classList.remove("is-flash"), 900);
  });

  $$(".up-thumb", $("#paramForm")).forEach((b) => b.addEventListener("click", () => {
    state.upscaleSrc = b.dataset.src;
    renderUpscaleTab();
  }));

  const clr = $("#upClear");
  if (clr) clr.addEventListener("click", () => {
    state.upscaleSrc = null;
    renderUpscaleTab();
  });
}

async function loadUpscaleInfo() {
  const res = await api("/api/upscale/models");
  const d = res.data || {};
  state.upscaleModels = d.models || [];
  const def = d.defaults || {};
  state.upscaleSel = normalizeUpscaleSel({
    model: def.model || d.default || (state.upscaleModels[0] && state.upscaleModels[0].name) || "",
    scale: def.scale || 4,
    tile: def.tile || 0,
    gpuid: def.gpuid != null ? def.gpuid : 0,
  });
  state.upscaleDefaults = Object.assign({}, state.upscaleSel);
}

async function loadUpscaleHistory() {
  const list = $("#upscaleList");
  if (!list) return;
  const res = await api("/api/upscale/list");
  const items = (res.data && res.data.items) || [];
  state.upscaleHistory = items;
  $("#upscaleEmpty").hidden = items.length > 0;
  list.innerHTML = items.map((it, i) => {
    const m = it.meta || {};
    const tag = m.model ? `${m.model} · x${m.scale || "?"}` : "";
    return `<div class="gal-item" data-idx="${i}" title="${esc(it.file)}">
      <img src="${esc(it.url)}?t=${it.mtime}" loading="lazy" alt="">
      ${tag ? `<span class="gal-tag">${esc(tag)}</span>` : ""}
      <button class="hist-del" data-file="${esc(it.file)}" title="L&ouml;schen">&#10005;</button>
    </div>`;
  }).join("");

  $$(".gal-item", list).forEach((el) => {
    el.addEventListener("click", (ev) => {
      if (ev.target.closest(".hist-del")) return;
      const it = state.upscaleHistory[Number(el.dataset.idx)];
      if (it) showImage(it.url + "?t=" + it.mtime, false);
    });
  });
  $$(".hist-del", list).forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      if (!confirm("Hochskaliertes Bild wirklich löschen?")) return;
      const r = await post("/api/upscale/delete", { file: btn.dataset.file });
      if (r.ok) { loadUpscaleHistory(); toast("Gelöscht", "ok"); }
      else toast((r.data && r.data.error) || "Löschen fehlgeschlagen", "err");
    });
  });
}

async function runUpscale() {
  if (state.running) return;
  if (!state.upscaleModels.length) {
    toast("Keine Real-ESRGAN-Modelle gefunden.", "err");
    return;
  }
  if (!state.upscaleSrc) {
    toast("Bitte zuerst ein Bild wählen: hochladen oder in der Galerie anklicken.", "warn");
    return;
  }
  const sel = state.upscaleSel;
  $("#btnGenerate").disabled = true;
  const res = await post("/api/upscale", {
    source: state.upscaleSrc,
    model: sel.model,
    scale: Number(sel.scale) || 4,
    tile: Number(sel.tile) || 0,
    gpuid: Number(sel.gpuid) || 0,
  });
  const data = res.data || {};
  if (!res.ok) {
    $("#btnGenerate").disabled = false;
    toast("Upscale: konnte nicht starten", "err", data.errors || [data.error || "Unbekannter Fehler"]);
    return;
  }
  resetStage(data.command);
  state.streamKind = "upscale";
  state.streamModel = "upscale";
  setRunning(true);
  startStream(data.job, "upscale");
}

/* ------------------------------------------------------------ Hochladen */

async function uploadSourceFile(file) {
  if (!file) return;
  const name = file.name || "";
  const okType = /^image\//.test(file.type || "") || /\.(png|jpe?g|webp|bmp|gif)$/i.test(name);
  if (!okType) {
    toast("Bitte eine Bilddatei auswählen (png, jpg, webp, bmp, gif).", "warn");
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await api("/api/upscale/upload", { method: "POST", body: fd });
    if (!res.ok) {
      toast("Upload fehlgeschlagen", "err",
        (res.data && res.data.errors) || [res.data && res.data.error || ("HTTP " + res.status)]);
      return;
    }
    state.upscaleSrc = res.data.url;
    if (isUpscale()) renderUpscaleTab(); else switchTab("upscale");
    toast("Bild geladen: " + res.data.name, "ok");
  } catch (err) {
    toast("Upload fehlgeschlagen: " + err, "err");
  }
}

/* ------------------------------------------------------------- Design */

const THEME_KEY = "sdwebui.theme";

function themeMode() {
  try { return localStorage.getItem(THEME_KEY) || "dark"; } catch (e) { return "dark"; }
}

function resolveTheme(mode) {
  if (mode === "dark" || mode === "light") return mode;
  return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches)
    ? "dark" : "light";
}

function applyTheme(mode) {
  const dark = resolveTheme(mode) !== "light";
  const root = document.documentElement;
  root.style.colorScheme = dark ? "dark" : "light";
  if (dark) root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", "light");
}

function setTheme(mode) {
  try { localStorage.setItem(THEME_KEY, mode); } catch (e) { /* ignore */ }
  applyTheme(mode);
}

/* --------------------------------------------------------- Galerie */

async function loadGallery() {
  const res = await api("/api/gallery");
  const items = (res.data && res.data.items) || [];
  state.gallery = items;
  const box = $("#gallery");
  $("#galleryEmpty").hidden = items.length > 0;
  box.innerHTML = items.map((it, i) => {
    const tag = it.meta && it.meta.seed != null ? it.meta.model + " · " + it.meta.seed : "";
    return `<div class="gal-item" data-idx="${i}" title="${esc((it.meta && it.meta.prompt) || it.file)}">
      <img src="${esc(it.url)}?t=${it.mtime}" loading="lazy" alt="">
      ${tag ? `<span class="gal-tag">${esc(tag)}</span>` : ""}
    </div>`;
  }).join("");

  $$(".gal-item", box).forEach((el) => {
    el.addEventListener("click", () => openLightbox(Number(el.dataset.idx)));
  });

  // Quell-Bild-Auswahl im Upscale-Reiter aktualisieren
  if (state.tab === "upscale" && !state.running) renderUpscaleTab();
}

function openLightbox(idx) {
  const it = state.gallery[idx];
  if (!it) return;
  const m = it.meta || {};
  $("#lbTitle").textContent = it.file;
  $("#lbImage").src = it.url + "?t=" + it.mtime;
  $("#lbPrompt").value = m.prompt || "";
  $("#lbNeg").value = m.negative_prompt || "";
  $("#lbDownload").href = it.url;
  $("#lbCmd").textContent = m.command || "";

  const stats = [
    ["Modell", m.model || "—"],
    ["Seed", m.seed != null ? m.seed : "—"],
    ["Größe", m.width ? `${m.width}×${m.height}` : "—"],
    ["Schritte", m.steps != null ? m.steps : "—"],
    ["CFG", m.cfg_scale != null ? m.cfg_scale : "—"],
    ["Sampler", m.sampling_method || "—"],
    ["Scheduler", m.scheduler || "—"],
    ["Datum", new Date(it.mtime * 1000).toLocaleString("de-DE")],
    ["Datei", fmtSize(it.size)],
  ];
  $("#lbStats").innerHTML = stats.map(([k, v]) =>
    `<div class="lb-stats-line">${esc(k)}: <b style="color:var(--text)">${esc(v)}</b></div>`).join("");

  $("#lightbox").hidden = false;
  $("#lightbox").dataset.idx = String(idx);
  $("#lightbox").dataset.file = it.file;

  $("#lbReuse").onclick = () => {
    if (m.model && state.cfg.models[m.model]) switchTab(m.model);
    const target = state.params[state.tab];
    Object.assign(target, {
      prompt: m.prompt || "", negative_prompt: m.negative_prompt || "",
      width: m.width || target.width, height: m.height || target.height,
      steps: m.steps || target.steps, cfg_scale: m.cfg_scale != null ? m.cfg_scale : target.cfg_scale,
      guidance: m.guidance != null ? m.guidance : target.guidance,
      sampling_method: m.sampling_method || target.sampling_method,
      scheduler: m.scheduler || target.scheduler,
      seed: m.seed != null ? m.seed : target.seed, random_seed: false,
    });
    renderForm();
    persist();
    close("lightbox");
    toast("Parameter übernommen", "ok");
  };

  $("#lbUpscale").onclick = () => {
    state.upscaleSrc = it.url;
    close("lightbox");
    if (state.tab === "upscale") renderUpscaleTab();
    else switchTab("upscale");
  };

  $("#lbDelete").onclick = async () => {
    if (!confirm("Bild wirklich löschen?")) return;
    const r = await post("/api/delete", { file: it.file });
    if (r.ok) { close("lightbox"); loadGallery(); toast("Gelöscht", "ok"); }
    else toast(r.data.error || "Löschen fehlgeschlagen", "err");
  };
}

/* --------------------------------------------------------- Einstellungen */

function renderSettings() {
  const cfg = state.cfg;
  $$("[data-cfg]").forEach((inp) => {
    const path = inp.dataset.cfg.split(".");
    let v = cfg;
    path.forEach((p) => { v = v ? v[p] : ""; });
    inp.value = v == null ? "" : v;
  });

  $("#modelSettings").innerHTML = Object.keys(cfg.models).map((key) => {
    const m = cfg.models[key];
    const fields = (PATH_FIELDS[key] || []).map((f) => {
      const val = (m.paths || {})[f.name] || "";
      const input = f.type === "select"
        ? `<select data-path="${key}.${f.name}" data-nopath="1">${f.options.map((o) =>
            `<option value="${esc(o)}" ${val === o ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`
        : `<div class="input-row">
             <input type="text" data-path="${key}.${f.name}" value="${esc(val)}" spellcheck="false">
             <button class="icon-btn" type="button" data-browse="${key}.${f.name}" title="Durchsuchen">&#8981;</button>
           </div>`;
      return `<label class="field"><span>${esc(f.label)}${f.hint ? `<em class="hint">${esc(f.hint)}</em>` : ""}</span>${input}</label>`;
    }).join("");

    return `<section class="settings-group">
      <h3><span>${esc(m.label || key)}</span><span class="path-state" data-state-for="${key}"></span></h3>
      <div class="grid2">${fields}</div>
    </section>`;
  }).join("");

  $$("[data-browse]").forEach((btn) => {
    btn.addEventListener("click", () => openPicker(btn));
  });
  checkPaths();
}

async function checkPaths() {
  const targets = {};
  $$("[data-cfg]").forEach((inp) => { if (inp.value) targets[inp.dataset.cfg] = inp.value; });
  $$("[data-path]").forEach((inp) => {
    if (inp.hasAttribute("data-nopath")) return;   // z.B. VAE-Format ist kein Pfad
    targets[inp.dataset.path] = inp.value;
  });
  const res = await post("/api/check", { paths: targets });
  const map = (res.data && res.data.paths) || {};
  Object.keys(state.cfg.models).forEach((key) => {
    const box = $(`[data-state-for="${key}"]`);
    if (!box) return;
    const entries = Object.entries(map).filter(([k]) => k.startsWith(key + "."));
    if (!entries.length) { box.innerHTML = ""; return; }
    const bad = entries.filter(([, ok]) => !ok);
    box.innerHTML = bad.length
      ? `<span class="bad">&#10007; ${bad.length} fehlt</span>`
      : `<span class="ok">&#10003; Pfade ok</span>`;
  });
}

async function saveSettings() {
  const payload = { models: {} };
  $$("[data-cfg]").forEach((inp) => {
    const parts = inp.dataset.cfg.split(".");
    if (parts.length === 1) payload[parts[0]] = inp.value;
    else { payload[parts[0]] = payload[parts[0]] || {}; payload[parts[0]][parts[1]] =
      inp.type === "number" ? Number(inp.value) : inp.value; }
  });
  $$("[data-path]").forEach((inp) => {
    const [key, field] = inp.dataset.path.split(".");
    payload.models[key] = payload.models[key] || { paths: {} };
    payload.models[key].paths[field] = inp.value;
  });

  const res = await post("/api/config", payload);
  if (!res.ok) { toast(res.data.error || "Speichern fehlgeschlagen", "err"); return; }
  state.cfg = res.data;
  renderSettings();
  const active = state.cfg.models[state.tab];
  if (active) $("#modelBadge").textContent = active.label;   // "upscale" hat kein Modell
  checkPaths();
  if (CKPT_TABS.includes(state.tab)) {   // Pfade im Dialog geaendert -> Dropdown neu einlesen
    await loadCheckpoints();
    renderForm();
  }
  toast("In config.json gespeichert", "ok");
}

function openSettings() {
  renderSettings();
  const themeSel = $("#themeSelect");
  if (themeSel) themeSel.value = themeMode();
  $("#settingsModal").hidden = false;
}

function close(id) {
  const el = document.getElementById(id);
  if (el) el.hidden = true;
}

/* --------------------------------------------------------- Datei-Wähler */

function openPicker(btn) {
  const input = btn.closest(".input-row, .field").querySelector("input");
  state.picker = { input, target: btn.dataset.browse };
  const current = (input.value || "").replace(/\/[^/]*$/, "") || "";
  $("#pickerModal").hidden = false;
  browse(current || "");
}

async function browse(dir) {
  const res = await api("/api/browse?models=1&dir=" + encodeURIComponent(dir || ""));
  const data = res.data || {};
  $("#pickerDir").value = data.dir || "";
  const list = $("#pickerList");
  const rows = [];
  if (data.parent) {
    rows.push(`<button class="picker-item" data-dir="${esc(data.parent)}">
      <span class="pi-icon">&#8617;</span> <span>&hellip; (nach oben)</span></button>`);
  }
  (data.dirs || []).forEach((d) => {
    rows.push(`<button class="picker-item" data-dir="${esc((data.dir ? data.dir + "/" : "") + d)}">
      <span class="pi-icon">&#128193;</span> <span>${esc(d)}</span></button>`);
  });
  (data.files || []).forEach((f) => {
    rows.push(`<button class="picker-item" data-file="${esc((data.dir ? data.dir + "/" : "") + f.name)}">
      <span class="pi-icon">&#9635;</span> <span>${esc(f.name)}</span>
      <span class="pi-size">${esc(fmtSize(f.size))}</span></button>`);
  });
  list.innerHTML = rows.join("") || `<div class="picker-item muted">Leerer Ordner</div>`;

  $$(".picker-item[data-dir]", list).forEach((b) =>
    b.addEventListener("click", () => browse(b.dataset.dir)));
  $$(".picker-item[data-file]", list).forEach((b) =>
    b.addEventListener("click", () => {
      if (state.picker && state.picker.input) {
        state.picker.input.value = b.dataset.file;
        state.picker.input.dispatchEvent(new Event("input", { bubbles: true }));
        state.picker.input.dispatchEvent(new Event("change", { bubbles: true }));
      }
      close("pickerModal");
      checkPaths();
    }));
}

/* --------------------------------------------------------- Init */

function startAction() {
  if (isUpscale()) runUpscale();
  else generate();
}

function bindStatic() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.key)));

  $("#btnGenerate").addEventListener("click", startAction);
  $("#btnCancel").addEventListener("click", cancel);
  $("#btnSettings").addEventListener("click", openSettings);
  $("#btnRefreshGallery").addEventListener("click", loadGallery);
  $("#btnRefreshUpscale").addEventListener("click", loadUpscaleHistory);
  $("#btnSaveSettings").addEventListener("click", saveSettings);
  $("#btnReset").addEventListener("click", () => {
    if (isUpscale()) {
      state.upscaleSel = Object.assign({}, state.upscaleDefaults);
      renderUpscaleTab();
      toast("Upscale-Werte zurückgesetzt", "ok");
      return;
    }
    if (!confirm("Eingaben für diesen Reiter auf die config-Defaults zurücksetzen?")) return;
    try { localStorage.removeItem("sdwebui.params." + state.tab); } catch (e) { /* ignore */ }
    state.params[state.tab] = Object.assign({}, state.cfg.models[state.tab].defaults || {});
    renderForm();
    toast("Zurückgesetzt", "ok");
  });
  $("#btnSaveDefaults").addEventListener("click", async () => {
    if (isUpscale()) {
      const res = await post("/api/config", { upscale: { defaults: Object.assign({}, state.upscaleSel) } });
      if (res.ok) {
        state.cfg = res.data;
        state.upscaleDefaults = Object.assign({}, state.upscaleSel);
        toast("Upscale-Standard in config.json gespeichert", "ok");
      } else toast(res.data.error || "Speichern fehlgeschlagen", "err");
      return;
    }
    const res = await post("/api/config", { models: { [state.tab]: { defaults: collectParams() } } });
    if (res.ok) {
      state.cfg = res.data;
      toast("Als Standard in config.json gespeichert", "ok");
    } else toast(res.data.error || "Speichern fehlgeschlagen", "err");
  });

  // Design / Styleswitcher
  const themeSel = $("#themeSelect");
  if (themeSel) {
    themeSel.value = themeMode();
    themeSel.addEventListener("change", () => {
      setTheme(themeSel.value);
      toast(themeSel.value === "light" ? "Helles Design aktiv"
        : themeSel.value === "dark" ? "Dunkles Design aktiv"
        : "Design folgt dem System", "ok");
    });
  }
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onSystemChange = () => { if (themeMode() === "system") applyTheme("system"); };
    if (mq.addEventListener) mq.addEventListener("change", onSystemChange);
    else if (mq.addListener) mq.addListener(onSystemChange);
  }

  // Upload fuer den Upscale-Reiter: Dateiauswahl + Drag & Drop
  $("#upFile").addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = "";
    if (file) uploadSourceFile(file);
  });

  const dropHost = $("#paramForm");
  ["dragenter", "dragover"].forEach((name) => {
    dropHost.addEventListener(name, (ev) => {
      if (!isUpscale()) return;
      ev.preventDefault();
      dropHost.classList.add("is-dropping");
    });
  });
  ["dragleave", "dragend", "drop"].forEach((name) => {
    dropHost.addEventListener(name, () => dropHost.classList.remove("is-dropping"));
  });
  dropHost.addEventListener("drop", (ev) => {
    if (!isUpscale()) return;
    ev.preventDefault();
    const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (file) uploadSourceFile(file);
  });

  $("#seedChip").addEventListener("click", () => {
    const v = $("#seedValue").textContent;
    if (v === "—") return;
    draft.seed = Number(v);
    draft.random_seed = false;
    renderForm();
    persist();
    toast("Seed übernommen", "ok");
  });

  $$("[data-close]").forEach((b) => b.addEventListener("click", () => close(b.dataset.close)));
  $$(".modal").forEach((m) => m.addEventListener("mousedown", (e) => {
    if (e.target === m) m.hidden = true;
  }));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $$(".modal").forEach((m) => { m.hidden = true; });
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); startAction(); }
  });

  $("#pickerUp").addEventListener("click", () => {
    const dir = $("#pickerDir").value.replace(/\/+$/, "");
    browse(dir.replace(/\/[^/]*$/, ""));
  });
  $("#pickerGo").addEventListener("click", () => browse($("#pickerDir").value));
  $("#pickerDir").addEventListener("keydown", (e) => {
    if (e.key === "Enter") browse($("#pickerDir").value);
  });

  window.addEventListener("hashchange", () => {
    const key = (location.hash || "").replace("#", "");
    if (key && key !== state.tab) switchTab(key);
  });
}

async function loadDevices() {
  const res = await api("/api/devices");
  const devs = (res.data && res.data.devices) || [];
  if (!devs.length) {
    $("#deviceInfo").textContent = "keine Geräte gefunden";
    return;
  }
  const gpu = devs.find((d) => /vulkan|cuda|hip/i.test(d.id)) || devs[0];
  $("#deviceInfo").textContent = `${gpu.name} · ${gpu.id}`;
}

async function init() {
  const cfg = await api("/api/config");
  if (!cfg.ok) { toast("config.json konnte nicht geladen werden", "err"); return; }
  state.cfg = cfg.data;
  applyTheme(themeMode());
  loadParams();
  bindStatic();
  await loadUpscaleInfo();
  await loadCheckpoints();

  // Reiter ueber URL-Anker waehlen, z.B. /#flux oder /#upscale
  const hash = (location.hash || "").replace("#", "");
  if (hash === "upscale" || state.cfg.models[hash]) {
    if (hash !== state.tab) {
      state.tab = hash;
      $$(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.key === hash));
    }
  }

  renderTab();
  loadGallery();
  loadDevices();

  const st = await api("/api/status");
  if (st.ok && st.data.running) {
    const kind = st.data.model === "upscale" ? "upscale" : "generate";
    setRunning(true);
    $("#seedValue").textContent = st.data.seed != null ? st.data.seed : "—";
    state.streamKind = kind;
    state.streamModel = st.data.model || state.tab;
    if (kind === "upscale" && state.tab !== "upscale") switchTab("upscale");
    startStream(st.data.job, kind);
    toast("Laufenden Job wieder angebunden", "ok");
  }
}

document.addEventListener("DOMContentLoaded", init);
