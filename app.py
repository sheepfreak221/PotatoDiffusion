#!/usr/bin/env python3
"""
PotatoDiffusion
===============
Web-Interface fuer stable-diffusion.cpp (sd-cli) mit Vulkan-Backend.

* Drei Modell-Reiter: SD 1.5, SDXL, FLUX - jeder mit eigenen Modellpfaden und Parametern
* Vierter Reiter "Upscale": Real-ESRGAN (Hochskalieren von Bildern, eigener Ausgabe-Ordner)
* Optionaler Nachlauf: "Bild nach dem Generieren hochskalieren" in den Modell-Reitern
* Pfade & Parameter-Defaults stehen in config.json (aenderbar ueber das Zahnrad-Menue)
* Live-Fortschritt (Log + Vorschau-Bild) per Server-Sent-Events
* Zufaelliger Seed (wird angezeigt und kann reproduziert werden)
* Galerie aller erzeugten Bilder inkl. Metadaten (JSON-Seitendatei);
  hochskalierte Bilder stehen in upscale/ und bleiben ausserhalb der Galerie

Start:  python3 app.py   ->  http://127.0.0.1:7860
"""

from __future__ import annotations

import copy
import json
import random
import re
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"

IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_SUFFIXES = {".gguf", ".safetensors", ".ckpt", ".pt", ".bin"}

SAMPLERS = [
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "dpm++2m_sde", "dpm++2m_sde_bt", "ipndm", "ipndm_v", "lcm", "ddim_trailing",
    "tcd", "res_multistep", "res_2s", "er_sde", "euler_cfg_pp", "euler_a_cfg_pp",
]
SCHEDULERS = [
    "auto", "discrete", "karras", "exponential", "ays", "gits", "smoothstep",
    "sgm_uniform", "simple", "kl_optimal", "lcm", "bong_tangent", "ltx2",
    "logit_normal", "flux2", "flux", "beta",
]
WEIGHT_TYPES = ["", "f32", "f16", "q8_0", "q6_K", "q5_1", "q5_0", "q4_K", "q4_1", "q4_0", "q3_K", "q2_K"]

# config- Feld  ->  sd-cli Argument
PATH_FLAGS = (
    ("model", "-m"),
    ("diffusion_model", "--diffusion-model"),
    ("clip_l", "--clip_l"),
    ("clip_g", "--clip_g"),
    ("t5xxl", "--t5xxl"),
    ("llm", "--llm"),
    ("vae", "--vae"),
)

DEFAULT_PARAMS = {
    "prompt": "", "negative_prompt": "", "width": 512, "height": 512, "steps": 20,
    "cfg_scale": 7.0, "guidance": 0.0, "sampling_method": "euler_a", "scheduler": "auto",
    "seed": -1, "random_seed": True, "batch_count": 1, "clip_skip": -1,
    "vae_tiling": False, "offload_to_cpu": False, "backend": "diffusion=vulkan0,te=cpu,vae=vulkan0",
    "weight_type": "", "max_vram": "", "threads": 0, "preview": True, "extra_args": "",
    # Nachlauf mit Real-ESRGAN (Ergebnis landet im upscale-Ordner, nicht in der Galerie)
    "upscale_after": False, "upscale_model": "realesrgan-x4plus", "upscale_scale": 4,
    "upscale_tile": 0, "upscale_gpuid": 0,
}

UPSCALE_DIR = BASE_DIR / "upscale"

DEFAULT_CONFIG = {
    "sd_cli": str(BASE_DIR / "bin" / "sd-cli"),
    "output_dir": str(BASE_DIR / "outputs"),
    "listen": {"host": "127.0.0.1", "port": 7860},
    "upscale": {
        "binary": str(BASE_DIR / "bin" / "realesrgan-ncnn-vulkan"),
        "model_dir": str(BASE_DIR / "models" / "upscale"),
        "output_dir": str(UPSCALE_DIR),
        "default_model": "realesrgan-x4plus",
        "tile": 0,
        "gpuid": 0,
        "defaults": {"model": "realesrgan-x4plus", "scale": 4, "tile": 0, "gpuid": 0},
    },
    "models": {
        "sd15": {
            "label": "SD 1.5",
            "paths": {"model": str(BASE_DIR / "models" / "checkpoints" / "SD1.5" / "mistoonAnime_v30.safetensors"),
                      "clip_l": ""},
            "defaults": {**DEFAULT_PARAMS, "steps": 20, "cfg_scale": 7.0,
                         "negative_prompt": "lowres, bad anatomy, bad hands, worst quality, jpeg artifacts, watermark"},
        },
        "sdxl": {
            "label": "SDXL",
            "paths": {"model": str(BASE_DIR / "models" / "checkpoints" / "SDXL" / "sdxlMergeheaven_betaM15-q4_0.gguf"),
                      "clip_l": "", "clip_g": ""},
            "defaults": {**DEFAULT_PARAMS, "width": 768, "height": 768, "steps": 20, "cfg_scale": 6.5,
                         "vae_tiling": True,
                         "negative_prompt": "lowres, bad anatomy, bad hands, worst quality, jpeg artifacts, watermark"},
        },
        "flux": {
            "label": "FLUX",
            "paths": {"diffusion_model": str(BASE_DIR / "models" / "diffusion-models" / "FLUX" / "flux-2-klein-4b-Q4_K_M.gguf"),
                      "llm": str(BASE_DIR / "models" / "textencoder" / "Qwen3-4B-Q4_K_M.gguf"),
                      "vae": str(BASE_DIR / "models" / "vae" / "ae.safetensors"),
                      "vae_format": "flux2",
                      "clip_l": "", "clip_g": "", "t5xxl": ""},
            "defaults": {**DEFAULT_PARAMS, "steps": 4, "cfg_scale": 1.0, "guidance": 3.5,
                         "sampling_method": "euler", "vae_tiling": True, "negative_prompt": ""},
        },
    },
}

app = Flask(__name__)

_config_lock = threading.Lock()
_config: dict = {}
job_lock = threading.Lock()
JOBS: dict[str, "Job"] = {}
current_job: "Job | None" = None

# --------------------------------------------------------------------------- config


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (over or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


# Nur diese Felder enthalten Pfade - alles andere (Labels, "flux2", Sampler …)
# darf nie umgerechnet werden.
MODEL_PATH_FIELDS = {name for name, _ in PATH_FLAGS}
TOP_PATH_FIELDS = ("sd_cli", "output_dir")
UPSCALE_PATH_FIELDS = ("binary", "model_dir", "output_dir")


def _map_paths(data: dict, fn) -> dict:
    """Kopie der Config, in der alle bekannten Pfad-Felder durch fn() laufen."""
    out = copy.deepcopy(data)
    for key in TOP_PATH_FIELDS:
        if isinstance(out.get(key), str):
            out[key] = fn(out[key])
    up = out.get("upscale")
    if isinstance(up, dict):
        for key in UPSCALE_PATH_FIELDS:
            if isinstance(up.get(key), str):
                up[key] = fn(up[key])
    for model in (out.get("models") or {}).values():
        paths = model.get("paths") if isinstance(model, dict) else None
        if isinstance(paths, dict):
            for key in list(paths):
                if key in MODEL_PATH_FIELDS and isinstance(paths[key], str):
                    paths[key] = fn(paths[key])
    return out


def resolve_path(value: str) -> str:
    """Relativen Pfad gegen das Projektverzeichnis aufloesen, ~ erweitern."""
    if not value:
        return value
    p = Path(value).expanduser()
    return str(p) if p.is_absolute() else str(BASE_DIR / p)


def portable_path(value: str) -> str:
    """Pfade, die im Projektordner liegen, relativ ablegen -> die UI ist portabel.

    Damit wachsen die Pfade in config.json nicht mit dem Speicherort: der ganze
    Ordner kann verschoben/umbenannt/anderswoher gestartet werden.
    """
    if not value:
        return value
    p = Path(value).expanduser()
    if not p.is_absolute():
        return value                      # schon relativ
    try:
        return p.resolve().relative_to(BASE_DIR).as_posix()
    except (ValueError, OSError):
        return value                      # liegt ausserhalb -> bleibt absolut


def load_config() -> dict:
    global _config
    data = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text("utf-8"))
        except Exception as exc:  # pragma: no cover - kaputte Config nicht crashen lassen
            print(f"[config] {CONFIG_FILE} nicht lesbar ({exc}), nutze Defaults")
    merged = _map_paths(_deep_merge(DEFAULT_CONFIG, data), portable_path)  # Migration: relativ
    _config = merged
    if not CONFIG_FILE.exists() or data != merged:
        write_config(merged)
    return merged


def write_config(data: dict) -> None:
    """Atomar speichern - Projektpfade landen relativ in der Datei."""
    portable = _map_paths(data, portable_path)
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(portable, indent=2, ensure_ascii=False) + "\n", "utf-8")
    tmp.replace(CONFIG_FILE)


def get_config() -> dict:
    """Config-Kopie mit aufgeloesten, absoluten Pfaden (fuer API und Befehls-Bau)."""
    with _config_lock:
        return _map_paths(_config, resolve_path)


# --------------------------------------------------------------------------- Jobs

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
BAR_RE = re.compile(r"\|\s*[#=>.\s]*\|\s*(\d+)/(\d+)\s*-\s*([0-9.]+)\s*([A-Za-z/]+)")
SEED_RE = re.compile(r"seed\s+(\d+)", re.IGNORECASE)
BAD_RE = re.compile(r"\b(error|failed|invalid|cannot|unable|no such file|not found)\b", re.IGNORECASE)

PHASES = ["Start", "Modell laden", "Prompt codieren", "Sampling", "Bild dekodieren",
          "Speichern", "Hochskalieren"]
PHASE_BASE = {"Start": 0, "Modell laden": 4, "Prompt codieren": 30, "Sampling": 45,
              "Bild dekodieren": 86, "Speichern": 92, "Hochskalieren": 94}

# Fortschrittszeile von realesrgan-ncnn-vulkan ("22.22%")
PERCENT_RE = re.compile(r"^(\d{1,3}(?:\.\d+)?)%$")


class BaseJob:
    """Gemeinsame Event-, Log- und Cancel-Logik aller Laeufe (sd-cli und Real-ESRGAN)."""

    kind = "generate"

    def __init__(self, command: list[str], dry_cmd: str, params: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.command = command
        self.dry_cmd = dry_cmd
        self.params = params or {}
        self.model_key = ""
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.finished = False
        self.cancelled = False
        self.started_at = time.time()

    # -- Events ------------------------------------------------------------
    def emit(self, type_: str, **data) -> None:
        with self.lock:
            self.events.append({"type": type_, "t": round(time.time(), 3), **data})
            if len(self.events) > 50000:
                del self.events[:40000]

    # -- Log-Verarbeitung --------------------------------------------------
    def handle_fragment(self, frag: str) -> None:
        """Eine Ausgabezeile des Kind-Prozesses verarbeiten (von den Klassen ueberschrieben)."""
        raise NotImplementedError

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        buf = b""
        stream = self.proc.stdout
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                self.handle_fragment(raw.decode("utf-8", "replace"))
        if buf:
            self.handle_fragment(buf.decode("utf-8", "replace"))

    # -- Ausfuehrung -------------------------------------------------------
    def cancel(self) -> None:
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass

            def _kill() -> None:
                time.sleep(4)
                if self.proc and self.proc.poll() is None:
                    try:
                        self.proc.kill()
                    except OSError:
                        pass
            threading.Thread(target=_kill, daemon=True).start()

    def finish(self) -> None:
        elapsed = round(time.time() - self.started_at, 1)
        self.emit("done", seconds=elapsed, cancelled=self.cancelled)
        self.finished = True


class Job(BaseJob):
    """Ein sd-cli Lauf inkl. Log-Parsing, Fortschritt, Vorschau-Bild und Real-ESRGAN-Nachlauf."""

    kind = "generate"

    def __init__(self, model_key: str, params: dict, command: list[str], out_dir: Path,
                 stem: str, preview_path: Path, dry_cmd: str):
        super().__init__(command, dry_cmd, params)
        self.model_key = model_key
        self.out_dir = out_dir
        self.stem = stem
        self.preview_path = preview_path
        self.phase = "Start"
        self.progress = 0.0
        self.upscaling = False          # True waehrend des Real-ESRGAN-Nachlaufs
        self.result_images: list[Path] = []

    def set_phase(self, phase: str, progress: float | None = None, force: bool = False) -> None:
        changed = False
        if PHASES.index(phase) >= PHASES.index(self.phase):
            changed = phase != self.phase
            self.phase = phase
        if progress is not None:
            new = max(self.progress, min(99.5, progress))
            if new - self.progress >= 0.4 or force:
                changed = True
            self.progress = new
        if changed or force:
            self.emit("status", phase=self.phase, progress=round(self.progress, 1))

    def _bump(self, progress: float) -> None:
        self.set_phase(self.phase, progress)

    # -- Log-Verarbeitung --------------------------------------------------
    def handle_fragment(self, frag: str) -> None:
        line = ANSI_RE.sub("", frag).replace("\r", "").strip()
        if not line:
            return

        percent = PERCENT_RE.match(line)
        if percent and self.upscaling:
            # Fortschrittszeile von realesrgan-ncnn-vulkan ("22.22%")
            self._upscale_progress(float(percent.group(1)))
            return

        bar = BAR_RE.search(line)
        if bar:
            cur, total, rate, unit = int(bar.group(1)), int(bar.group(2)), bar.group(3), bar.group(4)
            if total > 0 and cur <= total:
                ratio = cur / total
                in_sampling = PHASES.index(self.phase) <= PHASES.index("Sampling")
                if unit in ("it/s", "s/it") and in_sampling:
                    self.set_phase("Sampling", PHASE_BASE["Sampling"] + 40 * ratio)
                    self.emit("steps", step=cur, total=total, rate=rate)
                elif self.phase == "Bild dekodieren":
                    self._bump(PHASE_BASE["Bild dekodieren"] + 11 * ratio)
                elif self.phase == "Prompt codieren":
                    self._bump(PHASE_BASE["Prompt codieren"] + 14 * ratio)
                elif unit in ("it/s", "s/it"):
                    pass  # Balken gehoert nicht zum Sampling (z.B. VAE-Decode)
                else:
                    self._bump(PHASE_BASE["Modell laden"] + 25 * ratio)
            return

        low = line.lower()
        if "loading model from" in low or re.search(r"\bload \S+ using", low):
            self.set_phase("Modell laden", PHASE_BASE["Modell laden"])
        elif "get_learned_condition" in low or "text encoder" in low:
            self.set_phase("Prompt codieren", PHASE_BASE["Prompt codieren"])
        elif "sampling using" in low:
            self.set_phase("Prompt codieren", PHASE_BASE["Prompt codieren"])
            self.emit("log", level="info", line=line)
            return
        elif "generating image" in low and "seed" in low:
            match = SEED_RE.search(line)
            if match:
                self.params["seed"] = int(match.group(1))
                self.emit("seed", seed=int(match.group(1)))
            self.set_phase("Sampling", PHASE_BASE["Sampling"])
        elif low.startswith("[info") and "decoding" in low and "latent" in low:
            self.set_phase("Bild dekodieren", PHASE_BASE["Bild dekodieren"])
        elif "sampling completed" in low:
            self.set_phase("Bild dekodieren", PHASE_BASE["Bild dekodieren"])
        elif "generate_image completed in" in low:
            match = re.search(r"completed in ([\d.]+)s", line)
            if match:
                self.emit("timing", seconds=float(match.group(1)))
            self.set_phase("Speichern", PHASE_BASE["Speichern"])
        elif "images saved" in low or "save result image" in low:
            self.set_phase("Speichern", PHASE_BASE["Speichern"])
        elif "warn" in low:
            self.emit("log", level="warn", line=line)
            return

        if BAD_RE.search(line):
            self.emit("log", level="error", line=line)
        else:
            self.emit("log", level="info", line=line)

    def _preview_watcher(self) -> None:
        last_stamp = None
        while not self.finished:
            try:
                stat = self.preview_path.stat()
            except OSError:
                time.sleep(0.4)
                continue
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp != last_stamp and stat.st_size > 0:
                last_stamp = stamp
                self.emit("preview", url=f"/outputs/{self.preview_path.name}?v={stat.st_mtime_ns}")
            time.sleep(0.4)

    # -- Ausfuehrung -------------------------------------------------------
    def run(self) -> None:
        out_dir = self.out_dir
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.emit("fail", message=f"Ausgabe-Ordner nicht beschreibbar: {out_dir} ({exc})")
            self.finish()
            return
        self.emit("cmd", command=self.dry_cmd)
        self.emit("status", phase="Start", progress=1.0)
        try:
            self.proc = subprocess.Popen(
                self.command, cwd=str(out_dir), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0,
                start_new_session=True,
            )
        except OSError as exc:
            self.emit("fail", message=f"sd-cli konnte nicht gestartet werden: {exc}")
            self.finish()
            return

        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._preview_watcher, daemon=True).start()
        code = self.proc.wait()

        if self.cancelled:
            self.emit("fail", message="Abgebrochen.")
        elif code != 0:
            self.emit("fail", message=f"sd-cli beendet mit Fehlercode {code}.")
        else:
            images = self._collect_results()
            if images and not self.cancelled and self.params.get("upscale_after"):
                self._post_upscale(images[0])
        self.finish()

    def _collect_results(self) -> list[Path]:
        now = self.started_at - 5
        images = []
        for path in sorted(out_dir_files(self.out_dir)):
            if path.suffix.lower() not in IMG_SUFFIXES:
                continue
            if path.name.startswith("preview_"):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= now and path.name.startswith(self.stem):
                images.append(path)
        if not images:
            self.emit("fail", message="Kein Bild gefunden - sd-cli hat nichts geschrieben.")
            return []
        meta_path = self.out_dir / f"{self.stem}.json"
        meta = {"model": self.model_key, "seed": self.params.get("seed"),
                "prompt": self.params.get("prompt", ""),
                "negative_prompt": self.params.get("negative_prompt", ""),
                "width": self.params.get("width"), "height": self.params.get("height"),
                "steps": self.params.get("steps"), "cfg_scale": self.params.get("cfg_scale"),
                "guidance": self.params.get("guidance"),
                "sampling_method": self.params.get("sampling_method"),
                "scheduler": self.params.get("scheduler"),
                "backend": self.params.get("backend"),
                "command": self.dry_cmd,
                "created": time.time()}
        try:
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        except OSError:
            pass
        self.result_images = images
        self.emit("images", images=[f"/outputs/{p.name}" for p in images])
        return images

    # -- Real-ESRGAN-Nachlauf ---------------------------------------------
    def _post_upscale(self, src: Path) -> None:
        """Optionales Nachupscaling des frisch erzeugten Bildes (Ergebnis: upscale-Ordner)."""
        ucfg = upscale_cfg()
        if not ucfg["binary"].is_file():
            self.emit("log", level="warn",
                      line=f"Real-ESRGAN nicht gefunden: {ucfg['binary']} - Nachlauf uebersprungen.")
            return
        models = {m["name"]: m for m in list_upscale_models()}
        model = str(self.params.get("upscale_model") or ucfg["raw"].get("default_model") or "")
        info = models.get(model)
        if info is None:
            self.emit("log", level="warn", line=f"Upscale-Modell nicht verfuegbar: {model} - uebersprungen.")
            return
        scale = as_int(self.params.get("upscale_scale"), info["default_scale"], 1, 8)
        if scale not in info["scales"]:
            scale = info["default_scale"]
        tile = as_int(self.params.get("upscale_tile"), ucfg["raw"].get("tile") or 0, 0, 2048)
        gpuid = as_int(self.params.get("upscale_gpuid"), ucfg["raw"].get("gpuid") or 0, 0, 8)

        dst = unique_path(ucfg["out_dir"], f"{src.stem}_{model}_x{scale}", ".png")
        cmd, pretty = build_upscale_command(src, dst, model, scale, tile, gpuid)
        self.emit("log", level="info",
                  line=f"Nachlauf Real-ESRGAN: {model} x{scale}  ({' '.join(shlex.quote(c) for c in cmd)})")

        self.upscaling = True
        self.set_phase("Hochskalieren", PHASE_BASE["Hochskalieren"], force=True)
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(ucfg["out_dir"]),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
                start_new_session=True,
            )
        except OSError as exc:
            self.upscaling = False
            self.emit("log", level="warn", line=f"Real-ESRGAN konnte nicht gestartet werden: {exc}")
            return
        self._reader()
        code = self.proc.wait()
        self.upscaling = False
        self.proc = None

        def discard() -> None:
            try:
                dst.unlink(missing_ok=True)
            except OSError:
                pass

        if self.cancelled:
            discard()
            self.emit("log", level="warn", line="Nachlauf abgebrochen.")
            return
        if code != 0:
            discard()
            self.emit("log", level="warn", line=f"realesrgan-ncnn-vulkan beendet mit Fehlercode {code}.")
            return
        if not dst.is_file():
            self.emit("log", level="warn", line="Nachlauf: kein Ausgabebild geschrieben.")
            return

        write_upscale_meta(dst, {
            "kind": "upscale", "origin": self.model_key, "source": f"/outputs/{src.name}",
            "model": model, "scale": scale, "tile": tile, "gpuid": gpuid,
            "seed": self.params.get("seed"), "command": pretty, "created": time.time(),
        })
        self.emit("upscaled", images=[f"/upscales/{dst.name}"], model=model, scale=scale)

    def _upscale_progress(self, pct: float) -> None:
        base = max(self.progress, PHASE_BASE["Hochskalieren"])
        value = base + (99.0 - base) * min(pct, 100.0) / 100.0
        self.set_phase("Hochskalieren", value, force=True)

    def finish(self) -> None:
        self._bump(100.0)
        super().finish()
        threading.Thread(target=self._cleanup_preview, daemon=True).start()

    def _cleanup_preview(self) -> None:
        """Vorschaubild nach Laufende entfernen (Datei ist nur Zwischenstand)."""
        time.sleep(3)
        try:
            self.preview_path.unlink(missing_ok=True)
        except OSError:
            pass


class UpscaleJob(BaseJob):
    """Ein Real-ESRGAN Lauf aus dem eigenen Upscale-Reiter."""

    kind = "upscale"

    def __init__(self, params: dict, command: list[str], dry_cmd: str,
                 src: Path, dst: Path, src_url: str):
        super().__init__(command, dry_cmd, params)
        self.model_key = "upscale"
        self.src = src
        self.dst = dst
        self.src_url = src_url
        self.progress = 2.0

    def set_status(self, phase: str, progress: float) -> None:
        self.progress = max(self.progress, min(100.0, progress))
        self.emit("status", phase=phase, progress=round(self.progress, 1))

    def handle_fragment(self, frag: str) -> None:
        line = ANSI_RE.sub("", frag).replace("\r", "").strip()
        if not line:
            return
        percent = PERCENT_RE.match(line)
        if percent:
            self.set_status("Real-ESRGAN", float(percent.group(1)))
            return
        level = "error" if BAD_RE.search(line) else "info"
        self.emit("log", level=level, line=line)

    def run(self) -> None:
        out_dir = self.dst.parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.emit("fail", message=f"Upscale-Ordner nicht beschreibbar: {out_dir} ({exc})")
            self.finish()
            return
        self.emit("cmd", command=self.dry_cmd)
        self.emit("status", phase="Start", progress=2.0)
        self.emit("log", level="info", line=f"{self.src.name} -> {self.dst.name}")
        try:
            self.proc = subprocess.Popen(
                self.command, cwd=str(out_dir), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0, start_new_session=True,
            )
        except OSError as exc:
            self.emit("fail", message=f"realesrgan-ncnn-vulkan konnte nicht gestartet werden: {exc}")
            self.finish()
            return

        reader = threading.Thread(target=self._reader, daemon=True)
        reader.start()
        code = self.proc.wait()

        if not self.cancelled and code == 0 and self.dst.is_file():
            write_upscale_meta(self.dst, {
                "kind": "upscale", "origin": "upscale",
                "source": self.src_url,
                "model": self.params.get("model"), "scale": self.params.get("scale"),
                "tile": self.params.get("tile"), "gpuid": self.params.get("gpuid"),
                "command": self.dry_cmd, "created": time.time(),
            })
            self.emit("images", images=[f"/upscales/{self.dst.name}"])
            self.set_status("Fertig", 100.0)
        else:
            # unvollstaendiges Ergebnis nicht stehen lassen
            try:
                self.dst.unlink(missing_ok=True)
            except OSError:
                pass
            if self.cancelled:
                self.emit("fail", message="Abgebrochen.")
            elif code != 0:
                self.emit("fail", message=f"realesrgan-ncnn-vulkan beendet mit Fehlercode {code}.")
            else:
                self.emit("fail", message="Kein Ausgabebild geschrieben - Modell passt nicht zur Skalierung?")
        self.finish()


def out_dir_files(out_dir: Path) -> list[Path]:
    try:
        return [p for p in out_dir.iterdir() if p.is_file()]
    except OSError:
        return []


# ------------------------------------------------------------------ Real-ESRGAN

ANIME_V3_RE = re.compile(r"^(realesr-animevideov3)-x(\d+)$")


def upscale_cfg() -> dict:
    """Pfade rund um Real-ESRGAN aus der Config ableiten."""
    u = get_config().get("upscale") or {}
    out_dir = Path(u.get("output_dir") or UPSCALE_DIR).expanduser()
    return {
        "raw": u,
        "binary": Path(u.get("binary") or (BASE_DIR / "bin" / "realesrgan-ncnn-vulkan")).expanduser(),
        "model_dir": Path(u.get("model_dir") or (BASE_DIR / "models" / "upscale")).expanduser(),
        "out_dir": out_dir,
        "upload_dir": out_dir / "uploads",
        "default_model": str(u.get("default_model") or "realesrgan-x4plus"),
        "tile": as_int(u.get("tile"), 0, 0, 2048),
        "gpuid": as_int(u.get("gpuid"), 0, 0, 8),
    }


def list_upscale_models() -> list[dict]:
    """Verfuegbare Real-ESRGAN-Modelle (animevideov3-Varianten werden zu einem Eintrag zusammengefasst)."""
    ucfg = upscale_cfg()
    try:
        bin_files = sorted(ucfg["model_dir"].glob("*.bin"))
    except OSError:
        bin_files = []

    found: dict[str, dict] = {}
    for bin_file in bin_files:
        if not (ucfg["model_dir"] / f"{bin_file.stem}.param").is_file():
            continue                      # ohne .param kann das Binary nicht laden
        match = ANIME_V3_RE.match(bin_file.stem)
        name = match.group(1) if match else bin_file.stem
        entry = found.setdefault(name, {"name": name, "scales": set(), "size": 0})
        # das Binary baut fuer realesr-animevideov3 den Namen "<modell>-x<scale>.param"
        entry["scales"].add(int(match.group(2)) if match else 4)
        try:
            entry["size"] += bin_file.stat().st_size
        except OSError:
            pass

    models = []
    for entry in found.values():
        scales = sorted(entry["scales"])
        entry["scales"] = scales
        entry["default_scale"] = scales[-1]
        entry["label"] = f"{entry['name']} (x{'/'.join(str(s) for s in scales)})"
        models.append(entry)
    models.sort(key=lambda m: m["name"])
    return models


def build_upscale_command(src: Path, dst: Path, model: str, scale: int,
                          tile: int, gpuid: int) -> tuple[list[str], str]:
    ucfg = upscale_cfg()
    cmd = [str(ucfg["binary"]), "-i", str(src), "-o", str(dst), "-n", str(model),
           "-m", str(ucfg["model_dir"]), "-s", str(int(scale)), "-g", str(int(gpuid))]
    if tile:
        cmd += ["-t", str(int(tile))]
    pretty = " \\\n  ".join(shlex.quote(c) for c in cmd)
    return cmd, pretty


def unique_path(folder: Path, stem: str, suffix: str) -> Path:
    """Pfad im Zielordner, der noch nicht belegt ist."""
    candidate = folder / f"{stem}{suffix}"
    n = 1
    while candidate.exists() or candidate.with_suffix(".json").exists():
        n += 1
        candidate = folder / f"{stem}-{n}{suffix}"
    return candidate


def image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except Exception:                     # PIL ist optional
        return None, None
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None, None


def write_upscale_meta(dst: Path, meta: dict) -> None:
    """Seitendatei fuer ein hochskaliertes Bild schreiben (Anzeige im Verlauf)."""
    w, h = image_size(dst)
    if w:
        meta["width"], meta["height"] = w, h
    try:
        dst.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    except OSError:
        pass


def resolve_source(raw) -> tuple[Path, str]:
    """Bildquelle aus der UI (URL) auf eine erlaubte Datei abbilden.

    Erlaubt sind ausschliesslich Dateien im Ausgabe-Ordner (Galerie) und im
    Upscale-Ordner inklusive Uploads. Rueckgabe: (Pfad, URL).
    """
    cfg = get_config()
    ucfg = upscale_cfg()
    text = str(raw or "").strip()
    if text.startswith("/"):
        text = text[1:]
    if text.startswith("outputs/"):
        root, rest, url_prefix = Path(cfg["output_dir"]).expanduser(), text[len("outputs/"):], "/outputs/"
    elif text.startswith("upscales/"):
        root, rest, url_prefix = ucfg["out_dir"], text[len("upscales/"):], "/upscales/"
    else:
        raise ValueError("Bildquelle muss /outputs/... oder /upscales/... sein.")
    if not rest or ".." in Path(rest).parts:
        raise ValueError("Ungueltiger Bildpfad.")
    try:
        resolved = (root / rest).resolve()
        root_resolved = root.resolve()
    except OSError as exc:
        raise ValueError(f"Bildpfad nicht lesbar: {exc}")
    if root_resolved not in resolved.parents:
        raise ValueError("Pfad liegt ausserhalb der erlaubten Ordner.")
    if not resolved.is_file():
        raise ValueError(f"Bild nicht gefunden: {rest}")
    if resolved.suffix.lower() not in IMG_SUFFIXES:
        raise ValueError(f"Kein Bilddatei-Format: {resolved.name}")
    return resolved, url_prefix + rest


# --------------------------------------------------------------------------- Kommando

def as_int(val, fallback, lo, hi):
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return fallback


def as_float(val, fallback, lo, hi):
    try:
        return max(lo, min(hi, float(val)))
    except (TypeError, ValueError):
        return fallback


def sanitize_params(model_key: str, raw: dict) -> dict:
    cfg = get_config()
    base = dict(cfg["models"][model_key].get("defaults", {}))
    raw = raw or {}
    p = {**DEFAULT_PARAMS, **base}

    for key in ("prompt", "negative_prompt", "backend", "weight_type", "max_vram",
                "extra_args", "sampling_method", "scheduler"):
        if key in raw:
            p[key] = str(raw.get(key) or "")

    for key in ("width", "height"):
        if key in raw:
            val = as_int(raw.get(key), p[key], 64, 4096)
            p[key] = max(64, val - val % 8)

    if "steps" in raw:
        p["steps"] = as_int(raw.get("steps"), p["steps"], 1, 200)
    if "batch_count" in raw:
        p["batch_count"] = as_int(raw.get("batch_count"), p["batch_count"], 1, 8)
    if "clip_skip" in raw:
        p["clip_skip"] = as_int(raw.get("clip_skip"), p["clip_skip"], -1, 6)
    if "cfg_scale" in raw:
        p["cfg_scale"] = as_float(raw.get("cfg_scale"), p["cfg_scale"], 0.0, 50.0)
    if "guidance" in raw:
        p["guidance"] = as_float(raw.get("guidance"), p.get("guidance", 0.0), 0.0, 30.0)
    for key in ("vae_tiling", "offload_to_cpu", "preview", "random_seed", "upscale_after"):
        if key in raw:
            p[key] = bool(raw.get(key))
    if "seed" in raw:
        p["seed"] = as_int(raw.get("seed"), -1, -1, 2**31 - 1)

    if "upscale_model" in raw:
        p["upscale_model"] = str(raw.get("upscale_model") or "")
    if "upscale_scale" in raw:
        p["upscale_scale"] = as_int(raw.get("upscale_scale"), p.get("upscale_scale", 4), 1, 8)
    if "upscale_tile" in raw:
        p["upscale_tile"] = as_int(raw.get("upscale_tile"), 0, 0, 2048)
    if "upscale_gpuid" in raw:
        p["upscale_gpuid"] = as_int(raw.get("upscale_gpuid"), 0, 0, 8)

    if p.get("random_seed") or p["seed"] < 0:
        p["seed"] = random.randint(1, 2**31 - 1)
    if not p.get("sampling_method") in SAMPLERS:
        p["sampling_method"] = "euler_a"
    if not p.get("scheduler"):
        p["scheduler"] = "auto"
    return p


def validate(model_key: str, params: dict) -> tuple[list[str], list[str]]:
    cfg = get_config()
    errors: list[str] = []
    warnings: list[str] = []

    cli = Path(cfg["sd_cli"]).expanduser()
    if not cli.is_file():
        errors.append(f"sd-cli nicht gefunden: {cli}")
    elif not os_access_x(cli):
        errors.append(f"sd-cli ist nicht ausfuehrbar: {cli}")

    paths = cfg["models"][model_key].get("paths", {})

    def check(field: str, required: bool, label: str) -> None:
        val = (paths.get(field) or "").strip()
        if not val:
            if required:
                errors.append(f"{label} fehlt - bitte im Zahnrad-Menue eintragen.")
            return
        if not Path(val).expanduser().exists():
            errors.append(f"{label} existiert nicht: {val}")

    if model_key == "flux":
        check("diffusion_model", True, "Diffusions-Modell")
        has_te = any((paths.get(k) or "").strip() for k in ("llm", "t5xxl", "clip_l", "clip_g"))
        if not has_te:
            errors.append("Text-Encoder (llm / t5xxl / clip_l) fehlt - bitte im Zahnrad-Menue eintragen.")
        else:
            for k, label in (("llm", "LLM (Text-Encoder)"), ("t5xxl", "t5xxl"), ("clip_l", "clip_l")):
                val = (paths.get(k) or "").strip()
                if val and not Path(val).expanduser().exists():
                    errors.append(f"{label} existiert nicht: {val}")
        vae = (paths.get("vae") or "").strip()
        if vae and not Path(vae).expanduser().exists():
            errors.append(f"VAE existiert nicht: {vae}")
        if not vae:
            warnings.append("Kein VAE hinterlegt - FLUX braucht normalerweise --vae.")
    else:
        check("model", True, "Modell")
        for k, label in (("clip_l", "Text-Encoder clip_l"), ("clip_g", "Text-Encoder clip_g")):
            val = (paths.get(k) or "").strip()
            if val and not Path(val).expanduser().exists():
                warnings.append(f"{label} existiert nicht: {val}")

    if not (params.get("prompt") or "").strip():
        errors.append("Prompt ist leer.")
    px = params["width"] * params["height"]
    if px > 1600 * 1600:
        warnings.append(f"{params['width']}x{params['height']} ist fuer 4 GB VRAM sehr gross.")

    if params.get("upscale_after"):
        ucfg = upscale_cfg()
        if not ucfg["binary"].is_file():
            errors.append(f"Real-ESRGAN nicht gefunden: {ucfg['binary']}")
        elif not os_access_x(ucfg["binary"]):
            errors.append(f"Real-ESRGAN ist nicht ausfuehrbar: {ucfg['binary']}")
        else:
            names = {m["name"] for m in list_upscale_models()}
            if params.get("upscale_model") not in names:
                warnings.append(f"Upscale-Modell nicht gefunden: {params.get('upscale_model')}")
    return errors, warnings


def os_access_x(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def build_command(model_key: str, params: dict, out_path: Path, preview_path: Path) -> tuple[list[str], str]:
    cfg = get_config()
    paths = cfg["models"][model_key].get("paths", {})
    cmd: list[str] = [cfg["sd_cli"]]

    for field, flag in PATH_FLAGS:
        val = (paths.get(field) or "").strip()
        if val:
            cmd += [flag, str(Path(val).expanduser())]

    if (paths.get("vae") or "").strip() and paths.get("vae_format"):
        cmd += ["--vae-format", str(paths["vae_format"])]

    if (params.get("backend") or "").strip():
        cmd += ["--backend", params["backend"].strip()]
    if params.get("weight_type"):
        cmd += ["--type", str(params["weight_type"])]
    if (params.get("max_vram") or "").strip():
        cmd += ["--max-vram", str(params["max_vram"]).strip()]
    if params.get("threads"):
        cmd += ["-t", str(int(params["threads"]))]
    if params.get("vae_tiling"):
        cmd += ["--vae-tiling"]
    if params.get("offload_to_cpu"):
        cmd += ["--offload-to-cpu"]

    cmd += ["-W", str(params["width"]), "-H", str(params["height"])]
    cmd += ["--steps", str(params["steps"])]
    cmd += ["--cfg-scale", f"{params['cfg_scale']:g}"]
    if params.get("guidance"):
        cmd += ["--guidance", f"{params['guidance']:g}"]
    cmd += ["--sampling-method", params["sampling_method"]]
    if params.get("scheduler") and params["scheduler"] != "auto":
        cmd += ["--scheduler", params["scheduler"]]
    if params.get("clip_skip", -1) >= 1:
        cmd += ["--clip-skip", str(int(params["clip_skip"]))]
    cmd += ["-s", str(params["seed"])]
    if params.get("batch_count", 1) > 1:
        cmd += ["-b", str(int(params["batch_count"]))]

    if params.get("preview"):
        cmd += ["--preview", "proj", "--preview-interval", "1", "--preview-path", str(preview_path)]

    if (params.get("extra_args") or "").strip():
        try:
            cmd += shlex.split(params["extra_args"].strip())
        except ValueError:
            pass

    cmd += ["-p", params["prompt"]]
    if (params.get("negative_prompt") or "").strip():
        cmd += ["-n", params["negative_prompt"]]
    cmd += ["-o", str(out_path)]

    pretty = " \\\n  ".join(shlex.quote(c) for c in cmd)
    return cmd, pretty


# --------------------------------------------------------------------------- API


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def api_config():
    return jsonify(get_config())


@app.post("/api/config")
def api_config_save():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Ungueltiges JSON"}), 400
    with _config_lock:
        merged = _deep_merge(_config, payload)
        # Modell-Defaults vollstaendig ersetzen, wenn mitgeliefert
        for key, model in (payload.get("models") or {}).items():
            if key in merged["models"] and isinstance(model, dict):
                if isinstance(model.get("defaults"), dict):
                    merged["models"][key]["defaults"] = {**DEFAULT_PARAMS, **model["defaults"]}
                if isinstance(model.get("paths"), dict):
                    merged["models"][key]["paths"] = model["paths"]
        portable = _map_paths(merged, portable_path)  # im Projekt liegende Pfade relativ ablegen
        _config.clear()
        _config.update(portable)
        write_config(portable)
        out = _map_paths(portable, resolve_path)      # Antwort mit absoluten Pfaden
    try:
        Path(str(out["output_dir"])).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    try:
        ucfg = upscale_cfg()
        ucfg["out_dir"].mkdir(parents=True, exist_ok=True)
        ucfg["upload_dir"].mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return jsonify(out)


@app.get("/api/devices")
def api_devices():
    cfg = get_config()
    try:
        res = subprocess.run([cfg["sd_cli"], "--list-devices"], capture_output=True,
                             text=True, timeout=15)
        text = (res.stdout or "") + (res.stderr or "")
        devices = []
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                devices.append({"id": parts[0], "name": parts[1]})
        return jsonify({"devices": devices})
    except (OSError, subprocess.SubprocessError) as exc:
        return jsonify({"devices": [], "error": str(exc)})


@app.post("/api/check")
def api_check():
    """Existenz-Pruefung fuer die Pfade im Einstellungs-Dialog."""
    payload = request.get_json(force=True, silent=True) or {}
    paths = payload.get("paths") or {}
    result = {}
    for key, val in paths.items():
        text = str(val or "").strip()
        if not text:
            result[key] = True
            continue
        p = Path(text).expanduser()
        try:
            result[key] = p.exists()
        except OSError:
            result[key] = False
    return jsonify({"paths": result})


@app.get("/api/browse")
def api_browse():
    raw = request.args.get("dir") or str(Path.home())
    target = Path(raw).expanduser()
    if not target.is_dir():
        target = Path.home()
    dirs, files = [], []
    try:
        entries = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return jsonify({"dir": str(target), "parent": None, "dirs": [], "files": [], "error": "Kein Zugriff"})
    only_models = request.args.get("models") == "1"
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append(entry.name)
            elif entry.suffix.lower() in MODEL_SUFFIXES or not only_models:
                if entry.suffix.lower() in MODEL_SUFFIXES or entry.suffix.lower() in IMG_SUFFIXES or not only_models:
                    files.append({"name": entry.name, "size": entry.stat().st_size,
                                  "is_model": entry.suffix.lower() in MODEL_SUFFIXES})
        except OSError:
            continue
    if only_models:
        files = [f for f in files if f["is_model"]]
    parent = str(target.parent) if target.parent != target else None
    return jsonify({"dir": str(target), "parent": parent, "dirs": dirs, "files": files})


# ----------------------------------------------------------- Checkpoint-Auswahl
# Dropdown wie bei Real-ESRGAN: Checkpoints des Reiters aus dem Ordner des
# eingestellten Modells auflisten und per Klick umschalten (z.B. Stil-Wahl).

CKPT_PATH_FIELD = {"sd15": "model", "sdxl": "model"}


def ckpt_dir(tab: str, cfg: dict) -> Path:
    """Ordner, in dem die Checkpoints des Reiters liegen (VZ des eingestellten Modells)."""
    paths = ((cfg.get("models") or {}).get(tab) or {}).get("paths") or {}
    current = str(paths.get("model") or "").strip()
    if current:
        parent = Path(current).expanduser().parent
        if parent.is_dir():
            return parent
    guess = BASE_DIR / "models" / "checkpoints" / ("SD1.5" if tab == "sd15" else "SDXL")
    return guess if guess.is_dir() else BASE_DIR / "models" / "checkpoints"


def list_checkpoints(tab: str) -> dict:
    """Alle Modell-Dateien im Ordner des Reiters (+ der eingestellte Pfad, falls er woanders liegt)."""
    cfg = get_config()
    paths = ((cfg.get("models") or {}).get(tab) or {}).get("paths") or {}
    current = str(paths.get("model") or "").strip()
    folder = ckpt_dir(tab, cfg)

    items: list[dict] = []
    try:
        entries = sorted(folder.rglob("*"), key=lambda p: str(p).lower())
    except OSError:
        entries = []
    for p in entries:
        try:
            if not p.is_file() or p.suffix.lower() not in MODEL_SUFFIXES:
                continue
            size = p.stat().st_size
        except OSError:
            continue
        items.append({"name": p.relative_to(folder).as_posix(), "path": str(p), "size": size})
    if current and not any(i["path"] == current for i in items):
        cp = Path(current).expanduser()
        items.insert(0, {"name": f"{cp.name} (anderer Ordner)", "path": current, "size": 0})
    return {"tab": tab, "dir": str(folder), "current": current, "models": items}


@app.get("/api/checkpoints/<tab>")
def api_checkpoints(tab: str):
    if tab not in CKPT_PATH_FIELD or tab not in (get_config().get("models") or {}):
        return jsonify({"error": "Unbekannter Reiter."}), 404
    return jsonify(list_checkpoints(tab))


@app.post("/api/checkpoints/select")
def api_checkpoint_select():
    """Checkpoint fuer einen Reiter waehlen (Pfad landet in config.json)."""
    payload = request.get_json(force=True, silent=True) or {}
    tab = str(payload.get("tab") or "")
    field = CKPT_PATH_FIELD.get(tab)
    if not field or tab not in (get_config().get("models") or {}):
        return jsonify({"error": "Unbekannter Reiter."}), 400
    path = str(payload.get("path") or "").strip()
    if not path:
        return jsonify({"error": "Kein Modell angegeben."}), 400
    target = Path(path).expanduser()
    try:
        if not target.is_file():
            return jsonify({"error": f"Datei nicht gefunden: {target}"}), 400
        if target.suffix.lower() not in MODEL_SUFFIXES:
            return jsonify({"error": f"Kein Modell-Format: {target.name}"}), 400
    except OSError:
        return jsonify({"error": "Pfad nicht lesbar."}), 400
    with _config_lock:
        paths = _config.setdefault("models", {}).setdefault(tab, {}).setdefault("paths", {})
        paths[field] = portable_path(str(target))      # relativ ablegen (portabel)
        write_config(_config)
        out = _map_paths(_config, resolve_path)
    return jsonify({"ok": True, "tab": tab, "path": resolve_path(str(target)), "config": out})


@app.get("/api/status")
def api_status():
    with job_lock:
        job = current_job
    if not job:
        return jsonify({"running": False})
    with job.lock:
        return jsonify({
            "running": not job.finished,
            "job": job.id,
            "model": job.model_key,
            "seed": job.params.get("seed"),
            "events": len(job.events),
        })


@app.post("/api/generate")
def api_generate():
    global current_job
    payload = request.get_json(force=True, silent=True) or {}
    model_key = payload.get("model")
    cfg = get_config()
    if model_key not in cfg["models"]:
        return jsonify({"errors": ["Unbekannter Modell-Reiter."]}), 400

    params = sanitize_params(model_key, payload.get("params") or {})
    errors, warnings = validate(model_key, params)
    if errors:
        return jsonify({"errors": errors, "warnings": warnings}), 422

    out_dir = Path(cfg["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{model_key}_{params['seed']}"
    out_path = out_dir / f"{stem}.png"
    preview_path = out_dir / f"preview_{stem}.png"

    cmd, pretty = build_command(model_key, params, out_path, preview_path)
    job = Job(model_key, params, cmd, out_dir, stem, preview_path, pretty)

    with job_lock:
        if current_job and not current_job.finished:
            return jsonify({"errors": ["Es laeuft bereits ein Job - bitte zuerst abwarten oder abbrechen."]}), 409
        current_job = job
        JOBS[job.id] = job
        for old in [k for k, j in JOBS.items() if j.finished][:10]:
            JOBS.pop(old, None)

    threading.Thread(target=job.run, daemon=True).start()
    return jsonify({"job": job.id, "seed": params["seed"], "warnings": warnings, "command": pretty})


@app.post("/api/cancel")
def api_cancel():
    with job_lock:
        job = current_job
    if job and not job.finished:
        job.cancel()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Kein Job aktiv."})


@app.get("/api/stream")
def api_stream():
    job_id = request.args.get("job")
    after = request.headers.get("Last-Event-ID") or request.args.get("after") or "0"
    try:
        start_idx = int(after)
    except ValueError:
        start_idx = 0

    if not job_id:
        with job_lock:
            job = current_job
        job_id = job.id if job is not None else ""

    def generate():
        idx = start_idx
        last_ping = time.time()
        while True:
            job = JOBS.get(job_id)
            if job is None:
                yield "event: gone\ndata: {}\n\n"
                return
            with job.lock:
                events = job.events[idx:]
                finished = job.finished
            for ev in events:
                idx += 1
                yield f"id: {idx - 1}\nevent: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                last_ping = time.time()
            if finished:
                return
            if time.time() - last_ping > 15:
                yield ": keepalive\n\n"
                last_ping = time.time()
            time.sleep(0.12)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.get("/api/gallery")
def api_gallery():
    cfg = get_config()
    out_dir = Path(cfg["output_dir"]).expanduser()
    items = []
    if out_dir.is_dir():
        for path in out_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMG_SUFFIXES:
                continue
            if path.name.startswith("preview_"):
                continue
            meta = None
            side = path.with_suffix(".json")
            if side.exists():
                try:
                    meta = json.loads(side.read_text("utf-8"))
                except (OSError, ValueError):
                    meta = None
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append({"file": path.name, "url": f"/outputs/{path.name}",
                          "mtime": int(stat.st_mtime), "size": stat.st_size, "meta": meta})
    items.sort(key=lambda i: i["mtime"], reverse=True)
    return jsonify({"items": items[:300]})


@app.post("/api/delete")
def api_delete():
    payload = request.get_json(force=True, silent=True) or {}
    name = str(payload.get("file") or "")
    cfg = get_config()
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    target = (out_dir / name).resolve()
    if target.parent != out_dir or not target.is_file():
        return jsonify({"error": "Datei nicht gefunden."}), 404
    target.unlink()
    side = target.with_suffix(".json")
    if side.exists():
        side.unlink()
    return jsonify({"ok": True})


@app.get("/outputs/<path:name>")
def outputs(name):
    cfg = get_config()
    resp = send_from_directory(cfg["output_dir"], name, max_age=0)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/upscales/<path:name>")
def upscales(name):
    ucfg = upscale_cfg()
    resp = send_from_directory(str(ucfg["out_dir"]), name, max_age=0)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# --------------------------------------------------------------------- Upscale


@app.get("/api/upscale/models")
def api_upscale_models():
    ucfg = upscale_cfg()
    return jsonify({
        "models": list_upscale_models(),
        "default": ucfg["default_model"],
        "defaults": ucfg["raw"].get("defaults") or {},
        "binary": str(ucfg["binary"]),
        "binary_ok": ucfg["binary"].is_file() and os_access_x(ucfg["binary"]),
        "model_dir": str(ucfg["model_dir"]),
        "output_dir": str(ucfg["out_dir"]),
    })


@app.post("/api/upscale/upload")
def api_upscale_upload():
    """Bild fuer den Upscale-Reiter hochladen (landet in upscale/uploads)."""
    ucfg = upscale_cfg()
    file = request.files.get("file")
    if file is None or not (file.filename or "").strip():
        return jsonify({"errors": ["Keine Datei mitgesendet."]}), 400
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in IMG_SUFFIXES:
        return jsonify({"errors": [f"Kein Bilddatei-Format: {name}"]}), 400
    try:
        ucfg["upload_dir"].mkdir(parents=True, exist_ok=True)
        target = unique_path(ucfg["upload_dir"], Path(name).stem, Path(name).suffix.lower())
        file.save(str(target))
    except OSError as exc:
        return jsonify({"errors": [f"Upload fehlgeschlagen: {exc}"]}), 500
    return jsonify({"file": target.name, "name": name, "url": f"/upscales/uploads/{target.name}"})


@app.post("/api/upscale")
def api_upscale_run():
    """Real-ESRGAN Lauf starten (eigener Reiter)."""
    global current_job
    payload = request.get_json(force=True, silent=True) or {}
    ucfg = upscale_cfg()
    models = {m["name"]: m for m in list_upscale_models()}
    errors: list[str] = []

    if not ucfg["binary"].is_file():
        errors.append(f"Real-ESRGAN nicht gefunden: {ucfg['binary']}")
    elif not os_access_x(ucfg["binary"]):
        errors.append(f"Real-ESRGAN ist nicht ausfuehrbar: {ucfg['binary']}")
    if not models:
        errors.append(f"Keine Upscale-Modelle in {ucfg['model_dir']} gefunden.")

    model = str(payload.get("model") or ucfg["default_model"])
    info = models.get(model)
    if info is None and models:
        errors.append(f"Unbekanntes Upscale-Modell: {model}")

    scale = as_int(payload.get("scale"), 4, 1, 8)
    if info is not None and scale not in info["scales"]:
        scale = info["default_scale"]
    tile = as_int(payload.get("tile"), ucfg["tile"], 0, 2048)
    gpuid = as_int(payload.get("gpuid"), ucfg["gpuid"], 0, 8)

    src = None
    src_url = ""
    try:
        src, src_url = resolve_source(payload.get("source"))
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        return jsonify({"errors": errors}), 422

    try:
        ucfg["out_dir"].mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return jsonify({"errors": [f"Upscale-Ordner nicht beschreibbar: {ucfg['out_dir']} ({exc})"]}), 500

    dst = unique_path(ucfg["out_dir"], f"{src.stem}_{model}_x{scale}", ".png")
    cmd, pretty = build_upscale_command(src, dst, model, scale, tile, gpuid)
    params = {"model": model, "scale": scale, "tile": tile, "gpuid": gpuid}
    job = UpscaleJob(params, cmd, pretty, src, dst, src_url)

    with job_lock:
        if current_job and not current_job.finished:
            return jsonify({"errors": ["Es laeuft bereits ein Job - bitte zuerst abwarten oder abbrechen."]}), 409
        current_job = job
        JOBS[job.id] = job
        for old in [k for k, j in JOBS.items() if j.finished][:10]:
            JOBS.pop(old, None)

    threading.Thread(target=job.run, daemon=True).start()
    return jsonify({"job": job.id, "command": pretty, "target": f"/upscales/{dst.name}"})


@app.get("/api/upscale/list")
def api_upscale_list():
    """Verlauf der hochskalierten Bilder (eigener Ordner, nicht in der Galerie)."""
    ucfg = upscale_cfg()
    items = []
    for path in out_dir_files(ucfg["out_dir"]):
        if path.suffix.lower() not in IMG_SUFFIXES:
            continue
        meta = None
        side = path.with_suffix(".json")
        if side.exists():
            try:
                meta = json.loads(side.read_text("utf-8"))
            except (OSError, ValueError):
                meta = None
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append({"file": path.name, "url": f"/upscales/{path.name}",
                      "mtime": int(stat.st_mtime), "size": stat.st_size, "meta": meta})
    items.sort(key=lambda i: i["mtime"], reverse=True)
    return jsonify({"items": items[:300]})


@app.post("/api/upscale/delete")
def api_upscale_delete():
    payload = request.get_json(force=True, silent=True) or {}
    name = str(payload.get("file") or "")
    ucfg = upscale_cfg()
    root = ucfg["out_dir"].expanduser().resolve()
    target = (root / name).resolve()
    if target.parent != root or not target.is_file():
        return jsonify({"error": "Datei nicht gefunden."}), 404
    target.unlink()
    side = target.with_suffix(".json")
    if side.exists():
        side.unlink()
    return jsonify({"ok": True})


load_config()
try:
    Path(get_config()["output_dir"]).expanduser().mkdir(parents=True, exist_ok=True)
except OSError:
    pass
try:
    _ucfg = upscale_cfg()
    _ucfg["out_dir"].mkdir(parents=True, exist_ok=True)
    _ucfg["upload_dir"].mkdir(parents=True, exist_ok=True)
except OSError:
    pass


if __name__ == "__main__":
    conf = get_config()
    host = conf["listen"]["host"]
    port = int(conf["listen"]["port"])
    print(f"PotatoDiffusion  ->  http://{host}:{port}")
    app.config["TEMPLATES_AUTO_RELOAD"] = True  # index.html ohne Neustart aktualisieren
    app.run(host=host, port=port, threaded=True, debug=False)
