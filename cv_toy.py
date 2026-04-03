"""
CV Lab Toy — Computer vision demo for robotic laboratory equipment.
Works as a standalone script or in a Jupyter notebook.

Requirements:
    pip install gradio opencv-python-headless numpy ollama pillow

Ollama model (pick one based on your hardware):
    ollama pull qwen2.5vl:7b    # recommended — ~5 GB, good GPU or fast CPU
    ollama pull qwen2.5vl:3b    # lighter — ~2 GB, runs on most machines

Architecture:
  - Qwen2.5-VL does semantic detection (outputs bbox JSON natively)
  - Human reviews / edits / groups detections
  - Registry stores labelled examples (label + image crop)
  - Re-detection uses registry labels as few-shot hints in the prompt
"""

import base64
import io
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import ollama
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────

DETECT_MODEL  = "qwen2.5vl:7b"   # change to :3b for lighter hardware
REGISTRY_FILE = Path("cv_toy_registry.json")

# ── Colour palette ────────────────────────────────────────────────────────────

PALETTE = [
    (0, 255, 80),   (0, 180, 255),  (255, 180, 0),   (255, 60, 200),
    (80, 255, 255), (255, 100, 60), (160, 255, 0),    (200, 80, 255),
    (255, 220, 40), (0, 255, 180),  (255, 40, 80),    (40, 200, 255),
    (220, 255, 80), (255, 140, 200),(100, 100, 255),  (255, 200, 100),
    (0, 220, 140),  (255, 80, 140), (140, 220, 255),  (255, 160, 60),
]

def label_color(label: str) -> tuple:
    return PALETTE[hash(label) % len(PALETTE)]

# ── Registry ──────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {}

def save_registry(reg: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)

_registry: dict = load_registry()

def registry_labels() -> list[str]:
    return sorted(_registry.keys())

def add_to_registry(label: str, crop_pil: Image.Image, note: str = ""):
    global _registry
    b64 = encode_image_b64(crop_pil)
    _registry.setdefault(label, {"examples": []})
    _registry[label]["examples"].append({"crop_b64": b64, "note": note})
    save_registry(_registry)

def delete_from_registry(label: str):
    global _registry
    _registry.pop(label, None)
    save_registry(_registry)

# ── Image helpers ─────────────────────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def crop_pil(img: Image.Image, bbox: list) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return img.crop((x1, y1, x2, y2))

# ── LLM detection ─────────────────────────────────────────────────────────────

_DETECT_SYSTEM = """You are a precise object detection assistant for laboratory equipment analysis.
When given an image, detect every distinct physical object and return ONLY a JSON array.
Each element must have exactly: {"label": "<short_name>", "bbox": [x1, y1, x2, y2]}
- bbox values are ABSOLUTE pixel integers (not ratios, not percentages)
- x1,y1 = top-left corner; x2,y2 = bottom-right corner of the bounding box
- label should be concise snake_case: "pipette", "eppendorf_tube", "usb_drive", "ruler", "petri_dish"
Return ONLY the raw JSON array. No markdown, no code fences, no explanation."""


def _build_detect_prompt(img_w: int, img_h: int, guided: bool) -> str:
    base = (f"Detect all distinct physical objects in this image.\n"
            f"Image dimensions: {img_w}x{img_h} pixels.\n"
            f"Coordinates must be absolute pixel integers in range x:[0,{img_w}] y:[0,{img_h}].\n")
    if guided and _registry:
        labels = ", ".join(f'"{l}"' for l in registry_labels())
        base += (f"\nKnown object types — prefer these labels when applicable: {labels}\n"
                 f"You may also add new labels for objects not in the list.\n")
    base += "\nOutput ONLY a valid JSON array. Example format:\n"
    base += '[{"label": "pipette", "bbox": [120, 45, 310, 290]}, ...]'
    return base


def _parse_detections(raw: str, img_w: int, img_h: int) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "object")).strip()
        bbox  = item.get("bbox") or item.get("box") or item.get("coordinates")
        if not bbox or len(bbox) < 4:
            continue
        try:
            vals = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue
        # If model returned ratios (all ≤ 1.0), scale up to pixels
        if all(0.0 <= v <= 1.0 for v in vals):
            vals = [vals[0]*img_w, vals[1]*img_h, vals[2]*img_w, vals[3]*img_h]
        x1, y1, x2, y2 = [int(v) for v in vals]
        # Normalise corner order and clamp
        x1, x2 = sorted([max(0, min(x1, img_w)), max(0, min(x2, img_w))])
        y1, y2 = sorted([max(0, min(y1, img_h)), max(0, min(y2, img_h))])
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        results.append({"id": str(uuid.uuid4())[:8], "label": label,
                        "bbox": [x1, y1, x2, y2]})
    return results


def run_detection(img_pil: Image.Image, guided: bool = False) -> tuple[list[dict], str]:
    W, H = img_pil.size
    b64  = encode_image_b64(img_pil)
    prompt = _build_detect_prompt(W, H, guided)
    try:
        resp = ollama.chat(
            model=DETECT_MODEL,
            messages=[
                {"role": "system", "content": _DETECT_SYSTEM},
                {"role": "user",   "content": prompt, "images": [b64]},
            ],
        )
        raw  = resp["message"]["content"].strip()
        dets = _parse_detections(raw, W, H)
        return dets, raw
    except Exception as e:
        return [], f"[Detection error: {e}]"


# ── Annotation drawing ────────────────────────────────────────────────────────

def draw_detections(img_pil: Image.Image, detections: list[dict],
                    selected_ids: set | None = None) -> Image.Image:
    if img_pil is None:
        return img_pil
    if not detections:
        return img_pil.copy()

    base    = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    sel     = selected_ids or set()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label    = det["label"]
        r, g, b  = label_color(label)
        is_sel   = det["id"] in sel
        thickness = 4 if is_sel else 2
        fill_a    = 60 if is_sel else 25

        draw.rectangle([x1, y1, x2, y2],
                       fill=(r, g, b, fill_a),
                       outline=(r, g, b, 230), width=thickness)

        if is_sel:
            draw.rectangle([x1-3, y1-3, x2+3, y2+3],
                           outline=(255, 255, 255, 160), width=1)

        # Label pill
        char_w, char_h = 7, 13
        tw = len(label) * char_w + 10
        th = char_h + 6
        draw.rectangle([x1, max(0, y1-th), x1+tw, y1],
                       fill=(r, g, b, 220))
        draw.text((x1+4, max(0, y1-th)+2), label, fill=(0, 0, 0, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


# ── Webcam stream ─────────────────────────────────────────────────────────────

class WebcamStream:
    def __init__(self):
        self.cap = None; self.frame = None
        self._lock = threading.Lock(); self._running = False; self._thread = None

    def start(self, index: int = 0):
        if self._running: return "Already running."
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened(): return f"Cannot open camera {index}."
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return f"Camera {index} started."

    def stop(self):
        self._running = False
        if self.cap: self.cap.release(); self.cap = None
        self.frame = None
        return "Camera stopped."

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock: self.frame = frame
            time.sleep(0.03)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

webcam = WebcamStream()

# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
body, .gradio-container {
    background: #0a0e17 !important; color: #c8d8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important; }
.gr-button {
    background: #111827 !important; border: 1px solid #1e3a5f !important;
    color: #60a5fa !important; font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 4px !important; }
.gr-button:hover { background: #1a2f4a !important; border-color: #3b82f6 !important; }
.gr-button.primary { background: #1d4ed8 !important; color: #fff !important;
    border-color: #2563eb !important; }
.gr-button.stop { background: #7f1d1d !important; color: #fca5a5 !important;
    border-color: #991b1b !important; }
h1, h2, h3 { color: #93c5fd !important; font-family: 'IBM Plex Mono', monospace !important; }
label { color: #64748b !important; font-size: 0.75em !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important; }
textarea, input[type=text], input[type=number] {
    background: #070b12 !important; color: #93c5fd !important;
    border: 1px solid #1e3a5f !important;
    font-family: 'IBM Plex Mono', monospace !important; }
.tab-nav button { color: #64748b !important; font-family: 'IBM Plex Mono', monospace !important; }
.tab-nav button.selected { color: #93c5fd !important;
    border-bottom: 2px solid #3b82f6 !important; }
"""


def build_ui():
    with gr.Blocks(css=CSS, title="CV Lab Toy") as demo:
        gr.Markdown("# 🔬 CV Lab Toy\n*Semantic object detection · human-in-the-loop · local LLM*")

        current_image_state = gr.State(None)   # PIL Image
        detections_state    = gr.State([])     # list of {id, label, bbox}
        selected_ids_state  = gr.State(set())  # set of selected IDs

        # ════════════════════════════════════════════════════════════════════
        # Tab 1 — Input
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("📷 Input"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Still Image")
                    img_upload = gr.Image(label="Upload image", type="pil", height=240)
                    btn_use_upload = gr.Button("Use this image ▶", variant="primary")
                with gr.Column():
                    gr.Markdown("### Webcam")
                    cam_index = gr.Number(label="Camera index", value=0, precision=0)
                    with gr.Row():
                        btn_cam_start = gr.Button("Start")
                        btn_cam_stop  = gr.Button("Stop")
                    btn_cam_snap = gr.Button("📸 Grab frame", variant="primary")
                    cam_status = gr.Textbox(label="Status", interactive=False)

            current_display = gr.Image(label="Current working frame",
                                       type="pil", height=360, interactive=False)

            def use_upload(img):
                if img is None: return gr.update(), None, "No image uploaded."
                return img, img, "Image loaded."

            def cam_start(idx): return webcam.start(int(idx))
            def cam_stop():     return webcam.stop()
            def cam_snap():
                f = webcam.get_frame()
                if f is None: return gr.update(), None, "No frame — is camera running?"
                pil = cv_to_pil(f)
                return pil, pil, "Frame grabbed."

            btn_use_upload.click(use_upload, [img_upload],
                                 [current_display, current_image_state, cam_status])
            btn_cam_start.click(cam_start, [cam_index], [cam_status])
            btn_cam_stop .click(cam_stop,  [],          [cam_status])
            btn_cam_snap .click(cam_snap,  [],
                                [current_display, current_image_state, cam_status])

        # ════════════════════════════════════════════════════════════════════
        # Tab 2 — Detect & Review
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Detect & Review"):
            gr.Markdown(
                "**Auto-detect** asks the local LLM to find every object and draw boxes. "
                "**Click** a box to select it. **Shift-select** multiple boxes. "
                "Rename, group, or delete — then register what you want to track."
            )

            with gr.Row():
                btn_detect      = gr.Button("🤖 Detect all objects", variant="primary")
                btn_detect_guide = gr.Button("🗂️ Detect (guided by registry)")

            detect_status = gr.Textbox(label="Status", interactive=False, lines=2)

            detect_canvas = gr.Image(
                label="Detected objects — click a box to select",
                type="pil", height=500, interactive=False, show_download_button=False,
            )

            with gr.Row():
                # ── Edit panel ──────────────────────────────────────────────
                with gr.Column(scale=3):
                    gr.Markdown("#### Edit selection")
                    with gr.Row():
                        edit_label  = gr.Textbox(label="Rename selected to", placeholder="new_label")
                        btn_rename  = gr.Button("✏️ Rename")
                    with gr.Row():
                        group_label = gr.Textbox(label="Group selected → label", placeholder="group_label")
                        btn_group   = gr.Button("🔗 Group into one box")
                    with gr.Row():
                        btn_del_sel   = gr.Button("🗑️ Delete selected")
                        btn_clear_all = gr.Button("✕ Clear all boxes")

                # ── Manual add ──────────────────────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("#### Add box manually")
                    man_label = gr.Textbox(label="Label", placeholder="label")
                    with gr.Row():
                        man_x1 = gr.Number(label="X1", value=0, precision=0)
                        man_y1 = gr.Number(label="Y1", value=0, precision=0)
                    with gr.Row():
                        man_x2 = gr.Number(label="X2", value=100, precision=0)
                        man_y2 = gr.Number(label="Y2", value=100, precision=0)
                    btn_man_add = gr.Button("➕ Add")

            gr.Markdown("---")
            gr.Markdown("#### Register confirmed detections")
            with gr.Row():
                btn_reg_sel = gr.Button("💾 Register selected", variant="primary")
                btn_reg_all = gr.Button("💾 Register all visible")
            reg_status = gr.Textbox(label="Registration result", interactive=False)

            with gr.Accordion("🐛 Raw LLM output", open=False):
                det_raw = gr.Textbox(label="Raw response", lines=6, interactive=False)

            # ── Helpers ─────────────────────────────────────────────────────

            def do_detect(img, guided):
                if img is None:
                    return gr.update(), [], set(), "No image loaded — go to Input tab first.", ""
                dets, raw = run_detection(img, guided)
                n = len(dets)
                msg = (f"Found {n} object{'s' if n!=1 else ''}."
                       if dets else
                       "No objects detected. Check Ollama is running: `ollama serve`")
                return draw_detections(img, dets), dets, set(), msg, raw

            def on_click(img, dets, sel, evt: gr.SelectData):
                if not dets or img is None: return gr.update(), sel
                cx, cy = int(evt.index[0]), int(evt.index[1])
                hit, best = None, float("inf")
                for d in dets:
                    x1,y1,x2,y2 = d["bbox"]
                    if x1<=cx<=x2 and y1<=cy<=y2:
                        a = (x2-x1)*(y2-y1)
                        if a < best: best=a; hit=d["id"]
                if hit is None:
                    new_sel = set()
                elif hit in sel:
                    new_sel = sel - {hit}
                else:
                    new_sel = sel | {hit}
                return draw_detections(img, dets, new_sel), new_sel

            def rename_sel(img, dets, sel, label):
                if not label.strip(): return gr.update(), dets, "Enter a label."
                label = label.strip()
                dets = [dict(d, label=label) if d["id"] in sel else d for d in dets]
                return draw_detections(img, dets, sel), dets, f"Renamed to '{label}'."

            def group_sel(img, dets, sel, label):
                if not label.strip(): return gr.update(), dets, sel, "Enter a group label."
                label = label.strip()
                chosen = [d for d in dets if d["id"] in sel]
                rest   = [d for d in dets if d["id"] not in sel]
                if len(chosen) < 2:
                    return gr.update(), dets, sel, "Select ≥2 boxes to group."
                xs = [v for d in chosen for v in [d["bbox"][0], d["bbox"][2]]]
                ys = [v for d in chosen for v in [d["bbox"][1], d["bbox"][3]]]
                merged = {"id": str(uuid.uuid4())[:8], "label": label,
                          "bbox": [min(xs), min(ys), max(xs), max(ys)]}
                new_dets = rest + [merged]
                new_sel  = {merged["id"]}
                return draw_detections(img, new_dets, new_sel), new_dets, new_sel, \
                       f"Grouped {len(chosen)} → '{label}'."

            def delete_sel(img, dets, sel):
                new_dets = [d for d in dets if d["id"] not in sel]
                return draw_detections(img, new_dets, set()), new_dets, set(), \
                       f"Deleted {len(sel)} box(es)."

            def clear_all_dets(img):
                return (img.copy() if img else gr.update()), [], set(), "Cleared."

            def add_manual(img, dets, sel, label, x1, y1, x2, y2):
                if not label.strip(): return gr.update(), dets, sel, "Enter a label."
                det = {"id": str(uuid.uuid4())[:8], "label": label.strip(),
                       "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                new_dets = dets + [det]
                return draw_detections(img, new_dets, sel), new_dets, sel, \
                       f"Added '{label}'."

            def _register(img, dets, ids):
                if img is None: return "No image loaded."
                if not ids:     return "Nothing selected."
                n = 0
                for d in dets:
                    if d["id"] not in ids: continue
                    add_to_registry(d["label"], crop_pil(img, d["bbox"]))
                    n += 1
                return f"Registered {n} detection(s). Registry: {len(_registry)} labels."

            def reg_sel(img, dets, sel): return _register(img, dets, sel)
            def reg_all(img, dets):      return _register(img, dets, {d["id"] for d in dets})

            # ── Wiring ──────────────────────────────────────────────────────

            _det_outs = [detect_canvas, detections_state, selected_ids_state,
                         detect_status, det_raw]

            btn_detect      .click(lambda i: do_detect(i, False), [current_image_state], _det_outs)
            btn_detect_guide.click(lambda i: do_detect(i, True),  [current_image_state], _det_outs)

            detect_canvas.select(on_click,
                                 [current_image_state, detections_state, selected_ids_state],
                                 [detect_canvas, selected_ids_state])

            btn_rename.click(rename_sel,
                             [current_image_state, detections_state, selected_ids_state, edit_label],
                             [detect_canvas, detections_state, detect_status])

            btn_group.click(group_sel,
                            [current_image_state, detections_state, selected_ids_state, group_label],
                            [detect_canvas, detections_state, selected_ids_state, detect_status])

            btn_del_sel  .click(delete_sel,
                                [current_image_state, detections_state, selected_ids_state],
                                [detect_canvas, detections_state, selected_ids_state, detect_status])

            btn_clear_all.click(clear_all_dets, [current_image_state],
                                [detect_canvas, detections_state, selected_ids_state, detect_status])

            btn_man_add.click(add_manual,
                              [current_image_state, detections_state, selected_ids_state,
                               man_label, man_x1, man_y1, man_x2, man_y2],
                              [detect_canvas, detections_state, selected_ids_state, detect_status])

            btn_reg_sel.click(reg_sel,
                              [current_image_state, detections_state, selected_ids_state],
                              [reg_status])
            btn_reg_all.click(reg_all,
                              [current_image_state, detections_state],
                              [reg_status])

        # ════════════════════════════════════════════════════════════════════
        # Tab 3 — Points & Distance
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("📏 Points & Distance"):
            gr.Markdown(
                "Click to place named points. Distances computed between consecutive pairs. "
                "Set a mm/px scale for real-world units."
            )
            poi_state   = gr.State([])
            scale_state = gr.State(None)

            btn_poi_load = gr.Button("⬆ Load current image", variant="primary")
            poi_canvas   = gr.Image(label="Click to place points", type="pil",
                                    height=460, interactive=False, show_download_button=False)
            poi_hint     = gr.Textbox(value="Load an image first.", label="Status",
                                      interactive=False)

            with gr.Row():
                poi_label_in  = gr.Textbox(label="Next point label", value="P1")
                scale_in      = gr.Number(label="Scale (mm / px)", value=None, precision=5)
                btn_set_scale = gr.Button("Set scale")
            btn_clear_poi = gr.Button("🗑️ Clear points")
            poi_table     = gr.Textbox(label="Points + distances", lines=8, interactive=False)

            # ── helpers ─────────────────────────────────────────────────────

            def _draw_pois(base, pois, scale):
                if base is None: return base
                cv_img = pil_to_cv(base)
                colors = [(0,200,255),(255,100,0),(200,0,255),(0,255,150),(255,220,0)]
                for i,(x,y,lbl) in enumerate(pois):
                    c = colors[i % len(colors)]
                    cv2.circle(cv_img,(x,y),7,c,-1)
                    cv2.circle(cv_img,(x,y),9,(255,255,255),2)
                    cv2.putText(cv_img,lbl,(x+12,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
                for i in range(0,len(pois)-1,2):
                    if i+1<len(pois):
                        x1,y1,_=pois[i]; x2,y2,_=pois[i+1]
                        cv2.line(cv_img,(x1,y1),(x2,y2),(255,255,255),1,cv2.LINE_AA)
                        d=math.sqrt((x2-x1)**2+(y2-y1)**2)
                        txt=f"{d*scale:.2f}mm" if scale else f"{d:.1f}px"
                        cv2.putText(cv_img,txt,((x1+x2)//2+5,(y1+y2)//2-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                return cv_to_pil(cv_img)

            def _poi_text(pois, scale):
                lines=["Points:"]
                for i,(x,y,l) in enumerate(pois): lines.append(f"  [{i}] {l} ({x},{y})")
                lines.append("\nDistances (pairs):")
                for i in range(0,len(pois)-1,2):
                    if i+1<len(pois):
                        x1,y1,l1=pois[i]; x2,y2,l2=pois[i+1]
                        d=math.sqrt((x2-x1)**2+(y2-y1)**2)
                        s=f"{d*scale:.3f}mm" if scale else f"{d:.1f}px"
                        lines.append(f"  {l1} ↔ {l2}: {s}")
                return "\n".join(lines)

            def poi_load(img):
                if img is None: return gr.update(), [], "Load an image in Input tab."
                return img, [], "Ready — click to place points."

            def poi_click(base, pois, scale, label, evt: gr.SelectData):
                if base is None: return gr.update(), pois, "No image.", label, ""
                x,y = int(evt.index[0]), int(evt.index[1])
                new_pois = pois + [(x, y, label.strip() or f"P{len(pois)+1}")]
                m = re.match(r"(.*?)(\d+)$", label)
                next_lbl = (m.group(1)+str(int(m.group(2))+1)) if m else label
                return (_draw_pois(base, new_pois, scale), new_pois,
                        f"Added {label} at ({x},{y}).", next_lbl, _poi_text(new_pois, scale))

            def set_scale(base, pois, val):
                scale = float(val) if val else None
                return _draw_pois(base, pois, scale), scale, _poi_text(pois, scale)

            def clear_poi(base):
                return (base or gr.update()), [], "Cleared.", ""

            btn_poi_load .click(poi_load, [current_image_state], [poi_canvas, poi_state, poi_hint])
            poi_canvas.select(poi_click,
                              [current_image_state, poi_state, scale_state, poi_label_in],
                              [poi_canvas, poi_state, poi_hint, poi_label_in, poi_table])
            btn_set_scale.click(set_scale, [current_image_state, poi_state, scale_in],
                                [poi_canvas, scale_state, poi_table])
            btn_clear_poi.click(clear_poi, [current_image_state],
                                [poi_canvas, poi_state, poi_hint, poi_table])

        # ════════════════════════════════════════════════════════════════════
        # Tab 4 — Registry
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("🗃️ Registry"):
            gr.Markdown(
                "Registered labels are used as hints during **guided detection**. "
                "Each label accumulates example crops — more examples = better recall over time."
            )
            btn_ref_reg      = gr.Button("↻ Refresh")
            registry_display = gr.Textbox(label="Contents", lines=16, interactive=False)
            with gr.Row():
                del_lbl_in  = gr.Textbox(label="Label to delete")
                btn_del_lbl = gr.Button("🗑️ Delete", variant="stop")
            del_status = gr.Textbox(label="Status", interactive=False)

            def show_reg():
                if not _registry: return "Registry is empty."
                return "\n".join(
                    f"  • {lbl}  ({len(v['examples'])} example{'s' if len(v['examples'])!=1 else ''})"
                    for lbl,v in sorted(_registry.items())
                )

            def del_label(name):
                name = name.strip()
                if name in _registry:
                    delete_from_registry(name)
                    return f"Deleted '{name}'.", show_reg()
                return f"'{name}' not found.", show_reg()

            btn_ref_reg .click(show_reg, [], [registry_display])
            btn_del_lbl .click(del_label, [del_lbl_in], [del_status, registry_display])
            demo.load(show_reg, [], [registry_display])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def launch(**kwargs):
    """Launch the Gradio app. Pass share=True for a public link."""
    demo = build_ui()
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch(inbrowser=True)
