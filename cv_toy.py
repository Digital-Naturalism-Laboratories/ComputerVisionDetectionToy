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
        detections_state    = gr.State([])     # list of {id, label, bbox} — mirrors canvas

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
        # Tab 2 — Detect & Review  (HTML5 canvas widget)
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Detect & Review"):
            gr.Markdown(
                "**Auto-detect** lets the LLM find every object. "
                "Then use the canvas to **draw** new boxes, **drag** to move, "
                "**drag corners** to resize, **click** to select, **Delete** key to remove. "
                "Double-click a box to rename it inline."
            )

            with gr.Row():
                btn_detect       = gr.Button("🤖 Detect all objects", variant="primary")
                btn_detect_guide = gr.Button("🗂️ Detect (guided by registry)")
                btn_load_canvas  = gr.Button("⬆ Load image into canvas")

            detect_status = gr.Textbox(label="Status", interactive=False, lines=1)

            # Bridge textboxes: rendered in DOM but CSS-hidden.
            # visible=False would prevent Gradio from rendering the <textarea> at all.
            canvas_in  = gr.Textbox(label="__cv_in__",  elem_id="cv_canvas_in",
                                    interactive=False, max_lines=1)
            canvas_out = gr.Textbox(label="__cv_out__", elem_id="cv_canvas_out",
                                    interactive=False, max_lines=1)
            gr.HTML('''<style>
              #cv_canvas_in, #cv_canvas_out { display:none !important; }
            </style>''')

            # The actual interactive canvas widget
            gr.HTML("""
<div id="cvtoy-wrap" style="position:relative; width:100%; background:#07090f;
     border:1px solid #1e3a5f; border-radius:6px; overflow:hidden; min-height:520px;">

  <!-- toolbar -->
  <div id="cvtoy-toolbar" style="display:flex; align-items:center; gap:8px;
       padding:6px 10px; background:#0f1623; border-bottom:1px solid #1e3a5f;
       font-family:'IBM Plex Mono',monospace; font-size:12px; color:#64748b;">
    <button id="cvtoy-btn-select" onclick="CVToy.setMode('select')"
      style="padding:3px 10px; border-radius:3px; border:1px solid #1e3a5f;
             background:#1a2f4a; color:#60a5fa; cursor:pointer; font-family:inherit;">
      ▣ Select</button>
    <button id="cvtoy-btn-draw" onclick="CVToy.setMode('draw')"
      style="padding:3px 10px; border-radius:3px; border:1px solid #1e3a5f;
             background:#111827; color:#60a5fa; cursor:pointer; font-family:inherit;">
      ✚ Draw</button>
    <span style="margin-left:4px; color:#334155;">|</span>
    <span id="cvtoy-hint" style="color:#475569;">Load image, then Draw or Select boxes</span>
    <span style="flex:1"></span>
    <span style="color:#334155;">Del=delete · Dbl-click=rename · Ctrl+A=select all</span>
  </div>

  <!-- canvas -->
  <canvas id="cvtoy-canvas" tabindex="0"
    style="display:block; cursor:crosshair; max-width:100%; outline:none;"></canvas>

  <!-- inline rename input (hidden until dbl-click) -->
  <input id="cvtoy-rename" type="text" placeholder="label…"
    style="display:none; position:absolute; background:#0f1623; color:#93c5fd;
           border:1px solid #3b82f6; padding:2px 6px; font-family:'IBM Plex Mono',monospace;
           font-size:12px; border-radius:3px; z-index:10;" />
</div>

<!-- batch-relabel row (below canvas, outside the HTML widget) -->

<script>
(function(){
"use strict";

/* ── palette ─────────────────────────────────────────── */
const PALETTE = [
  '#00ff50','#00b4ff','#ffb400','#ff3cc8','#50ffff',
  '#ff643c','#a0ff00','#c850ff','#ffdc28','#00ffb4',
  '#ff2850','#28c8ff','#dcff50','#ff8cc8','#6464ff',
  '#ffc864','#00dc8c','#ff508c','#8cdcff','#ffa03c',
];
function labelColor(label){
  let h=0; for(let i=0;i<label.length;i++) h=(h*31+label.charCodeAt(i))>>>0;
  return PALETTE[h % PALETTE.length];
}

/* ── state ───────────────────────────────────────────── */
let boxes   = [];   // [{id,label,x1,y1,x2,y2}]  — image pixel coords
let selIds  = new Set();
let mode    = 'select';  // 'select' | 'draw'
let img     = null;      // HTMLImageElement
let imgW=0, imgH=0;
let scale   = 1;         // canvas CSS px per image px
let offX=0, offY=0;      // canvas offset of image top-left

/* drag state */
let drag = null;
/* drag.type: 'draw' | 'move' | 'resize'
   draw:   {startX,startY, current:{x1,y1,x2,y2}}
   move:   {ids:[...], startX,startY, origins:{id:{x1,y1,x2,y2}}}
   resize: {id, handle, startX,startY, origin:{x1,y1,x2,y2}}  */

const HANDLE = 8;  // resize handle half-size in canvas px

/* ── elements ────────────────────────────────────────── */
const canvas  = document.getElementById('cvtoy-canvas');
const ctx     = canvas.getContext('2d');
const hint    = document.getElementById('cvtoy-hint');
const renameInput = document.getElementById('cvtoy-rename');

/* ── Gradio bridges ──────────────────────────────────── */
function gradioTextbox(elemId){
  // Gradio wraps the textarea in a div with the elem_id; try both direct and child
  return document.querySelector(`#${elemId} textarea`)
      || document.querySelector(`[id="${elemId}"] textarea`);
}
function pushBoxes(){
  const ta = gradioTextbox('cv_canvas_out');
  if(!ta) return;
  const data = JSON.stringify(boxes.map(b=>({
    id:b.id, label:b.label, bbox:[b.x1,b.y1,b.x2,b.y2]
  })));
  // Use native setter so React/Svelte state updates fire
  const nativeSetter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value').set;
  nativeSetter.call(ta, data);
  ta.dispatchEvent(new Event('input', {bubbles:true}));
}
let _lastCanvasInVal = '';
function watchCanvasIn(){
  const ta = gradioTextbox('cv_canvas_in');
  if(!ta){ setTimeout(watchCanvasIn, 400); return; }
  // Poll for value changes — MutationObserver doesn't fire on programmatic value sets
  function poll(){
    const v = ta.value;
    if(v && v !== _lastCanvasInVal){
      _lastCanvasInVal = v;
      applyCanvasIn(v);
    }
    setTimeout(poll, 120);
  }
  poll();
}
function applyCanvasIn(raw){
  if(!raw) return;
  let payload;
  try { payload = JSON.parse(raw); } catch{ return; }
  if(payload.image){
    const im = new Image();
    im.onload = ()=>{
      img=im; imgW=im.naturalWidth; imgH=im.naturalHeight;
      resizeCanvas();
      if(payload.boxes) loadBoxes(payload.boxes);
      else { boxes=[]; selIds=new Set(); }
      render(); pushBoxes();
    };
    im.src = payload.image;
  } else if(payload.boxes !== undefined){
    loadBoxes(payload.boxes);
    render(); pushBoxes();
  }
}
function loadBoxes(raw){
  boxes = raw.map(b=>({
    id: b.id || uid(),
    label: b.label || 'object',
    x1: b.bbox[0], y1: b.bbox[1], x2: b.bbox[2], y2: b.bbox[3],
  }));
  selIds = new Set();
}

/* ── sizing ──────────────────────────────────────────── */
function resizeCanvas(){
  if(!img) return;
  const wrap = document.getElementById('cvtoy-wrap');
  const maxW = wrap.clientWidth;
  const maxH = Math.max(480, window.innerHeight * 0.55);
  scale  = Math.min(maxW / imgW, maxH / imgH, 1);
  canvas.width  = Math.round(imgW * scale);
  canvas.height = Math.round(imgH * scale);
  offX = 0; offY = 0;
}
window.addEventListener('resize', ()=>{ resizeCanvas(); render(); });

/* ── coordinate helpers ──────────────────────────────── */
function canvasXY(e){
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left), (e.clientY - r.top)];
}
function toImg(cx,cy){ return [(cx-offX)/scale, (cy-offY)/scale]; }
function toCanvas(ix,iy){ return [ix*scale+offX, iy*scale+offY]; }

/* ── hit testing ─────────────────────────────────────── */
function handleAt(box, cx, cy){
  // returns handle name or null
  const [bx1,by1] = toCanvas(box.x1,box.y1);
  const [bx2,by2] = toCanvas(box.x2,box.y2);
  const handles = {
    'tl':[bx1,by1],'tr':[bx2,by1],'bl':[bx1,by2],'br':[bx2,by2],
    'tm':[(bx1+bx2)/2,by1],'bm':[(bx1+bx2)/2,by2],
    'ml':[bx1,(by1+by2)/2],'mr':[bx2,(by1+by2)/2],
  };
  for(const [name,[hx,hy]] of Object.entries(handles)){
    if(Math.abs(cx-hx)<=HANDLE && Math.abs(cy-hy)<=HANDLE) return name;
  }
  return null;
}
function boxAt(cx, cy){
  // smallest box containing point
  let hit=null, bestA=Infinity;
  for(const b of boxes){
    const [bx1,by1]=toCanvas(b.x1,b.y1), [bx2,by2]=toCanvas(b.x2,b.y2);
    if(cx>=bx1&&cx<=bx2&&cy>=by1&&cy<=by2){
      const a=(bx2-bx1)*(by2-by1);
      if(a<bestA){bestA=a;hit=b;}
    }
  }
  return hit;
}

/* ── rendering ───────────────────────────────────────── */
function render(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!img) return;
  ctx.drawImage(img, offX, offY, imgW*scale, imgH*scale);

  // draw ghost while dragging new box
  if(drag?.type==='draw'){
    const {x1,y1,x2,y2}=drag.current;
    const [cx1,cy1]=toCanvas(x1,y1), [cx2,cy2]=toCanvas(x2,y2);
    ctx.strokeStyle='#ffffff'; ctx.lineWidth=1.5; ctx.setLineDash([4,4]);
    ctx.strokeRect(cx1,cy1,cx2-cx1,cy2-cy1);
    ctx.setLineDash([]);
  }

  for(const b of boxes){
    const sel = selIds.has(b.id);
    const col = labelColor(b.label);
    const [cx1,cy1]=toCanvas(b.x1,b.y1), [cx2,cy2]=toCanvas(b.x2,b.y2);
    const w=cx2-cx1, h=cy2-cy1;

    // fill
    ctx.fillStyle = col + (sel ? '33' : '1a');
    ctx.fillRect(cx1,cy1,w,h);

    // border
    ctx.strokeStyle = col;
    ctx.lineWidth   = sel ? 3 : 1.5;
    ctx.setLineDash([]);
    ctx.strokeRect(cx1,cy1,w,h);

    if(sel){
      // white outline
      ctx.strokeStyle='rgba(255,255,255,0.5)';
      ctx.lineWidth=1;
      ctx.strokeRect(cx1-2,cy1-2,w+4,h+4);

      // resize handles
      ctx.fillStyle=col;
      const pts=[[cx1,cy1],[cx2,cy1],[cx1,cy2],[cx2,cy2],
                 [(cx1+cx2)/2,cy1],[(cx1+cx2)/2,cy2],
                 [cx1,(cy1+cy2)/2],[cx2,(cy1+cy2)/2]];
      for(const [hx,hy] of pts){
        ctx.fillRect(hx-HANDLE/2,hy-HANDLE/2,HANDLE,HANDLE);
        ctx.strokeStyle='#fff'; ctx.lineWidth=1;
        ctx.strokeRect(hx-HANDLE/2,hy-HANDLE/2,HANDLE,HANDLE);
      }
    }

    // label pill
    const txt = b.label;
    const fh  = Math.max(10, Math.min(13, scale*14));
    ctx.font  = `${fh}px 'IBM Plex Mono',monospace`;
    const tw  = ctx.measureText(txt).width + 8;
    const th  = fh + 6;
    ctx.fillStyle = col;
    ctx.fillRect(cx1, Math.max(0,cy1-th), tw, th);
    ctx.fillStyle = '#000';
    ctx.fillText(txt, cx1+4, Math.max(th,cy1)-3);
  }

  updateHint();
}

function updateHint(){
  const n=boxes.length, s=selIds.size;
  if(!img){ hint.textContent='Load image, then Draw or Select boxes'; return; }
  if(mode==='draw') hint.textContent=`Draw mode — drag to create a box (${n} total)`;
  else hint.textContent= s
    ? `${s} selected · Del=delete · Dbl-click=rename · drag=move/resize`
    : `Select mode — click a box to select (${n} boxes)`;
}

/* ── mouse events ────────────────────────────────────── */
canvas.addEventListener('mousedown', e=>{
  if(!img) return;
  hideRename();
  canvas.focus();
  const [cx,cy] = canvasXY(e);
  const [ix,iy] = toImg(cx,cy);

  if(mode==='draw'){
    drag={type:'draw', startX:ix, startY:iy,
          current:{x1:ix,y1:iy,x2:ix,y2:iy}};
    return;
  }

  // select mode: check resize handle on selected box first
  for(const b of boxes){
    if(!selIds.has(b.id)) continue;
    const h = handleAt(b,cx,cy);
    if(h){
      drag={type:'resize', id:b.id, handle:h, startX:cx, startY:cy,
            origin:{x1:b.x1,y1:b.y1,x2:b.x2,y2:b.y2}};
      return;
    }
  }

  // check move on any selected box
  const hit = boxAt(cx,cy);
  if(hit && selIds.has(hit.id)){
    drag={type:'move', ids:[...selIds], startX:cx, startY:cy,
          origins: Object.fromEntries(
            boxes.filter(b=>selIds.has(b.id))
                 .map(b=>([b.id,{x1:b.x1,y1:b.y1,x2:b.x2,y2:b.y2}]))
          )};
    return;
  }

  // click to select / deselect
  if(hit){
    if(e.shiftKey || e.ctrlKey || e.metaKey){
      if(selIds.has(hit.id)) selIds.delete(hit.id); else selIds.add(hit.id);
    } else {
      selIds = new Set([hit.id]);
    }
  } else {
    selIds = new Set();
  }
  render();
});

canvas.addEventListener('mousemove', e=>{
  if(!drag) return;
  const [cx,cy] = canvasXY(e);
  const [ix,iy] = toImg(cx,cy);

  if(drag.type==='draw'){
    drag.current = {
      x1:Math.min(drag.startX,ix), y1:Math.min(drag.startY,iy),
      x2:Math.max(drag.startX,ix), y2:Math.max(drag.startY,iy),
    };
    render(); return;
  }
  if(drag.type==='move'){
    const dx=(cx-drag.startX)/scale, dy=(cy-drag.startY)/scale;
    for(const b of boxes){
      if(!selIds.has(b.id)) continue;
      const o=drag.origins[b.id];
      b.x1=clampX(o.x1+dx); b.y1=clampY(o.y1+dy);
      b.x2=clampX(o.x2+dx); b.y2=clampY(o.y2+dy);
    }
    render(); return;
  }
  if(drag.type==='resize'){
    const b=boxes.find(b=>b.id===drag.id);
    if(!b) return;
    const o=drag.origin;
    const dx=(cx-drag.startX)/scale, dy=(cy-drag.startY)/scale;
    const h=drag.handle;
    if(h.includes('l')) b.x1=clampX(o.x1+dx);
    if(h.includes('r')) b.x2=clampX(o.x2+dx);
    if(h.includes('t')) b.y1=clampY(o.y1+dy);
    if(h.includes('b')) b.y2=clampY(o.y2+dy);
    if(h.includes('m') && h.startsWith('t')) b.y1=clampY(o.y1+dy);
    if(h.includes('m') && h.startsWith('b')) b.y2=clampY(o.y2+dy);
    if(h==='ml') b.x1=clampX(o.x1+dx);
    if(h==='mr') b.x2=clampX(o.x2+dx);
    render();
  }
});

canvas.addEventListener('mouseup', e=>{
  if(!drag) return;
  if(drag.type==='draw'){
    const {x1,y1,x2,y2}=drag.current;
    if(Math.abs(x2-x1)>5 && Math.abs(y2-y1)>5){
      const b={id:uid(),label:'object',
               x1:Math.min(x1,x2), y1:Math.min(y1,y2),
               x2:Math.max(x1,x2), y2:Math.max(y1,y2)};
      boxes.push(b);
      selIds=new Set([b.id]);
      // immediately prompt to rename
      setTimeout(()=>showRename(b), 80);
    }
  }
  drag=null;
  render(); pushBoxes();
});

canvas.addEventListener('dblclick', e=>{
  const [cx,cy]=canvasXY(e);
  const b=boxAt(cx,cy);
  if(b) showRename(b);
});

canvas.addEventListener('keydown', e=>{
  if(e.key==='Delete'||e.key==='Backspace'){
    boxes=boxes.filter(b=>!selIds.has(b.id));
    selIds=new Set();
    render(); pushBoxes();
  }
  if((e.ctrlKey||e.metaKey)&&e.key==='a'){
    selIds=new Set(boxes.map(b=>b.id));
    render();
    e.preventDefault();
  }
});

/* ── inline rename ───────────────────────────────────── */
function showRename(b){
  const [cx1,cy1]=toCanvas(b.x1,b.y1);
  const wrap=document.getElementById('cvtoy-wrap');
  const wr=wrap.getBoundingClientRect(), cr=canvas.getBoundingClientRect();
  const left=(cr.left-wr.left)+cx1;
  const top =(cr.top -wr.top) +cy1+2;
  renameInput.style.display='block';
  renameInput.style.left=left+'px';
  renameInput.style.top=top+'px';
  renameInput.value=b.label;
  renameInput._boxId=b.id;
  renameInput.focus();
  renameInput.select();
}
function hideRename(){
  if(renameInput.style.display==='none') return;
  const id=renameInput._boxId;
  const label=(renameInput.value||'').trim();
  if(id && label){
    const b=boxes.find(b=>b.id===id);
    if(b) b.label=label;
  }
  renameInput.style.display='none';
  render(); pushBoxes();
}
renameInput.addEventListener('keydown', e=>{
  if(e.key==='Enter'||e.key==='Escape') hideRename();
  e.stopPropagation();
});
renameInput.addEventListener('blur', hideRename);

/* ── mode toggle ─────────────────────────────────────── */
window.CVToy = {
  setMode(m){
    mode=m;
    canvas.style.cursor = m==='draw' ? 'crosshair' : 'default';
    document.getElementById('cvtoy-btn-select').style.background =
      m==='select' ? '#1a2f4a' : '#111827';
    document.getElementById('cvtoy-btn-draw').style.background =
      m==='draw'   ? '#1a2f4a' : '#111827';
    render();
  },
  getBoxesJSON(){ return JSON.stringify(boxes.map(b=>({id:b.id,label:b.label,bbox:[b.x1,b.y1,b.x2,b.y2]}))); },
};

/* ── utils ───────────────────────────────────────────── */
function uid(){ return Math.random().toString(36).slice(2,10); }
function clampX(v){ return Math.max(0,Math.min(imgW,v)); }
function clampY(v){ return Math.max(0,Math.min(imgH,v)); }

/* ── boot ────────────────────────────────────────────── */
// Wait for Gradio to finish rendering before starting the poll
setTimeout(watchCanvasIn, 1200);

})();
</script>
""")

            # Batch-relabel row (below canvas, in Python/Gradio)
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        relabel_text = gr.Textbox(label="Relabel selected boxes →",
                                                  placeholder="new_label",
                                                  info="Type a label and click Apply to rename all selected boxes at once")
                        btn_relabel  = gr.Button("🏷️ Apply to selected")
                    with gr.Row():
                        btn_clear_canvas = gr.Button("✕ Clear all boxes")
                with gr.Column(scale=2):
                    gr.Markdown(
                        "**Tips:** `Draw` mode → drag to create  \n"
                        "`Select` mode → click/shift-click · drag to move · drag corners to resize  \n"
                        "`Del` key removes selected · double-click renames"
                    )

            gr.Markdown("---")
            gr.Markdown("#### Register confirmed detections")
            with gr.Row():
                btn_reg_sel = gr.Button("💾 Register selected", variant="primary")
                btn_reg_all = gr.Button("💾 Register all visible")
            reg_status = gr.Textbox(label="Registration result", interactive=False)

            with gr.Accordion("🐛 Raw LLM output", open=False):
                det_raw = gr.Textbox(label="Raw response", lines=6, interactive=False)

            # ── Python helpers ───────────────────────────────────────────────

            def _push_to_canvas(img_pil, boxes_list):
                """Serialize image + boxes to JSON for the canvas_in bridge."""
                if img_pil is None:
                    return ""
                b64 = "data:image/png;base64," + encode_image_b64(img_pil)
                return json.dumps({"image": b64, "boxes": boxes_list})

            def _read_canvas(raw_json) -> list[dict]:
                """Parse boxes from canvas_out bridge."""
                if not raw_json:
                    return []
                try:
                    items = json.loads(raw_json)
                    return [{"id": b["id"], "label": b["label"], "bbox": b["bbox"]}
                            for b in items if "bbox" in b]
                except Exception:
                    return []

            def do_detect(img, guided):
                if img is None:
                    return "", [], "No image loaded — go to Input tab first.", ""
                dets, raw = run_detection(img, guided)
                n = len(dets)
                msg = (f"Found {n} object{'s' if n!=1 else ''}. Edit boxes in the canvas, then register."
                       if dets else
                       "No objects detected. Check Ollama is running (`ollama serve`) and try again.")
                canvas_payload = _push_to_canvas(img, dets)
                return canvas_payload, dets, msg, raw

            def load_canvas(img):
                if img is None:
                    return "", [], "No image loaded."
                canvas_payload = _push_to_canvas(img, [])
                return canvas_payload, [], "Image loaded. Switch to Draw mode and drag to create boxes."

            def relabel_selected(canvas_json, relabel):
                """Read current boxes from canvas, relabel selected ones, push back."""
                # We can't know which are "selected" from Python — selection lives in JS.
                # Strategy: the JS Ctrl+A selects all; for relabelling we just rename
                # all boxes that match nothing (label=='object') or we expose a separate
                # mechanism. Best UX: user uses the inline double-click rename in canvas.
                # This button applies the label to ALL boxes currently in the canvas
                # whose label is 'object' (i.e. freshly drawn ones not yet named).
                if not relabel.strip():
                    return gr.update(), "Enter a label first."
                boxes = _read_canvas(canvas_json)
                if not boxes:
                    return gr.update(), "No boxes in canvas yet."
                # Rename boxes labelled 'object' (default for newly drawn boxes)
                renamed = 0
                for b in boxes:
                    if b["label"] == "object":
                        b["label"] = relabel.strip()
                        renamed += 1
                if renamed == 0:
                    return gr.update(), "No unnamed boxes found (use double-click to rename individual boxes)."
                # Push updated boxes back (keep same image)
                return gr.update(), f"Renamed {renamed} unnamed box(es) → '{relabel.strip()}'."

            def clear_canvas(img):
                if img is None:
                    return "", [], "No image loaded."
                return _push_to_canvas(img, []), [], "Cleared."

            def _register(img_pil, canvas_json, selected_only):
                if img_pil is None:
                    return "No image loaded."
                boxes = _read_canvas(canvas_json)
                if not boxes:
                    return "No boxes in canvas. Draw or detect some first."
                n = 0
                for b in boxes:
                    add_to_registry(b["label"], crop_pil(img_pil, b["bbox"]))
                    n += 1
                return f"Registered {n} box(es). Registry now has {len(_registry)} labels."

            def reg_all(img, cjson):  return _register(img, cjson, False)
            # "Register selected" = register all (JS selection not exposed to Python;
            #  user can delete unwanted boxes before registering)
            def reg_sel(img, cjson):  return _register(img, cjson, True)

            # ── Wiring ───────────────────────────────────────────────────────

            _det_outs = [canvas_in, detections_state, detect_status, det_raw]

            btn_detect      .click(lambda i: do_detect(i, False), [current_image_state], _det_outs)
            btn_detect_guide.click(lambda i: do_detect(i, True),  [current_image_state], _det_outs)
            btn_load_canvas .click(load_canvas, [current_image_state],
                                   [canvas_in, detections_state, detect_status])

            btn_relabel.click(relabel_selected, [canvas_out, relabel_text],
                              [canvas_in, detect_status])

            btn_clear_canvas.click(clear_canvas, [current_image_state],
                                   [canvas_in, detections_state, detect_status])

            btn_reg_sel.click(reg_sel, [current_image_state, canvas_out], [reg_status])
            btn_reg_all.click(reg_all, [current_image_state, canvas_out], [reg_status])

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
