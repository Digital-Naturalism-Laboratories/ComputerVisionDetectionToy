"""
CV Lab Toy — Computer vision demo for robotic laboratory equipment.
Works as a standalone script or in a Jupyter notebook.

Requirements:
    pip install gradio opencv-python-headless numpy ollama pillow

Ollama model:
    ollama pull moondream
"""

import base64
import io
import json
import math
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import ollama
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "moondream"
TEMPLATES_FILE = Path("cv_toy_objects.json")

# ── Persistent object store ───────────────────────────────────────────────────

def load_objects() -> dict:
    if TEMPLATES_FILE.exists():
        with open(TEMPLATES_FILE) as f:
            return json.load(f)
    return {}

def save_objects(objects: dict):
    with open(TEMPLATES_FILE, "w") as f:
        json.dump(objects, f, indent=2)

# In-memory state
_objects: dict = load_objects()   # name → {"descriptor_bytes": ..., "keypoints_data": ...}
_poi: list = []                    # list of (x, y, label) for current frame
_scale_mm_per_px: float | None = None

# ── Image helpers ─────────────────────────────────────────────────────────────

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── ORB feature matching ──────────────────────────────────────────────────────

orb = cv2.ORB_create(nfeatures=1000)
FLANN_INDEX_LSH = 6
flann = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_LSH, "table_number": 6, "key_size": 12, "multi_probe_level": 1},
    {"checks": 50},
)

def compute_features(cv_img: np.ndarray):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des

def serialize_keypoints(kp) -> list:
    return [
        {"pt": list(k.pt), "size": k.size, "angle": k.angle,
         "response": k.response, "octave": k.octave, "class_id": k.class_id}
        for k in kp
    ]

def deserialize_keypoints(data: list):
    return [
        cv2.KeyPoint(
            x=d["pt"][0], y=d["pt"][1], size=d["size"],
            angle=d["angle"], response=d["response"],
            octave=d["octave"], class_id=d["class_id"],
        )
        for d in data
    ]

def register_object(name: str, crop_cv: np.ndarray):
    """Store ORB descriptors for a named object."""
    kp, des = compute_features(crop_cv)
    if des is None or len(des) < 5:
        return False, "Too few features detected in crop — try a larger or more textured region."
    _objects[name] = {
        "keypoints": serialize_keypoints(kp),
        "descriptors": des.tolist(),
    }
    save_objects(_objects)
    return True, f"Registered '{name}' with {len(kp)} keypoints."

def find_object_in_frame(name: str, frame_cv: np.ndarray):
    """Return annotated frame + bounding polygon or None."""
    if name not in _objects:
        return frame_cv, "Object not found in registry."

    obj = _objects[name]
    kp_obj = deserialize_keypoints(obj["keypoints"])
    des_obj = np.array(obj["descriptors"], dtype=np.uint8)

    kp_scene, des_scene = compute_features(frame_cv)
    if des_scene is None or len(des_scene) < 5:
        return frame_cv, "Not enough features in scene."

    try:
        matches = flann.knnMatch(des_obj, des_scene, k=2)
    except cv2.error:
        return frame_cv, "Matching failed — try retraining the object."

    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

    out = frame_cv.copy()
    if len(good) >= 8:
        src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None:
            h_obj, w_obj = frame_cv.shape[:2]  # use scene size as proxy
            # get object template bounding box and project it
            kp_pts = np.float32([k.pt for k in kp_obj])
            x0, y0 = kp_pts.min(axis=0)
            x1, y1 = kp_pts.max(axis=0)
            corners = np.float32([[x0, y0],[x1, y0],[x1, y1],[x0, y1]]).reshape(-1, 1, 2)
            scene_corners = cv2.perspectiveTransform(corners, H)
            pts = np.int32(scene_corners)
            cv2.polylines(out, [pts], True, (0, 255, 80), 3)
            cx = int(scene_corners[:, 0, 0].mean())
            cy = int(scene_corners[:, 0, 1].mean())
            cv2.putText(out, name, (cx - 30, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)
            return out, f"Found '{name}' with {len(good)} good matches."
    return out, f"Object not confidently found ({len(good)} matches — need ≥8)."

# ── Ollama vision ─────────────────────────────────────────────────────────────

def ask_vision(img: Image.Image, prompt: str) -> str:
    try:
        b64 = encode_image_b64(img)
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [b64],
            }],
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama error: {e}]"

# ── Point-of-interest & distance ──────────────────────────────────────────────

def draw_pois(base_cv: np.ndarray, pois: list, scale: float | None) -> np.ndarray:
    out = base_cv.copy()
    colors = [
        (0, 200, 255), (255, 100, 0), (200, 0, 255),
        (0, 255, 150), (255, 220, 0), (255, 50, 50),
    ]
    for i, (x, y, label) in enumerate(pois):
        color = colors[i % len(colors)]
        cv2.circle(out, (x, y), 7, color, -1)
        cv2.circle(out, (x, y), 9, (255, 255, 255), 2)
        cv2.putText(out, label, (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw distances between consecutive pairs
    for i in range(0, len(pois) - 1, 2):
        if i + 1 < len(pois):
            x1, y1, l1 = pois[i]
            x2, y2, l2 = pois[i + 1]
            cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
            dx, dy = x2 - x1, y2 - y1
            px_dist = math.sqrt(dx * dx + dy * dy)
            mid = ((x1 + x2) // 2 + 5, (y1 + y2) // 2 - 5)
            if scale:
                label_d = f"{px_dist * scale:.2f} mm"
            else:
                label_d = f"{px_dist:.1f} px"
            cv2.putText(out, label_d, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out

# ── Webcam thread ─────────────────────────────────────────────────────────────

class WebcamStream:
    def __init__(self):
        self.cap = None
        self.frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self, index: int = 0):
        if self._running:
            return "Already running."
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            return f"Cannot open camera {index}."
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return f"Camera {index} started."

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame = None
        return "Camera stopped."

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.frame = frame
            time.sleep(0.03)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    @property
    def running(self):
        return self._running

webcam = WebcamStream()

# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #0e1117 !important; color: #e0e6f0 !important; font-family: 'JetBrains Mono', monospace; }
.gr-button { background: #1a2235 !important; border: 1px solid #2a3f6f !important; color: #7eb8ff !important; font-family: monospace !important; }
.gr-button:hover { background: #243050 !important; border-color: #4a7fbf !important; }
.gr-button.primary { background: #1a4a8a !important; color: #ffffff !important; }
h1, h2, h3 { color: #7eb8ff !important; font-family: 'JetBrains Mono', monospace !important; }
.gr-box, .gr-panel { background: #141a27 !important; border: 1px solid #1e2d4a !important; }
label { color: #8899bb !important; font-size: 0.8em !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
textarea, input[type=text], input[type=number] { background: #0a0f1a !important; color: #c8d8f0 !important; border: 1px solid #1e2d4a !important; font-family: monospace !important; }
"""

def build_ui():
    global _poi, _scale_mm_per_px

    with gr.Blocks(css=CSS, title="CV Lab Toy") as demo:
        gr.Markdown("# 🔬 CV Lab Toy\n*Local vision AI for robotic laboratory equipment*")

        # ── Shared state ──
        current_image_state = gr.State(None)   # PIL Image
        poi_state = gr.State([])               # list of (x, y, label)
        scale_state = gr.State(None)           # mm/px float

        # ══════════════════════════════════════════════════════════════════════
        # Tab 1 — Input
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📷 Input"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Still Image")
                    img_upload = gr.Image(label="Upload image", type="pil", height=260)
                    btn_use_upload = gr.Button("Use this image ▶", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Webcam")
                    cam_index = gr.Number(label="Camera index", value=0, precision=0)
                    with gr.Row():
                        btn_cam_start = gr.Button("Start camera")
                        btn_cam_stop  = gr.Button("Stop camera")
                    btn_cam_snap  = gr.Button("📸 Grab frame", variant="primary")
                    cam_status = gr.Textbox(label="Status", interactive=False)

            current_display = gr.Image(label="Current working frame", type="pil", height=380, interactive=False)

            def use_upload(img):
                if img is None:
                    return gr.update(), None, "No image uploaded."
                return img, img, "Image loaded."

            def cam_start(idx):
                msg = webcam.start(int(idx))
                return msg

            def cam_stop():
                return webcam.stop()

            def cam_snap():
                f = webcam.get_frame()
                if f is None:
                    return gr.update(), None, "No frame — is camera running?"
                pil = cv_to_pil(f)
                return pil, pil, "Frame grabbed."

            btn_use_upload.click(use_upload, [img_upload], [current_display, current_image_state, cam_status])
            btn_cam_start.click(cam_start, [cam_index], [cam_status])
            btn_cam_stop.click(cam_stop, [], [cam_status])
            btn_cam_snap.click(cam_snap, [], [current_display, current_image_state, cam_status])

        # ══════════════════════════════════════════════════════════════════════
        # Tab 2 — Detect & Name
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🏷️ Detect & Name"):
            gr.Markdown(
                "Crop a region by selecting coordinates, then ask the local vision LLM what it is. "
                "Give it a name and register it for tracking."
            )
            with gr.Row():
                x1_in = gr.Number(label="Crop X1", value=0, precision=0)
                y1_in = gr.Number(label="Crop Y1", value=0, precision=0)
                x2_in = gr.Number(label="Crop X2", value=200, precision=0)
                y2_in = gr.Number(label="Crop Y2", value=200, precision=0)

            with gr.Row():
                btn_crop_ask = gr.Button("🔍 Crop & Ask Vision LLM", variant="primary")

            crop_display = gr.Image(label="Cropped region", type="pil", height=200, interactive=False)
            vision_answer = gr.Textbox(label="LLM says…", lines=3, interactive=False)

            with gr.Row():
                object_name_in = gr.Textbox(label="Name this object", placeholder="e.g. pipette_tip")
                btn_register = gr.Button("💾 Register object", variant="primary")

            register_status = gr.Textbox(label="Registration status", interactive=False)

            def crop_and_ask(img, x1, y1, x2, y2):
                if img is None:
                    return None, "No image loaded — go to Input tab first."
                cv_img = pil_to_cv(img)
                h, w = cv_img.shape[:2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    return None, "Invalid crop coordinates."
                crop_cv = cv_img[y1:y2, x1:x2]
                crop_pil = cv_to_pil(crop_cv)
                answer = ask_vision(
                    crop_pil,
                    "You are analysing laboratory equipment. Identify the object in this image. "
                    "Be concise: give the object name, key features, and any measurement markings visible. "
                    "Two sentences maximum."
                )
                return crop_pil, answer

            def register(img, x1, y1, x2, y2, name):
                if not name.strip():
                    return "Please enter a name first."
                if img is None:
                    return "No image loaded."
                cv_img = pil_to_cv(img)
                h, w = cv_img.shape[:2]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                crop_cv = cv_img[y1:y2, x1:x2]
                ok, msg = register_object(name.strip(), crop_cv)
                return msg

            btn_crop_ask.click(crop_and_ask,
                               [current_image_state, x1_in, y1_in, x2_in, y2_in],
                               [crop_display, vision_answer])
            btn_register.click(register,
                               [current_image_state, x1_in, y1_in, x2_in, y2_in, object_name_in],
                               [register_status])

        # ══════════════════════════════════════════════════════════════════════
        # Tab 3 — Find Objects
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🔎 Find Objects"):
            gr.Markdown("Search the current frame for a previously registered object using ORB feature matching.")

            def get_object_names():
                return list(_objects.keys()) if _objects else ["(no objects registered)"]

            with gr.Row():
                obj_dropdown = gr.Dropdown(
                    label="Object to find",
                    choices=get_object_names(),
                    value=None,
                    interactive=True,
                )
                btn_refresh_list = gr.Button("↻ Refresh list")

            btn_find = gr.Button("🔍 Find in current frame", variant="primary")
            find_display = gr.Image(label="Detection result", type="pil", height=400, interactive=False)
            find_status  = gr.Textbox(label="Result", interactive=False)

            def refresh_list():
                names = list(_objects.keys()) if _objects else ["(no objects registered)"]
                return gr.update(choices=names, value=names[0] if names else None)

            def find_object(img, name):
                if img is None:
                    return None, "No image loaded."
                if not name or name == "(no objects registered)":
                    return None, "Select an object first."
                cv_img = pil_to_cv(img)
                annotated, msg = find_object_in_frame(name, cv_img)
                return cv_to_pil(annotated), msg

            btn_refresh_list.click(refresh_list, [], [obj_dropdown])
            btn_find.click(find_object, [current_image_state, obj_dropdown], [find_display, find_status])

        # ══════════════════════════════════════════════════════════════════════
        # Tab 4 — Points of Interest & Distance
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📏 Points & Distance"):
            gr.Markdown(
                "Drop named points of interest onto the image. "
                "Distances are computed between consecutive pairs (P1↔P2, P3↔P4, …). "
                "Optionally set a scale (mm per pixel) for real-world units."
            )

            with gr.Row():
                poi_x = gr.Number(label="X (px)", value=0, precision=0)
                poi_y = gr.Number(label="Y (px)", value=0, precision=0)
                poi_label_in = gr.Textbox(label="Label", placeholder="e.g. tip_A")

            with gr.Row():
                btn_add_poi    = gr.Button("➕ Add point", variant="primary")
                btn_clear_pois = gr.Button("🗑️ Clear all points")

            with gr.Row():
                scale_in  = gr.Number(label="Scale (mm / pixel)", value=None, precision=4)
                btn_set_scale = gr.Button("Set scale")

            poi_display = gr.Image(label="Annotated image", type="pil", height=400, interactive=False)
            poi_table   = gr.Textbox(label="Point list + distances", lines=8, interactive=False)

            def render_pois(img, pois, scale):
                if img is None:
                    return None, "No image loaded."
                cv_img = pil_to_cv(img)
                annotated = draw_pois(cv_img, pois, scale)

                lines = ["Point list:"]
                for i, (x, y, lbl) in enumerate(pois):
                    lines.append(f"  [{i}] {lbl}  ({x}, {y})")
                lines.append("")
                lines.append("Distances (consecutive pairs):")
                for i in range(0, len(pois) - 1, 2):
                    if i + 1 < len(pois):
                        x1, y1, l1 = pois[i]
                        x2, y2, l2 = pois[i + 1]
                        d_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if scale:
                            lines.append(f"  {l1} ↔ {l2}: {d_px:.1f} px  =  {d_px * scale:.3f} mm")
                        else:
                            lines.append(f"  {l1} ↔ {l2}: {d_px:.1f} px  (no scale set)")

                if len(pois) == 0:
                    return cv_to_pil(annotated), "No points added yet."
                return cv_to_pil(annotated), "\n".join(lines)

            def add_poi(img, pois, scale, x, y, label):
                if img is None:
                    return None, pois, "No image loaded."
                if not label.strip():
                    label = f"P{len(pois)+1}"
                new_pois = pois + [(int(x), int(y), label.strip())]
                display, table = render_pois(img, new_pois, scale)
                return display, new_pois, table

            def clear_pois(img, scale):
                display, table = render_pois(img, [], scale)
                return display, [], table

            def set_scale(img, pois, val):
                scale = float(val) if val else None
                display, table = render_pois(img, pois, scale)
                return display, scale, table

            btn_add_poi.click(
                add_poi,
                [current_image_state, poi_state, scale_state, poi_x, poi_y, poi_label_in],
                [poi_display, poi_state, poi_table],
            )
            btn_clear_pois.click(
                clear_pois,
                [current_image_state, scale_state],
                [poi_display, poi_state, poi_table],
            )
            btn_set_scale.click(
                set_scale,
                [current_image_state, poi_state, scale_in],
                [poi_display, scale_state, poi_table],
            )

        # ══════════════════════════════════════════════════════════════════════
        # Tab 5 — Object Registry
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🗃️ Registry"):
            gr.Markdown("Manage registered objects stored in `cv_toy_objects.json`.")
            btn_reload_registry = gr.Button("↻ Reload from disk")
            registry_display    = gr.Textbox(label="Registered objects", lines=12, interactive=False)
            del_name_in         = gr.Textbox(label="Name to delete", placeholder="exact name")
            btn_delete          = gr.Button("🗑️ Delete object", variant="stop")
            del_status          = gr.Textbox(label="Status", interactive=False)

            def show_registry():
                if not _objects:
                    return "No objects registered yet."
                lines = []
                for name, obj in _objects.items():
                    n_kp = len(obj.get("keypoints", []))
                    lines.append(f"  • {name}  ({n_kp} keypoints)")
                return "\n".join(lines)

            def delete_object(name):
                name = name.strip()
                if name in _objects:
                    del _objects[name]
                    save_objects(_objects)
                    return f"Deleted '{name}'.", show_registry()
                return f"'{name}' not found.", show_registry()

            btn_reload_registry.click(show_registry, [], [registry_display])
            btn_delete.click(delete_object, [del_name_in], [del_status, registry_display])

        # Populate registry on load
        demo.load(show_registry, [], [registry_display])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def launch(**kwargs):
    """Launch the Gradio app. Pass share=True for a public link."""
    demo = build_ui()
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch(inbrowser=True)
