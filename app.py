"""
Vehicle Counting and Tracking - Dashboard Streamlit
Lancer avec :  py -m streamlit run app.py
"""
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import theme

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE = Path(__file__).parent.resolve()
RUNS_DIR = BASE / "runs"
RUNS_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE / "yolov8m.pt"
IMGSZ = 1280
TRACKER_CFG = str(BASE / "bytetrack_custom.yaml")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LABELS_FR = {"car": "Voitures", "truck": "Camions", "bus": "Bus", "motorcycle": "Motos"}
COLORS = {"car": "#2F5D8A", "truck": "#F0A500", "bus": "#0F8B8D", "motorcycle": "#8A5FBF",
          "Up": "#0F8B8D", "Down": "#D9514E"}

st.set_page_config(page_title="Vehicle Counting & Tracking", page_icon="🚗",
                   layout="wide", initial_sidebar_state="expanded")

theme.inject_css()


def title(text: str, sub: str = ""):
    theme.page_header(text, sub)


# ----------------------------------------------------------------------------
# Lecture flexible des résultats (CSV / JSON)
# ----------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _norm_class(v) -> str:
    s = str(v).lower().strip()
    if s in ("motorbike", "moto", "motorcycle", "motos"):
        return "motorcycle"
    if s in ("car", "voiture", "cars"):
        return "car"
    if s in ("truck", "camion", "trucks"):
        return "truck"
    if s in ("bus", "buses"):
        return "bus"
    return s


def _norm_dir(v) -> str:
    s = str(v).lower().strip()
    if "up" in s or "haut" in s or "north" in s:
        return "Up"
    if "down" in s or "bas" in s or "south" in s:
        return "Down"
    return str(v)


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes en : track_id, class, direction, frame (si trouvées)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["track_id", "class", "direction", "frame"])
    out = pd.DataFrame(index=df.index)
    mapping = {
        "track_id": ["track_id", "id", "vehicle_id", "object_id", "tracker_id"],
        "class": ["class", "class_name", "label", "type", "vehicle_type", "category", "name"],
        "direction": ["direction", "dir", "sens"],
        "frame": ["frame", "frame_id", "frame_number", "frame_idx"],
        "time_s": ["time_s", "time", "timestamp", "seconds"],
    }
    for target, cands in mapping.items():
        col = _find_col(df, cands)
        if col is not None:
            out[target] = df[col]
    if "class" in out:
        out["class"] = out["class"].map(_norm_class)
    if "direction" in out:
        out["direction"] = out["direction"].map(_norm_dir)
    return out


def _deep_find(obj, keys):
    """Cherche récursivement la première clé (insensible à la casse) dans un dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys and isinstance(v, (int, float)):
                return v
        for v in obj.values():
            r = _deep_find(v, keys)
            if r is not None:
                return r
    return None


def _deep_find_dict(obj, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in keys and isinstance(v, dict):
                return v
        for v in obj.values():
            r = _deep_find_dict(v, keys)
            if r is not None:
                return r
    return None


def compute_stats(events: pd.DataFrame, summary: dict) -> dict:
    stats = {"total": 0, "up": 0, "down": 0, "by_class": {}, "fps": None, "frames": None}
    if not events.empty:
        stats["total"] = len(events)
        if "direction" in events:
            vc = events["direction"].value_counts()
            stats["up"] = int(vc.get("Up", 0))
            stats["down"] = int(vc.get("Down", 0))
        if "class" in events:
            stats["by_class"] = events["class"].value_counts().to_dict()
    if summary:
        t = _deep_find(summary, {"total", "total_count", "total_vehicles", "total_counted"})
        u = _deep_find(summary, {"up", "up_count", "count_up"})
        d = _deep_find(summary, {"down", "down_count", "count_down"})
        if t is not None:
            stats["total"] = int(t)
        if u is not None:
            stats["up"] = int(u)
        if d is not None:
            stats["down"] = int(d)
        if t is None and (u is not None or d is not None):
            stats["total"] = stats["up"] + stats["down"]
        f = _deep_find(summary, {"fps", "processing_fps", "avg_fps"})
        n = _deep_find(summary, {"frames", "frames_processed", "total_frames", "frame_count"})
        stats["fps"] = float(f) if f is not None else None
        stats["frames"] = int(n) if n is not None else None
        bc = _deep_find_dict(summary, {"by_class", "classes", "class_counts", "per_class", "counts"})
        if bc and not stats["by_class"]:
            stats["by_class"] = {_norm_class(k): int(v) for k, v in bc.items()
                                 if isinstance(v, (int, float))}
    return stats


# ----------------------------------------------------------------------------
# Gestion des "runs" (projet existant + vidéos importées)
# ----------------------------------------------------------------------------
def best_video(folder: Path, names):
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None


def list_runs() -> dict:
    runs = {}
    legacy_video = best_video(BASE, ["result_h264.mp4", "result.mp4"])
    if legacy_video or (BASE / "results.csv").exists() or (BASE / "summary.json").exists():
        runs["Projet existant (traffic.mp4)"] = {
            "dir": BASE,
            "original": BASE / "traffic.mp4" if (BASE / "traffic.mp4").exists() else None,
            "result": legacy_video,
            "csv": BASE / "results.csv" if (BASE / "results.csv").exists() else None,
            "json": BASE / "summary.json" if (BASE / "summary.json").exists() else None,
            "screens": BASE / "screenshots",
        }
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "summary.json").exists():
            runs[d.name] = {
                "dir": d,
                "original": best_video(d, ["original.mp4"]),
                "result": best_video(d, ["result_h264.mp4", "result.mp4"]),
                "csv": d / "results.csv" if (d / "results.csv").exists() else None,
                "json": d / "summary.json",
                "screens": d / "screenshots",
            }
    return runs


@st.cache_data(show_spinner=False)
def load_run_data(csv_path, json_path, _mtime):
    df = pd.read_csv(csv_path) if csv_path else pd.DataFrame()
    summary = {}
    if json_path:
        try:
            summary = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    events = normalize_events(df)
    return df, events, summary


def get_run(run: dict):
    mt = sum(p.stat().st_mtime for p in (run["csv"], run["json"]) if p)
    df, events, summary = load_run_data(str(run["csv"]) if run["csv"] else None,
                                        str(run["json"]) if run["json"] else None, mt)
    return df, events, summary, compute_stats(events, summary)


# ----------------------------------------------------------------------------
# Détection YOLOv8 + tracking + comptage par ligne
# ----------------------------------------------------------------------------
def to_h264(src: Path, dst: Path) -> bool:
    """Convertit en H.264 pour une lecture dans le navigateur."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    try:
        subprocess.run([exe, "-y", "-i", str(src), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "veryfast", "-an", str(dst)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst.exists()
    except Exception:
        return False


def run_detection(video_path: Path, out_dir: Path, conf: float, line_ratio: float, progress_cb):
    """Même logique que vehicle_counting.py : YOLOv8 + ByteTrack + franchissement de ligne."""
    from collections import defaultdict
    from ultralytics import YOLO

    model = load_model()  # nouvelle instance => tracker réinitialisé
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    (out_dir / "screenshots").mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "result.mp4"
    out = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    LINE_A, LINE_B = (0, int(h * line_ratio)), (w, int(h * line_ratio))

    def side(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    last_side, counted = {}, set()
    counts, directions = defaultdict(int), defaultdict(int)
    events, snapshots, frame_idx = [], 0, 0
    MAX_SNAPSHOTS = 6
    t0 = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        res = model.track(frame, persist=True, tracker=TRACKER_CFG,
                          classes=list(VEHICLE_CLASSES), conf=conf, imgsz=IMGSZ, verbose=False)[0]

        if res.boxes.id is not None:
            for box, tid, cls in zip(res.boxes.xyxy.tolist(),
                                     res.boxes.id.int().tolist(),
                                     res.boxes.cls.int().tolist()):
                x1, y1, x2, y2 = map(int, box)
                c = ((x1 + x2) // 2, (y1 + y2) // 2)
                s = side(c, LINE_A, LINE_B)
                name = VEHICLE_CLASSES[cls]
                color = (0, 255, 0)

                if s != 0:
                    prev = last_side.get(tid)
                    if prev is not None and prev * s < 0 and tid not in counted:
                        counted.add(tid)
                        direction = "down" if s > 0 else "up"
                        counts[name] += 1
                        directions[direction] += 1
                        events.append({"frame": frame_idx,
                                       "time_s": round(frame_idx / fps, 2),
                                       "track_id": tid, "class": name,
                                       "direction": direction})
                        color = (0, 165, 255)
                        if snapshots < MAX_SNAPSHOTS:
                            snap = frame.copy()
                            cv2.rectangle(snap, (x1, y1), (x2, y2), color, 3)
                            cv2.line(snap, LINE_A, LINE_B, (0, 0, 255), 3)
                            cv2.imwrite(str(out_dir / "screenshots" / f"crossing_{tid}_{direction}.jpg"), snap)
                            snapshots += 1
                    last_side[tid] = s

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} #{tid}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, c, 4, (0, 0, 255), -1)

        cv2.line(frame, LINE_A, LINE_B, (0, 0, 255), 3)
        y = 30
        cv2.putText(frame, f"Total: {sum(counts.values())}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 28
        cv2.putText(frame, f"Haut: {directions['up']}  Bas: {directions['down']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for name, n in counts.items():
            y += 26
            cv2.putText(frame, f"{name}: {n}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        progress_cb(min(frame_idx / total_frames, 1.0), frame_idx, total_frames)

    cap.release()
    out.release()
    elapsed = time.time() - t0

    progress_cb(1.0, frame_idx, total_frames, "Encodage H.264...")
    to_h264(raw_path, out_dir / "result_h264.mp4")

    summary = {
        "video": video_path.name,
        "model": MODEL_PATH.name,
        "frames_processed": frame_idx,
        "frames_total": total_frames,
        "total_count": sum(counts.values()),
        "by_class": dict(counts),
        "by_direction": dict(directions),
        "processing_fps": round(frame_idx / elapsed, 1) if elapsed > 0 else None,
        "processing_time_s": round(elapsed, 1),
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    pd.DataFrame(events, columns=["frame", "time_s", "track_id", "class", "direction"]) \
        .to_csv(out_dir / "results.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return summary


def load_model():
    """Charge YOLOv8. Si le fichier est absent ou corrompu, il est supprimé puis retéléchargé."""
    from ultralytics import YOLO

    try:
        return YOLO(str(MODEL_PATH))
    except Exception:
        try:
            MODEL_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        old = os.getcwd()
        os.chdir(BASE)  # le téléchargement se fait dans le dossier courant
        try:
            return YOLO("yolov8m.pt")
        finally:
            os.chdir(old)


def first_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_line_preview(frame, line_ratio: float):
    """Dessine la ligne de comptage sur une frame (retourne une image RGB)."""
    img = frame.copy()
    h, w = img.shape[:2]
    y = int(h * line_ratio)
    cv2.line(img, (0, y), (w, y), (0, 0, 255), 3)
    cv2.putText(img, "Haut (up)", (10, max(y - 14, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(img, "Bas (down)", (10, min(y + 34, h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_image(data: bytes, conf: float):
    """Détection YOLOv8 sur une image fixe. Retourne (image RGB annotée, comptes par classe)."""
    import numpy as np
    from ultralytics import YOLO

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible.")
    model = load_model()
    res = model.predict(img, conf=conf, classes=list(VEHICLE_CLASSES), imgsz=IMGSZ, verbose=False)[0]
    counts = {}
    for box, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(),
                               res.boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, box)
        name = VEHICLE_CLASSES[cls]
        counts[name] = counts.get(name, 0) + 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{name} {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), counts


# ----------------------------------------------------------------------------
# Composants d'affichage
# ----------------------------------------------------------------------------
def kpi_row(stats: dict):
    theme.hero_total(stats["total"], stats["up"], stats["down"])
    bc = stats["by_class"]
    theme.stat_strip([
        ("Voitures", bc.get("car", 0)), ("Camions", bc.get("truck", 0)),
        ("Bus", bc.get("bus", 0)), ("Motos", bc.get("motorcycle", 0)),
        ("Frames traitées", stats["frames"] if stats["frames"] else "N/A"),
        ("FPS traitement", f"{stats['fps']:.1f}" if stats["fps"] else "N/A"),
    ])


def class_pie(by_class: dict):
    if not by_class:
        st.info("Aucune donnée de classe disponible.")
        return
    d = pd.DataFrame({"Classe": [LABELS_FR.get(k, k) for k in by_class], "Nombre": list(by_class.values()),
                      "key": list(by_class)})
    fig = px.pie(d, names="Classe", values="Nombre", hole=0.5, color="key",
                 color_discrete_map=COLORS, title="Répartition par classe")
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(theme.style_fig(fig), width="stretch")


def direction_bar(stats: dict):
    d = pd.DataFrame({"Direction": ["Up", "Down"], "Nombre": [stats["up"], stats["down"]]})
    fig = px.bar(d, x="Direction", y="Nombre", color="Direction", text="Nombre",
                 color_discrete_map=COLORS, title="Répartition par direction")
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(theme.style_fig(fig), width="stretch")


def cumulative_chart(events: pd.DataFrame):
    if events.empty or "frame" not in events:
        return
    e = events.sort_values("frame").copy()
    e["Cumul"] = range(1, len(e) + 1)
    fig = px.line(e, x="frame", y="Cumul", title="Comptage cumulé au fil des frames")
    if "direction" in e:
        for dname in ("Up", "Down"):
            sub = e[e["direction"] == dname].copy()
            if not sub.empty:
                sub["c"] = range(1, len(sub) + 1)
                fig.add_trace(go.Scatter(x=sub["frame"], y=sub["c"], mode="lines", name=dname,
                                         line=dict(color=COLORS[dname], dash="dot")))
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(theme.style_fig(fig), width="stretch")


def time_bin_chart(events: pd.DataFrame):
    if events.empty or "time_s" not in events:
        return
    bin_s = st.select_slider("Intervalle de temps (secondes)", options=[1, 2, 5, 10, 30, 60],
                             value=5, key="bin_s")
    e = events.copy()
    e["Intervalle"] = (pd.to_numeric(e["time_s"], errors="coerce") // bin_s * bin_s).astype("Int64")
    if "direction" in e:
        g = e.groupby(["Intervalle", "direction"]).size().reset_index(name="Véhicules")
        fig = px.bar(g, x="Intervalle", y="Véhicules", color="direction", barmode="stack",
                     color_discrete_map=COLORS, title=f"Véhicules comptés par tranche de {bin_s} s")
    else:
        g = e.groupby("Intervalle").size().reset_index(name="Véhicules")
        fig = px.bar(g, x="Intervalle", y="Véhicules", title=f"Véhicules comptés par tranche de {bin_s} s")
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), xaxis_title="Temps (s)")
    st.plotly_chart(theme.style_fig(fig), width="stretch")


def crossing_schema():
    """Schéma pédagogique du comptage par franchissement de ligne."""
    fig = go.Figure()
    fig.add_hline(y=0.6, line=dict(color="red", width=3),
                  annotation_text="Ligne de comptage", annotation_position="top left")
    fig.add_trace(go.Scatter(x=[1, 1, 1], y=[0.9, 0.6, 0.3], mode="lines+markers+text",
                             text=["t", "t+1", "t+2"], textposition="middle right",
                             name="Véhicule A (Up)", line=dict(color=COLORS["Up"], width=4),
                             marker=dict(size=[8, 8, 16])))
    fig.add_trace(go.Scatter(x=[2, 2, 2], y=[0.3, 0.6, 0.9], mode="lines+markers+text",
                             text=["t", "t+1", "t+2"], textposition="middle right",
                             name="Véhicule B (Down)", line=dict(color=COLORS["Down"], width=4),
                             marker=dict(size=[8, 8, 16])))
    fig.update_layout(height=340, margin=dict(t=20, b=10, l=10, r=10),
                      xaxis=dict(visible=False, range=[0.4, 2.8]),
                      yaxis=dict(range=[1, 0], title="Position verticale (0 = haut)"),
                      legend=dict(orientation="h", y=-0.05))
    return fig


def select_run(key: str):
    runs = list_runs()
    if not runs:
        st.warning("Aucun résultat trouvé. Importez une vidéo dans la page « Video Upload ».")
        return None, None
    name = st.sidebar.selectbox("Résultat affiché", list(runs), key=key)
    return name, runs[name]


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    title("Vehicle Counting and Tracking",
          "Détection, tracking et comptage par franchissement de ligne · Image / Vidéo")

    steps = [
        ("1 · Détection", "YOLOv8n repère dans chaque image les voitures, motos, bus et camions "
                          "et renvoie une boîte englobante avec un score de confiance."),
        ("2 · Tracking", "ByteTrack associe les boîtes d'une image à l'autre : chaque véhicule garde "
                         "un identifiant unique (#id) tout au long de la vidéo."),
        ("3 · Comptage par ligne", "Le centre de chaque boîte est comparé à une ligne virtuelle. "
                                   "Quand il la traverse, le véhicule est compté une seule fois, "
                                   "avec sa direction (Up / Down)."),
    ]
    for col, (t, d) in zip(st.columns(3), steps):
        with col:
            with st.container(border=True):
                st.markdown(f"### {t}")
                st.write(d)

    st.subheader("Court exemple")
    runs = list_runs()
    left, right = st.columns([3, 2])
    run = next(iter(runs.values()), None)
    with left:
        if run and run["result"]:
            st.video(str(run["result"]))
            st.caption("Vidéo annotée : boîtes, identifiants, ligne rouge et compteurs en direct.")
        else:
            st.info("Importez une vidéo dans « Image / Video Upload » pour afficher un exemple ici.")
    with right:
        st.plotly_chart(crossing_schema(), width="stretch")
        st.caption("Un véhicule est compté quand son centre change de côté de la ligne.")

    if run:
        _, _, _, stats = get_run(run)
        kpi_row(stats)
        scr = run["screens"]
        imgs = sorted(Path(scr).glob("*.jpg"))[:3] if scr and Path(scr).exists() else []
        if imgs:
            st.markdown("**Captures automatiques des passages**")
            for col, p in zip(st.columns(len(imgs)), imgs):
                col.image(str(p), caption=p.stem, width="stretch")

    with st.expander("Comment fonctionne le comptage ?"):
        st.markdown(
            "- Une ligne horizontale est placée à une hauteur réglable (par défaut 60 % de l'image).\n"
            "- À chaque image, on calcule de quel côté de la ligne se trouve le centre de chaque véhicule.\n"
            "- Si le côté change entre deux images, le véhicule a franchi la ligne : "
            "il passe vers le **haut** (up) ou vers le **bas** (down).\n"
            "- Un identifiant déjà compté ne l'est plus : pas de doublons.\n"
            "- Sur une **image fixe**, seule la détection est possible (pas de tracking ni de ligne)."
        )
    st.caption("Utilisez le menu de gauche pour explorer le Dashboard, importer une image ou une vidéo.")


def page_dashboard():
    title("Dashboard", "Vue d'ensemble du comptage et du suivi des véhicules")
    name, run = select_run("run_dash")
    if not run:
        return
    df, events, summary, stats = get_run(run)
    kpi_row(stats)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        class_pie(stats["by_class"])
    with c2:
        direction_bar(stats)
    cumulative_chart(events)
    time_bin_chart(events)


def page_detection():
    title("Vehicle Detection", "Vidéo originale vs vidéo annotée par YOLOv8")
    name, run = select_run("run_det")
    if not run:
        return
    df, events, summary, stats = get_run(run)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vidéo originale")
        if run["original"]:
            st.video(str(run["original"]))
        else:
            st.info("Vidéo originale introuvable.")
    with c2:
        st.subheader("Détection YOLOv8 (boxes + IDs)")
        if run["result"]:
            st.video(str(run["result"]))
        else:
            st.info("Vidéo de résultat introuvable.")
    st.divider()
    st.subheader("Résultats du comptage")
    kpi_row(stats)
    scr = run["screens"]
    if scr and Path(scr).exists():
        imgs = sorted(Path(scr).glob("*.jpg"))
        if imgs:
            st.subheader("Captures des passages")
            cols = st.columns(3)
            for i, p in enumerate(imgs[:12]):
                cols[i % 3].image(str(p), caption=p.stem, width="stretch")


def page_upload():
    title("Image / Video Upload",
          "Vidéo : détection + tracking + comptage par ligne · Image : détection")
    if not MODEL_PATH.exists():
        st.warning("Modèle absent : il sera téléchargé automatiquement au premier lancement.")
    tab_v, tab_i = st.tabs(["Vidéo", "Image"])
    with tab_v:
        _upload_video_tab()
    with tab_i:
        _upload_image_tab()


def _upload_image_tab():
    img = st.file_uploader("Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="img_up")
    conf = st.slider("Seuil de confiance", 0.1, 0.9, 0.25, 0.05, key="conf_img")
    if img is not None and st.button("Start Detection", type="primary", key="btn_img"):
        with st.spinner("Détection en cours..."):
            try:
                rgb, counts = detect_image(img.getvalue(), conf)
            except Exception as e:
                st.error(f"Erreur pendant la détection : {e}")
                return
        st.session_state["img_result"] = (img.getvalue(), rgb, counts)

    res = st.session_state.get("img_result")
    if res:
        orig, rgb, counts = res
        c1, c2 = st.columns(2)
        c1.image(orig, caption="Image originale", width="stretch")
        c2.image(rgb, caption="Détection YOLOv8", width="stretch")
        m = st.columns(5)
        m[0].metric("Total", sum(counts.values()))
        m[1].metric("Voitures", counts.get("car", 0))
        m[2].metric("Camions", counts.get("truck", 0))
        m[3].metric("Bus", counts.get("bus", 0))
        m[4].metric("Motos", counts.get("motorcycle", 0))
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            st.download_button("Image annotée (PNG)", buf.tobytes(), "detection.png", "image/png")
        st.caption("Sur une image fixe, il n'y a ni tracking ni comptage par ligne : "
                   "seule la détection est appliquée.")


def _upload_video_tab():
    up = st.file_uploader("Vidéo MP4", type=["mp4"])
    c1, c2 = st.columns(2)
    conf = c1.slider("Seuil de confiance", 0.1, 0.9, 0.20, 0.05)
    line_ratio = c2.slider("Position de la ligne de comptage (hauteur)", 0.2, 0.9, 0.60, 0.05)

    if up is not None:
        pv = RUNS_DIR / "_preview.mp4"
        pv.write_bytes(up.getbuffer())
        frame = first_frame(pv)
        p1, p2 = st.columns(2)
        p1.video(up)
        if frame is not None:
            p2.image(draw_line_preview(frame, line_ratio),
                     caption="Aperçu de la ligne de comptage (rouge)", width="stretch")
    if up is not None and st.button("Start Detection", type="primary", key="btn_video"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RUNS_DIR / f"{stamp}_{Path(up.name).stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        src = out_dir / "original.mp4"
        src.write_bytes(up.getbuffer())

        bar = st.progress(0.0, text="Initialisation...")

        def cb(p, i, n, msg=None):
            bar.progress(p, text=msg or f"Frame {i}/{n}")

        try:
            summary = run_detection(src, out_dir, conf, line_ratio, cb)
        except Exception as e:
            st.error(f"Erreur pendant la détection : {e}")
            return
        bar.progress(1.0, text="Terminé")
        st.success(f"Détection terminée en {summary['processing_time_s']} s "
                   f"({summary['processing_fps']} FPS).")
        st.session_state["last_run"] = out_dir.name

    last = st.session_state.get("last_run")
    runs = list_runs()
    if last and last in runs:
        run = runs[last]
        df, events, summary, stats = get_run(run)
        st.subheader("Vidéo traitée")
        if run["result"]:
            st.video(str(run["result"]))
        kpi_row(stats)
        d1, d2 = st.columns(2)
        if run["csv"]:
            d1.download_button("results.csv", Path(run["csv"]).read_bytes(), "results.csv", "text/csv")
        if run["json"]:
            d2.download_button("summary.json", Path(run["json"]).read_bytes(), "summary.json",
                               "application/json")


def page_analytics():
    title("Analytics", "Analyse détaillée par classe, direction et entre vidéos")
    name, run = select_run("run_ana")
    if not run:
        return
    df, events, summary, stats = get_run(run)
    c1, c2 = st.columns(2)
    with c1:
        class_pie(stats["by_class"])
    with c2:
        direction_bar(stats)

    if not events.empty and {"class", "direction"} <= set(events.columns):
        st.subheader("Classe × Direction")
        g = events.groupby(["class", "direction"]).size().reset_index(name="Nombre")
        g["Classe"] = g["class"].map(lambda k: LABELS_FR.get(k, k))
        fig = px.bar(g, x="Classe", y="Nombre", color="direction", barmode="group",
                     color_discrete_map=COLORS)
        st.plotly_chart(theme.style_fig(fig), width="stretch")
        st.dataframe(g.pivot_table(index="Classe", columns="direction", values="Nombre",
                                   fill_value=0, aggfunc="sum"), width="stretch")

    runs = list_runs()
    if len(runs) > 1:
        st.subheader("Comparaison entre vidéos")
        rows = []
        for n, r in runs.items():
            _, _, _, s = get_run(r)
            rows.append({"Vidéo": n, "Up": s["up"], "Down": s["down"], "Total": s["total"],
                         **{LABELS_FR.get(k, k): v for k, v in s["by_class"].items()}})
        comp = pd.DataFrame(rows).fillna(0)
        fig = px.bar(comp, x="Vidéo", y=["Up", "Down"], barmode="stack", title="Up / Down par vidéo",
                     color_discrete_map=COLORS)
        st.plotly_chart(theme.style_fig(fig), width="stretch")
        st.dataframe(comp, width="stretch", hide_index=True)
    else:
        st.info("Importez d'autres vidéos pour activer la comparaison.")


def page_reports():
    title("Reports", "Résultats détaillés et téléchargements")
    name, run = select_run("run_rep")
    if not run:
        return
    df, events, summary, stats = get_run(run)
    kpi_row(stats)
    st.subheader("Résultats détaillés")
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("Aucun fichier CSV pour ce résultat.")
    if summary:
        with st.expander("Résumé (JSON)"):
            st.json(summary)
    st.subheader("Téléchargements")
    c1, c2, c3 = st.columns(3)
    if run["csv"]:
        c1.download_button("CSV", Path(run["csv"]).read_bytes(), "results.csv", "text/csv",
                           width="stretch")
    if run["json"]:
        c2.download_button("JSON", Path(run["json"]).read_bytes(), "summary.json",
                           "application/json", width="stretch")
    if run["result"]:
        c3.download_button("Vidéo annotée", Path(run["result"]).read_bytes(), "result.mp4",
                           "video/mp4", width="stretch")


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
PAGES = {
    "Présentation": page_home,
    "Dashboard": page_dashboard,
    "Vehicle Detection": page_detection,
    "Image / Video Upload": page_upload,
    "Analytics": page_analytics,
    "Reports": page_reports,
}

st.sidebar.markdown("### Vehicle Counting")
st.sidebar.caption("YOLOv8 · OpenCV · Streamlit")
choice = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Projet : Vehicle Counting and Tracking · Image / Vidéo")
PAGES[choice]()