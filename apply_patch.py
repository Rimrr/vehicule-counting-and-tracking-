"""Applique les corrections de détection et le nouveau design à app.py et vehicle_counting.py.

Usage (depuis le dossier du projet, avec theme.py à côté) :  py apply_patch.py
Les originaux sont sauvegardés en .bak. Chaque ligne affiche OK ou ABSENT (déjà modifié ou texte différent).
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")


def sub(text, pattern, repl, label):
    new, n = re.subn(pattern, lambda m: repl, text, count=1, flags=re.S | re.M)
    print(("OK      " if n else "ABSENT  ") + label)
    return new


def rep(text, old, new, label):
    ok = old in text
    print(("OK      " if ok else "ABSENT  ") + label)
    return text.replace(old, new, 1) if ok else text


TRACKER_YAML = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 60
match_thresh: 0.8
fuse_score: True
"""

NEW_COLORS = '''COLORS = {"car": "#2F5D8A", "truck": "#F0A500", "bus": "#0F8B8D", "motorcycle": "#8A5FBF",
          "Up": "#0F8B8D", "Down": "#D9514E"}'''

NEW_KPI = '''def kpi_row(stats: dict):
    theme.hero_total(stats["total"], stats["up"], stats["down"])
    bc = stats["by_class"]
    theme.stat_strip([
        ("Voitures", bc.get("car", 0)), ("Camions", bc.get("truck", 0)),
        ("Bus", bc.get("bus", 0)), ("Motos", bc.get("motorcycle", 0)),
        ("Frames traitées", stats["frames"] if stats["frames"] else "N/A"),
        ("FPS traitement", f"{stats['fps']:.1f}" if stats["fps"] else "N/A"),
    ])

'''


def patch_app(p):
    t = p.read_text(encoding="utf-8")
    # Détection
    t = rep(t, 'MODEL_PATH = BASE / "yolov8n.pt"',
            'MODEL_PATH = BASE / "yolov8s.pt"\nIMGSZ = 1280\nTRACKER_CFG = str(BASE / "bytetrack_custom.yaml")', "modèle yolov8s + IMGSZ + tracker")
    t = rep(t, 'return YOLO("yolov8n.pt")', 'return YOLO("yolov8s.pt")', "téléchargement yolov8s")
    t = sub(t, r'model\.track\(frame, persist=True, tracker="bytetrack\.yaml",\s*classes=list\(VEHICLE_CLASSES\), conf=conf, verbose=False\)',
            'model.track(frame, persist=True, tracker=TRACKER_CFG,\n                          classes=list(VEHICLE_CLASSES), conf=conf, imgsz=IMGSZ, verbose=False)', "appel model.track")
    t = rep(t, 'model.predict(img, conf=conf, classes=list(VEHICLE_CLASSES), verbose=False)',
            'model.predict(img, conf=conf, classes=list(VEHICLE_CLASSES), imgsz=IMGSZ, verbose=False)', "appel model.predict")
    t = rep(t, 'c1.slider("Seuil de confiance", 0.1, 0.9, 0.30, 0.05)', 'c1.slider("Seuil de confiance", 0.1, 0.9, 0.20, 0.05)', "seuil vidéo 0.20")
    t = rep(t, 'st.slider("Seuil de confiance", 0.1, 0.9, 0.30, 0.05, key="conf_img")',
            'st.slider("Seuil de confiance", 0.1, 0.9, 0.25, 0.05, key="conf_img")', "seuil image 0.25")
    # Design
    if "import theme" not in t:
        t = rep(t, "import streamlit as st\n", "import streamlit as st\n\nimport theme\n", "import theme")
    t = sub(t, r'st\.markdown\(\s*"""\s*<style>.*?</style>\s*""",\s*unsafe_allow_html=True,?\s*\)', "theme.inject_css()", "ancien CSS -> theme.inject_css()")
    t = sub(t, r'def title\(text: str, sub: str = ""\):\s+st\.markdown\(f\'<div class="page-title">.*?unsafe_allow_html=True\)',
            'def title(text: str, sub: str = ""):\n    theme.page_header(text, sub)', "fonction title")
    t = sub(t, r'COLORS = \{.*?\}', NEW_COLORS, "palette des graphiques")
    t = sub(t, r'def kpi_row\(stats: dict\):.*?(?=\ndef class_pie)', NEW_KPI, "kpi_row (bloc total + bandeau)")
    t = t.replace('st.plotly_chart(fig, width="stretch")', 'st.plotly_chart(theme.style_fig(fig), width="stretch")')
    for old, new in [('"🏠 Présentation"', '"Présentation"'), ('"📊 Dashboard"', '"Dashboard"'),
                     ('"🎯 Vehicle Detection"', '"Vehicle Detection"'), ('"⬆️ Image / Video Upload"', '"Image / Video Upload"'),
                     ('"📈 Analytics"', '"Analytics"'), ('"📄 Reports"', '"Reports"'),
                     ('st.sidebar.markdown("## 🚗 Vehicle Counting")', 'st.sidebar.markdown("### Vehicle Counting")')]:
        t = rep(t, old, new, f"menu {new}")
    p.write_text(t, encoding="utf-8")


def patch_counting(p):
    t = p.read_text(encoding="utf-8")
    t = rep(t, 'MODEL = "yolov8n.pt"', 'MODEL = "yolov8s.pt"', "modèle yolov8s")
    t = sub(t, r'^CONF = 0\.30', 'IMGSZ = 1280                     # résolution d\'inférence (plus haut = petits véhicules mieux vus)\nCONF = 0.20', "IMGSZ + CONF 0.20")
    t = sub(t, r'tracker="bytetrack\.yaml",\s*classes=list\(VEHICLES\), conf=CONF, verbose=False',
            'tracker="bytetrack_custom.yaml",\n                      classes=list(VEHICLES), conf=CONF, imgsz=IMGSZ, verbose=False', "appel model.track")
    p.write_text(t, encoding="utf-8")


def main():
    yaml = ROOT / "bytetrack_custom.yaml"
    if not yaml.exists():
        yaml.write_text(TRACKER_YAML, encoding="utf-8")
        print("CRÉÉ    bytetrack_custom.yaml")
    for name, fn in [("app.py", patch_app), ("vehicle_counting.py", patch_counting)]:
        p = ROOT / name
        if not p.exists():
            print(f"INTROUVABLE {name} (lancez le script depuis le dossier du projet)")
            continue
        bak = p.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy(p, bak)
        print(f"--- {name} (sauvegarde : {bak.name})")
        fn(p)
    print("\nTerminé. Si une ligne affiche ABSENT, la modification est à faire à la main ou était déjà faite.")


main()