"""
Vehicle Counting and Tracking
Détection (YOLOv8) + Tracking (ByteTrack) + comptage par franchissement de ligne
avec comptage directionnel (haut / bas) et export des résultats.

Installation :  pip install -r requirements.txt
Usage        :  py vehicle_counting.py
Touche q     :  arrêter (laisse la vidéo finir pour obtenir les résultats complets)
"""
import csv
import json
import os
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

# ---------- Paramètres à adapter ----------
VIDEO_IN = "traffic.mp4"        # vidéo d'entrée
VIDEO_OUT = "result.mp4"        # vidéo annotée
MODEL = "yolov8m.pt"            # yolov8n (rapide) ou yolov8s (plus précis)
IMGSZ = 1280                     # résolution d'inférence (plus haut = petits véhicules mieux vus)
CONF = 0.20                     # seuil de confiance de la détection
LINE_RATIO = 0.60               # position de la ligne : 0.0 (haut) -> 1.0 (bas)
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # classes COCO
SHOW = True                     # afficher la fenêtre en direct
MAX_SNAPSHOTS = 6               # nombre max de captures automatiques
# ------------------------------------------

os.makedirs("screenshots", exist_ok=True)

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise SystemExit(f"Impossible d'ouvrir la vidéo : {VIDEO_IN}")

w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Ligne de comptage horizontale
LINE_A, LINE_B = (0, int(h * LINE_RATIO)), (w, int(h * LINE_RATIO))


def side(p, a, b):
    """> 0 : le point est sous la ligne ; < 0 : au-dessus (pour une ligne horizontale)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


last_side = {}                      # id -> côté précédent
counted = set()                     # ids déjà comptés (évite les doublons)
counts = defaultdict(int)           # par type de véhicule
directions = defaultdict(int)       # "up" / "down"
events = []                         # journal des franchissements
snapshots = 0
frame_idx = 0
t0 = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    res = model.track(frame, persist=True, tracker="bytetrack_custom.yaml",
                      classes=list(VEHICLES), conf=CONF, imgsz=IMGSZ, verbose=False)[0]

    if res.boxes.id is not None:
        for box, tid, cls in zip(res.boxes.xyxy.tolist(),
                                 res.boxes.id.int().tolist(),
                                 res.boxes.cls.int().tolist()):
            x1, y1, x2, y2 = map(int, box)
            c = ((x1 + x2) // 2, (y1 + y2) // 2)    # centre de la boîte
            s = side(c, LINE_A, LINE_B)
            name = VEHICLES[cls]
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
                        cv2.imwrite(f"screenshots/crossing_{tid}_{direction}.jpg", snap)
                        snapshots += 1
                last_side[tid] = s

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} #{tid}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, c, 4, (0, 0, 255), -1)

    # Ligne et compteurs
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
    if SHOW:
        cv2.imshow("Vehicle counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Arrêt manuel : résultats partiels.")
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# ---------- Export des résultats ----------
elapsed = time.time() - t0
summary = {
    "video": VIDEO_IN,
    "model": MODEL,
    "frames_processed": frame_idx,
    "frames_total": total_frames,
    "total_count": sum(counts.values()),
    "by_class": dict(counts),
    "by_direction": dict(directions),
    "processing_fps": round(frame_idx / elapsed, 1) if elapsed > 0 else None,
}

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id", "class", "direction"])
    writer.writeheader()
    writer.writerows(events)

with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n===== RÉSULTATS =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("Fichiers générés : result.mp4, results.csv, summary.json, screenshots/")