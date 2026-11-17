# Vehicle Counting and Tracking
#  cd vehicle-counting 
#  py -m streamlit run app.py
Projet de vision artificielle : **détection**, **tracking** et **comptage par franchissement de ligne** des véhicules dans une vidéo (ou détection sur une image), avec un tableau de bord web Streamlit.

## Fonctionnalités

- **Détection** des voitures, motos, bus et camions avec YOLOv8 (`yolov8s.pt`, résolution d'analyse 960).
- **Tracking** avec ByteTrack (configuration `bytetrack_custom.yaml`) : chaque véhicule garde un identifiant unique (`#id`).
- **Comptage par ligne virtuelle** : un véhicule est compté une seule fois quand le centre de sa boîte traverse la ligne, avec sa direction (**up** / **down**).
- **Export des résultats** : vidéo annotée, `results.csv`, `summary.json` et captures automatiques des passages.
- **Application web Streamlit** (6 pages) : Présentation, Dashboard, Vehicle Detection, Image / Video Upload, Analytics, Reports.

## Structure du projet

```
vehicle-counting/
├── app.py                  # Application web Streamlit
├── theme.py                # Thème visuel (CSS et composants d'affichage)
├── vehicle_counting.py     # Script de détection / tracking / comptage
├── bytetrack_custom.yaml   # Réglages du tracker ByteTrack
├── requirements.txt        # Dépendances Python
├── .streamlit/
│   └── config.toml         # Couleurs et options du thème Streamlit
├── yolov8s.pt              # Modèle YOLOv8 small (fichier binaire, ne pas modifier)
├── traffic.mp4             # Vidéo d'exemple
├── result.mp4              # Vidéo annotée (générée)
├── result_h264.mp4         # Version H.264, lisible dans le navigateur
├── results.csv             # Journal des franchissements (généré)
├── summary.json            # Résumé des résultats (généré)
├── screenshots/            # Captures automatiques des passages
└── runs/                   # Résultats des vidéos importées via l'application
```

## Installation

Python 3.9 ou plus est recommandé.

```bash
pip install -r requirements.txt
```

Le modèle `yolov8s.pt` est téléchargé automatiquement au premier lancement (connexion internet nécessaire). S'il est absent ou corrompu, il peut être retéléchargé avec :

```bash
py -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

## Utilisation

### 1. Script en ligne de commande

```bash
py vehicle_counting.py
```

Les paramètres se règlent en haut du fichier :

| Paramètre | Rôle | Valeur par défaut |
|---|---|---|
| `VIDEO_IN` | Vidéo d'entrée | `traffic.mp4` |
| `MODEL` | Modèle YOLO | `yolov8s.pt` |
| `IMGSZ` | Résolution d'analyse (plus haut = meilleure détection des petits véhicules, mais plus lent) | `960` |
| `CONF` | Seuil de confiance | `0.20` |
| `LINE_RATIO` | Position de la ligne (0 = haut, 1 = bas) | `0.60` |
| `SHOW` | Afficher la fenêtre en direct | `True` |
| `MAX_SNAPSHOTS` | Nombre de captures automatiques | `6` |

Appuyez sur `q` pour arrêter ; laissez la vidéo se terminer pour obtenir des résultats complets.

### 2. Application web

Depuis le dossier `vehicle-counting` :

```bash
py -m streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

| Page | Contenu |
|---|---|
| Présentation | Principe du projet et court exemple |
| Dashboard | Total, classes, directions, FPS, frames, graphiques interactifs |
| Vehicle Detection | Vidéo originale et vidéo annotée côte à côte, comptage |
| Image / Video Upload | Import d'une vidéo MP4 (détection, tracking, comptage) ou d'une image (détection) |
| Analytics | Statistiques par classe et direction, comparaison entre vidéos |
| Reports | Résultats détaillés, téléchargement CSV / JSON / vidéo |

## Principe du comptage

1. YOLOv8 détecte les véhicules dans chaque image.
2. ByteTrack associe les détections d'une image à l'autre pour garder les identifiants.
3. On calcule de quel côté de la ligne se trouve le centre de chaque véhicule.
4. Si le côté change entre deux images, le véhicule a franchi la ligne : il est compté une fois, vers le haut (`up`) ou vers le bas (`down`).

## Réglages de détection

Une première version utilisait `yolov8n.pt`, un seuil de confiance de `0.30` et la résolution par défaut (640). Sur la vidéo de test, les motos n'étaient pas détectées. Les réglages actuels visent à corriger ce point :

| Réglage | Avant | Après | Effet attendu |
|---|---|---|---|
| Modèle | `yolov8n.pt` | `yolov8s.pt` | Meilleure détection, traitement plus lent |
| Résolution (`IMGSZ`) | 640 (défaut) | 960 | Meilleure détection des petits véhicules |
| Seuil de confiance (`CONF`) | 0,30 | 0,20 | Moins de véhicules ignorés |
| Mémoire du tracker (`track_buffer`) | 30 (défaut) | 60 | Identifiants conservés plus longtemps |

## Résultats

### Première version (`yolov8n.pt`)

Obtenus sur `traffic.mp4` (`summary.json`) :

| Indicateur | Valeur |
|---|---|
| Frames traitées | 198 |
| Véhicules comptés | 6 (6 voitures) |
| Direction up / down | 3 / 3 |
| FPS de traitement | 10,3 |

### Version actuelle (`yolov8s.pt`, résolution 960)

À compléter après avoir relancé `traffic.mp4` :

| Indicateur | Valeur |
|---|---|
| Frames traitées | |
| Véhicules comptés | |
| Dont motos | |
| Direction up / down | |
| FPS de traitement | |

### Comparaison avec un comptage manuel

Pour mesurer la précision, comptez à la main les véhicules qui franchissent la ligne rouge et comparez :

| Classe | Comptage manuel | Système |
|---|---|---|
| Voitures | | |
| Motos | | |
| Bus | | |
| Camions | | |
| Total | | |

Format de `results.csv` : `frame, time_s, track_id, class, direction`.

## Conseils et limites

- Utilisez une **caméra fixe**, placée en hauteur : les mouvements de caméra dégradent le tracking.
- Les véhicules doivent traverser la ligne horizontale (circulation de haut en bas ou de bas en haut).
- Une image fixe permet uniquement la détection (pas de tracking ni de ligne).
- La vidéo de test est courte (198 frames, environ 8 secondes) et filmée de nuit : les résultats ne sont pas représentatifs d'un trafic réel.
- Le comptage dépend de la qualité de la détection : des véhicules masqués, très petits ou peu éclairés (motos de nuit en particulier) peuvent être manqués.
- Si le tracker change l'identifiant d'un véhicule en cours de route, celui-ci peut être compté deux fois ou manqué.
- `yolov8s` à la résolution 960 est plus lent que `yolov8n` à 640 sur processeur. En cas de lenteur, baissez `IMGSZ` (par exemple à 800).
- Ne pas ouvrir `yolov8s.pt` ni les vidéos dans un éditeur de texte : un enregistrement accidentel corrompt le fichier.

## Technologies

Python, YOLOv8 (Ultralytics), ByteTrack, OpenCV, Pandas, Plotly, Streamlit