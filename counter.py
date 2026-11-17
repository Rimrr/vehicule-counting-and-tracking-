"""Comptage par franchissement de ligne, robuste aux oscillations et aux frames perdues."""

# Classes COCO utiles pour le trafic : voiture, moto, bus, camion
TRAFFIC_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class LineCounter:
    """Compte chaque identifiant une seule fois quand son centre change de côté d'une ligne horizontale.

    - margin : zone neutre (en pixels) autour de la ligne. Un véhicule qui hésite dans cette
      bande n'est pas compté deux fois.
    - Le changement de côté suffit : même si le tracker saute des frames, le passage est compté.
    """

    def __init__(self, line_y: float, margin: float = 6):
        self.line_y = line_y
        self.margin = margin
        self.last_side: dict[int, int] = {}
        self.counted: set[int] = set()
        self.by_direction = {"up": 0, "down": 0}
        self.by_class: dict[str, int] = {}

    def update(self, track_id: int, cy: float, class_name: str) -> str | None:
        """Met à jour un véhicule. Retourne 'up', 'down' si un passage vient d'être compté."""
        if cy < self.line_y - self.margin:
            side = -1  # au-dessus de la ligne
        elif cy > self.line_y + self.margin:
            side = 1   # en dessous
        else:
            return None  # zone neutre : on garde le dernier côté connu

        previous = self.last_side.get(track_id)
        self.last_side[track_id] = side
        if previous is None or previous == side or track_id in self.counted:
            return None

        self.counted.add(track_id)
        direction = "up" if previous == 1 else "down"
        self.by_direction[direction] += 1
        self.by_class[class_name] = self.by_class.get(class_name, 0) + 1
        return direction

    @property
    def total(self) -> int:
        return len(self.counted)