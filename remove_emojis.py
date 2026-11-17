"""Supprime les emojis de app.py (titres, menu, boutons, onglets, messages).

Usage (depuis le dossier du projet) :  py remove_emojis.py
Une sauvegarde est faite dans app.py.emoji.bak. La ligne page_icon (icône de l'onglet du navigateur) est conservée.
"""
import re
import shutil
from pathlib import Path

EMOJI = r"[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+"
PATTERN = re.compile(rf"( ?)({EMOJI})( ?)")


def clean(line: str) -> str:
    if "page_icon" in line:
        return line
    return PATTERN.sub(lambda m: " " if (m.group(1) and m.group(3)) else "", line)


p = Path("app.py")
if not p.exists():
    raise SystemExit("app.py introuvable : lancez le script depuis le dossier du projet.")
src = p.read_text(encoding="utf-8")
bak = Path("app.py.emoji.bak")
if not bak.exists():
    shutil.copy(p, bak)
lines = src.splitlines(keepends=True)
out = [clean(l) for l in lines]
changed = sum(a != b for a, b in zip(lines, out))
p.write_text("".join(out), encoding="utf-8")
print(f"{changed} lignes modifiées. Sauvegarde : {bak.name}")