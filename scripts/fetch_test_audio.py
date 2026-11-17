"""Télécharge des versets récités (everyayah.com) et leur texte de référence (api.alquran.cloud).

Usage : python scripts/fetch_test_audio.py <dossier_audio>
Les fichiers audio servent uniquement au test automatique ; ils ne sont pas publiés.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RECITERS = ["Husary_64kbps", "Minshawy_Murattal_128kbps", "Alafasy_64kbps"]
# Versets courts et moyens ; les premiers versets des sourates (hors Al-Fatiha) sont évités
# car le texte de référence peut y inclure la Basmala.
AYAHS = [(1, a) for a in range(1, 8)] + [(2, 2), (2, 3), (2, 4), (112, 2), (112, 3), (112, 4), (114, 2), (114, 3)]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "karabol-models-ci"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    refs: dict[str, str] = {}
    texts = {}
    for s in sorted({s for s, _ in AYAHS}):
        data = json.loads(get(f"https://api.alquran.cloud/v1/surah/{s}/quran-simple-clean"))
        for ayah in data["data"]["ayahs"]:
            texts[(s, ayah["numberInSurah"])] = ayah["text"]
    for i, reciter in enumerate(RECITERS, 1):
        for s, a in AYAHS:
            key = f"r{i}-{s:03d}{a:03d}"
            mp3 = out / f"{key}.mp3"
            try:
                mp3.write_bytes(get(f"https://everyayah.com/data/{reciter}/{s:03d}{a:03d}.mp3"))
            except Exception as exc:  # un fichier manquant ne bloque pas le test
                print(f"ignoré {reciter} {s}:{a} : {exc}")
                continue
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out / f"{key}.wav")], check=True)
            mp3.unlink()
            refs[key] = texts[(s, a)]
    (out / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(refs)} versets prêts ; récitants : {', '.join(RECITERS)}")
    if len(refs) < 20:
        raise SystemExit("trop peu de versets téléchargés")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
