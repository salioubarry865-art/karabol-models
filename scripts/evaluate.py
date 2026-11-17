"""Compare le modèle d'origine (fp32) et la version int8 sur des récitations réelles.

Usage : python scripts/evaluate.py <dossier_fp32> <dossier_int8> <dossier_audio> <rapport.md> [--offline]
Pour chaque modèle, dans un processus séparé : mémoire maximale (RSS), temps de chargement,
vitesse de décodage et taux d'erreur sur les mots et les lettres (texte sans voyelles brèves).
Sans --offline : décodage en flux (morceaux de 100 ms). Avec --offline : un verset = un segment
(OfflineRecognizer), comme un découpage par pauses dans l'application.
"""
import json
import re
import resource
import subprocess
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^ء-ي ]", " ", text)
    return text.split()


def distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def run_one(model_dir: str, suffix: str, audio_dir: str, offline: bool) -> None:
    """Exécuté dans un processus séparé pour mesurer la mémoire de ce seul modèle."""
    import sherpa_onnx

    d = Path(model_dir)
    t0 = time.time()
    if offline:
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder{suffix}.onnx"),
            decoder=str(d / f"decoder{suffix}.onnx"), joiner=str(d / f"joiner{suffix}.onnx"),
            num_threads=2, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            model_type="nemo_transducer",
        )
    else:
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder{suffix}.onnx"),
            decoder=str(d / f"decoder{suffix}.onnx"), joiner=str(d / f"joiner{suffix}.onnx"),
            num_threads=2, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
        )
    load_s = time.time() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results, audio_s, decode_s = {}, 0.0, 0.0
    for wav in sorted(Path(audio_dir).glob("*.wav")):
        samples = read_wav(wav)
        audio_s += len(samples) / 16000
        t1 = time.time()
        if offline:
            stream = rec.create_stream()
            stream.accept_waveform(16000, samples)
            rec.decode_stream(stream)
            decode_s += time.time() - t1
            results[wav.stem] = stream.result.text
            continue
        stream = rec.create_stream()
        for start in range(0, len(samples), 1600):
            stream.accept_waveform(16000, samples[start:start + 1600])
            while rec.is_ready(stream):
                rec.decode_stream(stream)
        stream.accept_waveform(16000, np.zeros(int(16000 * 1.5), dtype=np.float32))
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        decode_s += time.time() - t1
        results[wav.stem] = rec.get_result(stream)
    print(json.dumps({
        "load_s": round(load_s, 2), "rss_load_mb": round(rss_after_load),
        "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "rtf": round(decode_s / audio_s, 3) if audio_s else None, "texts": results,
    }, ensure_ascii=False))


def score(texts: dict[str, str], refs: dict[str, str]) -> tuple[float, float]:
    we = wn = ce = cn = 0
    for key, ref in refs.items():
        r, h = normalize(ref), normalize(texts.get(key, ""))
        we += distance(r, h); wn += len(r)
        rc, hc = list("".join(r)), list("".join(h))
        ce += distance(rc, hc); cn += len(rc)
    return 100 * we / max(wn, 1), 100 * ce / max(cn, 1)


def main(fp32: str, int8: str, audio_dir: str, report: str, offline: bool) -> None:
    refs = json.loads((Path(audio_dir) / "references.json").read_text(encoding="utf-8"))
    runs = {}
    for label, model_dir, suffix in (("Original (fp32)", fp32, ""), ("int8", int8, ".int8")):
        out = subprocess.run([sys.executable, __file__, "--one", model_dir, suffix, audio_dir, "--offline" if offline else "--stream"], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"{label} : échec\n{out.stderr[-4000:]}")
        runs[label] = json.loads(out.stdout.strip().splitlines()[-1])
        runs[label]["wer"], runs[label]["cer"] = score(runs[label]["texts"], refs)

    sizes = {lbl: sum(p.stat().st_size for p in Path(d).glob(f"*{suf}.onnx")) / 2**20
             for lbl, d, suf in (("Original (fp32)", fp32, ""), ("int8", int8, ".int8"))}
    lines = [
        f"# Comparaison fp32 / int8 — FastConformer-Quran ({'par segments' if offline else 'en flux'})", "",
        f"{len(refs)} versets, {len({k.split('-')[0] for k in refs})} récitants (everyayah.com). Texte comparé sans voyelles brèves.", "",
        "| Mesure | Original (fp32) | int8 |", "|---|---|---|",
    ]
    a, b = runs["Original (fp32)"], runs["int8"]
    lines += [
        f"| Taille des fichiers ONNX | {sizes['Original (fp32)']:.0f} Mo | {sizes['int8']:.0f} Mo |",
        f"| Mémoire après chargement (RSS max) | {a['rss_load_mb']} Mo | {b['rss_load_mb']} Mo |",
        f"| Mémoire maximale totale | {a['rss_peak_mb']} Mo | {b['rss_peak_mb']} Mo |",
        f"| Temps de chargement | {a['load_s']} s | {b['load_s']} s |",
        f"| Vitesse (temps de calcul / durée audio) | {a['rtf']} | {b['rtf']} |",
        f"| Erreurs sur les mots (WER) | {a['wer']:.1f} % | {b['wer']:.1f} % |",
        f"| Erreurs sur les lettres (CER) | {a['cer']:.1f} % | {b['cer']:.1f} % |",
        "", "Mesures faites sur le serveur GitHub (x86, 2 fils), pas sur un téléphone : comparer les deux colonnes entre elles.",
        "Les récitations d'everyayah.com font partie des données d'entraînement du modèle : les taux absolus sont optimistes,",
        "seul l'écart entre fp32 et int8 est significatif.", "",
        "## Détail par verset", "", "| Verset | Référence | fp32 | int8 |", "|---|---|---|---|",
    ]
    for key, ref in refs.items():
        lines.append(f"| {key} | {ref} | {a['texts'].get(key, '')} | {b['texts'].get(key, '')} |")
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        run_one(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "--offline")
    else:
        main(*sys.argv[1:5], offline="--offline" in sys.argv[5:])
