"""Vérifie la chaîne complète de l'application : détecteur de pauses (Silero VAD) puis décodage par segment.

Usage : python scripts/evaluate_vad.py <dossier_int8> <silero_vad.onnx> <dossier_audio> <rapport.md>
Pour chaque récitant, les versets d'Al-Fatiha sont mis bout à bout avec des pauses de 0,6 s,
comme une récitation continue. Le VAD découpe, chaque segment est décodé, et le texte complet
est comparé à la référence.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import distance, normalize, read_wav  # noqa: E402

PAUSE_S = 0.6


def main(model_dir: str, vad_path: str, audio_dir: str, report: str) -> None:
    d = Path(model_dir)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=str(d / "tokens.txt"), encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"), joiner=str(d / "joiner.int8.onnx"),
        num_threads=2, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
        model_type="nemo_transducer",
    )
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = vad_path
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.5
    config.silero_vad.min_speech_duration = 0.25
    config.silero_vad.max_speech_duration = 20
    config.sample_rate = 16000
    window = config.silero_vad.window_size

    refs = json.loads((Path(audio_dir) / "references.json").read_text(encoding="utf-8"))
    lines = ["# Récitation continue : détecteur de pauses + décodage par segment (int8)", "",
             f"Al-Fatiha complète, versets séparés par {PAUSE_S} s de silence.", "",
             "| Récitant | Durée | Segments | Délai moyen après la pause | Erreurs mots | Texte reconnu |", "|---|---|---|---|---|---|"]
    for reciter in sorted({k.split("-")[0] for k in refs}):
        keys = [f"{reciter}-001{a:03d}" for a in range(1, 8) if f"{reciter}-001{a:03d}" in refs]
        pause = np.zeros(int(16000 * PAUSE_S), np.float32)
        audio = np.concatenate([np.concatenate([read_wav(Path(audio_dir) / f"{k}.wav"), pause]) for k in keys])
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=120)
        texts, delays = [], []

        def drain() -> None:
            while not vad.empty():
                t0 = time.time()
                stream = rec.create_stream()
                stream.accept_waveform(16000, np.array(vad.front.samples, dtype=np.float32))
                rec.decode_stream(stream)
                delays.append(time.time() - t0)
                texts.append(stream.result.text)
                vad.pop()

        for start in range(0, len(audio), window):
            vad.accept_waveform(audio[start:start + window])
            drain()
        vad.flush()
        drain()
        ref = normalize(" ".join(refs[k] for k in keys))
        hyp = normalize(" ".join(texts))
        wer = 100 * distance(ref, hyp) / max(len(ref), 1)
        lines.append(f"| {reciter} | {len(audio) / 16000:.1f} s | {len(texts)} | {1000 * np.mean(delays):.0f} ms | {wer:.1f} % | {' / '.join(texts)} |")
    lines += ["", "Délai mesuré sur le serveur GitHub (x86, 2 fils) ; il sera plus long sur un téléphone."]
    Path(report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:5])
