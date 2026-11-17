"""Cherche la taille de fenêtre acceptée par l'encodeur en flux (diagnostic).

Usage : python scripts/probe_encoder.py <encoder.onnx>
Essaie plusieurs longueurs d'entrée (en trames de 10 ms) et tailles de cache, et affiche celles qui passent.
"""
import sys

import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
print("entrées :", [(i.name, i.shape) for i in sess.get_inputs()])
ok = []
for cache in (70, 0, 56, 64, 80, 100, 140):
    for frames in range(9, 400):
        feeds = {
            names[0]: np.zeros((1, 80, frames), np.float32),
            names[1]: np.array([frames], np.int64),
            names[2]: np.zeros((1, 17, cache, 512), np.float32),
            names[3]: np.zeros((1, 17, 512, 8), np.float32),
            names[4]: np.array([0], np.int64),
        }
        try:
            out = sess.run(None, feeds)
        except Exception:
            continue
        ok.append((cache, frames, [o.shape for o in out]))
        print(f"OK cache={cache} trames={frames} -> sorties {[o.shape for o in out]}")
        if sum(1 for c, _, _ in ok if c == cache) >= 6:
            break
print(f"{len(ok)} combinaisons valides")
