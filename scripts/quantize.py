"""Convertit le modèle FastConformer-Quran (sherpa-onnx) en int8 (quantification dynamique des poids).

Usage : python scripts/quantize.py <dossier_fp32> <dossier_int8>
- Quantification QUInt8 avec les opérations par défaut, exactement comme le script d'export NeMo de sherpa-onnx.
- Les métadonnées ONNX lues par sherpa-onnx (taille de fenêtre, cache, etc.) sont recopiées et vérifiées.
"""
import shutil
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def metadata(path: Path) -> dict[str, str]:
    return {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}


def main(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("encoder", "decoder", "joiner"):
        fp32, int8 = src / f"{name}.onnx", dst / f"{name}.int8.onnx"
        quantize_dynamic(model_input=str(fp32), model_output=str(int8), weight_type=QuantType.QUInt8)
        before, after = metadata(fp32), metadata(int8)
        if before != after:
            model = onnx.load(str(int8))
            del model.metadata_props[:]
            for key, value in before.items():
                model.metadata_props.add(key=key, value=value)
            onnx.save(model, str(int8))
            after = metadata(int8)
        assert before == after, f"métadonnées différentes pour {name}"
        a, b = onnx.load(str(fp32), load_external_data=False), onnx.load(str(int8), load_external_data=False)
        assert b.ir_version <= a.ir_version, f"version IR plus récente pour {name}"
        print(f"{name}: {fp32.stat().st_size / 2**20:.1f} Mo -> {int8.stat().st_size / 2**20:.1f} Mo ; "
              f"IR {a.ir_version} -> {b.ir_version} ; opset {[o.version for o in b.opset_import]} ; métadonnées : {sorted(before)}")
    shutil.copy(src / "tokens.txt", dst / "tokens.txt")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
