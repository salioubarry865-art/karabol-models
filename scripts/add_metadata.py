"""Ajoute à l'encodeur les métadonnées exigées par sherpa-onnx pour un transducteur NeMo en flux.

Usage : python scripts/add_metadata.py <dossier_source> <dossier_sortie>

Les fichiers publiés sur Hugging Face n'ont aucune métadonnée : sherpa-onnx 1.12.35 s'arrête alors
avec « 'vocab_size' does not exist in the metadata » (exit -1), ce qui fermait l'application.
Les valeurs suivent le script officiel sherpa-onnx
scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer.py :
- dimensions statiques des caches et de l'état du décodeur : lues dans le graphe ONNX ;
- taille du cache d'attention : att_context_size[0] = 70 (fiche du modèle) ;
- taille du cache de convolution : noyau de la convolution « depthwise » − 1 (lu dans les poids) ;
- vocab_size : nombre de jetons sans le blanc ; tokens.txt ne contient pas <blk>, il est ajouté
  à la fin (sortie du joiner = vocab_size + 1) ;
- window_size / chunk_shift : configuration de flux NeMo « chunked_limited » avec att_context_size = [70, 13]
  (fiche du modèle) et un sous-échantillonnage de 8 : chunk = 8 + 8 × 13 = 112 trames,
  cache de pré-encodage = 8 + 1 = 9 trames, fenêtre = 121 trames.
"""
import shutil
import sys
from pathlib import Path

import onnx

ATT_CONTEXT = (70, 13)
SUBSAMPLING = 8


def shape(value_info) -> list:
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in value_info.type.tensor_type.shape.dim]


def describe(path: Path) -> onnx.ModelProto:
    model = onnx.load(str(path), load_external_data=False)
    print(f"== {path.name} (IR {model.ir_version}, opset {[o.version for o in model.opset_import]})")
    for io, items in (("entrée", model.graph.input), ("sortie", model.graph.output)):
        for v in items:
            print(f"   {io} {v.name}: {shape(v)}")
    print(f"   métadonnées : {[(p.key, p.value) for p in model.metadata_props]}")
    return model


def static(dims: list, index: int, what: str) -> int:
    value = dims[index]
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f"dimension non statique pour {what} : {dims}")
    return value


def main(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    encoder = describe(src / "encoder.onnx")
    decoder = describe(src / "decoder.onnx")
    describe(src / "joiner.onnx")

    enc_inputs = list(encoder.graph.input)
    if len(enc_inputs) != 5:
        raise SystemExit("encodeur inattendu : 5 entrées attendues (audio, longueur, 2 caches, longueur du cache)")
    channel, time = shape(enc_inputs[2]), shape(enc_inputs[3])
    states = [shape(v) for v in decoder.graph.input if len(shape(v)) == 3]
    if not states:
        raise SystemExit("état du décodeur introuvable")

    tokens = [t for t in (src / "tokens.txt").read_text(encoding="utf-8").split("\n") if t.strip()]
    for i, line in enumerate(tokens):
        if int(line.split()[-1]) != i:
            raise SystemExit(f"tokens.txt : identifiant inattendu à la ligne {i + 1} : {line!r}")
    add_blank = tokens[-1].split()[0] != "<blk>"
    vocab_size = len(tokens) if add_blank else len(tokens) - 1
    joiner = onnx.load(str(src / "joiner.onnx"), load_external_data=False)
    joiner_out = shape(joiner.graph.output[0])[-1]
    if joiner_out != vocab_size + 1:
        raise SystemExit(f"sortie du joiner ({joiner_out}) différente de vocab_size + 1 ({vocab_size + 1})")

    full = onnx.load(str(src / "encoder.onnx"))
    d_model = static(channel, 3, "cache_last_channel")
    kernels = {int(t.dims[2]) for t in full.graph.initializer if len(t.dims) == 3 and t.dims[0] == d_model and t.dims[1] == 1}
    if len(kernels) != 1:
        raise SystemExit(f"noyau de convolution depthwise ambigu : {kernels}")
    conv_cache = kernels.pop() - 1

    chunk = SUBSAMPLING + SUBSAMPLING * ATT_CONTEXT[1]
    pre_encode_cache = SUBSAMPLING + 1
    meta = {
        "vocab_size": vocab_size,
        "window_size": chunk + pre_encode_cache,
        "chunk_shift": chunk,
        "normalize_type": "per_feature",
        "cache_last_channel_dim1": static(channel, 1, "cache_last_channel"),
        "cache_last_channel_dim2": ATT_CONTEXT[0],
        "cache_last_channel_dim3": static(channel, 3, "cache_last_channel"),
        "cache_last_time_dim1": static(time, 1, "cache_last_time"),
        "cache_last_time_dim2": static(time, 2, "cache_last_time"),
        "cache_last_time_dim3": conv_cache,
        "pred_rnn_layers": static(states[0], 0, "état du décodeur"),
        "pred_hidden": static(states[0], 2, "état du décodeur"),
        "subsampling_factor": SUBSAMPLING,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo (NVIDIA), streaming fine-tune by Ahmed Hany",
        "url": "https://huggingface.co/dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-sherpa",
        "comment": "Metadata added by Karabol (values from the sherpa-onnx NeMo export script)",
    }
    if meta["cache_last_time_dim2"] != d_model:
        raise SystemExit("dimension du modèle incohérente entre les deux caches")
    if meta["cache_last_channel_dim1"] != meta["cache_last_time_dim1"]:
        raise SystemExit("nombre de couches incohérent entre les deux caches")
    print("métadonnées ajoutées :", meta)

    del encoder.metadata_props[:]
    for key, value in meta.items():
        encoder.metadata_props.add(key=key, value=str(value))
    del full.metadata_props[:]
    full.metadata_props.extend(encoder.metadata_props)
    onnx.save(full, str(dst / "encoder.onnx"))
    for name in ("decoder.onnx", "joiner.onnx"):
        shutil.copy(src / name, dst / name)
    if add_blank:
        tokens.append(f"<blk> {vocab_size}")
        print(f"tokens.txt : ajout de « <blk> {vocab_size} »")
    (dst / "tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
