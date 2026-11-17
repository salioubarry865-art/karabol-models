"""Exporte le point de contrôle NeMo officiel en 3 fichiers ONNX pour sherpa-onnx (transducteur en flux).

Usage : python scripts/export_nemo.py <modele.nemo> <dossier_sortie> [--offline]

--offline : export non streaming (script officiel export-onnx-transducer-non-streaming.py), décodé
par segments (OfflineRecognizer, type nemo_transducer). Sans l'option : export en flux (cache-aware).

Reprend le script officiel de sherpa-onnx
scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer.py (v1.12.35),
en chargeant le fichier .nemo local au lieu d'un modèle du catalogue NVIDIA.
Les fichiers ONNX publiés par l'auteur n'avaient pas de métadonnées et leur encodeur refusait
les fenêtres de flux : on repart donc du point de contrôle d'origine.
"""
import os
import sys
from pathlib import Path

import nemo.collections.asr as nemo_asr
import onnx
import torch


def add_meta_data(filename: str, meta_data: dict) -> None:
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


@torch.no_grad()
def main(nemo_path: str, out: Path, offline: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    asr_model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    print("classe :", type(asr_model).__name__)
    print("préprocesseur :", dict(asr_model.cfg.preprocessor))
    print("att_context_size :", asr_model.encoder.att_context_size, "style :", asr_model.encoder.att_context_style)

    os.chdir(out)
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i + 1}\n")

    asr_model.change_decoding_strategy(decoder_type="rnnt")
    asr_model.eval()
    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""
    if offline:
        asr_model.set_export_config({"decoder_type": "rnnt"})
        asr_model.encoder.export("encoder.onnx")
        asr_model.decoder.export("decoder.onnx")
        asr_model.joint.export("joiner.onnx")
        meta_data = {
            "vocab_size": asr_model.decoder.vocab_size,
            "normalize_type": normalize_type,
            "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
            "pred_hidden": asr_model.decoder.pred_hidden,
            "subsampling_factor": 8,
            "model_type": "EncDecHybridRNNTCTCBPEModel",
            "version": "1",
            "model_author": "NeMo (NVIDIA base model), fine-tune by Ahmed Hany",
            "url": "https://huggingface.co/dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-mirror",
            "comment": "Only the transducer branch is exported (Karabol, sherpa-onnx non-streaming export script)",
        }
        add_meta_data("encoder.onnx", meta_data)
        print("métadonnées :", meta_data)
        return
    if asr_model.encoder.streaming_cfg is None:
        asr_model.encoder.setup_streaming_params()
    cfg = asr_model.encoder.streaming_cfg
    print("streaming_cfg :", cfg)

    chunk_size = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
    pre_encode_cache_size = cfg.pre_encode_cache_size[1] if isinstance(cfg.pre_encode_cache_size, list) else cfg.pre_encode_cache_size
    window_size = chunk_size + pre_encode_cache_size
    chunk_shift = chunk_size

    asr_model.set_export_config({"decoder_type": "rnnt", "cache_support": True})
    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "window_size": window_size,
        "chunk_shift": chunk_shift,
        "normalize_type": normalize_type,
        "cache_last_channel_dim1": len(asr_model.encoder.layers),
        "cache_last_channel_dim2": cfg.last_channel_cache_size,
        "cache_last_channel_dim3": asr_model.encoder.d_model,
        "cache_last_time_dim1": len(asr_model.encoder.layers),
        "cache_last_time_dim2": asr_model.encoder.d_model,
        "cache_last_time_dim3": asr_model.encoder.conv_context_size[0],
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo (NVIDIA base model), streaming fine-tune by Ahmed Hany",
        "url": "https://huggingface.co/dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-mirror",
        "comment": "Only the transducer branch is exported (Karabol, sherpa-onnx export script)",
    }
    add_meta_data("encoder.onnx", meta_data)
    print("métadonnées :", meta_data)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]).resolve(), "--offline" in sys.argv[3:])
