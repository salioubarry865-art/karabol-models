# Mentions et licences

## Modèle FastConformer-Quran (fichiers des Releases)

- Modèle de base : *stt_ar_fastconformer_hybrid_large_pcd* — NVIDIA, licence CC-BY-4.0.
- Variante « streaming » réentraînée : Ahmed Hany (`dev-ahmedhany` sur Hugging Face), licence CC-BY-4.0,
  point de contrôle `streaming_v11_epoch7.nemo` du dépôt `dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-mirror`.
- **Modification apportée par Karaböl** : export ONNX en 3 fichiers (encodeur, décodeur, joiner) avec le script officiel de sherpa-onnx, puis quantification dynamique int8 des poids (QUInt8) avec ONNX Runtime.
  Aucun réentraînement ; le vocabulaire (`tokens.txt`) est inchangé.
- Licence de la version modifiée : CC-BY-4.0 — https://creativecommons.org/licenses/by/4.0/

## Détecteur de pauses (silero_vad.onnx)

- Silero VAD — https://github.com/snakers4/silero-vad — licence MIT, © Silero Team.
- Copie non modifiée du fichier distribué par sherpa-onnx
  (https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx).

## Scripts de ce dépôt

Les scripts de conversion et de test sont fournis par Karaböl sous licence MIT.

## Audio de test

Les récitations utilisées par le test automatique proviennent d'everyayah.com ; elles sont téléchargées
au moment du test et ne sont pas redistribuées dans ce dépôt.
