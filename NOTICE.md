# Mentions et licences

## Modèle FastConformer-Quran (fichiers des Releases)

- Modèle de base : *stt_ar_fastconformer_hybrid_large_pcd* — NVIDIA, licence CC-BY-4.0.
- Variante « streaming » pour sherpa-onnx : Ahmed Hany (`dev-ahmedhany` sur Hugging Face), licence CC-BY-4.0,
  dépôt `dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-sherpa`, commit `2efcdd746b5470480de0787ba781e5622c080fc1`.
- **Modification apportée par Karaböl** : ajout des métadonnées sherpa-onnx à l'encodeur (dimensions des caches, fenêtre de flux), puis quantification dynamique int8 des poids (QUInt8) avec ONNX Runtime.
  Aucun réentraînement ; le vocabulaire (`tokens.txt`) est inchangé.
- Licence de la version modifiée : CC-BY-4.0 — https://creativecommons.org/licenses/by/4.0/

## Scripts de ce dépôt

Les scripts de conversion et de test sont fournis par Karaböl sous licence MIT.

## Audio de test

Les récitations utilisées par le test automatique proviennent d'everyayah.com ; elles sont téléchargées
au moment du test et ne sont pas redistribuées dans ce dépôt.
