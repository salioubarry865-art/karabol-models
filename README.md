# karabol-models

Modèles de reconnaissance vocale téléchargés par l'application mobile **Karaböl** (mémorisation du Coran)
à la première récitation contrôlée. Le code de l'application est dans un dépôt séparé.

## FastConformer-Quran int8

| | |
|---|---|
| Source | [`dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-sherpa`](https://huggingface.co/dev-ahmedhany/stt_ar_fastconformer_hybrid_large_streaming_pcd_v1.1-sherpa), commit `2efcdd746b5470480de0787ba781e5622c080fc1` |
| Modèle de base | FastConformer hybride arabe de NVIDIA, adapté au temps réel par Ahmed Hany |
| Licence | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — voir [NOTICE.md](NOTICE.md) |
| Modification | Ajout des métadonnées sherpa-onnx à l'encodeur (absentes de la source), puis quantification dynamique int8 des poids (QUInt8) avec ONNX Runtime 1.24.4 |
| Moteur | sherpa-onnx 1.12.35 (via react-native-sherpa-onnx 0.4.4) |

Les fichiers sont publiés dans les **Releases** de ce dépôt, avec leurs empreintes SHA-256.
L'application vérifie la taille et l'empreinte de chaque fichier après le téléchargement.

## Reproduire

Le workflow [`int8.yml`](.github/workflows/int8.yml) (lancement manuel) :

1. télécharge le modèle d'origine à la version figée et vérifie ses empreintes ;
2. ajoute les métadonnées exigées par sherpa-onnx (`scripts/add_metadata.py`), puis convertit en int8 (`scripts/quantize.py`) ;
3. compare l'original et l'int8 sur des versets récités (`scripts/fetch_test_audio.py`, `scripts/evaluate.py`) :
   mémoire, temps de chargement, vitesse, taux d'erreur sur les mots et les lettres ;
4. crée une Release **en brouillon** avec les fichiers int8 et le rapport, publiée seulement après validation.
