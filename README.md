

# 🐋 **Real-Time Beluga & Marine Animals Detection Tool**


<div align="center">

**Détection en temps réel de bélugas, dauphins et autres animaux marins à partir de vidéos aériennes**
Basé sur YOLO, ViT & SAM2 — Entraînement multi-phases et segmentation avancée.

</div>

---

## **Table des matières**

* [Introduction](#-introduction)
* [Objectifs du projet](#-objectifs-du-projet)
* [Structure du projet](#-structure-du-projet)
* [Méthodologie & pipeline](#-méthodologie--pipeline)
* [Résultats](#-résultats)
* [Installation & Environnement](#-installation--environnement)
* [Scripts principaux](#-scripts-principaux)
* [Utilisation pour les phases YOLO](#-utilisation)
* [Utilisation pour la phase SAM2 + ViT](#-utilisation)
* [Auteur](#-auteur)

---

## **Introduction**

Ce projet propose un outil complet de détection automatique de bélugas, de dauphins et d’autres espèces marines à partir de vidéos aériennes prises par drone.

Le système repose sur :

* **YOLOv11n** pour la détection rapide
* **Vision Transformers (ViT)** pour la classification robuste
* **SAM2** pour la segmentation et le cropping 
* un pipeline d’entraînement en **6 phases progressives**, intégrant de nouvelles espèces et davantage d’images pour améliorer la robustesse du modèle.

---

## **Objectifs du projet**

* Détecter des **bélugas en temps réel** dans des vidéos aériennes.
* Étendre la détection à **d’autres animaux marins** (dauphins, phoques, requins…).
* Construire un pipeline **itératif** et explicable : augmentation, transfert de connaissances, fine-tuning, segmentation.
* Fournir des outils d’évaluation, de comparaison et d’analyse de performance.

---

## **Structure du projet**

Voici une vue simplifiée de l'arborescence (réduite pour éviter 300 lignes dans le README) :

```txt
Real-Time-Beluga-Whales-Detection-Tool/
├── data/                  # Jeux de données (bruts, augmentés, multi-espèces)
├── models/                # Checkpoints des modèles YOLO etViT
├── outputs/               # Prédictions, évaluations, comparaisons
├── src/                   # Scripts de training, testing, segmentation
└── README.md
```

---

## **Méthodologie & pipeline**

Le pipeline complet est divisé en plusieurs phases :

1. **Transfer Learning sur les bélugas**
2. **Fine-tuning avancé**
3. **Ajout d’une nouvelle espèce : dauphins**
4. **Extension des données bélugas**
5. **Segmentation SAM2 + ViT pour classification fine**
6. **Modèle final multi-espèces (2526 images)**

Schéma du pipeline YOLO (phases 1 à 4 et 6) :

```
Images → YOLO → Évaluation  (Entraînement / Validation / Test sur images)
Vidéos →  YOLO → Prédictions finales (Test en temps réel)
```

Schéma du pipeline SAM2 + ViT (phase 5) :

```
Images → Segmentation (SAM2) → Crops → ViT → Évaluation Entraînement / Validation / Test sur images)
Vidéos → Segmentation (SAM2) → Crops → ViT → Prédictions finales  (Test en temps réel)
```


---

## **Résultats**


| Phase   | Dataset     | Modèle     | Accuracy | mAP50 |
|---------|-------------|------------|----------|-------|
| Phase 1 | 275 images  | YOLO       | 90.65%   | 0.88  |
| Phase 2 | 275 images  | YOLO       | 91.46%   | 0.90  |
| Phase 3 | 744 images  | YOLO       | 91.78%   | 0.91  |
| Phase 4 | 652 images  | YOLO       | 88.65%   | 0.92  |
| Phase 5 | 150 images  | SAM2 + ViT | 100.0%   | /     |
| Phase 6 | 2526 images | YOLO       | 91.12%   | 0.91  |

---

## **Installation & Environnement**

### Prérequis

* Python 3.10+
* CUDA (optionnel)
* PyTorch
* Ultralytics YOLO
* OpenCV
* SAM2 (Meta AI)
---

## **Scripts principaux**

```txt
src/
├── training_yolo/       # entraînement YOLO (phases 1 à 4 et 6)
├── testing_yolo/        # évaluation et prédictions (phases 1 à 4 et 6)
├── sam2_vit/            # segmentation + ViT (phase 5)
└── videos/              # vidéos de test
```

## **Utilisation pour les phases YOLO**

#### Pour entraîner les modèles YOLO

```bash
python src/training_yolo/train_phase1.py
python src/training_yolo/train_phase2.py
python src/training_yolo/train_phase3.py
python src/training_yolo/train_phase4.py
python src/training_yolo/train_phase6.py
```

#### Pour comparer les modèles

```bash
python src/testing/compare_models.py
```

#### Pour évaluer les modèles sur des images de test

```bash
python src/testing/evaluate.py
```

#### Pour prédire sur des vidéos

```bash
python src/testing/predict.py
```

#### **Sorties pour les phases 1 à 4 et 6**
```txt
outputs/Phase_X/
├── evaluation/             # résultats d'évaluation du meilleure modèle (metrics, courbes)
├── prediction/             # prédictions du meilleure modèle sur images de test et sur les vidéos
└── models_comparison.csv   # comparaison des modèles selon la métrique mAP50
```
## **Utilisation pour la phase SAM2 + ViT**

#### Pour segmenter et croper les images d'entraînement avec SAM2'

```bash
python src/sam2_vit/segment_and_crop.py
```

#### Pour entraîner le modèle ViT

```bash
python src/sam2_vit/train_and_evaluate_sam2_vit.py
```

#### Pour prédire sur des vidéos

```bash
python src/sam2_vit/predict_videos_sam2_vit.py
```

### **Résultats pour la phase 5**
```txt
outputs/Phase_5/
├── evaluation/           # résultats d'évaluation du modèle ViT (metrics, courbes)
├── prediction_images/    # prédictions du modèle ViT sur images de test
├── prediction_vidéos/    # prédictions du modèle SAM2 + ViT sur vidéos de test
└──  segmentation/        # résultats de segmentation SAM2 (bounding boxes, crops) lors de l'entraînement
```


---

## **Auteure**

**Howet Marie**, UQAC – Maitrise en Informatique (Intelligence Artificielle) \
Contact : *howet.marie@gmail.com* / *mhowet@etu.uqac.ca*
---

