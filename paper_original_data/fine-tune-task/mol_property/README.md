# Molecular Property Prediction Benchmark

This repository provides benchmark results for molecular property prediction using various pretraining strategies, including molecular-level and reaction-level representations.

## 📂 Data & Scripts Overview

| File/Directory | Description |
|----------------|-------------|
| `./data_MoIR/` | Raw and processed datasets for molecular property prediction tasks |
| `mol_property_molecular.ipynb` | Fine-tuning **molecular pre-trained models ChemBART-Mol**  for property prediction • 3 independent runs with different random seeds|
| `mol_property_reaction.ipynb` | Fine-tuning **reaction pre-trained models** for molecular property prediction:<br>• `ChemBART-M`: Pre-trained on reaction data<br>• `ChemBART-R`: Same architecture, randomly initialized weights<br>• *3 independent runs with different random seeds* |
| `mol_property_reaction_encoder.ipynb` | Fine-tuning **only the encoder** of ChemBART-M (frozen decoder):<br>• 3 independent runs + 5-fold cross-validation<br>• Evaluates encoder-only transferability |

## 🎯 Reported Results for ChemBART-M

The final reported performance of **ChemBART-M** is selected as the **best result** across:
- `mol_property_reaction.ipynb` (full model fine-tuning)
- `mol_property_reaction_encoder.ipynb` (encoder-only fine-tuning)

This strategy ensures fair comparison by leveraging the optimal adaptation protocol for each downstream task.

