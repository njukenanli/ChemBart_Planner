# Buchwald-Hartwig Reaction Yield Benchmark

This repository contains the original data and scripts for the Buchwald-Hartwig C–N coupling yield prediction benchmark test.

## 📂 Dataset
The dataset we used is located in the following directory: - `./bh_C_N_yield_dataset/`


## Model Optimization
Model performance was optimized via a grid search over the learning rate (`lr`) and decay (`decay`) hyperparameters.

For a complete workflow example, please refer to:
- `example_YieldBH_test2.py`

## Experimental Results
- **70/30 random Split and Test 1–3 :** Results were obtained through automated hyperparameter optimization (grid search).
- **Test 4:** Results were further refined via manual hyperparameter tuning. The detailed analysis, code, and outputs are documented separately in:
  - `Yield_BH_test4.ipynb`

## 📝 Notes
- Ensure all required dependencies are installed before executing the scripts.
- If your local directory structure differs from the default layout, please adjust the file paths accordingly.
- For reproducibility, random seeds and environment configurations are recommended to be documented in your execution logs.