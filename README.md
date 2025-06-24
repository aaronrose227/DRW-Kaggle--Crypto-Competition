# DRW Crypto Market Prediction

A self‐contained Kaggle project using DRW’s proprietary and public minute‐level crypto data to forecast next‐minute price moves.  This README explains each step of the pipeline, so you (or anyone else) can understand and reproduce the work.

---

## 1. Competition Overview  
- **Goal:** Predict the continuous “label” (next‐minute price movement) for each timestamp.  
- **Data period:** March 1 2023 – February 29 2024 (train); hidden minutes afterwards (test).  
- **Metric:** Pearson correlation between your predictions and true labels on held‐out test data.  
- **Result:** Top 7.4% finish (rank 464 / 6291).

---

## 2. Data Description  
- **timestamp**: UTC minute index (train only).  
- **bid_qty, ask_qty**: Quantity available at best bid/ask.  
- **buy_qty, sell_qty**: Volume executed on the ask (aggressive buys) and bid (aggressive sells).  
- **volume**: Total traded volume that minute.  
- **X₁…X₈₉₀**: 890 anonymized “production” features—proprietary order‐book and flow signals DRW generates in real time.  
- **label**: The target you predict (future price change).

---

## 3. Raw Feature Selection  
1. **Correlation screening** on a 10% random sample: compute |Pearson( Xᵢ, label )|.  
2. **Feature‐importance** from a quick LightGBM run on that sample.  
3. **Ablation tests** with small retrains to confirm consistent CV gains.  
4. **Community consensus**: adopt the ~24 highest‐signal X-columns that repeatedly improved OOF correlation.  

**Final raw feature list** (24 columns):  
X863, X856, X598, X862, X385, X852,
X603, X860, X674, X415, X345, X855,
X174, X302, X178, X168, X612,
buy_qty, sell_qty, volume,
X888, X421, X333,
bid_qty, ask_qty




---

## 4. Feature Engineering  
Added four customizable microstructure signals to amplify order‐flow and volume imbalances:

| New Feature               | Formula                                                   |
|---------------------------|-----------------------------------------------------------|
| volume_weighted_sell      | sell_qty × volume                                         |
| buy_sell_ratio            | buy_qty / (sell_qty + ε)                                  |
| selling_pressure          | sell_qty / (volume + ε)                                   |
| effective_spread_proxy    | |buy_qty − sell_qty| / (volume + ε)                       |

- Replace infinities with NaN, then fill NaN → 0.  
- Final input matrix has **24 raw + 4 engineered = 28 features**.

---

## 5. Model & Hyperparameters  
- **Algorithm:** XGBoost (histogram‐based) with **GPU** predictor for speed.  
- **Key params:**  
  - `max_depth=20, max_leaves=12` (deep, flexible trees)  
  - `learning_rate≈0.022`, `n_estimators=1667`  
  - Subsampling: `subsample≈0.066`, `colsample_bytree≈0.71`  
  - Regularization: `gamma≈1.71`, `reg_alpha≈39`, `reg_lambda≈75`  

---

## 6. Handling Non‐Stationarity  
1. **Time‐decay sample weights**  
   - Weight each training row by `w[i] ∝ DECAY**(1 - i/(N-1))`, so recent minutes count more in the loss.  
2. **Temporal Slicing**  
   - For each CV fold, collect out‐of‐fold (OOF) predictions on three “eras”:  
     - **full** (all samples)  
     - **last 75%** (drop earliest 25%)  
     - **last 50%** (drop earliest 50%)  

---

## 7. Cross‐Validation & OOF  
- **3-fold time-series CV** (no shuffling; contiguous splits).  
- In each fold:  
  1. Train on two chunks (with decay weights).  
  2. Predict on the held‐out chunk → store OOF for each slice.  
  3. Predict on test set → accumulate per slice.  
- After all folds, average test predictions across folds for each slice.

---

## 8. Ensembling  
- Compute Pearson r between true `label` and each slice’s OOF → gives three slice‐scores.  
- Form two ensembles of test‐set predictions:  
  1. **Simple mean** of the three slices.  
  2. **Weighted mean** (weights ∝ slice OOF r).  
- Choose whichever ensemble had the higher overall OOF r as your final submission.

---

## 9. Notebook Structure  
- **Cell 1:** Imports & `Config` (paths, feature lists, hyper‐params).  
- **Cell 2:** `feature_engineering()`, `create_time_decay_weights()`, `get_model_slices()`.  
- **Cell 3:** `load_data()` → read raw columns, engineer, return train/test/sub.  
- **Cell 4:** `train_and_predict()` → CV + time-decay + slicing + OOF + test preds.  
- **Cell 5:** `ensemble_and_submit()` → build final predictions, write CSV.  
- **Cell 6:** `if __name__=="__main__"` → run load → train/predict → ensemble.

---

## 10. How to Run  
1. Attach the Kaggle dataset in your Notebook settings.  
2. Select **GPU** accelerator.  
3. Run all cells in order; the final submission (`submission.csv`) will appear in your working directory.  

---

That’s the full end-to-end pipeline—from raw minute-level data to a top‐percentile Kaggle submission—documented so you can revisit, tweak, or extend at any time.

