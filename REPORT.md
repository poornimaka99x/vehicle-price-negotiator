# Predicting Used-Car Asking Prices in Sri Lanka with Stacked Gradient-Boosted Ensembles

*ML Lecture 7, Part 2: Final Assignment report. Every figure and number here comes from the executed notebook `vehicle_price.ipynb` (recorded in `results/metrics.json`).*

## Abstract

From 2020 to January 2025, Sri Lanka suspended private vehicle imports. Used-car prices rose far above international values, and buyers had little way to judge whether an asking price was fair. Using 9,770 de-duplicated listings from ikman.lk and Riyasewana.com (Dec 2024 – Feb 2025), this work predicts a car's asking price from its make, model, age, engine, transmission, fuel, town and equipment. The data analysis found that the dataset's mileage column is synthetic: it is exactly 11,000 km × age for every row, so it was removed. It also found 613 groups of listings with identical attributes (mostly reposts of the same car), so a split grouped by these was used. Seven models were compared under 5-fold grouped cross-validation. A stack of XGBoost, CatBoost and LightGBM was best and beat every alternative in all five folds. On unseen cars it achieves 13.0 % mean and 6.3 % median absolute percentage error (R² 0.89), against 16.3 % for the linear regression used in earlier Sri Lankan studies and 23.6 % for pricing from comparable listings. Plain bagging and equal-weight voting were added to make each ensemble technique explicit. Boosting beat bagging, and learned stacking weights added 0.1 points over voting. The stack is explained with exact SHAP values: the stack is a weighted sum of tree models, so its SHAP values are the same weighted sum of theirs. The model ships in a Streamlit app that gives an estimate, a calibrated 80 % range (79 % test coverage), a fair-price verdict, a per-car SHAP explanation, and a picker to compare every model.

## 1. Problem

**Task:** regression. Given a used car's attributes, predict its asking price in lakhs (1 lakh = Rs 100,000).

**Local relevance.** Sri Lanka suspended vehicle imports in 2020 to protect foreign reserves and lifted the suspension on 1 February 2025 (Gazette 2421/44). With no new supply for five years, used cars became the only option for most buyers. Local studies describe buyers being exploited by unrealistic dealer prices (Jayadeera & Jayamanne, 2019; Basnayaka, 2023). A data-driven price estimate helps buyers negotiate, sellers price realistically, and leasing companies value collateral.

## 2. Data

* **Source:** Prasad Nirmal, *SriLankan Vehicle (Car) Price Dataset*, Kaggle, version 3 (9 Feb 2025), MIT licence. It contains listings scraped from ikman.lk and Riyasewana.com. The file in `data/` is byte-identical to the Kaggle release (SHA-256 `4208d267…1ab`).
* **Size:** 9,788 rows and 16 columns: brand, model, year of manufacture, engine (cc), gear, fuel, mileage, town, date, leasing, condition, four equipment flags, and price. The listing dates run from 3 Dec 2024 to 5 Feb 2025.

## 3. Data analysis

| Check | Finding | Decision |
|---|---|---|
| Nulls | None | No imputation needed. The app validates its inputs |
| Duplicates | 18 exact duplicates | Removed (9,770 rows) |
| Reposts | 613 groups of listings with identical attributes (1,471 rows), mostly the same car reposted | Grouped split, so a reposted car is never in both train and test |
| **Mileage** | **Exactly 11,000 km × age in 100 % of rows** (`figures/mileage_is_synthetic.png`) | **Dropped.** It is a copy of age, not a measurement |
| Electric cars | 69 rows with made-up engine sizes (e.g. Tesla at 573 cc) | Kept, flagged as low-confidence |
| Target | Skew 3.46. log(price) skew −0.29. Median 43 lakhs, range 2.65–790 (`figures/target_distribution.png`) | Model log(price) |
| Categories | 50 brands, **1,555 models** (1,260 with fewer than 5 listings, 916 seen once), 107 towns | Categorical-aware learners. Rare levels pooled in one-hot encoding |
| Drivers | Age has Spearman −0.66 with log price, with a curved effect. Automatic median 61.5 vs manual 23.0 lakhs. Hybrid 73.8 vs petrol 38.5 (`figures/relationships.png`) | Non-linear tree ensembles. Interpret fuel conditionally |
| Time | Prices within the same make, model and year differ by < ±0.5 % across Dec, Jan and Feb | No time-series model. `date` dropped |

## 4. Methodology

**Features.** Age (2025 − year of manufacture), engine cc, and eleven categoricals: brand, model, gear, fuel, town, leasing, condition and four equipment flags. Target: log(price).

**Models.** Baselines representing existing approaches:
* A *market-comparable median*: the median price of the same make, model and year, backing off to make+model, then make. This is what buyers do by hand.
* Ridge regression on one-hot features, standing in for the multiple linear regression of earlier Sri Lankan studies.
* Random Forest and XGBoost on one-hot features (the Part 1 models).

Chosen candidates:
* **CatBoost** (ordered target statistics for high-cardinality categories).
* **LightGBM** (native categorical splits).
* A **stack** of XGBoost, CatBoost and LightGBM with a positive linear meta-learner trained on out-of-fold predictions.

Two further ensembles make each technique explicit (§5.1). Both were added after model selection, so they do not change the choice:
* **Plain bagging:** `BaggingRegressor`, 100 full-depth decision trees on bootstrap samples, with every feature available at every split. Random Forest is the same plus a random feature subset per split.
* **Voting:** the stack's three boosters averaged with fixed equal weights, which is what `VotingRegressor` computes. The only difference from the stack is learned vs fixed weights.

**Explainability.** SHAP values are computed exactly with TreeSHAP for every tree model (CatBoost and LightGBM natively; XGBoost natively on one-hot columns summed back to their feature; Random Forest and bagging with `shap.TreeExplainer`). Stacking and voting are weighted sums of base models and SHAP is additive, so an ensemble's SHAP is the same weighted sum. The notebook asserts that base value + Σ SHAP reproduces every prediction to within 10⁻⁴.

ARIMA hybrids were not used because the data is a two-month cross-section with no drift. Deep learning is out of scope per the brief.

**Hyperparameters** were designed rather than left at defaults (notebook §2.2). The depth check reused three of the five CV folds, so the CatBoost and LightGBM CV scores are slightly optimistic. The untouched test set confirms the ranking. Two key choices were then checked by grouped cross-validation on the training data:
* CatBoost depth: the designed depth 8 lost to depth 6 (validation RMSE 0.2130 vs 0.2147), so depth 6 with 2,900 trees was used. Depth 6 equals CatBoost's default but was chosen by evidence. CatBoost's other settings are non-default.
* LightGBM leaves: 31 and 48 were tied, so 48 leaves with 800 trees were used.

**Evaluation protocol.**
* One listing (a 2012 Toyota Aqua) is held out for single-sample inference.
* The remaining 9,769 rows are split 80/20 **by car**: 7,817 train and 1,952 test. 217 test rows are of models never seen in training.
* All model comparison and selection uses 5-fold `GroupKFold` on the training part. The test set is used once, for reporting.
* Metrics: MAE and RMSE (lakhs), MAPE and median APE (%), R² on price and on log price.

**Selection rule.** Prefer the simpler model unless the better-scoring one wins reliably in a paired fold-by-fold comparison (mean gap > one standard error of the per-fold gaps). *Disclosure:* the rule was first written as "within one fold SD of the best". After seeing the CV results, it was replaced, because fold SD mostly measures fold difficulty, which affects all models equally. Under the original rule, XGBoost would have been selected. Both are reported in the notebook (§3.1).

## 5. Results

| Model | CV MAPE % (± SD) | Test MAPE % | Test MdAPE % | Test MAE (lakhs) | Test RMSE (lakhs) | Test R² (price) |
|---|---|---|---|---|---|---|
| Market-comparable median | 23.27 ± 1.29 | 23.60 | **4.01** | 7.63 | 21.05 | 0.743 |
| Ridge (one-hot) | 15.97 ± 0.66 | 16.33 | 8.02 | 7.07 | 16.42 | 0.844 |
| Random Forest | 13.35 ± 0.56 | 13.76 | 5.79 | 5.82 | 14.82 | 0.873 |
| XGBoost | 12.72 ± 0.55 | 13.28 | 6.29 | 5.72 | 13.89 | 0.888 |
| LightGBM | 13.47 ± 0.46 | 14.01 | 6.41 | 6.07 | 14.61 | 0.876 |
| CatBoost | 13.42 ± 0.59 | 14.19 | 6.93 | 6.17 | 14.49 | 0.878 |
| **Stack (XGB + CatBoost + LGBM)** | **12.39 ± 0.53** | **13.04** | 6.28 | **5.62** | **13.67** | **0.892** |

The stack beats every other model in all five CV folds. Its mean gap to XGBoost is 0.33 points, with a standard error of 0.08. Meta-learner weights: XGBoost 0.50, CatBoost 0.38, LightGBM 0.12. See `figures/model_comparison.png`.

### 5.1 By ensemble technique

| Model | Technique | CV MAPE % (± SD) | Test MAPE % | Test MdAPE % | Test R² (price) |
|---|---|---|---|---|---|
| Bagging (decision trees) | bagging | 13.32 ± 0.82 | 13.82 | **5.37** | 0.872 |
| Random Forest | bagging + feature sampling | 13.35 ± 0.56 | 13.76 | 5.79 | 0.873 |
| XGBoost | boosting | 12.72 ± 0.55 | 13.28 | 6.29 | 0.888 |
| LightGBM | boosting | 13.47 ± 0.46 | 14.01 | 6.41 | 0.876 |
| CatBoost | boosting | 13.42 ± 0.59 | 14.19 | 6.93 | 0.878 |
| Voting (XGB + CatBoost + LGBM) | voting, equal weights | 12.48 ± 0.52 | 13.13 | 6.36 | 0.891 |
| **Stack (XGB + CatBoost + LGBM)** | stacking, learned weights | **12.39 ± 0.53** | **13.04** | 6.28 | **0.892** |

* **Bagging vs Random Forest:** the mean error is the same (13.3 % CV). Random feature sampling did not lower the error here. It made the error steadier across folds (SD 0.56 vs 0.82). The bagging models have the lowest median error of the ML models: averaging deep trees suits common cars.
* **Boosting vs bagging:** XGBoost is the best single model (12.7 % CV vs 13.3 %). Sequential shallow trees handle the long tail of rare models better.
* **Voting vs stacking:** voting the same three boosters with equal weights already beats every single model. Learning the weights adds about 0.1 points. Most of the gain comes from combining models whose errors differ. See `figures/ensemble_comparison.png`.

**Single-sample inference.** The held-out 2012 Toyota Aqua (hybrid, automatic, Boralesgamuwa) was listed at **70.95 lakhs** on 5 Feb 2025. The model predicts **65.68 lakhs** (−7.4 %), with an 80 % range of **56.2 – 77.7 lakhs**, which contains the true price.

## 6. Comparison with existing approaches

| Study | Data | Best reported | Our result on comparable terms |
|---|---|---|---|
| Jayadeera & Jayamanne (2019) | ~200 ikman Corolla listings | R² 0.83–0.89 (MLR, RF, DT) | All makes and models, unseen cars: R² 0.89 |
| Basnayaka (2023) | Kaggle SL used cars | "≈90 % accuracy" for tree ensembles. MLR lower | Same conclusion. Ensembles beat linear by 3.3 MAPE points |
| *sri-lanka-vehicle-price-ml* (2026), same Kaggle data | Random split, mileage used | RF R² 0.812, MAE 5.89 | R² 0.892, MAE 5.62, on a stricter grouped split and without synthetic mileage |
| *Sri_Lankan-Vehicle-Price-Predictor* (2026), apparently similar data | Random split | MAPE ≈ 13.5 %, R² ≈ 0.92, 80 % interval coverage ≈ 74 % | MAPE 13.0 %, coverage 79.4 %. Its R² is higher than ours (0.892), on a random split where reposts can leak, so not like-for-like |
| Jamalamadaka (2025), India | 8,128 CarDekho listings | LightGBM R² 0.965, linear 0.633, best neural net 0.916 | Same ordering: boosted > linear, and boosted > neural nets on tabular data |

Results across different datasets and splits are not like-for-like, and they are used here to position the approach, not to rank it.

## 7. Justification

1. **Lowest error, reliably.** The stack is best in CV and on test, and it wins in all five folds.
2. **It suits the data.** Its three ways of encoding the 1,555-value `model` field make different errors, which the stack combines, and every base model receives a non-zero weight.
3. **Large gains where it matters.** It cuts mean error by 10.6 points against comparable-listing pricing, and on never-seen models median error falls from 46.3 % to 14.3 %.
4. **Calibrated uncertainty.** The 80 % range covers 79.4 % of test prices.
5. **Cheap.** About 12 s to train and milliseconds to predict.

**Trade-off:** three models instead of one, for about 0.3 points over XGBoost alone. The gain is small but consistent. If maintenance cost dominated, XGBoost alone would be the fallback.

## 8. Interpretation

* **Importance** (permutation on the final model, MAPE increase when each input is shuffled): age 47.3, brand 18.3, gear 7.7, engine 7.0, model 6.8. Equipment and town are each under 1 point (`figures/permutation_importance.png`).
* **Effects** (CatBoost SHAP, relative to the average car; `figures/shap_dependence.png`):
  * Age: 0–5 years **+76 %**, 11–15 years **+26 %**, 21–30 years **−23 %**. In log terms this is close to a straight line, about 4–5 % per year from roughly 5 to 45 years, with an upturn for vintage cars beyond that.
  * Gear: automatic **+9 %**, manual **−12 %**.
  * Ongoing lease: **−9 %**.
  * Hybrid: **≈ 0 %** once model and age are known. The raw hybrid premium is confounding: hybrids are newer, specific models.
* **SHAP of the final stack** (the model the app uses; `figures/shap_stack_summary.png`). Mean |SHAP| in log units: age 0.33, gear 0.19, brand 0.17, engine 0.08, model 0.06, everything else ≤ 0.04. Gear ranks above brand in the stack, but below it in CatBoost alone (0.10 vs 0.20). The component that explains the most is not necessarily the model a user sees, so the stack itself is explained. A larger engine raises the price once age and brand are known. Fuel stays close to 0.
* **One car** (`figures/shap_waterfall_inference.png`): for the held-out Aqua, age (13 years) adds about +20 % vs the average car, Toyota +17 %, automatic +14 %, and the Aqua model −6 % (relative to the Toyota average).
* **Where it works:**
  * By how often the car's model appeared in training: median error 4.3 % when seen 20+ times, 8.0 % at 5–19 times, 12.9 % at 1–4 times, and 14.3 % when never seen.
  * Hardest segments: the cheapest quartile (11.6 %) and cars over 20 years old (10.8 %).
  * By brand: Suzuki (3.8 %) and Toyota (5.9 %) are priced well. Mazda (18.5 %) and Mitsubishi (15.8 %) are not.
* **Comparables vs ML:** for common cars, the median of identical listings is more accurate (2.5 % median error) because sellers copy each other's prices. The model's value is in the long tail, and the app therefore shows both.

## 9. Front end

`app.py` is a Streamlit app. Its features:
* **Inputs:** a brand → model dropdown, with a free-text option for unlisted models. Year, engine and gear default to typical values for the chosen model.
* **Estimate:** the price in lakhs and rupees, with the 80 % range.
* **Fair-price check:** enter an asking price to see whether it falls below, within or above the usual range.
* **Confidence note:** based on how many listings of that model exist.
* **Similar listings:** a table of the same model within ±2 years.
* **Prediction method:** a picker covering all nine models: the bagging, boosting, voting and stacking ensembles, plus the two baselines. The stack is the default. Each model has its own 80 % range from its own out-of-fold residuals, and its own error figures.
* **Why this price?** A SHAP waterfall for the entered car and the chosen model, with the three biggest effects in %. The lookup and linear baselines show a note instead.

Mileage is deliberately not asked for (see §3). Run it with `streamlit run app.py`.

## 10. Limitations and future work

1. The data shows **asking prices**, not sale prices.
2. There is **no real mileage or condition data**, which is the main cause of errors on old and cheap cars.
3. It is a **snapshot of a closed market**, taken just before imports resumed. Retrain on 2025–26 listings before trusting it on current prices.
4. There is a **long tail of rare models**.
5. **Electric vehicles** have unreliable engine sizes.
6. The 80 % range has **one width for all cars**. Quantile models would make it segment-aware.
7. The data was scraped. The Kaggle release is MIT-licensed, but the marketplaces' terms were not checked.

Future work:
* A comparables + stack hybrid.
* Quantile intervals.
* Measuring the price effect of import liberalisation.

## 11. Conclusion

A stacked ensemble of XGBoost, CatBoost and LightGBM predicts Sri Lankan used-car asking prices with 13.0 % mean and 6.3 % median error on cars it has never seen. It beats linear regression, single tree ensembles and comparable-listing pricing, and it comes with a calibrated range and a usable front end. Analysing the data before modelling paid off: it exposed synthetic mileage and reposted cars that would otherwise have inflated the reported accuracy.

## References

* Basnayaka, D. (2023). *Prediction of Market Price for Used Cars in Sri Lanka Based on Machine Learning Approaches.* MSc dissertation, Informatics Institute of Technology. http://dlib.iit.ac.lk/xmlui/handle/123456789/1634
* Jayadeera, T. P., & Jayamanne, D. J. (2019). Fair Price Prediction System for Used Cars in Sri Lanka Using Machine Learning and Robotic Process Automation. *International Conference on Business Innovation (ICOBI)*, NSBM Green University, 248–253. https://nspace.nsbm.ac.lk/handle/123456789/99
* Jamalamadaka, A. (2025). Practical Recommendations for Machine Learning-Based Tabular Regression Concerning Car Price Prediction. *National High School Journal of Science.* https://nhsjs.com/2025/practical-recommendations-for-machine-learning-based-tabular-regression-concerning-car-price-prediction/
* Nirmal, P. (2025). *SriLankan Vehicle (Car) Price Dataset*, v3. Kaggle. https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset
* Public repository: *sri-lanka-vehicle-price-ml* (2026). https://github.com/yulepaulusha/sri-lanka-vehicle-price-ml
* Public repository: *Sri_Lankan-Vehicle-Price-Predictor* (2026). https://github.com/pamindu-w/Sri_Lankan-Vehicle-Price-Predictor
* Ada Derana (2025). Gazette issued lifting suspension on vehicle imports to Sri Lanka. https://adaderana.lk/news/105432
* Ada Derana (2025). First batch of imported personal vehicles arrives at Colombo Port after 5-year hiatus. https://adaderana.lk/news/106112
