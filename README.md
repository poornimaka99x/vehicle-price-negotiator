# What is this car worth? Sri Lankan used-car price prediction

ML Lecture 7, Part 2 (Final Assignment). This project predicts the asking price of a used car in Sri Lanka from its make, model, age, engine, transmission, fuel, town and equipment. The models compared cover every ensemble technique: bagging (decision trees and Random Forest), boosting (XGBoost, LightGBM, CatBoost), voting and stacking. They are set against existing approaches, and explained with SHAP. A web front end turns the result into a fair-price check.

## Results (unseen cars, grouped 80/20 split)

| Model | Test MAPE | Test median APE | Test R² |
|---|---|---|---|
| Market-comparable median (how buyers price by hand) | 23.6 % | 4.0 % | 0.74 |
| Ridge regression (approach of earlier SL studies) | 16.3 % | 8.0 % | 0.84 |
| Bagging (decision trees) | 13.8 % | 5.4 % | 0.87 |
| Random Forest | 13.8 % | 5.8 % | 0.87 |
| XGBoost | 13.3 % | 6.3 % | 0.89 |
| LightGBM | 14.0 % | 6.4 % | 0.88 |
| CatBoost | 14.2 % | 6.9 % | 0.88 |
| Voting: equal-weight XGBoost + CatBoost + LightGBM | 13.1 % | 6.4 % | 0.89 |
| **Stack: XGBoost + CatBoost + LightGBM (final)** | **13.0 %** | **6.3 %** | **0.89** |

The full analysis, design, comparison with published work, interpretation and limitations are in [`REPORT.md`](REPORT.md) and [`vehicle_price.ipynb`](vehicle_price.ipynb).

## Data

The data is the [SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset) by Prasad Nirmal (Kaggle v3, Feb 2025, MIT licence): 9,788 listings scraped from ikman.lk and Riyasewana.com between Dec 2024 and Feb 2025. `data/vehicle_prices.csv` is the unchanged Kaggle file. Prices are in lakhs (1 lakh = Rs 100,000).

> The dataset's mileage column is exactly 11,000 km × age for every row, so it is not real data. The model and the app therefore do not use mileage.

## Files

| File | What it is |
|---|---|
| `vehicle_price.ipynb` | The full project: data analysis, model selection, hyperparameters, cross-validated comparison, test results, interpretation, export, single-sample inference |
| `REPORT.md` | The written report |
| `vehicle_model.py` | Feature engineering and model classes, shared by the notebook and the app |
| `app.py` | Streamlit front end |
| `model/price_model.joblib` | The trained final model (stack plus its 80 % interval) |
| `model/all_models.joblib` | All nine compared models, each with its own 80 % interval, for the app's model picker (~47 MB) |
| `model/listings.csv` | Training listings, used by the app for dropdowns and similar cars |
| `results/metrics.json` | Every number quoted in the report |
| `figures/` | Plots used in the report |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # macOS: LightGBM/XGBoost need `brew install libomp`
jupyter nbconvert --to notebook --execute --inplace vehicle_price.ipynb   # retrains everything (3–7 min)
streamlit run app.py                     # opens http://localhost:8501
```

Tested with Python 3.14. Results are deterministic (fixed seeds). Only the timing figures change between runs.

> `price_model.joblib` is a pickle, and loading a pickle can run code. Only load model files you trained yourself.

## Using the app

1. Pick the brand and model. Year, engine size and gear fill in with typical values for that model, and you can change them.
2. Set fuel, town, leasing status and equipment.
3. Optionally, enter the asking price you were quoted.

The app shows:
* the estimated market price, in lakhs and rupees;
* the range that 80 % of similar cars fall in;
* whether the quoted price is below, within or above that range;
* how reliable the estimate is for that model;
* a table of similar listings;
* **why this price:** a SHAP waterfall showing how each feature moved the estimate.

The **Prediction method** dropdown at the top switches between all nine models (the stack is the default), so you can see how much bagging, boosting, voting and stacking disagree on one car.

## MLOps

`.github/workflows/mlops.yml` runs on GitHub Actions:

| Job | When | What |
|---|---|---|
| `test` | every PR and push | `pytest`: feature engineering, the model prices the held-out car inside its 80 % range, quality bar |
| `train` | push to `main`, manual run | re-executes `vehicle_price.ipynb`, then the same tests act as a gate (test MAPE ≤ 14 %, R² ≥ 0.85) |
| `deploy` | after `train` | registers `vehicle-price-model` in Azure ML (tagged with the commit and its metrics), builds the image, pushes it to ACR, updates the Container App and checks `/_stcore/health` |

Azure (one resource group, `centralindia`): Container Registry (Basic), Azure ML workspace (model registry), and a Container App that scales to zero when nobody is using it. GitHub signs in to Azure with OIDC, so no Azure secret is stored in GitHub.

```bash
./infra/provision.sh    # one-off: resources, GitHub OIDC identity, repo variables, budget alert
./infra/teardown.sh     # delete everything when grading is done
```

Run the tests locally with `python -m pytest`, and the container with `docker build -t vehicle-price . && docker run -p 8501:8501 vehicle-price`.
