"""Sri Lanka used-car price checker. Run with:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from vehicle_model import PriceModel, add_features, explain, FEATURES, REFERENCE_YEAR

HERE = Path(__file__).parent
st.set_page_config(page_title="What is this car worth?", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model():
    return PriceModel.load(HERE / "model" / "price_model.joblib")


@st.cache_resource
def load_all_models():
    """Every compared model (written by notebook section 7), or just the final one if that file is absent."""
    path = HERE / "model" / "all_models.joblib"
    # joblib uses pickle, and loading a pickle can run code: load only models you trained yourself.
    return joblib.load(path) if path.exists() else {}


@st.cache_data
def load_listings():
    return pd.read_csv(HERE / "model" / "listings.csv")


@st.cache_data
def load_error_by_familiarity(final_name):
    """Typical (median) % error on the test set, by how often the car's model appeared in training."""
    m = json.loads((HERE / "results" / "metrics.json").read_text())
    rows = m["segments_mdape"]["model seen in train"]
    return {r["model seen in train"]: r[final_name] for r in rows}


def lakhs(x):
    return f"{x:,.1f} lakhs"


def rupees(x):
    return f"Rs {x * 100_000:,.0f}"


final_model, listings = load_model(), load_listings()
price_models = {final_model.name: final_model, **{n: m for n, m in load_all_models().items() if n != final_model.name}}

st.title("🚗 What is this car worth?")
st.caption("Estimate the market asking price of a used car in Sri Lanka, and check whether a listed price is fair. "
           "Based on 7,800+ ikman.lk and Riyasewana.com listings, Dec 2024 – Feb 2025.")

form_col, result_col = st.columns([1, 1.25], gap="large")

with form_col:
    if len(price_models) > 1:
        label = lambda n: (f"{n} (recommended)" if n == final_model.name
                           else f"{n} · {price_models[n].metrics.get('technique', '')}")
        model = price_models[st.selectbox("Prediction method", list(price_models), format_func=label,
                                    help="Compare how bagging, boosting, voting and stacking price the same car. "
                                         "The stack had the lowest error on unseen cars.")]
    else:
        model = final_model
    typical_error = model.metrics.get("mdape_by_familiarity") or load_error_by_familiarity(model.name)

    st.subheader("The car")
    brands = listings.brand.value_counts().index.tolist()
    brand = st.selectbox("Brand", brands, index=brands.index("TOYOTA") if "TOYOTA" in brands else 0)

    by_brand = listings[listings.brand == brand]
    models = by_brand.model.value_counts().index.tolist()
    choice = st.selectbox("Model", models + ["Other (type it)"])
    model_name = st.text_input("Model name", placeholder="e.g. AXIO").strip().upper() if choice == "Other (type it)" else choice
    same_model = by_brand[by_brand.model == model_name]
    ref = same_model if len(same_model) else by_brand    # defaults come from the same model when known

    c1, c2 = st.columns(2)
    yom = c1.number_input("Year of manufacture", 1950, REFERENCE_YEAR, int(ref.yom.median()), step=1)
    engine_cc = c2.number_input("Engine capacity (cc)", 500, 6000, int(ref.engine_cc.median()), step=50)
    gear = c1.radio("Transmission", ["Automatic", "Manual"],
                    index=0 if ref.gear.mode().iloc[0] == "Automatic" else 1, horizontal=True)
    fuels = ["Petrol", "Hybrid", "Diesel", "Electric"]
    fuel = c2.selectbox("Fuel", fuels, index=fuels.index(ref.fuel.mode().iloc[0]))
    towns = sorted(listings.town.unique())
    town = c1.selectbox("Town", towns, index=towns.index("Colombo") if "Colombo" in towns else 0)
    leasing = c2.radio("Leasing", ["No Leasing", "Ongoing Lease"], horizontal=True)
    condition = c1.radio("Condition", ["USED", "NEW"], horizontal=True, format_func=str.title)

    st.markdown("**Equipment**")
    e1, e2, e3, e4 = st.columns(4)
    ac = e1.checkbox("Air conditioning", True)
    ps = e2.checkbox("Power steering", True)
    pm = e3.checkbox("Power mirrors", True)
    pw = e4.checkbox("Power windows", True)

    asking = st.number_input("Asking price you were quoted (lakhs, optional)", 0.0, 5000.0, 0.0, step=0.5,
                             help="1 lakh = Rs 100,000. Leave at 0 to skip the fair-price check.")

with result_col:
    if not model_name:
        st.info("Type the model name to get an estimate.")
        st.stop()

    avail = lambda b: "Available" if b else "Not_Available"
    car = pd.DataFrame([{
        "brand": brand, "model": model_name, "yom": yom, "engine_cc": engine_cc, "gear": gear, "fuel": fuel,
        "town": town, "leasing": leasing, "condition": condition, "air_condition": avail(ac),
        "power_steering": avail(ps), "power_mirror": avail(pm), "power_window": avail(pw),
    }])
    X = add_features(car)
    price = float(model.predict(X)[0])
    lo, hi = (float(v[0]) for v in model.interval(X))

    st.subheader("Estimated market price")
    st.metric("Most likely asking price", lakhs(price), help=rupees(price))
    st.write(f"**{rupees(price)}**. 80 % of similar cars are listed between "
             f"**{lakhs(lo)}** and **{lakhs(hi)}** ({rupees(lo)} – {rupees(hi)}).")

    n_known = int((listings.model.str.upper() == str(model_name).strip().upper()).sum())
    band = "never" if n_known == 0 else "1–4 times" if n_known < 5 else "5–19 times" if n_known < 20 else "20+ times"
    note = {"never": "This model is not in the data, so the estimate leans on the brand and the car's age.",
            "1–4 times": "Only a few listings of this model exist, so treat the estimate as a rough guide.",
            "5–19 times": "A moderate number of listings of this model exist.",
            "20+ times": "This is a common model, so estimates are most reliable."}[band]
    st.caption(f"{n_known} listing(s) of this model in the data. Typical error for such cars: "
               f"about {typical_error[band]:.0f} %. {note}")

    if asking > 0:
        st.subheader("Is the asking price fair?")
        diff = (asking - price) / price * 100
        if asking > hi:
            st.error(f"**Above the usual range.** {lakhs(asking)} is {diff:+.0f} % vs the estimate. "
                     "Negotiate, or ask what justifies the premium (low mileage, condition, extras).")
        elif asking < lo:
            st.warning(f"**Below the usual range.** {lakhs(asking)} is {diff:+.0f} % vs the estimate. "
                       "It could be a good deal, but check the car's condition, papers and lease status carefully.")
        else:
            st.success(f"**Within the usual range.** {lakhs(asking)} is {diff:+.0f} % vs the estimate.")

    st.subheader("Why this price?")
    explained = explain(model.model, X)
    if explained is None:
        st.info(f"{model.name} has no per-feature breakdown: it is a lookup or a linear baseline. "
                "Pick a bagging, boosting, voting or stacking method to see one.")
    else:
        base, contrib = explained
        row = contrib.iloc[0]
        fig = plt.figure()
        shap.plots.waterfall(shap.Explanation(row.values, base_values=base, data=X.iloc[0].astype(str).values,
                                              feature_names=FEATURES), max_display=8, show=False)
        st.pyplot(fig, clear_figure=True)
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(3)
        effects = "; ".join(f"**{k.replace('_', ' ')}** {(np.exp(v) - 1) * 100:+.0f} %" for k, v in top.items())
        st.caption(f"SHAP values for {model.name}, in log price. The chart starts from the average car "
                   f"({lakhs(np.exp(base))}) and adds each feature's effect until it reaches this estimate. "
                   f"Biggest effects vs the average car: {effects}.")

    st.subheader("Similar listings")
    similar = listings[(listings.brand == brand) & (listings.model == model_name) & listings.yom.between(yom - 2, yom + 2)]
    if similar.empty:
        st.write("No listings of this model within ±2 years.")
    else:
        st.write(f"{len(similar)} listing(s) of a {brand.title()} {str(model_name).title()} from {yom - 2}–{yom + 2}. "
                 f"Median: **{lakhs(similar.price.median())}**.")
        show = similar.sort_values("yom", ascending=False)[["yom", "engine_cc", "gear", "fuel", "town", "price"]].head(15)
        st.dataframe(show.rename(columns={"yom": "Year", "engine_cc": "cc", "gear": "Gear", "fuel": "Fuel",
                                          "town": "Town", "price": "Price (lakhs)"}),
                     hide_index=True, width="stretch")

with st.expander("About this estimate"):
    mt = model.metrics
    st.markdown(f"""
* **Model:** {model.name}, trained on log price. On cars it had never seen, the typical error is {mt['test_mdape']:.1f} % (median) and {mt['test_mape']:.1f} % (mean), with R² {mt['test_r2']:.2f}. {mt['interval_coverage'] * 100:.0f} % of true prices fell inside the 80 % range.
* **Asking prices, not sale prices.** Listings show what sellers ask, and final prices are usually negotiated lower.
* **Mileage is not used.** In the source data, mileage is exactly 11,000 km × age for every car, so it adds nothing beyond the year. Real mileage and condition will move a real price.
* **Snapshot before imports resumed.** The data covers Dec 2024 – Feb 2025, just as Sri Lanka lifted its 2020–2025 vehicle import ban. Prices may have changed since.
""")
