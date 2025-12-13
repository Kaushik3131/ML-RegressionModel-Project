import streamlit as st
import pandas as pd
import requests
import plotly.express as px
# import boto3
import os
from pathlib import Path

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

# ============================
# Data loading
# ============================
HOLDOUT_URL = API_URL.replace("/predict", "/holdout_data")


@st.cache_data(show_spinner=False)
def load_data():
    resp = requests.get("http://fastapi-api:8000/holdout_data", timeout=60)
    resp.raise_for_status()
    data = resp.json()

    fe = pd.DataFrame(data["features"])
    meta = pd.DataFrame(data["meta"])

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = pd.to_datetime(meta["date"])
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp


with st.spinner("Loading housing data…"):
    fe_df, disp_df = load_data()

# ============================
# UI
# ============================
st.title("🏠 Housing Price Prediction — Holdout Explorer")

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(
            f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")

        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()

            out = resp.json()

            if "predictions" not in out:
                st.warning(out.get("message", "Model not ready yet"))
                st.stop()

            preds = out["predictions"]

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")

            if len(preds) != len(view):
                st.error("Prediction length mismatch. Try again in a moment.")
                st.stop()

            view["prediction"] = preds

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"])
                    ** 2).mean() ** 0.5
            avg_pct_error = (
                (view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price",
                      "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(
                    preds_all, index=yearly_data.index).astype(float)

            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (
                    disp_df["region"] == region)].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(
                    orient="records")

                resp_region = requests.post(
                    API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(
                    preds_region, index=yearly_data.index).astype(float)

            # Aggregate by month
            monthly_avg = yearly_data.groupby(
                "month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(
                lambda m: "Selected" if m == month else "Other")

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}"
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")
