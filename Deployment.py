import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

trainval_path = "Model/XGBoost/xgb__trainval_bundle.joblib"
full_path = "Model/XGBoost/xgb_full_bundle.joblib"

FEATURES_EXP_C = ["Close", "MA10", "MA20", "MA50", "RSI"]

FINAL_COL_ORDER = [
    "Date","Close","Open","High","Low","Volume","Return","MA10","MA20","MA50","RSI",
    "Volatility","Dev","MA10_Lag1","MA20_Lag1","MA50_Lag1","RSI_Lag1",
    "NextDate","Actual_NextClose","Pred_NextClose","Predicted_Close","Split"
]

# =========================
# Page config
# =========================
st.set_page_config(page_title="Stock Forecasting System", layout="wide")

# =========================
# original helpers
# =========================
def convert_volume(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").strip()
    if x.endswith("M"):
        return float(x[:-1]) * 1_000_000
    if x.endswith("K"):
        return float(x[:-1]) * 1_000
    return float(x)

@st.cache_data
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Price": "Close", "Vol.": "Volume"})

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = df["Volume"].apply(convert_volume)
    if "Change %" in df.columns:
        df = df.drop(columns=["Change %"])

    # Feature engineering (same as notebook)
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility"] = df["Return"].rolling(20).std()
    df["Dev"] = df["Close"] - df["MA20"]

    df = df.dropna().reset_index(drop=True)
    return df


# =========================
# Notebook functions
# =========================

def is_last_5_trading_days_consecutive(df, date_col="Date"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    dates = (
        d[date_col]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(dates) < 5:
        return False

    last5 = dates.iloc[-5:]
    last_date = last5.iloc[-1]

    expected = pd.bdate_range(end=last_date, periods=5)
    return last5.reset_index(drop=True).equals(pd.Series(expected))

def get_next_forecast_date(df, date_col="Date"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index(drop=True)

    last_date = d[date_col].iloc[-1]
    last_wd = last_date.weekday() 

    consecutive_5 = is_last_5_trading_days_consecutive(d, date_col=date_col)

    if last_wd == 4 and consecutive_5:
        next_date = last_date + pd.Timedelta(days=3)
    else:
        next_date = last_date + pd.Timedelta(days=1)
        if next_date.weekday() == 5:     
            next_date += pd.Timedelta(days=2)
        elif next_date.weekday() == 6:   
            next_date += pd.Timedelta(days=1)

    return next_date

def inject_fullmodel_forecast_to_df_step4_v2(
    df_step4,
    df_raw,
    feature_cols,
    model_full,
    forecast_date,
    split_label="Forecast"
):
    df_step4_out = df_step4.copy()
    df_step4_out["Date"] = pd.to_datetime(df_step4_out["Date"])
    df_step4_out = df_step4_out.sort_values("Date").reset_index(drop=True)

    forecast_date = pd.to_datetime(forecast_date)

    # use last row from step4 as the latest known day
    last_row = df_step4_out.iloc[-1].copy()
    last_date = pd.to_datetime(last_row["Date"])

    x_latest = last_row[feature_cols].values.astype(float).reshape(1, -1)
    pred_next_close = float(model_full.predict(x_latest)[0])

    for col in ["Pred_NextClose", "Predicted_Close", "Split"]:
        if col not in df_step4_out.columns:
            df_step4_out[col] = np.nan

    df_step4_out.loc[df_step4_out["Date"] == last_date, "Pred_NextClose"] = pred_next_close

    last_step4_row = df_step4_out.loc[df_step4_out["Date"] == last_date].iloc[-1]

    if not (df_step4_out["Date"] == forecast_date).any():
        new_row = {c: np.nan for c in df_step4_out.columns}
        new_row["Date"] = forecast_date

        lag_map = {
            "MA10_Lag1": "MA10",
            "MA20_Lag1": "MA20",
            "MA50_Lag1": "MA50",
            "RSI_Lag1": "RSI"
        }
        for lag_col, src_col in lag_map.items():
            if lag_col in df_step4_out.columns and src_col in df_step4_out.columns:
                new_row[lag_col] = last_step4_row[src_col]

        if "NextDate" in df_step4_out.columns:
            new_row["NextDate"] = pd.NaT
        if "Actual_NextClose" in df_step4_out.columns:
            new_row["Actual_NextClose"] = np.nan
        if "Pred_NextClose" in df_step4_out.columns:
            new_row["Pred_NextClose"] = np.nan

        new_row["Predicted_Close"] = pred_next_close
        new_row["Split"] = split_label

        df_step4_out = pd.concat([df_step4_out, pd.DataFrame([new_row])], ignore_index=True)

    else:
        df_step4_out.loc[df_step4_out["Date"] == forecast_date, "Predicted_Close"] = pred_next_close
        df_step4_out.loc[df_step4_out["Date"] == forecast_date, "Split"] = split_label

        nan_cols = ["Close","Open","High","Low","Volume","Return","MA10","MA20","MA50","RSI","Volatility","Dev"]
        for c in nan_cols:
            if c in df_step4_out.columns:
                df_step4_out.loc[df_step4_out["Date"] == forecast_date, c] = np.nan

        lag_map = {"MA10_Lag1":"MA10","MA20_Lag1":"MA20","MA50_Lag1":"MA50","RSI_Lag1":"RSI"}
        for lag_col, src_col in lag_map.items():
            if lag_col in df_step4_out.columns and src_col in df_step4_out.columns:
                df_step4_out.loc[df_step4_out["Date"] == forecast_date, lag_col] = last_step4_row[src_col]

    df_step4_out = df_step4_out.sort_values("Date").reset_index(drop=True)
    return df_step4_out, pred_next_close

def run_iforest_on_test_and_forecast(
    df_step4_updated,
    n_roll=10,
    contamination=0.08,
    n_estimators=500,
    random_state=42,
    date_col="Date",
    split_col="Split"
):
    df_ad = df_step4_updated.copy()
    df_ad[date_col] = pd.to_datetime(df_ad[date_col])
    df_ad = df_ad.sort_values(date_col).reset_index(drop=True)

    df_ad["Close_Lag1"] = df_ad["Close"].shift(1)

    df_ad["Pred_Return"] = (df_ad["Predicted_Close"] - df_ad["Close_Lag1"]) / df_ad["Close_Lag1"]

    df_ad["Roll_Mean"] = df_ad["Pred_Return"].rolling(window=n_roll, min_periods=n_roll).mean()
    df_ad["Roll_Std"] = df_ad["Pred_Return"].rolling(window=n_roll, min_periods=n_roll).std(ddof=0)
    df_ad["Z_Return"] = (df_ad["Pred_Return"] - df_ad["Roll_Mean"]) / df_ad["Roll_Std"]

    split_lower = df_ad[split_col].astype(str).str.lower()
    mask_fit = split_lower.isin(["validation", "testing"])
    mask_apply = split_lower.isin(["testing", "forecast"])

    if_features = ["Pred_Return", "Z_Return"]

    df_fit = df_ad.loc[mask_fit].dropna(subset=if_features).copy()
    if len(df_fit) == 0:
        raise ValueError("No rows available to fit Isolation Forest. Check Pred_Return/Z_Return NaNs or Split labels.")

    X_fit = df_fit[if_features].values

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    iso.fit(X_fit)

    df_apply = df_ad.loc[mask_apply].dropna(subset=if_features).copy()
    if len(df_apply) == 0:
        raise ValueError("No rows available to apply Isolation Forest. Check Pred_Return/Z_Return NaNs.")

    X_apply = df_apply[if_features].values
    df_apply["IF_Anomaly"] = (iso.predict(X_apply) == -1).astype(int)
    df_apply["IF_Score"] = iso.decision_function(X_apply)

    df_ad["IF_Anomaly"] = np.nan
    df_ad["IF_Score"] = np.nan
    df_ad.loc[df_apply.index, "IF_Anomaly"] = df_apply["IF_Anomaly"].values
    df_ad.loc[df_apply.index, "IF_Score"] = df_apply["IF_Score"].values

    return df_ad, iso

def build_step5_recommendation(df_TA: pd.DataFrame) -> pd.DataFrame:
    df_TA = df_TA.copy()
    df_TA["Date"] = pd.to_datetime(df_TA["Date"])
    df_TA = df_TA.sort_values("Date").reset_index(drop=True)

    df_TA["Pred_Up"] = df_TA["Predicted_Close"] > df_TA["Close_Lag1"]
    df_TA["Pred_Down"] = ~df_TA["Pred_Up"]
    df_TA["Anomaly_Flag"] = (
        pd.to_numeric(df_TA["IF_Anomaly"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(int)
    )
    ma10 = df_TA["MA10_Lag1"]
    ma50 = df_TA["MA50_Lag1"]

    df_TA["Bull_Cross"] = (ma10.shift(1) <= ma50.shift(1)) & (ma10 > ma50)
    df_TA["Bear_Cross"] = (ma10.shift(1) >= ma50.shift(1)) & (ma10 < ma50)

    rsi = df_TA["RSI_Lag1"]
    df_TA["TA_Buy"] = df_TA["Bull_Cross"] & (rsi < 65)
    df_TA["TA_Sell"] = df_TA["Bear_Cross"] | (rsi > 65)

    df_TA["Recommendation"] = "Hold"
    buy_mask = df_TA["Pred_Up"] & (df_TA["Anomaly_Flag"] == 0) & df_TA["TA_Buy"]
    sell_mask = df_TA["Pred_Down"] & (df_TA["Anomaly_Flag"] == 0) & df_TA["TA_Sell"]

    conflict = buy_mask & sell_mask
    buy_mask = buy_mask & ~conflict
    sell_mask = sell_mask & ~conflict

    df_TA.loc[buy_mask, "Recommendation"] = "Buy"
    df_TA.loc[sell_mask, "Recommendation"] = "Sell"

    df_TA["TA_Label"] = np.select(
        [df_TA["TA_Buy"], df_TA["TA_Sell"]],
        ["Buy", "Sell"],
        default="None"
    )
    df_TA["Forecast_Direction"] = np.where(df_TA["Pred_Up"], "Up", "Down")

    df_TA["Agreement"] = (
        (df_TA["TA_Buy"] & df_TA["Pred_Up"]) |
        (df_TA["TA_Sell"] & df_TA["Pred_Down"])
    )
    df_TA["Blocked_By_Anomaly"] = df_TA["Agreement"] & (df_TA["Anomaly_Flag"] == 1)

    needed = [
        "Close_Lag1", "Predicted_Close",
        "MA10_Lag1", "MA50_Lag1",
        "RSI_Lag1", "IF_Anomaly"
    ]
    df_TA = df_TA.dropna(subset=needed).reset_index(drop=True)

    return df_TA

def plot_step5_chart(df: pd.DataFrame):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    buy_pts = df[df["Recommendation"] == "Buy"]
    sell_pts = df[df["Recommendation"] == "Sell"]
    anom_pts = df[df["IF_Anomaly"] == 1]

    fig = plt.figure(figsize=(14, 8))
    plt.plot(df["Date"], df["Close"], label="Close")
    plt.plot(df["Date"], df["MA10"], label="MA10")
    plt.plot(df["Date"], df["MA50"], label="MA50")

    plt.scatter(buy_pts["Date"], buy_pts["Close"], marker="^", s=150, label="Buy")
    plt.scatter(sell_pts["Date"], sell_pts["Close"], marker="v", s=150, label="Sell")
    plt.scatter(anom_pts["Date"], anom_pts["Close"], marker="x", s=150, label="Anomaly")

    plt.title("Testing Period Price with MA Lines and Final Recommendations")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    return fig


# =========================
# Config from notebook
# =========================
FEATURES_EXP_C = ["Close", "MA10", "MA20", "MA50", "RSI"]

BEST_PARAMS = {
    "n_estimators": 800,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9
}

bundle_path = "./Model/XGBoost/xgb_full_bundle.joblib"
bundle = joblib.load(bundle_path)
model_full = bundle["model"]

def build_step4_no_train_with_alignment(df_raw: pd.DataFrame, feature_cols: list, model_full):
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Close_Lag1"] = df["Close"].shift(1)
    lag_map = {
        "MA10_Lag1": "MA10",
        "MA20_Lag1": "MA20",
        "MA50_Lag1": "MA50",
        "RSI_Lag1": "RSI"
    }
    for lag_col, src_col in lag_map.items():
        if src_col in df.columns:
            df[lag_col] = df[src_col].shift(1)
    df["NextDate"] = df["Date"].shift(-1)
    df["Actual_NextClose"] = df["Close"].shift(-1)

    X = df[feature_cols].astype(float).values
    pred = model_full.predict(X)
    df["Pred_NextClose"] = pred

    df_pred = df[["NextDate","Actual_NextClose","Pred_NextClose"]].copy()
    df_pred = df_pred.rename(columns={"NextDate":"Date", "Pred_NextClose":"Predicted_Close"})
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])

    df_out = df.merge(df_pred[["Date","Predicted_Close"]], on="Date", how="left")

    n = len(df_out)
    test_start = int(n * 0.85)
    df_out["Split"] = "Training"
    df_out.loc[test_start:, "Split"] = "Testing"

    df_out = df_out.dropna(subset=["Predicted_Close"]).reset_index(drop=True)
    return df_out

def get_model_and_features(bundle):
    model = bundle.get("model_train_val") or bundle.get("model_full") or bundle.get("model")
    feature_cols = bundle.get("feature_cols") or bundle.get("features")
    if model is None or feature_cols is None:
        raise ValueError("Bundle must contain model and feature_cols/features")
    return model, feature_cols

# =========================
# Pipeline runner
# =========================
def run_full_pipeline(df_raw, contamination, trainval_path, full_path):
    bundle_tv = joblib.load(trainval_path)
    bundle_full = joblib.load(full_path)

    model_tv, feature_cols_tv = get_model_and_features(bundle_tv)
    model_full, feature_cols_full = get_model_and_features(bundle_full)

    if list(feature_cols_tv) != list(feature_cols_full):
        st.warning("Train Val feature_cols and Full feature_cols are different. Using Train Val feature order for testing.")
    feature_cols = list(feature_cols_tv)

    df_step4 = build_step4_no_train_with_alignment(df_raw, feature_cols, model_tv)

    forecast_date = get_next_forecast_date(df_step4)
    df_step4_updated, _ = inject_fullmodel_forecast_to_df_step4_v2(
        df_step4=df_step4,
        df_raw=df_raw,
        feature_cols=feature_cols,
        model_full=model_full,
        forecast_date=forecast_date,
        split_label="Forecast"
    )

    # Step4 anomaly
    df_ad, _ = run_iforest_on_test_and_forecast(df_step4_updated, contamination=contamination)

    # Step5 TA
    df_step5 = build_step5_recommendation(df_ad)

    # outputs
    fig_out = plot_step5_chart(df_step5)
    table_out = (
        df_step5[df_step5["Split"].isin(["Testing", "Forecast"])]
        .sort_values("Date")
        [["Date","Close_Lag1","Predicted_Close","Forecast_Direction","IF_Anomaly","TA_Label","Recommendation"]]
        .tail(10)
        .reset_index(drop=True)
    )

    return df_step5, fig_out, table_out

# =========================
# UI
# =========================
st.title("Stock Forecasting, Anomaly Detection, and Recommendation System")

with st.sidebar:
    st.header("Controls")

    csv_path = st.text_input(
        "Dataset path",
        value="./Dataset/Public_Bank_Stock_Price_History.csv"
    )

    contamination = st.slider(
        "Isolation Forest contamination",
        min_value=0.01,
        max_value=0.20,
        value=0.08,
        step=0.01
    )

    predict_clicked = st.button("Predict")

# Load data
df = load_and_prepare_data(csv_path)

# Main layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Historical Data")
    fig_hist = plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["Close"])
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.tight_layout()
    st.pyplot(fig_hist, clear_figure=True)

with right:
    st.subheader("Quick view")
    st.write("Rows", len(df))
    st.write("Date range")
    st.write(df["Date"].min(), "to", df["Date"].max())

# Predict section
if predict_clicked:
    df_step5, fig_out, table_out = run_full_pipeline(df, contamination, trainval_path, full_path)
    st.pyplot(fig_out, clear_figure=True)
    st.dataframe(table_out, use_container_width=True)

    st.subheader("Download results")
    csv_bytes = df_step5.to_csv(index=False).encode("utf-8")
    st.download_button("Download Recommendation result as CSV", data=csv_bytes, file_name="Recommendation_Results.csv", mime="text/csv")
