"""
Fraud Detection — Streamlit Frontend
Talks to the FastAPI /score endpoint and renders a big visual
ALLOW / REVIEW / BLOCK decision plus a probability gauge.

Run this AFTER the FastAPI backend is already running:
    Terminal 1: python -m uvicorn main:app --host 0.0.0.0 --port 8000   (from api/)
    Terminal 2: streamlit run app.py                                    (from wherever this file is)
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/score"
DEMO_SAMPLE_PATH = "demo_sample.csv"


@st.cache_data
def load_demo_sample():
    try:
        return pd.read_csv(DEMO_SAMPLE_PATH)
    except FileNotFoundError:
        return None


demo_df = load_demo_sample()

st.set_page_config(
    page_title="Cost-Aware Fraud Detection",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Preset scenarios — pulled directly from the actual held-out dataset,
# verified against the trained model's real predictions. Using genuine
# transactions here (not hand-built synthetic ones) avoids the instability
# seen when testing artificial out-of-distribution feature combinations.
# ---------------------------------------------------------------------------
PRESETS = {
    "✅ Legit transaction": {
        "amount": 76431.17,
        "type": "CASH_OUT",
        "origin_balance_error": -76431.17,
        "destination_balance_error": 0.0,
        "destination_balance_is_zero": False,
        "destination_is_first_transaction": False,
        "destination_transactions_last_24h": 1.0,
        "destination_transactions_last_7d": 11.0,
        "destination_avg_previous_amount": 391891.78,
        "destination_amount_deviation": -315460.61,
        "total_transactions": 33528,
        "total_transaction_amount": 4693535568.26,
        "avg_transaction_amount": 139988.53,
    },
    "⛔ Fraud — low amount": {
        "amount": 48266.80,
        "type": "TRANSFER",
        "origin_balance_error": 0.0,
        "destination_balance_error": 48266.80,
        "destination_balance_is_zero": True,
        "destination_is_first_transaction": True,
        "destination_transactions_last_24h": 0.0,
        "destination_transactions_last_7d": 0.0,
        "destination_avg_previous_amount": 0.0,
        "destination_amount_deviation": 48266.80,
        "total_transactions": 40218,
        "total_transaction_amount": 6596385476.64,
        "avg_transaction_amount": 164015.75,
    },
    "⛔ Fraud — mid amount": {
        "amount": 500003.56,
        "type": "CASH_OUT",
        "origin_balance_error": 0.0,
        "destination_balance_error": 0.0,
        "destination_balance_is_zero": False,
        "destination_is_first_transaction": False,
        "destination_transactions_last_24h": 0.0,
        "destination_transactions_last_7d": 2.0,
        "destination_avg_previous_amount": 47108.30,
        "destination_amount_deviation": 452895.27,
        # This row came from an unusually quiet hour in the real data —
        # total_transactions=10 is a big outlier vs. the ~31,900 median.
        # That system-level context matters to the model, so it's carried
        # here explicitly rather than falling back to a "normal hour" default.
        "total_transactions": 10,
        "total_transaction_amount": 23250735.88,
        "avg_transaction_amount": 2325073.588,
    },
    "⛔ Fraud — high amount": {
        "amount": 4129482.96,
        "type": "CASH_OUT",
        "origin_balance_error": 0.0,
        "destination_balance_error": 0.0,
        "destination_balance_is_zero": False,
        "destination_is_first_transaction": False,
        "destination_transactions_last_24h": 1.0,
        "destination_transactions_last_7d": 9.0,
        "destination_avg_previous_amount": 222913.55,
        "destination_amount_deviation": 3906569.41,
        "total_transactions": 26927,
        "total_transaction_amount": 5172213957.19,
        "avg_transaction_amount": 192082.81,
    },
}

DEFAULTS = {
    "amount": 5000.0,
    "type": "CASH_OUT",
    "origin_balance_error": 0.0,
    "destination_balance_error": 0.0,
    "destination_balance_is_zero": False,
    "destination_is_first_transaction": False,
    "destination_transactions_last_24h": 0.0,
    "destination_transactions_last_7d": 0.0,
    "destination_avg_previous_amount": 0.0,
    "destination_amount_deviation": 0.0,
    # System-level features — not shown in the form, but carried through so
    # presets can override them with real values. Manual entries use the API's
    # own median defaults by simply not sending these keys at all.
    "total_transactions": None,
    "total_transaction_amount": None,
    "avg_transaction_amount": None,
}

# Initialize session state with defaults on first load
if "form_values" not in st.session_state:
    st.session_state.form_values = DEFAULTS.copy()
if "ground_truth" not in st.session_state:
    st.session_state.ground_truth = None  # None = unknown (manual entry), else 0/1


def apply_preset(preset_name):
    st.session_state.form_values = PRESETS[preset_name].copy()
    st.session_state.ground_truth = 1 if "Fraud" in preset_name else 0


def apply_random_row():
    if demo_df is None or len(demo_df) == 0:
        return
    row = demo_df.sample(1).iloc[0]
    type_col = next((c for c in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
                                   "type_PAYMENT", "type_TRANSFER"] if row.get(c)), "type_CASH_OUT")
    txn_type = type_col.replace("type_", "")

    st.session_state.form_values = {
        "amount": float(row["amount"]),
        "type": txn_type,
        "origin_balance_error": float(row["origin_balance_error"]),
        "destination_balance_error": float(row["destination_balance_error"]),
        "destination_balance_is_zero": bool(row["destination_balance_is_zero"]),
        "destination_is_first_transaction": bool(row["destination_is_first_transaction"]),
        "destination_transactions_last_24h": float(row["destination_transactions_last_24h"]),
        "destination_transactions_last_7d": float(row["destination_transactions_last_7d"]),
        "destination_avg_previous_amount": float(row["destination_avg_previous_amount"]),
        "destination_amount_deviation": float(row["destination_amount_deviation"]),
        "total_transactions": float(row["total_transactions"]),
        "total_transaction_amount": float(row["total_transaction_amount"]),
        "avg_transaction_amount": float(row["avg_transaction_amount"]),
    }
    st.session_state.ground_truth = int(row["actual_fraud"])


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .title-text {
        font-size: 2.1rem;
        font-weight: 700;
        color: #fafafa;
        margin-bottom: 0.1rem;
    }
    .subtitle-text {
        font-size: 0.95rem;
        color: #9aa0a6;
        margin-bottom: 1.6rem;
    }

    .decision-banner {
        padding: 2.2rem 1rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.2rem 0 1.6rem 0;
        border: 2px solid rgba(255,255,255,0.08);
    }
    .decision-label {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 0;
    }
    .decision-sub {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.4rem;
    }

    .allow   { background: linear-gradient(135deg, #0f3d2e, #12291f); }
    .allow   .decision-label { color: #4ade80; }

    .review  { background: linear-gradient(135deg, #4d3b0a, #2b2408); }
    .review  .decision-label { color: #fbbf24; }

    .block   { background: linear-gradient(135deg, #4a1414, #2b0d0d); }
    .block   .decision-label { color: #f87171; }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title-text">🛡️ Cost-Aware Fraud Detection</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle-text">Delay-aware decisioning on PaySim transactions — LightGBM, cost-optimized thresholds</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Preset buttons — real, verified transactions for a safe live demo
# ---------------------------------------------------------------------------
st.markdown("#### Quick Scenarios")
st.caption("Real transactions from the held-out evaluation set — verified against the model.")

preset_cols = st.columns(5)
for i, name in enumerate(PRESETS.keys()):
    with preset_cols[i]:
        if st.button(name, use_container_width=True):
            apply_preset(name)
            st.rerun()

with preset_cols[4]:
    random_disabled = demo_df is None
    if st.button("🎲 Random Real Transaction", use_container_width=True, disabled=random_disabled):
        apply_random_row()
        st.rerun()

if demo_df is None:
    st.warning(
        f"`{DEMO_SAMPLE_PATH}` not found — the Random Transaction button is disabled. "
        "Export a sample from the notebook and place it next to app.py to enable it."
    )

if st.session_state.ground_truth is not None:
    truth_label = "🚨 ACTUAL: FRAUD" if st.session_state.ground_truth == 1 else "✅ ACTUAL: LEGITIMATE"
    st.info(f"**Ground truth for this transaction:** {truth_label}  (from real held-out data — not shown to the model)")

st.markdown("---")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
fv = st.session_state.form_values

with st.form("txn_form"):
    st.markdown("#### Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=float(fv["amount"]), step=100.0)
        txn_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT", "TRANSFER"],
            index=["CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT", "TRANSFER"].index(fv["type"]),
        )
        origin_balance_error = st.number_input(
            "Origin Balance Error", value=float(fv["origin_balance_error"]), step=1.0,
            help="Discrepancy between expected and actual sender balance after the transaction."
        )
        destination_balance_error = st.number_input(
            "Destination Balance Error", value=float(fv["destination_balance_error"]), step=1.0,
            help="Discrepancy between expected and actual recipient balance after the transaction."
        )

    with col2:
        destination_balance_is_zero = st.checkbox(
            "Destination balance is zero", value=bool(fv["destination_balance_is_zero"])
        )
        destination_is_first_transaction = st.checkbox(
            "First transaction to this destination", value=bool(fv["destination_is_first_transaction"])
        )
        destination_transactions_last_24h = st.number_input(
            "Destination Txns (Last 24h)", min_value=0.0,
            value=float(fv["destination_transactions_last_24h"]), step=1.0
        )
        destination_transactions_last_7d = st.number_input(
            "Destination Txns (Last 7d)", min_value=0.0,
            value=float(fv["destination_transactions_last_7d"]), step=1.0
        )

    col3, col4 = st.columns(2)
    with col3:
        destination_avg_previous_amount = st.number_input(
            "Destination Avg Previous Amount", min_value=0.0,
            value=float(fv["destination_avg_previous_amount"]), step=100.0
        )
    with col4:
        destination_amount_deviation = st.number_input(
            "Destination Amount Deviation", value=float(fv["destination_amount_deviation"]), step=1.0,
            help="How far this amount deviates from the destination's historical pattern."
        )

    submitted = st.form_submit_button("Score Transaction")

# ---------------------------------------------------------------------------
# Build payload + call API
# ---------------------------------------------------------------------------
if submitted:
    type_flags = {
        "type_CASH_IN": txn_type == "CASH_IN",
        "type_CASH_OUT": txn_type == "CASH_OUT",
        "type_DEBIT": txn_type == "DEBIT",
        "type_PAYMENT": txn_type == "PAYMENT",
        "type_TRANSFER": txn_type == "TRANSFER",
    }

    payload = {
        "amount": amount,
        "origin_balance_error": origin_balance_error,
        "destination_balance_error": destination_balance_error,
        "destination_balance_is_zero": int(destination_balance_is_zero),
        "destination_transactions_last_24h": destination_transactions_last_24h,
        "destination_transactions_last_7d": destination_transactions_last_7d,
        "destination_avg_previous_amount": destination_avg_previous_amount,
        "destination_amount_deviation": destination_amount_deviation,
        "destination_is_first_transaction": int(destination_is_first_transaction),
        **type_flags,
    }

    # Carry through real system-level values from a preset, if set.
    # Manual entries leave these out entirely so the API falls back to
    # its own median defaults.
    for sys_field in ("total_transactions", "total_transaction_amount", "avg_transaction_amount"):
        if fv.get(sys_field) is not None:
            payload[sys_field] = fv[sys_field]

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        prob = result["fraud_probability"]
        tier = result["tier"]
        t_review = result["thresholds_used"]["t_review"]
        t_block = result["thresholds_used"]["t_block"]

        css_class = {"ALLOW": "allow", "REVIEW": "review", "BLOCK": "block"}[tier]
        icon = {"ALLOW": "✅", "REVIEW": "⚠️", "BLOCK": "⛔"}[tier]

        st.markdown(f"""
        <div class="decision-banner {css_class}">
            <p class="decision-label">{icon} {tier}</p>
            <p class="decision-sub">Fraud probability: {prob:.4%}</p>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Fraud Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#60a5fa", "thickness": 0.3},
                "steps": [
                    {"range": [0, t_review * 100], "color": "#12291f"},
                    {"range": [t_review * 100, t_block * 100], "color": "#2b2408"},
                    {"range": [t_block * 100, 100], "color": "#2b0d0d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": prob * 100,
                },
            },
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa"},
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw API response"):
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the API. Make sure the FastAPI backend is running:\n\n"
            "`python -m uvicorn main:app --host 0.0.0.0 --port 8000` (run from the `api/` folder)"
        )
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")

st.markdown("---")
st.caption("Cost-aware, delay-aware fraud decisioning · PaySim dataset · LightGBM")