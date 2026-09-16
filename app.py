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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/score"

st.set_page_config(
    page_title="Cost-Aware Fraud Detection",
    page_icon="🛡️",
    layout="centered",
)

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
# Input form
# ---------------------------------------------------------------------------
with st.form("txn_form"):
    st.markdown("#### Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
        txn_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "CASH_IN", "DEBIT", "PAYMENT", "TRANSFER"],
        )
        origin_balance_error = st.number_input(
            "Origin Balance Error", value=0.0, step=1.0,
            help="Discrepancy between expected and actual sender balance after the transaction."
        )
        destination_balance_error = st.number_input(
            "Destination Balance Error", value=0.0, step=1.0,
            help="Discrepancy between expected and actual recipient balance after the transaction."
        )

    with col2:
        destination_balance_is_zero = st.checkbox("Destination balance is zero")
        destination_is_first_transaction = st.checkbox("First transaction to this destination")
        destination_transactions_last_24h = st.number_input(
            "Destination Txns (Last 24h)", min_value=0.0, value=0.0, step=1.0
        )
        destination_transactions_last_7d = st.number_input(
            "Destination Txns (Last 7d)", min_value=0.0, value=0.0, step=1.0
        )

    col3, col4 = st.columns(2)
    with col3:
        destination_avg_previous_amount = st.number_input(
            "Destination Avg Previous Amount", min_value=0.0, value=0.0, step=100.0
        )
    with col4:
        destination_amount_deviation = st.number_input(
            "Destination Amount Deviation", value=0.0, step=1.0,
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
