
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from model_utils import implied_probs_from_odds, blend_probs, compute_form_from_last5, ensure_order, DEFAULT_CLASS_ORDER

st.set_page_config(page_title="Global Football Match Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Global Football Match Predictor")

st.markdown("""
Predict **Home/Draw/Away** using:
- **Recent form** (last 5), and/or
- **Bookmaker odds** (converted to implied probabilities).

Drop a trained **model.pkl** (from `train_model.py`) next to this file to use it; otherwise a transparent **form+odds** fallback is used.
""")

MODEL = None
LABEL_ORDER = DEFAULT_CLASS_ORDER  # ['H','D','A']
try:
    MODEL = load("model.pkl")
    st.success("Model loaded (model.pkl).")
except Exception:
    st.info("No model.pkl found — using fallback (form + odds blend).")

with st.form("inputs"):
    colA, colB = st.columns(2)
    with colA:
        home_team = st.text_input("Home Team", "Kaizer Chiefs")
        home_last5 = st.text_input("Home last 5 (W/D/L or GF-GA comma list)", "W,D,W,L,W")
        home_ppg5_default, home_gd5_default = compute_form_from_last5(home_last5)
        home_ppg5 = st.number_input("Home PPG last 5", 0.0, 3.0, float(home_ppg5_default), 0.05)
        home_gd5  = st.number_input("Home GD avg last 5", -5.0, 5.0, float(home_gd5_default), 0.05)
    with colB:
        away_team = st.text_input("Away Team", "Orlando Pirates")
        away_last5 = st.text_input("Away last 5 (W/D/L or GF-GA comma list)", "W,L,D,W,D")
        away_ppg5_default, away_gd5_default = compute_form_from_last5(away_last5)
        away_ppg5 = st.number_input("Away PPG last 5", 0.0, 3.0, float(away_ppg5_default), 0.05)
        away_gd5  = st.number_input("Away GD avg last 5", -5.0, 5.0, float(away_gd5_default), 0.05)

    st.markdown("#### Optional: Bookmaker Odds (Decimal)")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds = st.number_input("Home odds", 1.01, 100.0, 2.10, 0.01)
    with col2:
        draw_odds = st.number_input("Draw odds", 1.01, 100.0, 3.10, 0.01)
    with col3:
        away_odds = st.number_input("Away odds", 1.01, 100.0, 3.40, 0.01)
    use_odds = st.checkbox("Use odds in prediction", value=True)

    w = st.slider("Blend weight (model/form vs odds)", 0.0, 1.0, 0.6, 0.05)

    submitted = st.form_submit_button("Predict")

if submitted:
    p_odds = implied_probs_from_odds(home_odds, draw_odds, away_odds) if use_odds else None

    if MODEL is not None:
        X = pd.DataFrame([{
            "Home_ppg5": home_ppg5, "Home_gd5": home_gd5,
            "Away_ppg5": away_ppg5, "Away_gd5": away_gd5,
            "HomeTeam": home_team, "AwayTeam": away_team
        }])
        proba = MODEL.predict_proba(X)[0]
        if hasattr(MODEL, "classes_"):
            proba = ensure_order(proba, MODEL.classes_, LABEL_ORDER)
        p_model = np.array([proba[0], proba[1], proba[2]])
        source = "Model"
    else:
        p_home = 0.40 + 0.12*(home_ppg5 - away_ppg5) + 0.08*(home_gd5 - away_gd5)
        p_draw = 0.28
        p_away = 1 - p_home - p_draw
        p_model = np.clip([p_home, p_draw, p_away], 0.01, 0.98)
        p_model = p_model/np.sum(p_model)
        source = "Form-only fallback"

    if p_odds is None:
        p_final = p_model
        details = {"Model/Form": np.round(p_model, 3)}
    else:
        from model_utils import blend_probs
        p_final = blend_probs(p_model, p_odds, w)
        details = {"Model/Form": np.round(p_model, 3), "Odds": np.round(p_odds, 3), "Blend": np.round(p_final, 3)}

    labels = ["Home","Draw","Away"]
    st.subheader("Predicted Probabilities")
    st.write({labels[i]: float(np.round(p_final[i],3)) for i in range(3)})
    st.bar_chart(pd.DataFrame(details, index=labels))
    st.caption(f"Label order [H, D, A]. Source: {source}.")
