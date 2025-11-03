import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# ===== helper utilities you already have =====
from model_utils import (
    implied_probs_from_odds,
    blend_probs,
    compute_form_from_last5,
    ensure_order,
    DEFAULT_CLASS_ORDER,
)

# ===== UI SETUP =====
st.set_page_config(page_title="Global Football Match Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Global Football Match Predictor")
st.markdown("""
Predict **Home/Draw/Away** using:
- **Recent form** (last 5), and/or
- **Bookmaker odds** (converted to implied probabilities).

Drop a trained **model.pkl** next to this file to use it; otherwise a transparent **form+odds** fallback is used.
""")

# ===== COMPAT PATCHES =====
def patch_sklearn_compat(model):
    """
    Make older pickled sklearn pipelines run on newer sklearn (1.7.x).
    - SimpleImputer: add keep_empty_features if missing
    - OneHotEncoder: add private/public attrs and sparse_output shim
    """
    try:
        if not hasattr(model, "named_steps"):
            return model
        pre = model.named_steps.get("pre")
        if pre is None or not hasattr(pre, "transformers_"):
            return model

        for name, trans, cols in pre.transformers_:
            # Numeric branch: Pipeline(..., ("imp", SimpleImputer(...)))
            if hasattr(trans, "named_steps") and "imp" in trans.named_steps:
                imp = trans.named_steps["imp"]
                if not hasattr(imp, "keep_empty_features"):
                    imp.keep_empty_features = False

            # Find a fitted OneHotEncoder (could be direct or inside Pipeline as 'ohe')
            enc = None
            if hasattr(trans, "named_steps"):
                enc = trans.named_steps.get("ohe")
            elif "OneHotEncoder" in str(type(trans)):
                enc = trans
            if enc is None and hasattr(trans, "categories_") and hasattr(trans, "transform"):
                enc = trans  # heuristic

            if enc is not None:
                # newer sklearn checks this private attr
                if not hasattr(enc, "_drop_idx_after_grouping"):
                    enc._drop_idx_after_grouping = None
                # ensure public .drop exists
                if not hasattr(enc, "drop"):
                    enc.drop = None
                # shim for sparse -> sparse_output rename
                if hasattr(enc, "sparse") and not hasattr(enc, "sparse_output"):
                    try:
                        enc.sparse_output = bool(enc.sparse)
                    except Exception:
                        enc.sparse_output = False
                if hasattr(enc, "sparse_output") and not hasattr(enc, "sparse"):
                    try:
                        enc.sparse = bool(enc.sparse_output)
                    except Exception:
                        enc.sparse = False
    except Exception:
        # don't crash patching
        pass
    return model

def model_expects_odds(model) -> bool:
    """Return True if the trained preprocessor expects numeric columns Odds_H/D/A."""
    try:
        if not hasattr(model, "named_steps"):
            return False
        pre = model.named_steps.get("pre")
        if pre and hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                if name == "num":
                    cols_set = set([str(c) for c in cols])
                    return {"Odds_H", "Odds_D", "Odds_A"}.issubset(cols_set)
    except Exception:
        pass
    return False

# ===== LOAD MODEL =====
MODEL = None
LABEL_ORDER = DEFAULT_CLASS_ORDER  # ['H','D','A']
model_load_note = None

for candidate in ["model.pkl", "model_international.pkl"]:
    if MODEL is None:
        try:
            m = load(candidate)
            MODEL = patch_sklearn_compat(m)
            model_load_note = f"Model loaded ({candidate})."
        except Exception:
            MODEL = None

if MODEL is not None:
    st.success(model_load_note)
else:
    st.info("No model file found — using fallback (form + odds blend).")

expects_odds = model_expects_odds(MODEL) if MODEL is not None else False

# ===== FORM =====
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

# ===== PREDICT =====
if submitted:
    # NOTE: make sure users use dots not commas for decimals (e.g., 1.60 not 1,60)
    p_odds = implied_probs_from_odds(home_odds, draw_odds, away_odds) if use_odds else None

    if MODEL is not None:
        # build input row for the trained pipeline
        X = pd.DataFrame([{
            "Home_ppg5": home_ppg5, "Home_gd5": home_gd5,
            "Away_ppg5": away_ppg5, "Away_gd5": away_gd5,
            "HomeTeam": home_team, "AwayTeam": away_team
        }])

        # odds columns if expected
        if expects_odds:
            if use_odds and p_odds is not None:
                X["Odds_H"], X["Odds_D"], X["Odds_A"] = float(p_odds[0]), float(p_odds[1]), float(p_odds[2])
            else:
                X["Odds_H"], X["Odds_D"], X["Odds_A"] = np.nan, np.nan, np.nan

        # ensure numeric dtypes
        for c in ["Home_ppg5","Home_gd5","Away_ppg5","Away_gd5","Odds_H","Odds_D","Odds_A"]:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        try:
            proba = MODEL.predict_proba(X)[0]
            if hasattr(MODEL, "classes_"):
                proba = ensure_order(proba, MODEL.classes_, LABEL_ORDER)
            p_model = np.array([proba[0], proba[1], proba[2]])
            source = "Model"
        except Exception as e:
            st.error("Prediction failed. Falling back to form heuristic. See details below.")
            st.exception(e)
            p_home = 0.40 + 0.12*(home_ppg5 - away_ppg5) + 0.08*(home_gd5 - away_gd5)
            p_draw = 0.28
            p_away = 1 - p_home - p_draw
            p_model = np.clip([p_home, p_draw, p_away], 0.01, 0.98)
            p_model = p_model/np.sum(p_model)
            source = "Form-only fallback (after error)"
    else:
        # no model: heuristic
        p_home = 0.40 + 0.12*(home_ppg5 - away_ppg5) + 0.08*(home_gd5 - away_gd5)
        p_draw = 0.28
        p_away = 1 - p_home - p_draw
        p_model = np.clip([p_home, p_draw, p_away], 0.01, 0.98)
        p_model = p_model/np.sum(p_model)
        source = "Form-only fallback"

    # blend with odds if provided
    if p_odds is None:
        p_final = p_model
        details = {"Model/Form": np.round(p_model, 3)}
    else:
        p_final = blend_probs(p_model, p_odds, w)
        details = {"Model/Form": np.round(p_model, 3), "Odds": np.round(p_odds, 3), "Blend": np.round(p_final, 3)}

    labels = ["Home", "Draw", "Away"]
    st.subheader("Predicted Probabilities")
    st.write({labels[i]: float(np.round(p_final[i], 3)) for i in range(3)})
    st.bar_chart(pd.DataFrame(details, index=labels))
    st.caption(f"Label order [H, D, A]. Source: {source}.")
