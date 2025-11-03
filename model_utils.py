
import numpy as np

DEFAULT_CLASS_ORDER = ['H','D','A']

def implied_probs_from_odds(home_odds: float, draw_odds: float, away_odds: float):
    ip = np.array([1.0/home_odds, 1.0/draw_odds, 1.0/away_odds], dtype=float)
    s = ip.sum()
    return (ip/s) if s > 0 else np.array([1/3,1/3,1/3])

def blend_probs(p_model, p_odds, w: float):
    p_model = np.asarray(p_model, dtype=float)
    p_odds  = np.asarray(p_odds, dtype=float)
    p = w*p_model + (1-w)*p_odds
    p = np.clip(p, 1e-6, 1.0)
    return p / p.sum()

def compute_form_from_last5(txt: str):
    s = (txt or "").strip()
    if not s:
        return 1.5, 0.0
    parts = [p.strip() for p in s.split(",") if p.strip()]
    pts, gds = [], []
    for p in parts[:5]:
        if "-" in p and any(ch.isdigit() for ch in p):
            try:
                gf, ga = p.split("-")
                gf, ga = int(gf.strip()), int(ga.strip())
                gds.append(gf - ga)
                if gf > ga: pts.append(3)
                elif gf == ga: pts.append(1)
                else: pts.append(0)
            except:
                continue
        else:
            if p.upper().startswith("W"): pts.append(3); gds.append(1)
            elif p.upper().startswith("D"): pts.append(1); gds.append(0)
            elif p.upper().startswith("L"): pts.append(0); gds.append(-1)
    if not pts:
        return 1.5, 0.0
    import numpy as np
    return float(np.mean(pts)), float(np.mean(gds) if gds else 0.0)

def ensure_order(proba, model_classes, desired_order):
    idx = {c:i for i,c in enumerate(model_classes)}
    return np.array([proba[idx[o]] if o in idx else 0.0 for o in desired_order])
