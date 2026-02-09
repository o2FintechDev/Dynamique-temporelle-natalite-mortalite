# src/narrative/interpretation_engine.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from src.utils.run_reader import RunManager, read_metric_json


def _m(run_id: str, label: str) -> dict:
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if not p:
        return {}
    try:
        v = read_metric_json(p)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _parse_order(order: Any) -> Optional[Tuple[int, int, int]]:
    """
    Normalise order vers (p,d,q).
    Accepte:
      - tuple/list (p,d,q)
      - dict {"p":..,"d":..,"q":..} ou {"order":[p,d,q]}
      - string "ARIMA(4,1,1)" ou "(4, 1, 1)" ou "4,1,1"
    """
    if order is None:
        return None

    if isinstance(order, (tuple, list)) and len(order) == 3:
        try:
            p, d, q = order
            return int(p), int(d), int(q)
        except Exception:
            return None

    if isinstance(order, dict):
        if "order" in order:
            return _parse_order(order.get("order"))
        if all(k in order for k in ("p", "d", "q")):
            try:
                return int(order["p"]), int(order["d"]), int(order["q"])
            except Exception:
                return None
        return None

    s = str(order).strip()
    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 3:
        try:
            p, d, q = map(int, nums[:3])
            return p, d, q
        except Exception:
            return None
    return None


def _model_kind(p: int, d: int, q: int) -> str:
    if d > 0:
        return "ARIMA"
    if p > 0 and q == 0:
        return "AR"
    if p == 0 and q > 0:
        return "MA"
    if p > 0 and q > 0:
        return "ARMA"
    return "ARIMA"


def _fmt_p(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.3f}"


def _fmt2(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "NA"


def build_snippets_from_run(*, run_id: str, y: str) -> Dict[str, str]:
    """
    Génère des snippets LaTeX narratifs.
    IMPORTANT: pas de clés tbl_* ici (réservées aux tables).
    """
    tsds = _m(run_id, "m.diag.ts_vs_ds")
    uni = _m(run_id, "m.uni.best")
    coint = _m(run_id, "m.coint.meta")

    # ============================================================
    # Stationnarité (ADF)
    # ============================================================
    adf_c = tsds.get("adf_p_c")
    adf_ct = tsds.get("adf_p_ct")
    verdict = (tsds.get("verdict") or "").upper()

    st: list[str] = [r"\paragraph{Stationnarité.}"]
    pvals = [p for p in (adf_c, adf_ct) if isinstance(p, (int, float))]
    pmin = min(pvals) if pvals else None

    if pmin is not None:
        if pmin < 0.05:
            st.append(
                rf"Le test ADF rejette l’hypothèse de racine unitaire "
                rf"($p$-value minimale $\approx {_fmt_p(pmin)} < 0.05$). "
                r"La série est traitable comme stationnaire (en niveau ou autour d’une tendance selon la spécification)."
            )
        else:
            st.append(
                rf"Le test ADF ne rejette pas l’hypothèse de racine unitaire "
                rf"($p$-value minimale $\approx {_fmt_p(pmin)} \ge 0.05$). "
                r"La série est compatible avec un processus intégré, justifiant une différenciation (souvent $d=1$)."
            )
    else:
        st.append(
            rf"Le verdict ADF-only est \textbf{{{verdict or 'NA'}}}. "
            r"Il pilote la transformation retenue avant modélisation."
        )

    # ============================================================
    # Modélisation univariée (ARIMA)
    # ============================================================
    kp = (uni.get("key_points") or {})
    order_raw = kp.get("order") or (uni.get("best") or {}).get("order")
    order = _parse_order(order_raw)

    aic = kp.get("aic") or (uni.get("best") or {}).get("aic")
    bic = kp.get("bic") or (uni.get("best") or {}).get("bic")
    jb_p = kp.get("jb_p") or kp.get("jarque_bera_p")
    lb_p = kp.get("lb_p") or kp.get("ljungbox_p")

    uv: list[str] = [r"\paragraph{Modélisation univariée et choix du modèle.}"]

    if order is not None:
        p, d, q = order
        kind = _model_kind(p, d, q)
        uv.append(
            rf"Le modèle retenu est \textbf{{{kind}({p},{d},{q})}}, "
            rf"sélectionné via critères d’information (AIC={_fmt2(aic)}, BIC={_fmt2(bic)})."
        )
        if kind == "AR":
            uv.append(
                r"Lecture : dynamique dominée par l’inertie (persistance élevée), cohérente avec des ajustements démographiques lents."
            )
        elif kind == "MA":
            uv.append(
                r"Lecture : dynamique dominée par des chocs transitoires récents (effets courts)."
            )
        else:
            uv.append(
                r"Lecture : compromis entre persistance (AR) et chocs transitoires (MA)."
            )
    else:
        uv.append(
            r"Le modèle univarié retenu n’est pas exploitable (order absent ou non parsable dans les métriques)."
        )

    # Diagnostics résiduels (ton pro)
    if isinstance(jb_p, (int, float)) and jb_p < 0.05:
        uv.append(
            rf"Normalité : rejet (Jarque--Bera, $p$={_fmt_p(jb_p)}). "
            r"Interprétation : queues épaisses/chocs rares ; privilégier des conclusions robustes plutôt qu’une inférence gaussienne naïve."
        )

    if isinstance(lb_p, (int, float)) and lb_p < 0.05:
        uv.append(
            rf"Blancheur : rejet (Ljung--Box, $p$={_fmt_p(lb_p)}). "
            r"Interprétation : dynamique résiduelle ; la spécification est perfectible (ordres, saisonnalité, rupture)."
        )

    # ============================================================
    # Cointégration
    # ============================================================
    rank = coint.get("rank")
    choice = (coint.get("choice") or coint.get("selected_model") or "NA").upper()

    co: list[str] = [r"\paragraph{Cointégration et dynamique de long terme.}"]
    if isinstance(rank, int) and rank > 0:
        co.append(
            rf"Johansen indique un rang $r={rank}>0$. "
            r"Décision : \textbf{VECM}. Interprétation : relation(s) d’équilibre et mécanisme de rappel via le terme de correction d’erreur."
        )
    elif isinstance(rank, int) and rank == 0:
        co.append(
            r"Johansen ne détecte pas de cointégration ($r=0$). "
            r"Décision : \textbf{VAR en différences} pour la dynamique de court terme."
        )
    else:
        co.append(
            rf"Décision pipeline : \textbf{{{choice}}}, mais le rang n’est pas renseigné de manière exploitable."
        )

    # ============================================================
    # Sortie : uniquement des clés narratifs (pas tbl_*)
    # ============================================================
    return {
        "sec_stationarity": "\n".join(st) + "\n",
        "sec_univariate": "\n".join(uv) + "\n",
        "sec_cointegration": "\n".join(co) + "\n",
        # optionnel: si tu veux relier aux notes:
        "m.note.step3": "\n".join(st) + "\n",
        "m.note.step4": "\n".join(uv) + "\n",
        "m.note.step6": "\n".join(co) + "\n",
    }
