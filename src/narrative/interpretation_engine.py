# src/narrative/interpretation_engine.py
from __future__ import annotations
from typing import Any, Dict
from src.utils.run_reader import RunManager, read_metric_json


def _m(run_id: str, label: str) -> dict:
    """
    Lecture robuste d'une métrique JSON via le RunManager.
    """
    p = RunManager.get_artefact_path(label, run_id=run_id)
    if not p:
        return {}
    try:
        v = read_metric_json(p)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _model_type(order: Any) -> str:
    """
    Déduit AR / MA / ARMA / ARIMA à partir de (p,d,q).
    """
    if not isinstance(order, (list, tuple)) or len(order) != 3:
        return "ARIMA"

    p, d, q = order
    if d and d > 0:
        return "ARIMA"
    if p > 0 and q == 0:
        return "AR"
    if p == 0 and q > 0:
        return "MA"
    if p > 0 and q > 0:
        return "ARMA"
    return "ARIMA"


def build_snippets_from_run(*, run_id: str, y: str) -> Dict[str, str]:
    """
    Génère les snippets LaTeX dynamiques (interprétations)
    en fonction des métriques calculées.
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

    st = []
    st.append(r"\paragraph{Stationnarité.}")

    pvals = [p for p in (adf_c, adf_ct) if isinstance(p, (int, float))]
    pmin = min(pvals) if pvals else None

    if pmin is not None:
        if pmin < 0.05:
            st.append(
                rf"Le test de Dickey--Fuller augmenté rejette l’hypothèse de racine unitaire "
                rf"($p$-value minimale $\approx {pmin:.3f} < 0.05$). "
                r"La série peut être traitée comme stationnaire, ce qui autorise une modélisation en niveau "
                r"et suggère un mécanisme de retour vers une moyenne de long terme."
            )
        else:
            st.append(
                rf"Le test de Dickey--Fuller augmenté ne permet pas de rejeter l’hypothèse de racine unitaire "
                rf"($p$-value minimale $\approx {pmin:.3f} \ge 0.05$). "
                r"La série présente une non-stationnarité compatible avec un processus intégré, "
                r"justifiant une différenciation première ($d=1$)."
            )
    else:
        st.append(
            rf"Le verdict ADF-only est \textbf{{{verdict or 'NA'}}}. "
            r"Il pilote la transformation de la série avant modélisation."
        )

    # ============================================================
    # Modélisation univariée
    # ============================================================
    kp = (uni.get("key_points") or {})
    order = kp.get("order") or (uni.get("best") or {}).get("order")
    aic = kp.get("aic") or (uni.get("best") or {}).get("aic")
    bic = kp.get("bic") or (uni.get("best") or {}).get("bic")
    jb_p = kp.get("jb_p")
    lb_p = kp.get("lb_p")

    model_kind = _model_type(order)

    uv = []
    uv.append(r"\paragraph{Modélisation univariée et choix du modèle.}")

    if order:
        uv.append(
            rf"Le modèle retenu est un processus \textbf{{{model_kind}{tuple(order)}}}, "
            r"sélectionné par minimisation des critères d’information "
            + (
                rf"(AIC={aic:.2f}, BIC={bic:.2f})."
                if isinstance(aic, (int, float)) or isinstance(bic, (int, float))
                else "."
            )
        )

        if model_kind == "AR":
            uv.append(
                r"Cette spécification met en évidence une dynamique dominée par l’inertie : "
                r"les valeurs passées du solde naturel expliquent l’essentiel de son évolution courante, "
                r"traduisant une mémoire démographique marquée."
            )
        elif model_kind == "MA":
            uv.append(
                r"Le modèle suggère que la dynamique est principalement gouvernée par des chocs exogènes récents, "
                r"dont l’impact est transitoire et rapidement résorbé."
            )
        elif model_kind in ("ARMA", "ARIMA"):
            uv.append(
                r"La dynamique résulte d’un équilibre entre persistance endogène et chocs aléatoires, "
                r"la différenciation éventuelle permettant de stabiliser la série avant estimation."
            )
    else:
        uv.append(
            r"Le modèle univarié retenu n’est pas documenté de manière exploitable dans les métriques."
        )

    # Diagnostics résiduels
    if isinstance(jb_p, (int, float)) and jb_p < 0.05:
        uv.append(
            rf"Le test de Jarque--Bera rejette l’hypothèse de normalité des résidus "
            rf"($p$-value={jb_p:.3f}$). "
            r"Cette non-normalité est cohérente avec des épisodes extrêmes "
            r"(pics de mortalité, chutes de natalité), invitant à la prudence "
            r"dans l’interprétation des intervalles de prévision."
        )

    if isinstance(lb_p, (int, float)) and lb_p < 0.05:
        uv.append(
            rf"Le test de Ljung--Box met en évidence une autocorrélation résiduelle "
            rf"($p$-value={lb_p:.3f}$), "
            r"suggérant qu’une re-spécification du modèle pourrait être envisagée."
        )

    # ============================================================
    # Cointégration
    # ============================================================
    rank = coint.get("rank")
    choice = (coint.get("choice") or "").upper()

    co = []
    co.append(r"\paragraph{Cointégration et dynamique de long terme.}")

    if isinstance(rank, int) and rank > 0:
        co.append(
            rf"Le test de Johansen indique un rang de cointégration $r={rank}>0$. "
            r"Ce résultat justifie l’adoption d’un modèle VECM, mettant en évidence "
            r"une relation de long terme assimilable à une force de rappel : "
            r"les écarts à l’équilibre sont progressivement corrigés."
        )
    elif isinstance(rank, int) and rank == 0:
        co.append(
            r"Aucune relation de cointégration n’est détectée ($r=0$). "
            r"Une modélisation VAR en différences demeure appropriée pour analyser "
            r"la dynamique de court terme."
        )
    else:
        co.append(
            rf"Le pipeline indique un choix \textbf{{{choice or 'NA'}}}, "
            r"mais les informations de rang ne sont pas suffisamment renseignées."
        )

    # ============================================================
    # Snippets retournés (CLÉS COHÉRENTES AVEC normalize_key)
    # ============================================================
    return {
        # Section-level
        "sec_stationarity": "\n".join(st) + "\n",
        "sec_univariate": "\n".join(uv) + "\n",
        "sec_cointegration": "\n".join(co) + "\n",

        # Table-level (granularité fine)
        "tbl_diag_ts_vs_ds_decision": "\n".join(st) + "\n",
        "tbl_uni_arima": "\n".join(uv) + "\n",
        "tbl_coint_johansen": "\n".join(co) + "\n",
    }
