# src/narrative/renderer.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
from datetime import datetime, timezone

from src.utils.run_writer import RunWriter


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def append_to_narrative(runs_dir: str, run_id: str, step_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrichit narrative.json sous blocks[step_name] (append-only logique).
    """
    rw = RunWriter(Path(runs_dir), run_id)
    p = rw.paths.narrative_path
    obj = json.loads(p.read_text(encoding="utf-8"))

    blocks = obj.get("blocks", {})
    step_block = blocks.get(step_name, {})
    # merge simple (step-level). Si tu veux list-append, fais-le côté data.
    step_block.update(data)
    step_block["updated_at"] = _utc_ts()
    blocks[step_name] = step_block

    obj["blocks"] = blocks
    obj["updated_at"] = _utc_ts()

    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return obj

def render_anthropology(*, facts: Dict[str, Any]) -> Dict[str, Any]:
    y = facts.get("y", "Croissance_Naturelle")
    refs = facts.get("_refs") or []

    tsds = facts.get("m.diag.ts_vs_ds") or {}
    uni = facts.get("m.uni.best") or {}
    var = facts.get("m.var.meta") or {}
    coint = facts.get("m.coint.meta") or {}

    # ---- TS vs DS ----
    verdict = tsds.get("verdict") or tsds.get("key_points", {}).get("verdict") or "NA"
    p_c = tsds.get("adf_p_c") or tsds.get("key_points", {}).get("adf_p_c")
    p_ct = tsds.get("adf_p_ct") or tsds.get("key_points", {}).get("adf_p_ct")
    p_pp = tsds.get("pp_p") or tsds.get("key_points", {}).get("pp_p")

   

    # ---- Univarié ----
    # m.uni.best contient directement order/aic/bic (pas de champ "best")
    best = uni if isinstance(uni, dict) else {}
    order = best.get("order")
    aic = best.get("aic")
    bic = best.get("bic")
    family = best.get("family")

    # ---- VAR ----
    def _as_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return {}

    def _pick(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    var = _as_dict(facts.get("m.var.meta") or {})
    note5 = _as_dict(facts.get("m.note.step5") or {})
    audit = _as_dict(facts.get("m.var.audit") or {})

    kp5 = _as_dict(note5.get("key_points") or {})
    audit_data = _as_dict(audit.get("data") or {})

    # Vars: meta -> step5.key_points -> audit.data
    vars_ = _pick(
        var.get("vars"),
        kp5.get("vars"),
        var.get("variables"),
        kp5.get("variables"),
        audit_data.get("vars"),
    )

    # Lag: meta -> step5.key_points (noms alternatifs)
    p_var = _pick(
        var.get("selected_lag_aic"),
        kp5.get("selected_lag_aic"),
        var.get("lag_aic"),
        kp5.get("lag_aic"),
        var.get("p_aic"),
        kp5.get("p_aic"),
        var.get("selected_lag"),
        kp5.get("selected_lag"),
        var.get("lag"),
        kp5.get("lag"),
    )

    # nobs: meta -> step5.key_points -> audit.data
    nobs_var = _pick(
        var.get("nobs"),
        kp5.get("nobs"),
        var.get("n_obs"),
        kp5.get("n_obs"),
        audit_data.get("nobs_stationary"),
        audit_data.get("nobs_after_dropna"),
        audit_data.get("nobs_raw"),
    )


    # ---- Cointégration ----
    vars_c = coint.get("vars")
    rank = coint.get("rank")
    choice = coint.get("choice")  # dans ton cointegration_pack c’est dans note6.key_points, pas forcément dans meta
    # robust: chercher choix aussi dans m.note.step6 si l'utilisateur l'ajoute dans facts plus tard
    if choice is None:
        # fallback: si meta ne contient pas "choice"
        if isinstance(coint, dict) and "rank" in coint:
            choice = "VECM" if int(coint.get("rank") or 0) > 0 else "VAR_diff"


    # ---- Narratif (Todd-like) ----
    lines = []
    lines.append(f"**Étape 7 — Lecture anthropologique (Todd) — {y}**")
    lines.append("")
    lines.append("### 1) Propriétés de la série et implication « structurelle »")
    lines.append(f"- **Régime statistique (TS vs DS)** : **{verdict}**.")
    if p_c is not None and p_ct is not None:
        lines.append(f"  - ADF(c) p={float(p_c):.3g}, ADF(ct) p={float(p_ct):.3g}"
                     + (f", PP p={float(p_pp):.3g}." if p_pp is not None else ", PP indisponible."))
    lines.append(
        "- **Interprétation** : une série plutôt TS suggère un retour vers une trajectoire (chocs transitoires), "
        "alors qu’une DS suggère des ruptures/politiques/événements qui déplacent durablement la dynamique."
    )

    lines.append("")
    lines.append("### 2) Dynamique courte (cycle) — lecture « conjoncturelle »")
    if order and aic is not None and bic is not None:
        lines.append(f"- **ARIMA** : famille={family}, ordre={order}, AIC={float(aic):.2f}, BIC={float(bic):.2f}.")
        lines.append(
            "- **Interprétation** : capte la persistance de court terme (inertie démographique), utile pour dater "
            "les phases de dégradation/amélioration de la croissance naturelle."
        )
    else:
        lines.append("- **ARIMA** : non disponible (étape 4 non exécutée ou métriques absentes).")

    lines.append("")
    lines.append("### 3) Système interne (niveau/tendance/saisonnalité) — lecture « mécanismes »")
    if vars_ is not None and (p_var is not None or nobs_var is not None):
        lag_txt = f"lag AIC p={int(p_var)}" if p_var is not None else "lag non renseigné"
        nobs_txt = f"{int(nobs_var)}" if nobs_var is not None else "NA"
        lines.append(f"- **VAR** : variables={vars_}, {lag_txt}, nobs={nobs_txt}.")
        lines.append(
            "- **Interprétation** : mesure la propagation des chocs entre composantes STL. "
            "Un choc de tendance peut contaminer le niveau; la saisonnalité est souvent plus « mécanique »."
        )
        lines.append(
            "- **Prudence** : Granger = dépendance prédictive, pas causalité sociologique."
        )
    else:
        lines.append("- **VAR** : non disponible (étape 5 non exécutée ou métriques absentes).")

    lines.append("")
    lines.append("### 4) Long terme (régularités) — lecture « structure vs rupture »")
    if vars_c and rank is not None:
        lines.append(f"- **Johansen** : variables={vars_c}, rang={int(rank)} ⇒ choix **{choice}**.")
        if int(rank) > 0:
            lines.append(
                "- **Interprétation** : existence d’une ou plusieurs relations de long terme (cohérence structurelle) ; "
                "l’ajustement (alpha) indique la vitesse de retour après choc."
            )
        else:
            lines.append(
                "- **Interprétation** : absence de relation de long terme détectée ; dynamique surtout en différences "
                "(ruptures ou tendances non co-mouvantes)."
            )
    else:
        lines.append("- **Cointégration/VECM** : non disponible (étape 6 non exécutée ou meta absente).")

    lines.append("")
    lines.append("### 5) Ancrage historique")
    lines.append(
        "- **2008–2009** : choc macro → report des naissances, incertitude économique. "
        "- **2020–2021** : COVID → surmortalité + effets de calendrier sur naissances, puis rattrapages partiels. "
        "- **Post-2022** : inflation/énergie → pression sur fécondité, vieillissement → hausse tendancielle des décès."
    )

    if refs:
        lines.append("")
        lines.append(f"_Artefacts run utilisés_ : {', '.join(refs)}")

    md = "\n".join(lines)

    return {
        "markdown": md,
        "key_points": {
            "y": y,
            "ts_vs_ds_verdict": verdict,
            "arima_order": order,
            "var_lag_aic": p_var,
            "coint_rank": rank,
            "coint_choice": choice,
        },
    }
