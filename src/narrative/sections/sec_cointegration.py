# src/narrative/sections/sec_cointegration.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec,
    lookup,
    md_basic_to_tex,
    include_table_tex,
    narr_call,
)

def _fmt2(x: Any) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "NA"

def _fmt_p(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.3f}"


def render_sec_cointegration(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    # ---------- metrics ----------
    coint = metrics_cache.get("m.coint.meta") or {}
    audit = metrics_cache.get("m.coint.audit") or {}
    vecm_meta = metrics_cache.get("m.vecm.meta") or {}

    choice = coint.get("choice") or coint.get("selected_model") or "NA"
    rank = coint.get("rank")
    if rank is None:
        rank = coint.get("johansen_rank")
    rank = rank if rank is not None else "NA"

    # optional: store some p-values if present
    eg_p = coint.get("eg_p") or coint.get("engle_granger_p") or coint.get("pvalue")
    joh_case = coint.get("deterministic_case") or coint.get("johansen_case") or coint.get("case")
    joh_rank_sel = coint.get("rank_selected") or rank

    note = metrics_cache.get("m.note.step6") or {}
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = note_md.replace("−", "-")

    # ---------- artefacts (tables) ----------
    tbl_eg = lookup(manifest, "tables", "tbl.coint.eg")
    tbl_joh = lookup(manifest, "tables", "tbl.coint.johansen")
    tbl_choice = lookup(manifest, "tables", "tbl.coint.var_vs_vecm_choice")
    tbl_vecm = lookup(manifest, "tables", "tbl.vecm.params")

    lines: list[str] = []

    # ============================================================
    # SECTION : Cadre théorique (cointégration / VECM)
    # ============================================================
    lines += [
        r"\section{Cointégration et dynamique de long terme : le cadre VECM}",
        "",
        md_basic_to_tex(
            "Les séries macroéconomiques et démographiques sont fréquemment non stationnaires mais évoluent néanmoins selon des relations "
            "de long terme stables. La théorie de la cointégration concilie cette non-stationnarité avec l’existence d’équilibres structurels. "
            "Dans le cadre de la croissance naturelle, la cointégration traduit l’existence d’un lien durable entre natalité, mortalité et facteurs "
            "économiques, malgré des fluctuations de court terme."
        ),
        "",
        r"\subsection*{Motivation théorique de la cointégration}",
        md_basic_to_tex(
            "Soient deux séries $X_t$ et $Y_t$ intégrées d’ordre 1. Individuellement, elles dérivent, mais il peut exister une combinaison "
            "linéaire stationnaire $Z_t$ : dans ce cas, les séries sont cointégrées et un mécanisme de rappel empêche la divergence illimitée."
        ),
        "",
        r"\begin{equation}",
        r"Z_t = Y_t - \beta X_t \sim I(0)",
        r"\end{equation}",
        "",
        r"\subsection*{Lien entre VAR et VECM}",
        md_basic_to_tex(
            "Un VAR($p$) en variables $I(1)$ peut se réécrire en VECM, où la matrice $\Pi=\alpha\beta'$ concentre l’information de long terme : "
            r"$\beta$ décrit les relations d’équilibre, $\alpha$ la vitesse d’ajustement."
        ),
        "",
        r"\begin{equation}",
        r"\Delta Y_t = \Pi Y_{t-1} + \sum_{i=1}^{p-1}\Gamma_i \Delta Y_{t-i} + \varepsilon_t",
        r"\end{equation}",
        "",
        r"\subsection*{Test d’Engle–Granger}",
        md_basic_to_tex(
            "Procédure en deux étapes : estimation de la relation de long terme (MCO), puis test de racine unitaire sur les résidus. "
            "Limite : identification d’un seul vecteur de cointégration et biais de normalisation."
        ),
        "",
        r"\subsection*{Test de Johansen : cadre général}",
        md_basic_to_tex(
            "La méthode de Johansen détermine le rang de cointégration $r$ dans un système multivarié via l’analyse spectrale de $\\Pi$. "
            "Les conclusions dépendent du traitement de la constante et de la tendance (choix déterministe)."
        ),
        "",
        r"\subsection*{Statistiques de test}",
        "",
        r"\begin{equation}",
        r"\lambda_{\text{trace}}(r) = -T\sum_{i=r+1}^{k}\ln(1-\hat{\lambda}_i)",
        r"\end{equation}",
        r"\begin{equation}",
        r"\lambda_{\max}(r,r+1) = -T\ln(1-\hat{\lambda}_{r+1})",
        r"\end{equation}",
        "",
        r"\subsection*{Choix de la composante déterministe}",
        md_basic_to_tex(
            "Cinq cas standards (constante/tendance restreinte ou non). Un mauvais choix déforme le rang estimé. "
            "La décision doit rester cohérente avec le profil des séries et les diagnostics amont."
        ),
        "",
        r"\subsection*{Interprétation du rang}",
        md_basic_to_tex(
            r"$r=0$ : pas de relation de long terme (VAR en différences). "
            r"$r=1$ : équilibre unique (souvent plausible en démographie). "
            r"$r>1$ : plusieurs équilibres structurels."
        ),
        "",
        r"\subsection*{Estimation du VECM}",
        "",
        r"\begin{equation}",
        r"\Delta Y_t = \alpha\beta'Y_{t-1} + \sum_{i=1}^{p-1}\Gamma_i\Delta Y_{t-i} + \varepsilon_t",
        r"\end{equation}",
        "",
        r"\subsection*{Vitesse d’ajustement}",
        md_basic_to_tex(
            "Si un coefficient d’ajustement $\\alpha_i < 0$ et significatif, la variable contribue au retour vers l’équilibre. "
            "Une mesure intuitive du temps moyen de correction est $\\tau = 1/\\lvert\\alpha_i\\rvert$."
        ),
        "",
        r"\begin{equation}",
        r"\tau=\frac{1}{|\alpha|}",
        r"\end{equation}",
        "",
        r"\subsection*{Implications économiques}",
        md_basic_to_tex(
            "La cointégration implique une cohérence structurelle de long terme : absence de divergence illimitée, "
            "et articulation explicite entre dynamique de court terme et équilibre démographique. "
            "Le VECM est l’outil central pour formaliser cette articulation."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Résultats empiriques (Step6)
    # ============================================================
    lines += [
        r"\section{Résultats cointégration et VECM}",
        "",
        md_basic_to_tex(
            f"Synthèse quantitative : choix **{choice}** ; rang (Johansen) **{joh_rank_sel}**."
            + (f" Cas déterministe : **{joh_case}**." if joh_case not in (None, "", "NA") else "")
            + (f" Engle–Granger p={_fmt_p(eg_p)}." if eg_p not in (None, "", "NA") else "")
        ),
        narr_call("m.coint.meta"),
        "",
    ]

    if tbl_eg:
        lines += [
            r"\paragraph{Tableau 1 — Test d’Engle–Granger}",
            md_basic_to_tex(
                "Lecture : le test porte sur la stationnarité des résidus de la relation de long terme. "
                "Un rejet de racine unitaire sur les résidus soutient la cointégration. "
                "Ce résultat doit être cohérent avec Johansen (méthode système) et le choix déterministe."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_eg, caption="Test de cointégration d’Engle–Granger", label="tab:tbl-coint-eg"),
            narr_call("tbl.coint.eg"),
            "",
        ]

    if tbl_joh:
        lines += [
            r"\paragraph{Tableau 2 — Test de Johansen (rang de cointégration)}",
            md_basic_to_tex(
                "Lecture : déterminer $r$ via trace et valeur propre maximale. "
                "La conclusion dépend fortement du traitement de la constante/tendance et du nombre de retards. "
                "Une conclusion fragile (changements de rang selon les cas) doit être traitée comme un signal d’incertitude, "
                "pas comme un fait robuste."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_joh, caption="Tests de cointégration de Johansen", label="tab:tbl-coint-johansen"),
            narr_call("tbl.coint.johansen"),
            "",
        ]

    if tbl_choice:
        lines += [
            r"\paragraph{Tableau 3 — Arbitrage VAR en différences vs VECM}",
            md_basic_to_tex(
                "Lecture : si $r=0$, un VAR en différences est cohérent. "
                "Si $r\\ge 1$, le VECM est théoriquement supérieur car il réintroduit l’information de long terme via le terme de correction d’erreur. "
                "La décision finale doit intégrer la stabilité, la parcimonie et la cohérence économique."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_choice, caption="Choix du modèle : VAR en différences ou VECM", label="tab:tbl-coint-choice"),
            narr_call("tbl.coint.var_vs_vecm_choice"),
            "",
        ]

    if tbl_vecm:
        lines += [
            r"\paragraph{Tableau 4 — Paramètres du VECM (long terme et ajustement)}",
            md_basic_to_tex(
                "Lecture : $\\beta$ (cointégration) décrit l’équilibre de long terme ; $\\alpha$ la vitesse d’ajustement. "
                "Un ajustement significatif valide l’existence d’un mécanisme de rappel. "
                "Des coefficients d’ajustement faibles ou non significatifs suggèrent un équilibre économiquement peu opérant "
                "ou une spécification inadéquate (rang, retards, déterministes)."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_vecm, caption="Paramètres estimés du modèle VECM", label="tab:tbl-vecm-params"),
            narr_call("tbl.vecm.params"),
            "",
        ]

    # Audit (if any)
    if audit:
        lines += [
            md_basic_to_tex("**Audit de robustesse**"),
            md_basic_to_tex(
                "Les sorties d’audit cadrent la robustesse du rang et la cohérence du choix VAR/VECM. "
                "Toute instabilité du rang ou dépendance excessive au cas déterministe doit être explicitée."
            ),
            "",
            narr_call("m.coint.audit"),
            "",
        ]

    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée**"),
            md_basic_to_tex(
                "Cette note doit rester cohérente avec : (i) EG/Johansen, (ii) le cas déterministe, "
                "(iii) le choix VAR/VECM, (iv) les paramètres VECM (signes/ajustement). "
                "Toute affirmation de “relation de long terme” exige une conclusion robuste sur $r$."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step6"),
            "",
        ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            f"Décision long terme : **{choice}** (rang Johansen = **{joh_rank_sel}**). "
            "En présence de cointégration, le VECM est la spécification cohérente : il sépare l’ajustement de court terme "
            "du mécanisme de rappel vers l’équilibre. "
            "La solidité de l’interprétation dépend directement de la robustesse du rang et du choix déterministe."
        ),
        narr_call("m.coint.meta"),
        "",
    ]

    # Optional vecm meta reference if present
    if vecm_meta:
        lines += [narr_call("m.vecm.meta"), ""]

    return "\n".join(lines).strip() + "\n"
