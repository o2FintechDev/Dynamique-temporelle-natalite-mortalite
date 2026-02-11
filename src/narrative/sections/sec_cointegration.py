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
    # SECTION 1 : Cadre théorique (cointégration / VECM)
    # ============================================================

    lines += [
        r"\section{Cointégration et dynamique de long terme : le cadre VECM}",
        "",
        md_basic_to_tex(
            "Les séries macroéconomiques et démographiques sont fréquemment caractérisées par une non-stationnarité "
            "structurelle, résultant de tendances déterministes, de dérives stochastiques ou de ruptures de régime. "
            "Dans ce contexte, une modélisation purement en différences, telle qu’un VAR stationnaire, "
            "supprime l’information contenue dans les trajectoires de long terme.\n\n"

            "La théorie de la cointégration permet de concilier cette non-stationnarité individuelle "
            "avec l’existence éventuelle de relations d’équilibre structurelles. "
            "Il peut être observé que plusieurs variables évoluent selon des tendances propres "
            "tout en maintenant une combinaison linéaire stable au cours du temps.\n\n"

            "Dans le système étudié — Croissance Naturelle, Nombre de Mariages, IPC et Masse Monétaire M3 — "
            "des trajectoires de long terme sont observées. "
            "Il convient dès lors de déterminer si ces trajectoires sont indépendantes "
            "ou si elles sont liées par une ou plusieurs relations structurelles."
        ),
        "",

        r"\subsection*{Motivation du passage du VAR au VECM}",
        md_basic_to_tex(
            "Lorsque les variables sont intégrées d’ordre 1, un VAR estimé en différences "
            "ne capture que la dynamique de court terme. "
            "La composante de tendance est éliminée et toute relation d’équilibre potentielle disparaît.\n\n"

            "Le modèle à correction d’erreurs (VECM) constitue une reformulation du VAR "
            "permettant d’introduire explicitement un terme d’ajustement vers l’équilibre. "
            "Il représente la limite théorique de la mémoire courte : "
            "les déséquilibres de court terme sont corrigés par un mécanisme de rappel de long terme.\n\n"

            "Ainsi, alors que le VAR modélise la propagation des chocs, "
            "le VECM modélise également la force qui empêche une divergence illimitée."
        ),
        "",

        r"\subsection*{Notion formelle de cointégration}",
        md_basic_to_tex(
            "Soient deux séries $X_t$ et $Y_t$ intégrées d’ordre 1. "
            "Individuellement, ces séries présentent une dérive. "
            "Il peut néanmoins exister une combinaison linéaire stationnaire :"
        ),
        "",
        r"\begin{equation}",
        r"Z_t = Y_t - \beta X_t \sim I(0)",
        r"\end{equation}",
        "",

        md_basic_to_tex(
            "Dans ce cas, les séries sont dites cointégrées.\n\n"

            "Il convient de distinguer strictement :\n"
            "• Corrélation : co-mouvement instantané.\n"
            "• Cointégration : relation d’équilibre de long terme.\n\n"

            "La cointégration n’implique pas causalité. "
            "Elle indique qu’un équilibre structurel contraint l’évolution conjointe des variables."
        ),
        "",

        r"\subsection*{Approche d’Engle–Granger}",
        md_basic_to_tex(
            "Une première formalisation empirique de la cointégration a été proposée par Engle et Granger (1987). "
            "La méthode repose sur une procédure en deux étapes.\n\n"

            "1) Estimation par moindres carrés d’une relation de long terme en niveau.\n"
            "2) Application d’un test de racine unitaire sur les résidus estimés.\n\n"

            "Si les résidus sont stationnaires, une relation de cointégration est identifiée."
        ),
        "",
        r"\begin{equation}",
        r"Y_t = \beta_0 + \beta_1 X_t + u_t",
        r"\end{equation}",
        "",
        
        md_basic_to_tex(
            "La stationnarité de $u_t$ est alors testée.\n\n"

            "Cette approche présente cependant des limites majeures :\n"
            "• Identification d’un unique vecteur de cointégration.\n"
            "• Sensibilité à la normalisation.\n"
            "• Inadaptation aux systèmes multivariés.\n\n"

            "Dans un cadre à quatre variables, une approche système complète s’impose."
        ),
        "",

        r"\subsection*{Approche de Johansen : analyse spectrale du système}",
        md_basic_to_tex(
            "La méthode de Johansen repose sur la décomposition spectrale de la matrice "
            "issue de la représentation VAR en niveau. "
            "Le rang de cointégration $r$ est déterminé à partir des valeurs propres estimées.\n\n"

            "Tester la cointégration revient à déterminer le nombre de relations "
            "de long terme indépendantes reliant les variables du système."
        ),
        "",

        r"\begin{equation}",
        r"\lambda_{\text{trace}}(r) = -T\sum_{i=r+1}^{k}\ln(1-\hat{\lambda}_i)",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        r"\lambda_{\max}(r,r+1) = -T\ln(1-\hat{\lambda}_{r+1})",
        r"\end{equation}",
        "",

        md_basic_to_tex(
            "Ces statistiques reposent sur les valeurs propres $\hat{\lambda}_i$ de la matrice du système. "
            "Plus ces valeurs sont élevées, plus l’existence d’un vecteur de cointégration est probable.\n\n"

            "Une attention particulière doit être portée au choix de la composante déterministe "
            "(constante libre, constante restreinte, tendance). "
            "Une dérive mal spécifiée peut conduire à une mauvaise estimation du rang."
        ),
        "",

        r"\subsection*{Représentation VECM}",
        "",
        r"\begin{equation}",
        r"\Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{i=1}^{p-1}\Gamma_i \Delta Y_{t-i} + \varepsilon_t",
        r"\end{equation}",
        "",

        md_basic_to_tex(
            "La matrice $\beta$ contient les vecteurs de cointégration "
            "et décrit les relations d’équilibre.\n\n"

            "La matrice $\alpha$ contient les coefficients d’ajustement. "
            "Un coefficient $\alpha_i$ négatif et significatif indique "
            "que la variable contribue au rétablissement de l’équilibre.\n\n"

            "Le VECM est estimé à partir des séries en niveau "
            "afin de préserver l’information de long terme."
        ),
        "",

        r"\subsection*{Temps moyen d’ajustement}",
        "",
        r"\begin{equation}",
        r"\tau = \frac{1}{|\alpha|}",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Le paramètre $\tau$ fournit une approximation du temps moyen "
            "nécessaire pour corriger un déséquilibre.\n\n"

            "Un choc de court terme — sanitaire, géopolitique ou monétaire — "
            "est ainsi inscrit dans une dynamique de long terme. "
            "La correction vers l’équilibre structurel est progressive."
        ),
        "",

        r"\subsection*{Interprétation économique du rang}",
        md_basic_to_tex(
        r"$r=0$ : absence de relation de long terme (VAR en différences). "
            r"$r=1$ : existence d’un équilibre structurel unique. "
            r"$r>1$ : présence de plusieurs équilibres imbriqués.\n\n"

            "Dans un système à quatre variables, un rang strictement positif "
            "signifie qu’une combinaison stable relie la dynamique démographique "
            "aux variables économiques.\n\n"

            "Le VECM constitue ainsi l’outil central permettant d’articuler "
            "dynamique de court terme et équilibre de long terme."
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
