from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from src.narrative.sections.base import (
    SectionSpec,
    lookup,
    md_basic_to_tex,
    include_table_tex,
    include_figure,
    narr_call,
)

def _fmt_p(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "NA"
    if v < 1e-4:
        return f"{v:.2e}"
    return f"{v:.3f}"


def render_sec_stationarity(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    tsds = metrics_cache.get("m.diag.ts_vs_ds") or {}
    verdict = tsds.get("verdict", "NA")
    p_c = tsds.get("adf_p_c")
    p_ct = tsds.get("adf_p_ct")

    note = (metrics_cache.get("m.note.step3") or {})
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = note_md.replace("−", "-")

    # artefacts
    fig_acf = lookup(manifest, "figures", "fig.diag.acf_level")
    fig_pacf = lookup(manifest, "figures", "fig.diag.pacf_level")

    tbl_adf = lookup(manifest, "tables", "tbl.diag.adf")
    tbl_tsds = lookup(manifest, "tables", "tbl.diag.ts_vs_ds_decision")
    tbl_ljung = lookup(manifest, "tables", "tbl.diag.ljungbox_diff")
    tbl_band = lookup(manifest, "tables", "tbl.diag.band_df")

    lines: list[str] = []

    # ============================================================
    # SECTION : Cadre théorique (texte structurant)
    # ============================================================
    lines += [
        r"\section{Diagnostics statistiques et stationnarité}",
        "",
        md_basic_to_tex(
            "La stationnarité constitue une condition centrale de l’économétrie des séries temporelles. "
            "Une série non stationnaire utilisée sans précaution conduit à des régressions fallacieuses, "
            "avec des coefficients apparemment significatifs mais dénués d’interprétation économique. "
            "Cette section vise à caractériser la nature stochastique de la croissance naturelle et à déterminer "
            "les transformations nécessaires avant toute modélisation dynamique."
        ),
        "",
        r"\subsection*{Notion formelle de stationnarité}",
        md_basic_to_tex(
            "En pratique, on retient la stationnarité au second ordre : (i) une espérance constante, "
            "(ii) une variance finie et constante, et (iii) une autocovariance dépendant uniquement du décalage. "
            "La non-stationnarité viole ces conditions et empêche l’utilisation directe des outils ARMA classiques."
        ),
        "",
        r"\subsection*{Processus TS versus DS : enjeu conceptuel}",
        md_basic_to_tex(
            "Deux familles doivent être distinguées. "
            "Processus à tendance déterministe (TS) : les chocs sont transitoires et la série revient vers une tendance. "
            "Processus à tendance stochastique (DS) : les chocs sont permanents et la série ne revient pas vers un niveau déterminé. "
            "La conséquence est opérationnelle : un TS se détrend, un DS se différencie."
        ),
        "",
        r"\begin{equation}",
        r"Y_t = \alpha + \beta t + u_t",
        r"\end{equation}",
        r"\begin{equation}",
        r"Y_t = Y_{t-1} + \varepsilon_t",
        r"\end{equation}",
        "",
        r"\subsection*{Analyse corrélographique : ACF et PACF}",
        md_basic_to_tex(
            "Avant tout test formel, ACF/PACF fournissent une information qualitative sur la persistance. "
            "Une décroissance lente de l’ACF est typique d’une non-stationnarité, alors qu’une coupure rapide suggère une stationnarité. "
            "Ces outils restent indicatifs : la décision doit être statistique."
        ),
        "",
        r"\subsection*{Test de Dickey-Fuller augmenté (ADF)}",
        md_basic_to_tex(
            "Le test ADF est l’outil central de détection de racine unitaire. Il estime une régression en différence "
            "incluant des retards pour purger l’autocorrélation des résidus. Trois spécifications sont à considérer : "
            "sans constante, avec constante, et avec constante + tendance. "
            "Hypothèses : $H_0$ racine unitaire, $H_1$ stationnarité. "
            "Le choix du nombre de retards est critique : trop faible biaise le test, trop élevé réduit la puissance."
        ),
        "",
        r"\begin{equation}",
        r"\Delta Y_t = \phi Y_{t-1} + \sum_{i=1}^{k}\gamma_i \Delta Y_{t-i} + \varepsilon_t",
        r"\end{equation}",
        "",
        r"\subsection*{3.6 Test de la bande de Dickey-Fuller (robustesse)}",
        md_basic_to_tex(
            "Le test de bande examine la stabilité de la conclusion sur un intervalle de retards/spécifications. "
            "Il détecte les cas fragiles où le rejet/non-rejet de $H_0$ dépend fortement de la paramétrisation."
        ),
        "",
        r"\subsection*{Limites des tests de racine unitaire}",
        md_basic_to_tex(
            "Limites classiques : faible puissance, sensibilité aux ruptures structurelles, confusion entre mémoire longue et racine unitaire. "
            "Un non-rejet de $H_0$ n’est pas une preuve définitive de non-stationnarité."
        ),
        "",
        r"\subsection*{Décision finale et transformation de la série}",
        md_basic_to_tex(
            "La décision économétrique synthétise les diagnostics. Trois cas : stationnarité en niveau (rien), "
            "stationnarité autour d’une tendance (détrendage), stationnarité en différence (différenciation). "
            "En DS, la transformation opérationnelle est :"
        ),
        "",
        r"\begin{equation}",
        r"\Delta Y_t = Y_t - Y_{t-1}",
        r"\end{equation}",
        "",
        r"\subsection*{Implications pour la suite de l’analyse}",
        md_basic_to_tex(
            "La stationnarité pilote l’identification AR/MA, l’estimation univariée, la validité multivariée et la cointégration. "
            "Une erreur ici se propage mécaniquement à toutes les étapes suivantes."
        ),
        "",
    ]

    # ============================================================
    # SECTION 2 : Résultats empiriques (artefacts + analyses)
    # ============================================================
    lines += [
        r"\section{Résultats des diagnostics}",
        "",
        md_basic_to_tex(
            f"Synthèse quantitative : **verdict ADF-only = {verdict}** "
            f"(ADF(c) p={_fmt_p(p_c)}, ADF(ct) p={_fmt_p(p_ct)}). "
            "Cette décision pilote directement le traitement du niveau et le degré d’intégration pour ARIMA/VAR/VECM."
        ),
        narr_call("m.diag.ts_vs_ds"),
        "",
    ]

    if fig_acf:
        lines += [
            r"\paragraph{Figure 1 — ACF (niveau)}",
            md_basic_to_tex(
                "Lecture : une persistance élevée et une décroissance lente sont compatibles avec non-stationnarité ou mémoire longue. "
                "L’interprétation doit être corroborée par ADF et la robustesse (bande)."
            ),
            "",
            include_figure(fig_rel=fig_acf, caption="fig.diag.acf_level", label="fig:fig-diag-acf-level"),
            narr_call("fig.diag.acf_level"),
            "",
        ]

    if fig_pacf:
        lines += [
            r"\paragraph{Figure 2 — PACF (niveau)}",
            md_basic_to_tex(
                "Lecture : la PACF informe sur la structure autorégressive potentielle, mais elle devient trompeuse en présence de non-stationnarité. "
                "Elle sert ici de repérage qualitatif, pas de décision."
            ),
            "",
            include_figure(fig_rel=fig_pacf, caption="fig.diag.pacf_level", label="fig:fig-diag-pacf-level"),
            narr_call("fig.diag.pacf_level"),
            "",
        ]

    if tbl_adf:
        lines += [
            r"\paragraph{Tableau 1 — Test ADF (diagnostic principal)}",
            md_basic_to_tex(
                "Lecture : comparer les p-values et la cohérence entre spécifications (avec/sans tendance). "
                "Une conclusion instable selon la spécification signale une frontière TS/DS, une rupture non modélisée, ou une persistance élevée."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_adf, caption="tbl.diag.adf", label="tab:tbl-diag-adf"),
            narr_call("tbl.diag.adf"),
            "",
        ]

    if tbl_band:
        lines += [
            r"\paragraph{Tableau 2 — Bande Dickey-Fuller (robustesse)}",
            md_basic_to_tex(
                "Lecture : vérifier la stabilité de la conclusion sur un intervalle de retards. "
                "Une conclusion fragile impose prudence : la transformation retenue doit privilégier la robustesse plutôt que l’optimisme."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_band, caption="tbl.diag.band_df", label="tab:tbl-diag-band-df"),
            narr_call("tbl.diag.band_df"),
            "",
        ]

    if tbl_tsds:
        lines += [
            r"\paragraph{Tableau 3 — Décision TS vs DS (audit)}",
            md_basic_to_tex(
                "Lecture : ce tableau formalise la règle de décision opérationnelle. "
                "Il verrouille le traitement de la série : différenciation si DS, détrendage si TS, et aucun traitement si stationnaire en niveau."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_tsds, caption="tbl.diag.ts_vs_ds_decision", label="tab:tbl-diag-ts-vs-ds-decision"),
            narr_call("tbl.diag.ts_vs_ds_decision"),
            "",
        ]

    if tbl_ljung:
        lines += [
            r"\paragraph{Tableau 4 — Ljung–Box sur différence (contrôle)}",
            md_basic_to_tex(
                "Lecture : vérifier qu’après transformation (notamment différenciation), une autocorrélation résiduelle massive ne subsiste pas. "
                "Si l’autocorrélation persiste fortement, l’identification ARMA/ARIMA devra être plus structurée (retards, composantes)."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_ljung, caption="tbl.diag.ljungbox_diff", label="tab:tbl-diag-ljungbox-diff"),
            narr_call("tbl.diag.ljungbox_diff"),
            "",
        ]

    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée**"),
            md_basic_to_tex(
                "Cette note doit être strictement cohérente avec ADF, bande et tableau TS/DS. "
                "Toute mention de transformation retenue doit reprendre explicitement le verdict et les p-values clés."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step3"),
            "",
        ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            f"La décision opérationnelle est **{verdict}**. "
            "Elle fixe le traitement (niveau/détrendage/différenciation) et conditionne l’identification AR/MA, "
            "la stabilité des résidus et la validité des modèles ARIMA/VAR/VECM."
        ),
        narr_call("m.diag.ts_vs_ds"),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
