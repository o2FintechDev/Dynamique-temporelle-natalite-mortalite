# src/narrative/sections/sec_stationarity.py
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
    tbl_acfp  = lookup(manifest, "tables", "tbl.diag.acf_pacf")
    tbl_ljung = lookup(manifest, "tables", "tbl.diag.ljungbox_diff")
    tbl_band = lookup(manifest, "tables", "tbl.diag.band_df")

    lines: list[str] = []

    # ===========================
    # SECTION 1 : Cadre théorique
    # ===========================
    lines += [
        r"\section{Tests de racine unitaire et propriétés temporelles}",
        "",
        md_basic_to_tex(
            "L’analyse des séries temporelles repose sur une condition fondamentale : la stationnarité. "
            "L’utilisation de séries non stationnaires sans transformation appropriée conduit à des régressions fallacieuses, "
            "dans lesquelles des relations statistiques apparemment significatives ne correspondent à aucun lien économique réel. "
            "Cette section vise à caractériser la nature stochastique de la croissance naturelle française "
            "et à déterminer les transformations nécessaires avant toute modélisation dynamique."
        ),
        "",
        r"\subsection*{Stationnarité : définition et portée économétrique}",
        md_basic_to_tex(
            "En pratique, une série est dite stationnaire au second ordre si son espérance est constante dans le temps, "
            "si sa variance est finie et stable, et si sa structure de dépendance temporelle dépend uniquement du décalage "
            "entre observations et non du temps calendaire. "
            "Ces propriétés garantissent la validité des outils classiques de l’économétrie des séries temporelles "
            "(ARMA, VAR, tests statistiques usuels).\n\n"
            "Dans le cas de la croissance naturelle, la stationnarité ne peut être supposée a priori. "
            "Les évolutions de long terme liées au vieillissement de la population, aux transitions démographiques "
            "ou aux chocs sanitaires majeurs sont susceptibles d’introduire des composantes persistantes "
            "incompatibles avec une stationnarité stricte en niveau."
        ),
        "",
        r"\subsection*{Tendance déterministe versus tendance stochastique}",
        md_basic_to_tex(
            "Une distinction conceptuelle essentielle oppose deux types de dynamiques de long terme. "
            "Dans un processus à tendance déterministe (TS), la série fluctue autour d’une trajectoire prévisible : "
            "les chocs sont transitoires et la série tend à revenir vers sa tendance. "
            "À l’inverse, dans un processus à tendance stochastique (DS), les chocs ont des effets permanents "
            "et modifient durablement le niveau de la série."
        ),
        "",
        r"\begin{equation}",
        r"Y_t = \alpha + \beta t + u_t",
        r"\end{equation}",
        r"\begin{equation}",
        r"Y_t = Y_{t-1} + \varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "La distinction est économétriquement décisive : un processus TS doit être détrendé, "
            "tandis qu’un processus DS requiert une différenciation. "
            "Dans un contexte démographique, une dynamique DS signifie que certains chocs "
            "(politiques familiales, crises sanitaires, changements de comportements de fécondité) "
            "laissent des traces durables sur la trajectoire de la population."
        ),
        "",
        r"\subsection*{Premiers diagnostics : corrélogrammes ACF et PACF}",
        md_basic_to_tex(
            "Avant toute procédure formelle, l’analyse des fonctions d’autocorrélation (ACF) "
            "et d’autocorrélation partielle (PACF) fournit une lecture qualitative de la persistance. "
            "Une décroissance lente de l’ACF est typiquement associée à une non-stationnarité, "
            "alors qu’une coupure rapide suggère un processus stationnaire.\n\n"
            "Ces outils ne permettent pas de trancher définitivement, "
            "mais ils orientent l’interprétation des tests de racine unitaire "
            "et aident à détecter des structures de dépendance incompatibles "
            "avec une stationnarité en niveau."
        ),
        "",
        r"\subsection*{Test de Dickey--Fuller augmenté (ADF)}",
        md_basic_to_tex(
            "Le test de Dickey--Fuller augmenté constitue l’outil central de détection de racine unitaire. "
            "Il repose sur l’estimation d’une régression en différence incluant des retards "
            "afin de corriger l’autocorrélation des résidus. "
            "L’hypothèse nulle correspond à la présence d’une racine unitaire, "
            "tandis que l’hypothèse alternative indique la stationnarité."
        ),
        "",
        r"\begin{equation}",
        r"\Delta Y_t = \phi Y_{t-1} + \sum_{i=1}^{k} \gamma_i \Delta Y_{t-i} + \varepsilon_t",
        r"\end{equation}",
        "",

        md_basic_to_tex(
            "Le choix du nombre de retards est crucial : un nombre insuffisant biaise le test, "
            "tandis qu’un nombre excessif réduit sa puissance. "
            "Dans le cas de séries démographiques mensuelles, "
            "ce choix doit tenir compte à la fois de la saisonnalité "
            "et de la persistance structurelle liée aux évolutions de long terme."
        ),
        "",
        r"\subsection*{Robustesse et limites des tests de racine unitaire}",
        md_basic_to_tex(
            "Les tests de racine unitaire présentent plusieurs limites bien documentées : "
            "faible puissance en échantillon fini, sensibilité aux ruptures structurelles, "
            "et confusion possible entre mémoire longue et racine unitaire véritable. "
            "Un non-rejet de l’hypothèse nulle ne constitue donc pas une preuve définitive "
            "de non-stationnarité.\n\n"
            "Dans un contexte démographique, ces limites sont particulièrement importantes, "
            "car les ruptures observées peuvent résulter de transformations sociales profondes "
            "plutôt que d’une dynamique purement stochastique."
        ),
        "",
        r"\subsection*{Décision économétrique et implications pour la modélisation}",
        md_basic_to_tex(
            "La décision finale repose sur une synthèse des diagnostics graphiques et statistiques. "
            "Trois configurations sont envisagées : stationnarité en niveau, stationnarité autour d’une tendance, "
            "ou stationnarité en différence. "
            "Dans le cas d’un processus à tendance stochastique, "
            "la transformation opérationnelle retenue est la différenciation :"
        ),
       "",
        r"\begin{equation}",
        r"\Delta Y_t = Y_t - Y_{t-1}",
        r"\end{equation}",
        "",

        md_basic_to_tex(
            "Cette décision conditionne l’ensemble des étapes ultérieures : "
            "identification des modèles ARIMA, validité des diagnostics résiduels, "
            "et pertinence des analyses multivariées et de cointégration. "
            "Une erreur à ce stade se propage mécaniquement à toute la chaîne économétrique, "
            "ce qui justifie le soin particulier apporté à cette étape."
        ),
        "",
    ]

    # ================================
    # SECTION 2 : Résultats empiriques
    # ================================
    lines += [
        r"\section{Résultats de la stationnarité}",
        "",
        md_basic_to_tex(
            f"Synthèse quantitative : **verdict (diagnostics TS/DS) = {verdict}** "
            f"(ADF(c) p={_fmt_p(p_c)}, ADF(ct) p={_fmt_p(p_ct)}). "
            "Cette décision pilote directement le traitement du niveau et le degré d’intégration pour ARIMA/VAR/VECM."
        ),
        narr_call("m.diag.ts_vs_ds"),
        "",
    ]

    if fig_acf:
        lines += [
            include_figure(
                fig_rel=fig_acf,
                caption="Fonction d’autocorrélation",
                label="fig:fig-diag-acf-level",
            ),
            narr_call("fig.diag.acf_level"),
            "",

            md_basic_to_tex(
                "**Lecture — ACF (niveau).** "
                "La persistance élevée des autocorrélations et leur décroissance lente indiquent une dynamique fortement persistante. "
                "Ce comportement est compatible avec une non-stationnarité de type DS ou, alternativement, avec une mémoire longue. "
                "Cette lecture corrélographique ne constitue pas un critère de décision autonome et doit être systématiquement "
                "corroborée par les tests de racine unitaire (ADF) et l’analyse de robustesse via la bande de Dickey–Fuller."
            ),
            "",
        ]

    if fig_pacf:
        lines += [
            include_figure(
                fig_rel=fig_pacf,
                caption="Fonction d’autocorrélation partielle",
                label="fig:fig-diag-pacf-level",
            ),
            narr_call("fig.diag.pacf_level"),
            "",
            md_basic_to_tex(
                "**Lecture — PACF (niveau).** "
                "La PACF fournit des indications sur une éventuelle structure autorégressive sous-jacente. "
                "Cependant, en présence de non-stationnarité, son interprétation devient instable et potentiellement trompeuse. "
                "Elle est donc mobilisée ici uniquement à des fins exploratoires et qualitatives, sans rôle décisionnel."
            ),
            "",
        ]

    if tbl_acfp:
        lines += [
            
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_acfp,
                caption="Autocorrélations et autocorrélations partielles de la série",
                label="tab:tbl-diag-acf-pacf",
            ),
            narr_call("tbl.diag.acf_pacf"),
            "",
            md_basic_to_tex(
                "**Lecture — Synthèse ACF/PACF (niveau).** "
                "Ce tableau condense la structure corrélographique observée. "
                "Une ACF à décroissance lente est cohérente avec une non-stationnarité ou une mémoire longue, "
                "tandis qu’une coupure nette orienterait vers un processus autorégressif stationnaire. "
                "Il s’agit d’un outil de validation croisée ; la décision finale repose exclusivement sur les tests ADF "
                "et le diagnostic TS/DS."
            ),
            "",
        ]

    if tbl_adf:
        lines += [
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_adf,
                caption="Résultats du test de racine unitaire ADF",
                label="tab:tbl-diag-adf",
            ),
            narr_call("tbl.diag.adf"),
            "",
            md_basic_to_tex(
                "**Lecture — Test ADF (diagnostic principal).** "
                "Les p-values issues des spécifications avec constante et avec constante–tendance ne permettent pas "
                "de rejeter l’hypothèse de racine unitaire. "
                "Une éventuelle sensibilité à la spécification signalerait une frontière TS/DS ou une rupture non modélisée, "
                "ce qui justifie le recours à des outils de robustesse complémentaires."
            ),
            "",
        ]

    if tbl_band:
        lines += [
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_band,
                caption="Bande critique du test de Dickey–Fuller",
                label="tab:tbl-diag-band-df",
            ),
            narr_call("tbl.diag.band_df"),
            "",
            md_basic_to_tex(
                "**Lecture — Bande de Dickey–Fuller (robustesse).** "
                "La bande permet de vérifier la stabilité du verdict sur un ensemble de retards. "
                "Une conclusion robuste sur l’ensemble de l’intervalle renforce la décision TS/DS, "
                "tandis qu’une instabilité imposerait une approche prudente privilégiant la robustesse "
                "plutôt que l’optimisme économétrique."
            ),
            "",
        ]

    if tbl_tsds:
        lines += [
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_tsds,
                caption="Décision de stationnarité : processus TS ou DS",
                label="tab:tbl-diag-ts-vs-ds-decision",
            ),
            narr_call("tbl.diag.ts_vs_ds_decision"),
            "",

            md_basic_to_tex(
                "**Lecture — Décision TS vs DS (audit).** "
                "Ce tableau formalise la règle de décision opérationnelle et verrouille le traitement de la série. "
                "En cas de processus DS, une différenciation est requise ; en cas de processus TS, un détrendage est approprié ; "
                "enfin, aucune transformation n’est appliquée si la série est stationnaire en niveau."
            ),
            "",
        ]

    if tbl_ljung:
        lines += [
            include_table_tex(
                run_root=run_root,
                tbl_rel=tbl_ljung,
                caption="Test de Ljung–Box sur la série différenciée",
                label="tab:tbl-diag-ljungbox-diff",
            ),
            narr_call("tbl.diag.ljungbox_diff"),
            "",
            md_basic_to_tex(
                "**Lecture — Ljung–Box après transformation.** "
                "Ce test vérifie qu’après différenciation, aucune autocorrélation résiduelle massive ne subsiste. "
                "Une persistance significative signalerait une structure dynamique non capturée et imposerait "
                "une identification ARMA/ARIMA plus structurée."
            ),
            "",
        ]

    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée**"),
            md_basic_to_tex(
                "Cette note est strictement contrainte par les résultats ADF, la bande de Dickey–Fuller "
                "et la décision TS/DS. Toute transformation retenue doit être explicitement justifiée "
                "par le verdict et les p-values associées."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step3"),
            "",
        ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            f"La décision opérationnelle retenue est **{verdict}**. "
            "Les tests ADF et l’analyse corrélographique indiquent une persistance élevée compatible "
            "avec une dynamique intégrée. "
            "Cette décision impose la transformation appropriée de la série et conditionne "
            "l’identification AR/MA, la stabilité des résidus et la validité des modèles ARIMA/VAR/VECM."
        ),
        narr_call("m.diag.ts_vs_ds"),
        "",
    ]

    return "\n".join(lines).strip() + "\n"
