from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import re
from src.narrative.sections.base import (
    SectionSpec,
    lookup,
    md_basic_to_tex,
    include_table_tex,
    include_figure,
    narr_call,
)

def _parse_order(order: Any) -> tuple[int, int, int] | None:
    """
    Normalise l'order vers (p,d,q).
    Accepte:
      - tuple/list (p,d,q)
      - dict {"p":..,"d":..,"q":..} ou {"order":[p,d,q]}
      - string "ARIMA(4,1,1)" ou "(4,1,1)" ou "4,1,1"
    Retourne None si non parsable.
    """
    if order is None:
        return None

    # tuple / list
    if isinstance(order, (tuple, list)) and len(order) == 3:
        try:
            p, d, q = order
            return int(p), int(d), int(q)
        except Exception:
            return None

    # dict
    if isinstance(order, dict):
        if "order" in order:
            return _parse_order(order.get("order"))
        if all(k in order for k in ("p", "d", "q")):
            try:
                return int(order["p"]), int(order["d"]), int(order["q"])
            except Exception:
                return None
        return None

    # string
    s = str(order).strip()
    # extrait 3 entiers dans la chaîne
    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 3:
        try:
            p, d, q = map(int, nums[:3])
            return p, d, q
        except Exception:
            return None
    return None


def model_label(p: int, d: int, q: int) -> str:
    """
    Libellé professionnel selon la structure.
    """
    if d == 0 and q == 0:
        return f"AR({p})"
    if d == 0 and p == 0:
        return f"MA({q})"
    if d == 0:
        return f"ARMA({p},{q})"
    return f"ARIMA({p},{d},{q})"

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


def render_sec_univariate(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    sec: SectionSpec,
    metrics_cache: Dict[str, Dict[str, Any]],
) -> str:
    
    def _pick(*vals):
        for v in vals:
            if v is not None:
                return v
        return None
    
    uni = metrics_cache.get("m.uni.best") or {}
    kp = uni.get("key_points") or {}

    order_raw = kp.get("order") or (uni.get("best") or {}).get("order")
    order_parsed = _parse_order(order_raw)

    if order_parsed:
        p_best, d_best, q_best = order_parsed
        best_label = model_label(p_best, d_best, q_best)
        order_str = f"({p_best},{d_best},{q_best})"  # utile si tu veux l'afficher brut
    else:
        best_label = "Modèle univarié (order indisponible)"
        order_str = "NA"

    aic = _pick(kp.get("aic") or (uni.get("best") or {}).get("aic"))
    bic = _pick(kp.get("bic") or (uni.get("best") or {}).get("bic"))

    lb_p = _pick(kp.get("lb_p"), kp.get("ljungbox_p"))
    jb_p = _pick(kp.get("jb_p") or kp.get("jarque_bera_p"))
    arch_p = kp.get("arch_p")

    tsds = metrics_cache.get("m.diag.ts_vs_ds") or {}
    verdict = tsds.get("verdict", "NA")
    d_force = kp.get("d_force")
    if d_force is None:
        d_force = 1 if verdict == "DS" else 0 if verdict == "TS" else "auto"

    note = (metrics_cache.get("m.note.step4") or {})
    note_md = ""
    if isinstance(note, dict):
        note_md = note.get("markdown") or note.get("text") or note.get("summary") or ""
    elif isinstance(note, str):
        note_md = note
    note_md = note_md.replace("−", "-")

    # artefacts (figures)
    fig_fit = lookup(manifest, "figures", "fig.uni.fit")
    fig_resid_acf = lookup(manifest, "figures", "fig.uni.resid_acf")
    fig_qq = lookup(manifest, "figures", "fig.uni.qq")

    # artefacts (tables)
    tbl_summary = lookup(manifest, "tables", "tbl.uni.summary")
    tbl_arima = lookup(manifest, "tables", "tbl.uni.arima")
    tbl_resid = lookup(manifest, "tables", "tbl.uni.resid_diag")
    tbl_memory = lookup(manifest, "tables", "tbl.uni.memory")

    lines: list[str] = []

    # ============================================================
    # SECTION 1 : Cadre théorique (Box–Jenkins)
    # ============================================================
    lines += [
        r"\section{Analyse univariée : modèles AR, MA, ARMA et ARIMA}",
        "",
        md_basic_to_tex(
            "L’analyse univariée vise à modéliser la dynamique propre de la croissance naturelle, indépendamment de toute variable explicative. "
            "Elle repose sur l’hypothèse que l’information contenue dans l’historique de la série suffit à expliquer sa dynamique présente, "
            "une fois la stationnarité assurée. Cette étape constitue un préalable indispensable avant toute extension multivariée. "
            "La méthodologie de Box–Jenkins fournit un cadre systématique fondé sur : identification, estimation et validation."
        ),
        "",
        r"\subsection*{Cadre probabiliste général}",
        md_basic_to_tex(
            "On considère une série stationnaire au second ordre : espérance constante, variance finie, autocovariance ne dépendant que du retard. "
            "Le terme d’erreur est supposé bruit blanc, non autocorrélé et de variance constante."
        ),
        "",
        r"\subsection*{Modèles autorégressifs AR(p)}",
        "",
        r"\begin{equation}",
        r"Y_t = c + \sum_{i=1}^{p}\phi_i Y_{t-i} + \varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Interprétation : chaque observation dépend linéairement des valeurs passées ; les coefficients mesurent la persistance. "
            "Condition de stationnarité : les racines du polynôme caractéristique doivent être hors cercle unité. "
            "Identification : ACF décroît progressivement, PACF coupure au retard $p$. "
            "Lecture économique : un $p$ élevé traduit une inertie démographique importante."
        ),
        "",
        r"\subsection*{Modèles à moyenne mobile MA(q)}",
        "",
        r"\begin{equation}",
        r"Y_t = c + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Interprétation : la série dépend des chocs présents et passés (effets transitoires). "
            "Condition d’inversibilité : racines hors cercle unité. "
            "Identification : ACF coupure au retard $q$, PACF décroît progressivement. "
            "Lecture économique : capte des chocs ponctuels (ex. crises sanitaires temporaires)."
        ),
        "",
        r"\subsection*{Modèles ARMA(p,q)}",
        "",
        r"\begin{equation}",
        r"Y_t = c + \sum_{i=1}^{p}\phi_i Y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Intérêt : modéliser simultanément la persistance structurelle (AR) et les effets transitoires (MA). "
            "L’identification ACF/PACF est souvent ambiguë : la sélection finale repose sur des critères d’information."
        ),
        "",
        r"\subsection*{Modèles ARIMA(p,d,q)}",
        md_basic_to_tex(
            "Lorsque la série est intégrée d’ordre $d$, on estime un ARMA sur la série transformée. "
            "Un $d=1$ traduit l’existence de chocs permanents."
        ),
        "",
        r"\begin{equation}",
        r"\Delta Y_t = c + \sum_{i=1}^{p}\phi_i \Delta Y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}",
        r"\end{equation}",
        "",
        r"\subsection*{4.6 Sélection du modèle optimal}",
        md_basic_to_tex(
            "Le choix repose sur un arbitrage biais–variance. "
            "AIC pénalise faiblement la complexité ; BIC pénalise plus fortement et limite la surparamétrisation, "
            "souvent préférable en démographie."
        ),
        "",
        r"\begin{equation}",
        r"AIC = -2\ell + 2k",
        r"\end{equation}",
        r"\begin{equation}",
        r"BIC = -2\ell + k\ln(T)",
        r"\end{equation}",
        "",
        r"\subsection*{Diagnostics des résidus}",
        md_basic_to_tex(
            "Un modèle correctement spécifié produit des résidus assimilables à un bruit blanc : "
            "absence d’autocorrélation (Ljung–Box), normalité raisonnable, variance stable. "
            "Toute autocorrélation résiduelle signale une mauvaise spécification de $p$ ou $q$."
        ),
        "",
        r"\subsection*{Limites des modèles univariés}",
        md_basic_to_tex(
            "Limites : absence de déterminants explicites, confusion possible entre mémoire longue et racine unitaire, "
            "interprétation économique parfois restreinte. Ces limites justifient l’analyse de la mémoire et l’extension multivariée."
        ),
        "",
        r"\subsection*{Rôle dans la stratégie globale}",
        md_basic_to_tex(
            "Cette étape fournit un benchmark, stabilise l’identification, et sert de contrôle de cohérence "
            "avant les modèles multivariés."
        ),
        "",
    ]

    # ============================================================
    # SECTION : Résultats empiriques
    # ============================================================
    lines += [
        r"\section{Résultats de la modélisation univariée}",
        "",
        md_basic_to_tex(
            f"Synthèse quantitative : **{best_label}**. "
            f"Stationnarité imposée : $d={d_force}$ (verdict {verdict}). "
            f"AIC={_fmt2(aic)}, BIC={_fmt2(bic)}. "
            f"Diagnostics résiduels (si disponibles) : Ljung–Box p={_fmt_p(lb_p)}, JB p={_fmt_p(jb_p)}, ARCH p={_fmt_p(arch_p)}."

        ),
        narr_call("m.uni.best"),
        "",
    ]

    if fig_fit:
        lines += [
            r"\paragraph{Figure 1 — Ajustement du modèle}",
            md_basic_to_tex(
                "Lecture : vérifier l’adéquation globale (niveau/variations) et repérer les périodes mal expliquées. "
                "Des écarts systématiques signalent une spécification insuffisante ou un changement de régime."
            ),
            "",
            include_figure(fig_rel=fig_fit, caption="fig.uni.fit", label="fig:fig-uni-fit"),
            narr_call("fig.uni.fit"),
            "",
        ]

    if tbl_summary:
        lines += [
            r"\paragraph{Tableau 1 — Synthèse des candidats / sélection}",
            md_basic_to_tex(
                "Lecture : comparer les critères d’information et vérifier la stabilité du compromis biais–variance. "
                "Un gain marginal d’AIC au prix d’une explosion de paramètres est une mauvaise décision."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_summary, caption="tbl.uni.summary", label="tab:tbl-uni-summary"),
            narr_call("tbl.uni.summary"),
            "",
        ]

    if tbl_arima:
        lines += [
            r"\paragraph{Tableau 2 — Paramètres ARIMA retenus}",
            md_basic_to_tex(
                "Lecture : contrôler signes, significativité et plausibilité (persistance AR vs correction MA). "
                "Des coefficients instables ou non significatifs en bloc suggèrent un sur-ajustement."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_arima, caption="tbl.uni.arima", label="tab:tbl-uni-arima"),
            narr_call("tbl.uni.arima"),
            "",
        ]

    if fig_resid_acf:
        lines += [
            r"\paragraph{Figure 2 — ACF des résidus}",
            md_basic_to_tex(
                "Lecture : la corrélogramme des résidus doit être compatible avec un bruit blanc. "
                "Des pics persistants indiquent une dynamique non capturée (ordre $p$/$q$ insuffisant, saisonnalité, rupture)."
            ),
            "",
            include_figure(fig_rel=fig_resid_acf, caption="fig.uni.resid_acf", label="fig:fig-uni-resid-acf"),
            narr_call("fig.uni.resid_acf"),
            "",
        ]

    if fig_qq:
        lines += [
            r"\paragraph{Figure 3 — QQ-plot des résidus}",
            md_basic_to_tex(
                "Lecture : évaluer l’écart à la normalité (queues épaisses). "
                "Des queues épaisses sont cohérentes avec des chocs rares mais extrêmes et justifient une prudence sur l’inférence classique."
            ),
            "",
            include_figure(fig_rel=fig_qq, caption="fig.uni.qq", label="fig:fig-uni-qq"),
            narr_call("fig.uni.qq"),
            "",
        ]

    if tbl_resid:
        lines += [
            r"\paragraph{Tableau 3 — Diagnostics résiduels}",
            md_basic_to_tex(
                "Lecture : Ljung–Box (blancheur), Jarque–Bera (normalité), ARCH (hétéroscédasticité). "
                "Un rejet de blancheur invalide la spécification ; un rejet de normalité appelle une lecture robuste ; "
                "un signal ARCH indique variance conditionnelle non constante."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_resid, caption="tbl.uni.resid_diag", label="tab:tbl-uni-resid-diag"),
            narr_call("tbl.uni.resid_diag"),
            "",
        ]

    if tbl_memory:
        lines += [
            r"\paragraph{Tableau 4 — Indices de mémoire / persistance}",
            md_basic_to_tex(
                "Lecture : distinguer persistance ARIMA standard et mémoire longue potentielle. "
                "Une persistance élevée peut refléter des structures sociales lentes, mais elle peut aussi être confondue avec une racine unitaire "
                "ou des ruptures non modélisées."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_memory, caption="tbl.uni.memory", label="tab:tbl-uni-memory"),
            narr_call("tbl.uni.memory"),
            "",
        ]

    if note_md.strip():
        lines += [
            md_basic_to_tex("**Note d’interprétation automatisée**"),
            md_basic_to_tex(
                "Cette note doit rester cohérente avec : (i) le choix ARIMA, (ii) les critères d’information, "
                "(iii) les diagnostics résiduels. Toute conclusion de “bon modèle” exige une blancheur acceptable."
            ),
            "",
            md_basic_to_tex(note_md),
            narr_call("m.note.step4"),
            "",
        ]

    lines += [
        md_basic_to_tex("**Conclusion**"),
        md_basic_to_tex(
            f"Le modèle retenu est **{best_label}** avec $d={d_force}$ (verdict {verdict}). "
            "La décision finale repose sur l’acceptabilité des résidus (blancheur en priorité), "
            "puis sur la parcimonie (BIC) et la stabilité des paramètres."
        ),
        narr_call("m.uni.best"),
        "",
    ]
    # ============================================================
    # SECTION : Mémoire longue (cadre)
    # ============================================================
    lines += [
        r"\section{Analyse de la mémoire longue}",
        "",
        md_basic_to_tex(
            "L’analyse de la mémoire longue vise à caractériser la persistance temporelle des chocs au-delà du cadre classique des modèles ARMA. "
            "Dans de nombreuses séries macroéconomiques et démographiques, la dépendance temporelle ne décroît pas de manière exponentielle "
            "mais selon une loi hyperbolique, traduisant une inertie structurelle profonde. Cette propriété remet en cause l’hypothèse de mémoire "
            "courte implicite des modèles ARMA et justifie le recours à des outils spécifiques."
        ),
        "",
        r"\subsection*{Mémoire courte versus mémoire longue}",
        md_basic_to_tex(
            "Mémoire courte : la somme des autocorrélations est absolument convergente. "
            "Mémoire longue : la somme diverge et l’autocorrélation décroît selon une loi hyperbolique."
        ),
        "",
        r"\begin{equation}",
        r"\sum_{h=0}^{\infty}|\rho(h)| < \infty",
        r"\end{equation}",
        r"\begin{equation}",
        r"\sum_{h=0}^{\infty}|\rho(h)| = \infty",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        r"\rho(h) \sim C h^{2H-2}\quad \text{lorsque } h\to\infty",
        r"\end{equation}",
        "",
        r"\subsection*{Fondements théoriques de la mémoire longue}",
        md_basic_to_tex(
            "La mémoire longue traduit des mécanismes d’agrégation et des rigidités institutionnelles générant une persistance de long terme. "
            "En démographie : structures familiales stables, politiques publiques durables, inerties biologiques et sociales. "
            "Économétriquement, elle peut être confondue avec une racine unitaire, d’où la nécessité de diagnostics robustes."
        ),
        "",
        r"\subsection*{Statistique du Rescaled Range (R/S)}",
        md_basic_to_tex(
            "L’approche R/S (Hurst) examine la croissance de l’amplitude cumulée des écarts à la moyenne."
        ),
        "",
        r"\begin{equation}",
        r"X_k=\sum_{t=1}^{k}(Y_t-\bar{Y})",
        r"\end{equation}",
        r"\begin{equation}",
        r"R(n)=\max_{1\le k\le n}X_k-\min_{1\le k\le n}X_k",
        r"\end{equation}",
        r"\begin{equation}",
        r"\frac{R(n)}{S(n)}",
        r"\end{equation}",
        "",
        r"\begin{equation}",
        r"\mathbb{E}\left[\frac{R(n)}{S(n)}\right]=C n^{H}",
        r"\end{equation}",
        "",
        r"\subsection*{Exposant de Hurst : interprétation}",
        md_basic_to_tex(
            "H=0,5 : absence de mémoire longue (bruit blanc/ARMA). "
            "H>0,5 : persistance (les chocs tendent à se prolonger). "
            "H<0,5 : antipersistence (retour rapide vers la moyenne)."
        ),
        "",
        r"\subsection*{Lien entre Hurst et intégration fractionnaire}",
        "",
        r"\begin{equation}",
        r"H=d+\frac{1}{2}",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "d=0 : ARMA ; 0<d<0,5 : mémoire longue stationnaire ; d≥0,5 : non-stationnarité. "
            "Ce lien conduit naturellement aux modèles ARFIMA."
        ),
        "",
        r"\subsection*{Modèles ARFIMA(p,d,q)}",
        "",
        r"\begin{equation}",
        r"\Phi(L)(1-L)^{d}Y_t=\Theta(L)\varepsilon_t",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Le paramètre d capture l’intensité de la mémoire longue : persistance intermédiaire entre stationnarité et non-stationnarité."
        ),
        "",
        r"\subsection*{Pièges et limites}",
        md_basic_to_tex(
            "Risques : confusion entre mémoire longue et ruptures, sensibilité aux erreurs de mesure, instabilité en échantillon fini. "
            "Les diagnostics doivent être confrontés aux tests de racine unitaire et à l’analyse des ruptures."
        ),
        "",
        r"\subsection*{Implications économétriques et économiques}",
        md_basic_to_tex(
            "Une mémoire longue implique des effets persistants à très long horizon : "
            "les modèles ARMA sous-estiment la persistance, et les politiques publiques peuvent produire des effets différés durables. "
            "Cela justifie une extension multivariée pour analyser les canaux dynamiques."
        ),
        "",
    ]

    # ==== FAIRE CONCLU MAYBE ====
    return "\n".join(lines).strip() + "\n"
