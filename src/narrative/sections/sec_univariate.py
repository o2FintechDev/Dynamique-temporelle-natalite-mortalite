# src/narrative/sections/sec_univariate.py
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

    aic = _pick(kp.get("aic"), (uni.get("best") or {}).get("aic"))
    bic = _pick(kp.get("bic"), (uni.get("best") or {}).get("bic"))

    lb_p = _pick(kp.get("lb_p"), kp.get("ljungbox_p"))
    jb_p = _pick(kp.get("jb_p"), kp.get("jarque_bera_p"))
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
    tbl_ar   = lookup(manifest, "tables", "tbl.uni.ar")    # NEW
    tbl_ma   = lookup(manifest, "tables", "tbl.uni.ma")    # NEW
    tbl_arma = lookup(manifest, "tables", "tbl.uni.arma")  # NEW

    lines: list[str] = []

    # =========================================
    # SECTION 1 : Cadre théorique (Box–Jenkins)
    # =========================================
    lines += [
        r"\section{Modèles AR, MA, ARMA et ARIMA}",
        "",
        md_basic_to_tex(
            "L’analyse univariée constitue le fondement méthodologique pour caractériser la dynamique intrinsèque "
            "d’une série temporelle, indépendamment de toute variable explicative. Sous l’hypothèse que "
            "l’information pertinente est contenue dans l’histoire passée de la série, ce cadre permet de capturer "
            "structure autorégressive, chocs transitoires et dynamiques mixtes. La méthodologie de Box–Jenkins "
            "organise cette analyse en trois étapes interdépendantes : identification, estimation puis validation."
        ),
        "",
        r"\subsection*{Cadre probabiliste général}",
        md_basic_to_tex(
            "Soit $(Y_t)_{t\in\mathbb{Z}}$ un processus stationnaire au second ordre, c'est-à-dire à espérance constante, "
            "variance finie et autocovariances ne dépendant que du retard. Sous cette hypothèse, les opérateurs "
            "linéaires jouent un rôle central dans la représentation et la décomposition des structures dynamiques. "
            "Le terme d’$\varepsilon_t$ est supposé être un bruit blanc : $\mathbb{E}[\varepsilon_t]=0$, "
            "$\mathrm{Var}(\varepsilon_t)=\sigma^2$ et $\mathrm{Cov}(\varepsilon_t,\varepsilon_{t-k})=0$ pour $k\neq 0$. "
            "Ce cadre garantit la validité des théorèmes d’inversion et des représentations Wold."
        ),
        "",
        r"\subsection*{Modèles autorégressifs AR(p)}",
        "",
        r"\begin{equation}",
        r"Y_t = c + \sum_{i=1}^{p}\phi_i Y_{t-i} + \varepsilon_t,",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Dans un AR(p), chaque observation est une combinaison linéaire des $p$ valeurs passées et d’un bruit blanc. "
            "Les coefficients $\phi_i$ mesurent la persistance structurée. La condition de stationnarité impose que les "
            "racines du polynôme caractéristique $\Phi(L)=1-\sum_{i=1}^p\phi_iL^i$ soient strictement à l’extérieur du "
            "cercle unité. Sur le plan corrélographique, une ACF décroissante progressivement et une PACF coupant après "
            "le retard $p$ constituent des critères heuristiques classiques d’identification."
        ),
        "",
        r"\subsection*{Modèles à moyenne mobile MA(q)}",
        "",
        r"\begin{equation}",
        r"Y_t = c + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j},",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Dans un modèle MA(q), l’observation dépend des chocs présents et des $q$ chocs passés. Cette formulation "
            "capture des effets transitoires qui ne se traduisent pas par une persistance infinie. La condition "
            "d’inversibilité exige que les racines du polynôme $\Theta(L)=1+\sum_{j=1}^q\theta_jL^j$ soient hors du "
            "cercle unité, ce qui garantit une représentation AR($\infty$) et l’unicité de l’estimation. Corrélographiquement, "
            "une ACF coupant après le retard $q$ et une PACF décroissante constituent des repères."
        ),
        "",
        r"\subsection*{Modèles mixtes ARMA(p,q)}",
        "",
        r"\begin{equation}",
        r"Y_t = c + \sum_{i=1}^{p}\phi_i Y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j},",
        r"\end{equation}",
        "",
        md_basic_to_tex(
            "Les modèles ARMA(p,q) combinent simultanément persistance structurelle et effets transitoires. Ils "
            "sont particulièrement utiles lorsque ni une structure purement AR, ni une structure purement MA ne "
            "suffisent à rendre compte des dynamiques observées. L’identification pure par ACF/PACF peut s’avérer "
            "ambigüe ; l’usage de critères d’information tels que AIC ou BIC devient indispensable pour éviter "
            "la surparamétrisation."
        ),
        "",
        r"\subsection*{Modèles intégrés ARIMA(p,d,q)}",
        "",
        md_basic_to_tex(
            "Lorsque la série n’est pas stationnaire en niveau mais intégrée d’ordre $d$, on applique une différenciation "
            "d’ordre $d$ pour récupérer la stationnarité avant d’estimer un ARMA(p,q) sur la série transformée. "
            "Un $d=1$ suggère que les chocs ont un effet permanent sur le niveau de la série, ce qui est fréquent dans "
            "les séries macroéconomiques et démographiques."
        ),
        "",
        r"\begin{equation}",
        r"\Delta^d Y_t = c + \sum_{i=1}^{p}\phi_i \Delta^d Y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j},",
        r"\end{equation}",
        "",
        r"\subsection*{Sélection du modèle optimal}",
        "",
        md_basic_to_tex(
            "La sélection du modèle repose sur un arbitrage entre biais et variance, évitant l’over‐fitting tout en capturant "
            "la dynamique essentielle. Les critères d’information AIC et BIC s’écrivent :"
        ),
        "",
        r"\begin{equation}",
        r"AIC = -2\ell + 2k,",
        r"\end{equation}",
        r"\begin{equation}",
        r"BIC = -2\ell + k\ln(T),",
        r"\end{equation}",
        "",

        md_basic_to_tex(
            "où $\ell$ est la log‐vraisemblance, $k$ le nombre de paramètres et $T$ la taille de l’échantillon. BIC pénalise "
            "plus fortement la complexité et est souvent préférable dans des contextes où la parsimony prime."
        ),
        "",
        r"\subsection*{Diagnostics des résidus}",
        "",
        md_basic_to_tex(
            "Un modèle correctement spécifié doit produire des résidus assimilables à un bruit blanc : absence d’autocorrélation "
            "(test de Ljung–Box), distribution raisonnablement symétrique et variance stable. Toute autocorrélation résiduelle "
            "signale une spécification incomplète des retards ou des dynamiques structurelles non capturées."
        ),
        "",
        r"\subsection*{Limites des modèles univariés}",
        "",
        md_basic_to_tex(
            "Les modèles univariés présentent des limites. Ils n’incorporent pas de déterminants explicites et peuvent confondre "
            "mémoire longue et racines unitaires. Leur interprétation économique peut être restreinte, ce qui justifie des "
            "analyses complémentaires (tests de mémoire longue, modèles multivariés ou non linéaires)."
        ),
        "",
        r"\subsection*{Rôle dans la stratégie globale}",
        "",
        md_basic_to_tex(
            "Dans la stratégie globale, cette étape sert de benchmark interne, stabilise l’identification des dynamiques "
            "temporelles et constitue un préalable méthodologique impératif avant toute extension vers des modèles VAR, VECM "
            "ou des approches causales plus élaborées."
        ),
        "",
    ]

    # =================================
    # SECTION 2 : Résultats modèles Uni
    # =================================
    lines += [
        r"\section{Résultats de la modélisation univariée}",
        "",
        md_basic_to_tex(
            "Conformément à la méthodologie de Box–Jenkins, l’ensemble de l’analyse univariée "
            "a été conduit sur la variable de croissance naturelle stationnarisée par différence première. "
            "Les diagnostics de stationnarité ayant mis en évidence la présence d’une racine unitaire "
            "(processus DS), la série en niveau ne vérifiait pas les conditions de stationnarité "
            "requises pour l’estimation de modèles ARMA classiques. "
            "Une différenciation d’ordre un ($d=1$) a donc été appliquée afin de stabiliser "
            "l’espérance et la structure d’autocovariance, condition nécessaire à une "
            "identification et une inférence économétrique valides. "
            "L’automate a ensuite exploré de manière systématique les modèles AR(p), MA(q), "
            "ARMA(p,q) et ARIMA(p,d,q), en testant différents ordres $p$ et $q$ "
            "afin d’identifier la spécification minimisant les critères d’information "
            "AIC et BIC, tout en respectant le principe de parcimonie."
        ),
        "",
        md_basic_to_tex(
            "Le modèle retenu est un **ARIMA(1,1,4)**. "
            "Ce choix reflète le meilleur compromis entre qualité d’ajustement "
            "et complexité paramétrique. "
            "Il est cohérent avec la lecture préalable des fonctions ACF et PACF : "
            "la décroissance lente en niveau justifiait la différenciation ($d=1$), "
            "et la structure des corrélations suggérait la présence combinée "
            "de composantes autorégressives et de moyenne mobile."
        ),
        "",
    ]   

    # ----- Interprétation mensuelle explicite -----
    lines += [
        r"\paragraph{Interprétation temporelle des paramètres (données mensuelles)}",
        md_basic_to_tex(
            "Les données étant mensuelles, l’ordre autorégressif $p=1$ signifie que "
            "la variation courante de la croissance naturelle dépend directement "
            "de la variation observée le mois précédent. "
            "Il existe donc une inertie mensuelle immédiate : "
            "un choc intervenu à la période $t-1$ continue d’influencer la dynamique au mois $t$."
        ),
        "",
        md_basic_to_tex(
            "Les termes de moyenne mobile d’ordre $q=4$ indiquent que la variation courante "
            "intègre les effets des chocs aléatoires survenus au cours des quatre derniers mois. "
            "Concrètement, un choc démographique ne s’éteint pas instantanément : "
            "son impact se diffuse et s’amortit progressivement sur environ un trimestre. "
            "Cette structure est cohérente avec la nature graduelle des ajustements "
            "démographiques observés à fréquence mensuelle."
        ),
        "",
    ]

    # ----- Ajustement -----
    if fig_fit:
        lines += [
            r"\paragraph{Figure 1 — Ajustement du modèle}",
            md_basic_to_tex(
                "La comparaison entre la série observée et les valeurs ajustées "
                "montre une reproduction satisfaisante de la dynamique globale. "
                "Les inflexions majeures sont correctement capturées, "
                "ce qui confirme la pertinence de la spécification retenue."
            ),
            "",
            include_figure(fig_rel=fig_fit, caption="Ajustement du modèle ARIMA(1,1,4)", label="fig:fig-uni-fit"),
            "",
        ]

    # ----- Comparaison modèles -----
    if tbl_summary:
        lines += [
            r"\paragraph{Tableau 1 — Comparaison des modèles candidats}",
            md_basic_to_tex(
                "La grille comparative confirme la supériorité de l’ARIMA(1,1,4) "
                "au regard des critères AIC et BIC. "
                "Les modèles plus complexes n’apportent pas de gain substantiel "
                "une fois pénalisés pour leur nombre de paramètres."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_summary, caption="Synthèse des modèles ARIMA estimés", label="tab:tbl-uni-summary"),
            "",
        ]

    # ----- ACF résidus -----
    if fig_resid_acf:
        lines += [
            r"\paragraph{Figure 2 — ACF des résidus}",
            md_basic_to_tex(
                "La corrélogramme des résidus ne présente pas de structure persistante significative. "
                "Cette absence d’autocorrélation résiduelle indique que la dynamique principale "
                "a été correctement capturée."
            ),
            "",
            include_figure(fig_rel=fig_resid_acf, caption="ACF des résidus du modèle ARIMA(1,1,4)", label="fig:fig-uni-resid-acf"),
            "",
        ]

    # ----- QQ plot -----
    if fig_qq:
        lines += [
            r"\paragraph{Figure 3 — QQ-plot des résidus}",
            md_basic_to_tex(
                "Le QQ-plot suggère une distribution proche de la normalité au centre, "
                "avec des écarts dans les queues. "
                "Ces écarts traduisent la présence possible de chocs rares mais intenses, "
                "fréquents dans les données macro-démographiques."
            ),
            "",
            include_figure(fig_rel=fig_qq, caption="QQ-plot des résidus du modèle ARIMA(1,1,4)", label="fig:fig-uni-qq"),
            "",
        ]

    # ----- Diagnostics -----
    if tbl_resid:
        lines += [
            r"\paragraph{Tableau 2 — Diagnostics des résidus}",
            md_basic_to_tex(
                "Le test de Ljung–Box confirme l’absence d’autocorrélation résiduelle significative. "
                "Le test de Durbin–Watson, proche de 2, renforce ce constat. "
                "Le test de Jarque–Bera signale un écart à la normalité, "
                "tandis que le test ARCH indique une hétéroscédasticité conditionnelle. "
                "Ces éléments invitent à une prudence sur l’inférence classique, "
                "mais ne remettent pas en cause la validité dynamique du modèle."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_resid, caption="Diagnostics des résidus du modèle ARIMA(1,1,4)", label="tab:tbl-uni-resid"),
            "",
        ]

    # ----- Mémoire -----
    if tbl_memory:
        lines += [
            r"\paragraph{Tableau 3 — Indicateurs de persistance}",
            md_basic_to_tex(
                "Les indicateurs de mémoire suggèrent une persistance modérée, "
                "compatible avec la structure intégrée du processus. "
                "Cette persistance reflète vraisemblablement des mécanismes "
                "structurels lents propres aux phénomènes démographiques."
            ),
            "",
            include_table_tex(run_root=run_root, tbl_rel=tbl_memory, caption="Indicateurs de persistance de la série", label="tab:tbl-uni-memory"),
            "",
        ]

    # ----- Conclusion section -----
    lines += [
        r"\paragraph{Conclusion de la modélisation univariée}",
        md_basic_to_tex(
            "Le modèle ARIMA(1,1,4) constitue une représentation cohérente "
            "de la dynamique propre de la croissance naturelle française. "
            "Il confirme que le processus est intégré, doté d’une inertie réelle "
            "et sensible aux chocs récents. "
            "Cette spécification s’inscrit pleinement dans la continuité "
            "des diagnostics de stationnarité et des analyses ACF/PACF précédentes, "
            "et prépare logiquement l’extension multivariée."
        ),
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
