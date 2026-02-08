# src/narrative/latex_renderer.py
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------- utils ----------
def _escape_tex(s: str) -> str:
    """Escape texte courant (pas chemins)."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")  # ATTENTION: ne pas utiliser sur des contenus avec maths $...$
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("_", r"\_")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _escape_caption(s: str) -> str:
    """Caption = texte => escape complet (texte seulement)."""
    return _escape_tex(s)


def _label_tex_safe(s: str) -> str:
    """
    Labels LaTeX: éviter macros et caractères spéciaux.
    Normalise en [a-zA-Z0-9:-] et tronque.
    """
    if s is None:
        return "na"
    out = re.sub(r"[^a-zA-Z0-9:\-]+", "-", str(s))
    out = re.sub(r"-{2,}", "-", out).strip("-")
    return out[:80] if out else "na"


_MATH_TOKEN = "<<<MATH_BLOCK_{i}>>>"


def _escape_tex_keep_math(s: str) -> str:
    """
    Escape LaTeX sur les parties texte en préservant les segments $...$.
    Hypothèse: pas de dollars imbriqués.
    """
    if s is None:
        return ""

    src = str(s)
    math_blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        math_blocks.append(m.group(0))  # inclut les $
        return _MATH_TOKEN.format(i=len(math_blocks) - 1)

    tmp = re.sub(r"\$[^$]*\$", _stash, src)  # capture $...$
    tmp = _escape_tex(tmp)  # escape texte (OK car tokens ne contiennent pas $)
    for i, mb in enumerate(math_blocks):
        tmp = tmp.replace(_MATH_TOKEN.format(i=i), mb)
    return tmp


def _file_tex_safe(fname: str) -> str:
    """
    Pour includegraphics/input : protéger les underscores dans les noms de fichiers.
    Hypothèse repo: filenames sans espaces exotiques.
    """
    return str(fname).replace(" ", "_").replace("_", r"\_")


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


def _md_basic_to_tex(md: str) -> str:
    """Markdown minimal -> LaTeX (gras/italique) en préservant $...$."""
    s = (md or "").strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", s)

    # escape texte en conservant les maths $...$
    s = _escape_tex_keep_math(s)

    # réactive \textbf/\emph
    s = s.replace(r"\textbackslash{}textbf", r"\textbf").replace(r"\textbackslash{}emph", r"\emph")
    return s


# ---------- spec ----------
@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    intro_md: str
    figure_keys: List[str]
    table_keys: List[str]
    metric_keys: List[str]


def default_spec() -> List[SectionSpec]:
    return [
        SectionSpec(
            key="sec_data",
            title="Données, construction et qualité",
            intro_md=(
                "Cette section documente la construction de la variable d’intérêt $Y_t$ (solde naturel), "
                "la couverture temporelle, et les diagnostics de qualité (manquants, cohérence, métadonnées). "
                "Les tableaux ci-dessous établissent la traçabilité statistique du jeu de données."
            ),
            figure_keys=[],
            table_keys=["tbl.data.desc_stats", "tbl.data.missing_report", "tbl.data.coverage_report"],
            metric_keys=["m.data.dataset_meta", "m.note.step1"],
        ),
        SectionSpec(
            key="sec_descriptive",
            title="Analyse descriptive et décomposition",
            intro_md=(
                "On réalise une analyse descriptive (statistiques résumées) et une décomposition STL "
                "pour isoler tendance, saisonnalité et résidu. L’objectif est d’identifier des structures "
                "visuelles (tendance, cycles, volatilité) avant la phase de diagnostic de stationnarité."
            ),
            figure_keys=["fig.desc.level", "fig.desc.decomp"],
            table_keys=["tbl.desc.summary", "tbl.desc.seasonality", "tbl.desc.decomp_components"],
            metric_keys=["m.desc.seasonal_strength", "m.desc.seasonality_type", "m.note.step2"],
        ),
        SectionSpec(
            key="sec_stationarity",
            title="Diagnostics de stationnarité",
            intro_md=(
                "La spécification économétrique dépend de la stationnarité. "
                "La décision TS/DS est fondée exclusivement sur ADF (avec/sans tendance). "
                "On complète par une lecture corrélographique (ACF/PACF) et un test Ljung–Box sur la série différenciée."
            ),
            figure_keys=["fig.diag.acf_level", "fig.diag.pacf_level"],
            table_keys=["tbl.diag.adf", "tbl.diag.ts_vs_ds_decision", "tbl.diag.ljungbox_diff", "tbl.diag.band_df"],
            metric_keys=["m.diag.ts_vs_ds", "m.note.step3"],
        ),
        SectionSpec(
            key="sec_univariate",
            title="Modélisation univariée (ARIMA) et diagnostics",
            intro_md=(
                "On spécifie un modèle ARIMA selon une approche Box–Jenkins. "
                "Le degré d’intégration $d$ est imposé par le verdict TS/DS (ADF-only) : "
                "si DS, on travaille avec $d=1$ ; si TS, on travaille avec $d=0$ après détrend. "
                "La sélection repose sur AIC/BIC et la validité des résidus (Ljung–Box, normalité, ARCH)."
            ),
            figure_keys=["fig.uni.fit", "fig.uni.resid_acf", "fig.uni.qq"],
            table_keys=["tbl.uni.summary", "tbl.uni.arima", "tbl.uni.resid_diag", "tbl.uni.memory"],
            metric_keys=["m.uni.best", "m.note.step4", "m.diag.ts_vs_ds"],
        ),
        SectionSpec(
            key="sec_multivariate",
            title="Analyse multivariée (VAR)",
            intro_md=(
                "On estime un VAR sur les composantes (niveau/tendance/saisonnalité) issues de la décomposition STL, "
                "afin de caractériser les dépendances dynamiques. Les choix de retards sont guidés par les critères "
                "d’information, puis on examine causalités (Granger), tests de Sims/Wald, et réponses impulsionnelles."
            ),
            figure_keys=["fig.var.irf"],
            table_keys=["tbl.var.lag_selection", "tbl.var.granger", "tbl.var.sims", "tbl.var.fevd"],
            metric_keys=["m.var.meta", "m.var.sims", "m.var.audit", "m.note.step5"],
        ),
        SectionSpec(
            key="sec_cointegration",
            title="Cointégration et dynamique de long terme (VECM)",
            intro_md=(
                "On teste la cointégration (Engle–Granger, Johansen) sur les composantes STL. "
                "Si une relation de long terme est détectée, on privilégie un VECM ; sinon, on conserve une spécification VAR. "
                "L’interprétation porte sur les vecteurs de cointégration et la vitesse d’ajustement."
            ),
            figure_keys=[],
            table_keys=["tbl.coint.eg", "tbl.coint.johansen", "tbl.coint.var_vs_vecm_choice", "tbl.vecm.params"],
            metric_keys=["m.coint.meta", "m.coint.audit", "m.note.step6"],
        ),
        SectionSpec(
            key="sec_anthropology",
            title="Synthèse historique et anthropologique",
            intro_md=(
                "Cette section articule les résultats économétriques avec une lecture historique (chocs, ruptures) "
                "et une interprétation anthropologique (cadre toddien). Les affirmations sont contraintes aux ruptures "
                "observables dans les données et aux conclusions des modèles."
            ),
            figure_keys=[],
            table_keys=[],
            metric_keys=["m.anthro.todd_analysis"],
        ),
    ]


# ---------- manifest readers ----------
def _lookup_path(manifest: Dict[str, Any], kind: str, key: str) -> Optional[str]:
    lookup = (manifest.get("lookup") or {}).get(kind) or {}
    return lookup.get(key)


def _fig_path(manifest: Dict[str, Any], key: str) -> Optional[str]:
    return _lookup_path(manifest, "figures", key)


def _tbl_path(manifest: Dict[str, Any], key: str) -> Optional[str]:
    return _lookup_path(manifest, "tables", key)


def _metric_path(manifest: Dict[str, Any], key: str) -> Optional[str]:
    return _lookup_path(manifest, "metrics", key)


def _read_metric_json(run_root: Path, rel: str) -> Dict[str, Any]:
    p = (run_root / rel).resolve()
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- rendering per artefact ----------
def _include_figure(fig_rel: str, caption: str, label: str) -> str:
    fname = _file_tex_safe(Path(fig_rel).name)
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            r"\includegraphics[width=0.95\linewidth]{artefacts/figures/" + fname + r"}",
            r"\caption{" + _escape_caption(caption) + r"}",
            r"\label{" + _label_tex_safe(label) + r"}",
            r"\end{figure}",
            "",
        ]
    )


def _include_table(tbl_rel: str, caption: str, label: str) -> str:
    fname = _file_tex_safe(Path(tbl_rel).name)
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{" + _escape_caption(caption) + r"}",
            r"\label{" + _label_tex_safe(label) + r"}",
            r"\begin{adjustbox}{max width=\linewidth,center}",
            r"\input{artefacts/tables/" + fname + r"}",
            r"\end{adjustbox}",
            r"\end{table}",
            "",
        ]
    )


# ---------- metric synthesis helpers ----------
def _metric_synthesis_tex(sec_key: str, metrics_cache: Dict[str, Dict[str, Any]]) -> str:
    """
    Petit paragraphe de synthèse injectant des valeurs clefs (sans tables).
    """
    if sec_key == "sec_descriptive":
        st = metrics_cache.get("m.desc.seasonal_strength", {}).get("value")
        ty = metrics_cache.get("m.desc.seasonality_type", {}).get("value")
        return (
            _md_basic_to_tex(
                f"**Synthèse quantitative** — Force saisonnière (STL) : {_fmt2(st)}. "
                f"Qualification : **{ty}**. "
                "Ces éléments guident la pertinence d’un traitement saisonnier explicite."
            )
            + "\n"
        )

    if sec_key == "sec_stationarity":
        m = metrics_cache.get("m.diag.ts_vs_ds", {})
        verdict = m.get("verdict")
        return (
            _md_basic_to_tex(
                f"**Synthèse quantitative** — Verdict ADF-only : **{verdict}** "
                f"(ADF(c) p={_fmt_p(m.get('adf_p_c'))}, ADF(ct) p={_fmt_p(m.get('adf_p_ct'))})."
            )
            + "\n"
        )

    if sec_key == "sec_univariate":
        m_uni = metrics_cache.get("m.uni.best", {})
        kp = (m_uni.get("key_points") or {})
        return (
            _md_basic_to_tex(
                f"**Synthèse quantitative** — Modèle candidat : ARIMA{kp.get('order')} "
                f"(AIC={_fmt2(kp.get('aic'))}, BIC={_fmt2(kp.get('bic'))}). "
                f"Diagnostics résiduels : Ljung–Box p={_fmt_p(kp.get('lb_p'))}, "
                f"JB p={_fmt_p(kp.get('jb_p'))}, ARCH p={_fmt_p(kp.get('arch_p'))}."
            )
            + "\n"
        )

    return ""


# ---------- conclusions (pilotées par metrics) ----------
def _stationarity_conclusion(m_tsds: Dict[str, Any]) -> str:
    verdict = m_tsds.get("verdict", "NA")
    p_c = m_tsds.get("adf_p_c")
    p_ct = m_tsds.get("adf_p_ct")
    return (
        _md_basic_to_tex(
            f"**Conclusion (stationnarité)** — Décision ADF-only : verdict **{verdict}** "
            f"(ADF(c) p={_fmt_p(p_c)}, ADF(ct) p={_fmt_p(p_ct)}). "
            "Ce verdict pilote la transformation de la série avant modélisation (différenciation si DS, détrend si TS)."
        )
        + "\n"
    )


def _univariate_conclusion(m_uni: Dict[str, Any], m_tsds: Dict[str, Any]) -> str:
    kp = (m_uni.get("key_points") or {})
    order = kp.get("order") or (m_uni.get("best") or {}).get("order")
    aic = kp.get("aic") or (m_uni.get("best") or {}).get("aic")
    bic = kp.get("bic") or (m_uni.get("best") or {}).get("bic")
    lb_p = kp.get("lb_p")
    arch_p = kp.get("arch_p")
    jb_p = kp.get("jb_p")

    verdict = m_tsds.get("verdict", "NA")
    d_force = kp.get("d_force")
    if d_force is None:
        d_force = 1 if verdict == "DS" else 0 if verdict == "TS" else "auto"

    validity = []
    if lb_p is not None:
        validity.append(f"Ljung–Box p={_fmt_p(lb_p)}")
    if jb_p is not None:
        validity.append(f"JB p={_fmt_p(jb_p)}")
    if arch_p is not None:
        validity.append(f"ARCH p={_fmt_p(arch_p)}")
    valid_txt = ", ".join(validity) if validity else "diagnostics indisponibles"

    return (
        _md_basic_to_tex(
            f"**Conclusion (univarié)** — Le modèle retenu est **ARIMA{order}** "
            f"avec $d={d_force}$ (piloté par le verdict {verdict}). "
            f"Critères : AIC={_fmt2(aic)}, BIC={_fmt2(bic)}. "
            f"Validité résiduelle : {valid_txt}. "
            "Le choix final privilégie le compromis performance (AIC/BIC) / résidus (bruit blanc, homoscédasticité)."
        )
        + "\n"
    )


def _multivariate_conclusion(m_var_meta: Dict[str, Any], m_var_sims: Dict[str, Any], m_var_audit: Dict[str, Any]) -> str:
    cols = m_var_meta.get("vars") or []
    cols_txt = ", ".join(cols) if isinstance(cols, list) else str(cols)
    p = m_var_meta.get("selected_lag_aic")
    nobs = m_var_meta.get("nobs_used") or m_var_meta.get("nobs")
    maxlags = m_var_meta.get("maxlags")

    irf_h = m_var_meta.get("irf_h")
    fevd_h = m_var_meta.get("fevd_h")

    stable = m_var_meta.get("stable")
    roots_abs_max = m_var_meta.get("roots_abs_max")

    whiteness_p = m_var_meta.get("whiteness_p")
    normality_p = m_var_meta.get("normality_p")

    dropped = (m_var_meta.get("rows_dropped_dropna") or None)

    sims_errors = m_var_sims.get("n_errors")
    lead_q_tested = m_var_sims.get("lead_q_tested") or []
    q_min = min(lead_q_tested) if isinstance(lead_q_tested, list) and lead_q_tested else None
    q_max = max(lead_q_tested) if isinstance(lead_q_tested, list) and lead_q_tested else None

    sel = (m_var_audit.get("selection") or {})
    aic = sel.get("aic")
    bic = sel.get("bic")
    hqic = sel.get("hqic")
    fpe = sel.get("fpe")

    diag_bits = []
    if stable is not None:
        diag_bits.append(f"stabilité={'oui' if stable else 'non'}")
    if roots_abs_max is not None:
        diag_bits.append(f"max|root|={_fmt2(roots_abs_max)}")
    if whiteness_p is not None:
        diag_bits.append(f"blanchiment (whiteness) p={_fmt_p(whiteness_p)}")
    if normality_p is not None:
        diag_bits.append(f"normalité p={_fmt_p(normality_p)}")
    diag_txt = ", ".join(diag_bits) if diag_bits else "diagnostics indisponibles"

    sel_bits = []
    if p is not None:
        sel_bits.append(f"p={p}")
    if maxlags is not None:
        sel_bits.append(f"maxlags={maxlags}")
    if aic is not None:
        sel_bits.append(f"AIC={_fmt2(aic)}")
    if bic is not None:
        sel_bits.append(f"BIC={_fmt2(bic)}")
    if hqic is not None:
        sel_bits.append(f"HQIC={_fmt2(hqic)}")
    if fpe is not None:
        sel_bits.append(f"FPE={_fmt2(fpe)}")
    sel_txt = ", ".join(sel_bits) if sel_bits else "sélection indisponible"

    hz_bits = []
    if irf_h is not None:
        hz_bits.append(f"IRF horizon={irf_h}")
    if fevd_h is not None:
        hz_bits.append(f"FEVD horizon={fevd_h}")
    hz_txt = ", ".join(hz_bits) if hz_bits else "horizons non renseignés"

    sims_txt = "test Sims indisponible"
    if q_min is not None and q_max is not None:
        sims_txt = f"Sims leads q∈[{q_min},{q_max}] (erreurs={sims_errors})"

    return (
        _md_basic_to_tex(
            f"**Conclusion (VAR)** — Spécification : VAR({p}) sur **{cols_txt}**, n={nobs}"
            + (f", lignes supprimées (dropna)={dropped}" if dropped is not None else "")
            + f". Sélection : {sel_txt}. "
            f"Validité/stabilité : {diag_txt}. "
            f"Lecture dynamique : {hz_txt}. "
            f"Tests descriptifs : {sims_txt}. "
            "Décision : on retient la spécification VAR(p) si la stabilité est satisfaite et si les résidus "
            "ne contredisent pas fortement l’hypothèse de bruit blanc (whiteness). Dans le cas contraire, "
            "il faut réduire p, revoir la transformation des composantes (ex. différenciation), ou introduire "
            "des termes déterministes supplémentaires."
        )
        + "\n"
    )


def _cointegration_conclusion(m_coint: Dict[str, Any]) -> str:
    choice = (m_coint.get("choice") or "NA")
    rank = m_coint.get("rank")
    nobs = m_coint.get("nobs")
    vars_ = m_coint.get("vars") or []
    vars_txt = ", ".join(vars_) if isinstance(vars_, list) else str(vars_)

    det_order = m_coint.get("det_order")
    k_ar_diff = m_coint.get("k_ar_diff")

    trace0 = m_coint.get("trace_stat_0")
    crit50 = m_coint.get("trace_crit5_0")
    rej0 = m_coint.get("trace_reject5_0")

    eg_n = m_coint.get("eg_n_pairs")
    eg_fail = m_coint.get("eg_n_fail")
    eg_p_min = m_coint.get("eg_p_min")
    eg_p_q10 = m_coint.get("eg_p_q10")

    hdr_bits = []
    if vars_txt:
        hdr_bits.append(f"variables={vars_txt}")
    if nobs is not None:
        hdr_bits.append(f"n={nobs}")
    if det_order is not None:
        hdr_bits.append(f"det\\_order={det_order}")
    if k_ar_diff is not None:
        hdr_bits.append(f"k\\_ar\\_diff={k_ar_diff}")
    hdr = ", ".join(hdr_bits) if hdr_bits else "métadonnées indisponibles"

    joh_bits = []
    if trace0 is not None and crit50 is not None:
        joh_bits.append(f"trace(r=0)={_fmt2(trace0)} vs crit5%={_fmt2(crit50)}")
    if rej0 is not None:
        joh_bits.append("rejet@5%=" + ("oui" if bool(rej0) else "non"))
    joh_txt = "; ".join(joh_bits) if joh_bits else "résumé Johansen indisponible"

    eg_bits = []
    if eg_n is not None:
        eg_bits.append(f"paires={eg_n}")
    if eg_fail is not None and int(eg_fail) > 0:
        eg_bits.append(f"échecs={eg_fail}")
    if eg_p_min is not None:
        eg_bits.append(f"p_min={_fmt_p(eg_p_min)}")
    if eg_p_q10 is not None:
        eg_bits.append(f"p_q10={_fmt_p(eg_p_q10)}")
    eg_txt = ", ".join(eg_bits) if eg_bits else "résumé Engle–Granger indisponible"

    ch = str(choice).upper()
    if ch == "VECM":
        decision = (
            f"Décision : **VECM** car le rang Johansen est **{rank}**. "
            "Interprétation : existence de relations de long terme (vecteurs $\\beta$) et dynamique d’ajustement ($\\alpha$) "
            "via le terme de correction d’erreur."
        )
    elif ch in ("VAR", "VAR_DIFF", "VAR-DIFF"):
        decision = (
            "Décision : **VAR en différences** (pas de cointégration robuste). "
            "Interprétation centrée sur la dynamique de court terme (IRF/FEVD) sans contrainte d’équilibre."
        )
    else:
        decision = (
            "Décision non disponible. Le pipeline doit écrire `choice` dans `m.coint.meta` "
            "(valeurs attendues : VECM ou VAR_diff)."
        )

    return (
        _md_basic_to_tex(
            f"**Conclusion (cointégration)** — Paramétrage : {hdr}. "
            f"Johansen : {joh_txt}. Engle–Granger (indicatif) : {eg_txt}. "
            f"{decision}"
        )
        + "\n"
    )


# ---------- main ----------
def render_all_section_blocks(
    run_root: Path,
    manifest: Dict[str, Any],
    *,
    spec: Optional[List[SectionSpec]] = None,
) -> Dict[str, str]:
    r"""
    Génère latex/blocks/sec_*.tex comme CONTENU (pas de \chapter).
    Le wrapper latex/master.tex porte la structure des chapitres.
    """
    spec = spec or default_spec()

    blocks_dir = run_root / "latex" / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    out_map: Dict[str, str] = {}

    # précharge metrics utiles
    metrics_cache: Dict[str, Dict[str, Any]] = {}
    for sec in spec:
        for mk in sec.metric_keys:
            rel = _metric_path(manifest, mk)
            if rel and mk not in metrics_cache:
                metrics_cache[mk] = _read_metric_json(run_root, rel)

    for sec in spec:
        lines: List[str] = []

        # --- Titre interne du block (niveau section) ---
        lines.append(r"\section{" + _escape_tex(sec.title) + "}")
        lines.append("")
        lines.append(_md_basic_to_tex(sec.intro_md))
        lines.append("")

        # --- Synthèse métriques (insertion de valeurs) ---
        synth = _metric_synthesis_tex(sec.key, metrics_cache)
        if synth:
            lines.append(r"\subsection{Synthèse quantitative}")
            lines.append(synth)
            lines.append("")

        # --- Tables: analyse unitaire + inclusion ---
        for tk in sec.table_keys:
            rel = _tbl_path(manifest, tk)
            if not rel:
                continue
            lines.append(r"\subsection{Tableau : " + _escape_tex(tk) + "}")
            lines.append(
                _md_basic_to_tex(
                    "Lecture unitaire : niveaux, variations, ruptures éventuelles, puis implication méthodologique."
                )
            )
            lines.append("")
            lines.append(_include_table(rel, caption=tk, label=f"tab:{tk.replace('.', '-')[:50]}"))
            lines.append("")

        # --- Figures: analyse unitaire + inclusion ---
        for fk in sec.figure_keys:
            rel = _fig_path(manifest, fk)
            if not rel:
                continue
            lines.append(r"\subsection{Figure : " + _escape_tex(fk) + "}")
            lines.append(
                _md_basic_to_tex(
                    "Lecture unitaire : dynamique temporelle, changement de régime, cohérence avec les diagnostics."
                )
            )
            lines.append("")
            lines.append(_include_figure(rel, caption=fk, label=f"fig:{fk.replace('.', '-')[:50]}"))
            lines.append("")

        # --- Conclusions pilotées ---
        lines.append(r"\subsection{Conclusion de section}")

        if sec.key == "sec_stationarity":
            m_tsds = metrics_cache.get("m.diag.ts_vs_ds", {})
            lines.append(_stationarity_conclusion(m_tsds))

        elif sec.key == "sec_univariate":
            m_uni = metrics_cache.get("m.uni.best", {})
            m_tsds = metrics_cache.get("m.diag.ts_vs_ds", {})
            lines.append(_univariate_conclusion(m_uni, m_tsds))

        elif sec.key == "sec_anthropology":
            m = metrics_cache.get("m.anthro.todd_analysis", {})
            md = m.get("markdown") if isinstance(m, dict) else None
            if md:
                lines.append(_md_basic_to_tex(md))
            else:
                lines.append(_md_basic_to_tex("Aucune synthèse anthropologique disponible dans les métriques."))

        elif sec.key == "sec_multivariate":
            m_var_meta = metrics_cache.get("m.var.meta", {})
            m_var_sims = metrics_cache.get("m.var.sims", {})
            m_var_audit = metrics_cache.get("m.var.audit", {})
            lines.append(_multivariate_conclusion(m_var_meta, m_var_sims, m_var_audit))

        elif sec.key == "sec_cointegration":
            m_coint = metrics_cache.get("m.coint.meta", {})
            lines.append(_cointegration_conclusion(m_coint))

        else:
            note_key = f"m.note.{sec.key.replace('sec_', 'step')}"
            note = metrics_cache.get(note_key, {}).get("markdown")
            if note:
                lines.append(_md_basic_to_tex(note))
            else:
                lines.append(
                    _md_basic_to_tex(
                        "Conclusion automatique : les résultats ci-dessus structurent la décision de modélisation "
                        "et la section suivante exploite ces diagnostics."
                    )
                )

        out_path = blocks_dir / f"{sec.key}.tex"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        out_map[sec.key] = f"latex/blocks/{sec.key}.tex"

    return out_map
