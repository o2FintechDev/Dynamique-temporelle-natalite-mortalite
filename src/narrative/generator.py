from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, List

from .schema import NarrativePacket, Chapter, Section, Paragraph, EvidenceRef

def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.3f}"

def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"

def _metric_path(manifest: Dict[str, Any], key: str) -> Optional[str]:
    # cherche dans lookup plat ou lookup typé
    lookup = manifest.get("lookup", {}) or {}
    if isinstance(lookup, dict):
        if key in lookup and isinstance(lookup[key], str):
            return lookup[key]
        for v in lookup.values():
            if isinstance(v, dict) and key in v and isinstance(v[key], str):
                return v[key]
    # fallback: artefacts.metrics
    for it in (manifest.get("artefacts", {}) or {}).get("metrics", []) or []:
        if it.get("key") == key or it.get("label") == key:
            return it.get("path")
    return None

def _load_metric(manifest: Dict[str, Any], run_root: Path, key: str) -> Dict[str, Any]:
    p = _metric_path(manifest, key)
    if not p:
        return {}
    path = (run_root / p).resolve() if not Path(p).is_absolute() else Path(p)
    if not path.exists():
        return {}
    return _read_json(path)

def build_narrative_from_run(run_root: Path, manifest: Dict[str, Any]) -> NarrativePacket:
    rid = manifest.get("run_id", run_root.name)

    m1 = _load_metric(manifest, run_root, "m.note.step1")
    m2 = _load_metric(manifest, run_root, "m.desc.key_points")
    m3 = _load_metric(manifest, run_root, "m.diag.ts_vs_ds")
    m4 = _load_metric(manifest, run_root, "m.uni.best")
    m7 = _load_metric(manifest, run_root, "m.anthro.todd_analysis")

    chapters: List[Chapter] = []
    chapters.append(_ch_intro(m1))
    chapters.append(_ch_descriptive(m2))
    chapters.append(_ch_stationarity(m3))
    chapters.append(_ch_univariate(m4, m3))
    chapters.append(_ch_historical(m2))
    chapters.append(_ch_anthropology(m2, m7))

    return NarrativePacket(
        run_id=rid,
        chapters=chapters,
        meta={"generator": "narrative_v2_rule_based", "alpha": 0.05},
    )

def save_narrative_packet(run_root: Path, packet: NarrativePacket) -> Path:
    out_dir = run_root / "artefacts" / "narrative"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "narrative.json"

    # Pydantic v1
    out_path.write_text(
        packet.json(indent=2, exclude_none=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path

# ---------------- Chapters ----------------

def _ch_intro(m1: Dict[str, Any]) -> Chapter:
    md = m1.get("markdown")
    p = (
        "Ce rapport est généré automatiquement à partir des artefacts persistés (figures, tables, métriques) afin de garantir la traçabilité. "
        "L’objectif est de caractériser la dynamique temporelle du solde naturel et d’identifier les ruptures/régimes pertinents."
    )
    if isinstance(md, str) and md.strip():
        p += "\n\n" + md.strip()
    return Chapter(
        title="Introduction et préparation",
        sections=[Section(title="Cadre et données", paragraphs=[Paragraph(text_md=p, evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.note.step1.json", label="Note Step1")])])],
    )

def _ch_descriptive(m2: Dict[str, Any]) -> Chapter:
    kp = (m2.get("key_points") if isinstance(m2, dict) else {}) or {}
    trend = kp.get("trend")
    ws = kp.get("window_stats")
    p = "L’analyse descriptive caractérise tendance, volatilité et changements de régime, préalable à toute spécification."
    if trend:
        p += f"\n\nTendance qualitative : **{trend}**."
    if isinstance(ws, dict):
        pre = ws.get("2017-01..2019-12", {}).get("mean")
        covid = ws.get("2020-03..2021-12", {}).get("mean")
        p += f"\n\nMoyennes par fenêtres : pré-Covid={_fmt(pre)}, Covid={_fmt(covid)}."
    return Chapter(
        title="Analyse descriptive",
        sections=[Section(title="Tendance et structure empirique", paragraphs=[Paragraph(text_md=p, evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.desc.key_points.json", label="Key points Step2")])])],
    )

def _ch_stationarity(m3: Dict[str, Any]) -> Chapter:
    kp = m3.get("key_points", {}) or m3  # selon ta structure
    verdict = kp.get("verdict")
    adf_c = kp.get("adf_p_c")
    adf_ct = kp.get("adf_p_ct")

    p = (
    "La décision TS/DS est fondée exclusivement sur l’ADF. "
    f"ADF(c) p={_fmt_p(adf_c)}, ADF(ct) p={_fmt_p(adf_ct)}. "
    f"Verdict : **{verdict}**."
    )
    return Chapter(
        title="Diagnostics statistiques",
        sections=[Section(title="Stationnarité (TS vs DS)", paragraphs=[Paragraph(text_md=p, evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.diag.ts_vs_ds.json", label="Stationnarité Step3")])])],
    )

def _ch_univariate(m4: Dict[str, Any], m3: Dict[str, Any]) -> Chapter:
    kp = (m4.get("key_points") if isinstance(m4, dict) else {}) or {}
    order = kp.get("order")
    aic = kp.get("aic")
    bic = kp.get("bic")
    lb = kp.get("lb_p")
    jb = kp.get("jb_p")
    arch = kp.get("arch_p")
    hurst = kp.get("hurst")

    p = "La modélisation ARIMA capture la dynamique moyenne, puis les diagnostics résiduels valident (ou invalident) les hypothèses."
    if order:
        p += f"\n\nModèle retenu : **ARIMA{order}**, AIC={_fmt(aic)}, BIC={_fmt(bic)}."
    p += f"\n\nRésidus : Ljung–Box p={_fmt_p(lb)}, JB p={_fmt_p(jb)}, ARCH p={_fmt_p(arch)}."

    if lb is not None and lb < 0.05:
        p += "\n\nLes résidus ne sont pas blancs : la spécification est perfectible (saisonnalité, ruptures, dynamique multivariée)."
    if arch is not None and arch < 0.05:
        p += "\n\nHétéroscédasticité conditionnelle significative : une extension ARIMA–GARCH est économétriquement justifiée si l’objectif inclut la variance."
    if hurst is not None and abs(hurst) < 0.1:
        p += f"\n\nHurst≈{_fmt(hurst,3)} : pas d’évidence robuste de mémoire longue sur la série transformée."

    # note méthodo TS vs d=1 (si verdict TS)
    v = ((m3.get("key_points") if isinstance(m3, dict) else {}) or {}).get("verdict")
    if v == "TS" and isinstance(order, str) and ", 1," in order.replace(" ", ""):
        p += "\n\nPoint méthodologique : le recours à d=1 doit être comparé à une spécification au niveau avec tendance, afin de trancher TS (tendance déterministe) vs DS (intégration)."
    return Chapter(
        title="Modélisation univariée",
        sections=[Section(title="ARIMA et diagnostics", paragraphs=[Paragraph(text_md=p, evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.uni.best.json", label="ARIMA Step4")])])],
    )

def _ch_historical(m2: Dict[str, Any]) -> Chapter:
    kp = (m2.get("key_points") if isinstance(m2, dict) else {}) or {}
    ws = kp.get("window_stats") or {}
    bps = kp.get("breakpoints") or []
    if not isinstance(ws, dict) and not isinstance(bps, list):
        txt = "Aucun signal historique robuste n’est disponible dans les métriques descriptives."
        return Chapter(title="Lecture historique", sections=[Section(title="Contraintes empiriques", paragraphs=[Paragraph(text_md=txt)])])

    pre = ws.get("2017-01..2019-12", {}).get("mean") if isinstance(ws, dict) else None
    covid = ws.get("2020-03..2021-12", {}).get("mean") if isinstance(ws, dict) else None
    post2023 = ws.get("2023-01..2025-12", {}).get("mean") if isinstance(ws, dict) else None

    txt = "Lecture historique (strictement contrainte par les métriques)."
    if any(isinstance(bp, dict) and bp.get("tag") == "covid_signal" for bp in bps):
        txt += "\n\nUne rupture est détectée autour de 2020–2021, compatible avec un choc sanitaire affectant la mortalité et le solde naturel."
    if pre is not None and covid is not None:
        txt += f"\n\nMoyenne pré-Covid={_fmt(pre)} vs Covid={_fmt(covid)} (écart={_fmt(covid-pre)})."
    if any(isinstance(bp, dict) and bp.get("tag") == "inversion_2023_signal" for bp in bps):
        txt += "\n\nUn signal de changement de régime apparaît autour de 2023, cohérent avec une fragilisation structurelle du solde naturel."
    if pre is not None and post2023 is not None:
        txt += f"\n\nMoyenne pré-Covid={_fmt(pre)} vs 2023+={_fmt(post2023)} (écart={_fmt(post2023-pre)})."

    txt += "\n\nCette lecture ne conclut pas à une causalité : elle documente des coïncidences temporelles entre ruptures et événements."
    return Chapter(
        title="Lecture historique",
        sections=[Section(title="Covid et bascule 2023", paragraphs=[Paragraph(text_md=txt, evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.desc.key_points.json", label="Step2 windows/breakpoints")])])],
    )

def _ch_anthropology(m2: Dict[str, Any], m7: Dict[str, Any]) -> Chapter:
    kp = (m2.get("key_points") if isinstance(m2, dict) else {}) or {}
    has2023 = bool(kp.get("inversion_2023"))
    has_covid = bool(kp.get("covid_signal"))

    txt = (
        "Lecture anthropologique (cadre toddien) : interprétation **conditionnelle** des régularités observées.\n\n"
        "Si les ruptures/régimes sont robustes, elles suggèrent une dynamique démographique pilotée par phases historiques plutôt qu’un processus unique : "
        "transformations des structures familiales, arbitrages fécondité/activité, et sensibilité accrue aux chocs.\n\n"
        "Hypothèse structurante : la fragilisation du solde naturel correspond à une recomposition du contrat intergénérationnel (normes, politiques publiques, incertitude)."
    )
    if has_covid:
        txt += "\n\nLe choc 2020–2021 peut être lu comme révélateur d’une vulnérabilité du régime démographique : hausse de mortalité immédiate et effets différés sur la natalité."
    if has2023:
        txt += "\n\nLe signal 2023 est interprétable comme point de bascule : le solde naturel devient plus fragile, renforçant l’intérêt d’un raisonnement en régimes."

    # si step7 existe (markdown), on l’annexe comme “note”
    md7 = m7.get("markdown") if isinstance(m7, dict) else None
    paras = [Paragraph(text_md=txt, evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.desc.key_points.json", label="Base empirique Step2")])]
    if isinstance(md7, str) and md7.strip():
        paras.append(Paragraph(text_md="Note (Step7) :\n\n" + md7.strip(), importance="support", evidence=[EvidenceRef(kind="metric", path="artefacts/metrics/m.anthro.todd_analysis.json", label="Step7")]))
    return Chapter(
        title="Synthèse anthropologique augmentée",
        sections=[Section(title="Interprétation contrainte", paragraphs=paras)],
    )
