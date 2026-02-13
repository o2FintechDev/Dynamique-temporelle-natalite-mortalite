# app/streamlit_app.py
import sys
from pathlib import Path
import os
import json
import requests
import re
import streamlit as st
from dotenv import load_dotenv
load_dotenv() 
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.run_reader import list_runs, RunManager, read_metric_json
from src.utils.session_state import get_state
from src.agent.executor import AgentExecutor
from src.agent.schemas import Plan, ToolCall

st.set_page_config(page_title="AnthroDem Lab", layout="wide")
state = get_state()

Y = "Croissance_Naturelle"

STEP_TO_PAGE = {
    "step1_load_and_profile": "1_Exploration",
    "step2_descriptive":      "2_Analyse_Descriptive",
    "step3_stationarity":     "3_Modeles",
    "step4_univariate":       "3_Modeles",
    "step5_var":              "3_Modeles",
    "step6_cointegration":    "4_Resultats",
    "step7_anthropology":     "5_Analyse_Anthropologique",
    "export_latex_pdf":       "6_Historique_Artefacts",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

def _is_email(s: str) -> bool:
    return bool(EMAIL_RE.fullmatch((s or "").strip()))

def _extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def run_step(step_name: str, params: dict) -> str:
    # inject routing page AVANT création du Plan
    params = dict(params)  # copy
    params["_page"] = STEP_TO_PAGE.get(step_name, None)

    plan = Plan(
        intent=step_name,
        tool_calls=[ToolCall(tool_name=step_name, variables=[Y], params=params)],
    )
    ex = AgentExecutor(run_id=state.selected_run_id)
    res = ex.run(plan, user_query=f"{step_name} via chatbot")
    state.selected_run_id = res["run_id"]
    return res["run_id"]


def append_assistant(md: str) -> None:
    st.session_state.chat_messages.append({"role": "assistant", "content": md})


def show_note(run_id: str, note_label: str) -> None:
    p = RunManager.get_artefact_path(note_label, run_id=run_id)
    if p:
        payload = read_metric_json(p)
        md = payload.get("markdown")
        if md:
            append_assistant(md)

def _truncate(s: str, max_chars: int) -> str:
    s = "" if s is None else str(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n[...TRONQUÉ...]"

def _groq_post(api_key: str, payload: dict, *, timeout: int):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if r.status_code != 200:
        # DEBUG CRITIQUE: tu veux voir r.text, pas juste raise_for_status()
        raise RuntimeError(f"Groq {r.status_code}: {r.text}")
    return r.json()

# ---------------------------
# LLM Router (API) + fallback
# ---------------------------
PLAN_STEPS = [
    ("step1_load_and_profile", "m.note.step1", "Traitement des données"),
    ("step2_descriptive",      "m.note.step2", "Analyse descriptive + décomposition"),
    ("step3_stationarity",     "m.note.step3", "Diagnostics & stationnarité"),
    ("step4_univariate",       "m.note.step4", "Analyse univariée"),
    ("step5_var",              "m.note.step5", "Analyse multivariée VAR"),
    ("step6_cointegration",    "m.note.step6", "Cointégration & long terme"),
    ("step7_anthropology",     "m.note.step7", "Analyse anthropologique"),
]

def next_step_hint(step_name: str) -> str | None:
    names = [s[0] for s in PLAN_STEPS]
    if step_name in names:
        i = names.index(step_name)
        return names[i + 1] if i + 1 < len(names) else "export_latex_pdf"
    if step_name == "RUN_ALL":
        return "export_latex_pdf"
    return None

def fallback_route(user_text: str) -> dict:
    t = user_text.strip().lower()

    # global
    if any(k in t for k in ["run all", "run_all", "pipeline complet", "tout", "lance tout", "analyse complète"]):
        return {"action": "RUN_ALL"}
    if any(k in t for k in ["export", "latex", "pdf", "rapport"]):
        return {"action": "export_latex_pdf"}

    # steps
    if any(k in t for k in ["step1", "étape 1", "traitement", "nettoyage", "profil", "coverage"]):
        return {"action": "step1_load_and_profile"}
    if any(k in t for k in ["step2", "étape 2", "décomposition", "decomposition", "stl", "descriptive"]):
        return {"action": "step2_descriptive"}
    if any(k in t for k in ["step3", "étape 3", "stationnar", "adf", "phillips", "pp", "acf", "pacf", "diagnostic"]):
        return {"action": "step3_stationarity"}
    if any(k in t for k in ["step4", "étape 4", "arima", "univari", "hurst", "rs", "rescaled"]):
        return {"action": "step4_univariate"}
    if any(k in t for k in ["step5", "étape 5", "var", "granger", "irf", "fevd"]):
        return {"action": "step5_var"}
    if any(k in t for k in ["step6", "étape 6", "cointegr", "johansen", "engle", "vecm"]):
        return {"action": "step6_cointegration"}
    if any(k in t for k in ["step7", "étape 7", "todd", "anthrop"]):
        return {"action": "step7_anthropology"}

    if any(k in t for k in ["aide", "help", "?"]):
        return {"action": "HELP"}

    return {"action": None}

def llm_route(user_text: str) -> dict:
    """
    Retour: {"action": <tool_name|RUN_ALL|export_latex_pdf|HELP|None>, "confidence": float, "reason": str}
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        out = fallback_route(user_text)
        out["confidence"] = 0.50 if out["action"] else 0.0
        out["reason"] = "fallback(no_GROQ_API_KEY)"
        return out

    system = (
        "Tu es un routeur d'intentions pour une app économétrique. "
        "Choisis UNE action autorisée. Réponds STRICTEMENT en JSON minifié, sans texte autour."
    )

    # éviter une entrée trop longue qui casse le router
    user_text_small = _truncate(user_text, int(os.getenv("GROQ_ROUTER_USER_MAX_CHARS", "2000")))

    user = (
        "Actions autorisées:\n"
        "- step1_load_and_profile\n- step2_descriptive\n- step3_stationarity\n- step4_univariate\n"
        "- step5_var\n- step6_cointegration\n- step7_anthropology\n- RUN_ALL\n- export_latex_pdf\n- HELP\n- null\n\n"
        f"Texte utilisateur: {user_text_small}\n\n"
        'JSON minifié attendu: {"action":"<...|null>","confidence":0-1,"reason":"..."}'
    )

    allowed = {s[0] for s in PLAN_STEPS} | {"RUN_ALL", "export_latex_pdf", "HELP", None}

    # modèle router “safe”
    model = os.getenv("GROQ_MODEL_ROUTER") or "llama3-8b-8192"

    try:
        data = _groq_post(
            api_key,
            {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.0,
                "max_tokens": int(os.getenv("GROQ_ROUTER_MAX_TOKENS", "120")),
                # optionnel mais stabilise la conformité JSON si supporté
                # "response_format": {"type": "json_object"},
            },
            timeout=30,
        )

        content = (data["choices"][0]["message"]["content"] or "").strip()

        # Réparation légère: extrait le premier objet JSON si le modèle “bave”
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            content = m.group(0)

        obj = json.loads(content)

        action = obj.get("action", None)
        if action == "null":
            action = None

        if action not in allowed:
            out = fallback_route(user_text)
            out["confidence"] = 0.40 if out["action"] else 0.0
            out["reason"] = "llm_invalid_action_fallback"
            return out

        conf = obj.get("confidence", 0.7)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.7

        return {"action": action, "confidence": conf, "reason": obj.get("reason", "llm")}

    except Exception:
        out = fallback_route(user_text)
        out["confidence"] = 0.40 if out["action"] else 0.0
        out["reason"] = "llm_error_fallback"
        return out

def _collect_run_context(run_id: str | None) -> str:
    if not run_id:
        return "Aucun run actif. Réponds en définitions générales."

    per_section_max = int(os.getenv("GROQ_CTX_SECTION_MAX_CHARS", "3500"))

    chunks = []
    for _, note_label, title in PLAN_STEPS:
        p = RunManager.get_artefact_path(note_label, run_id=run_id)
        if not p:
            continue
        payload = read_metric_json(p) or {}
        md = payload.get("markdown")
        if md:
            md = _truncate(md, per_section_max)
            chunks.append(f"## {title}\n{md}")

    return "\n\n".join(chunks) if chunks else "Run actif mais aucune note disponible."

def llm_answer(user_text: str, *, run_id: str | None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    ctx = _collect_run_context(run_id)

    if not api_key:
        return (
            "Mode sans LLM (GROQ_API_KEY absent).\n\n"
            "Je peux: (1) exécuter des étapes via mots-clés, (2) expliquer concepts (ADF, VAR, cointégration, ARIMA)."
        )

    # modèle QA “safe”
    model = os.getenv("GROQ_MODEL_QA") or "llama3-70b-8192"

    # troncature agressive du contexte (c’est LE point critique)
    ctx_small = _truncate(ctx, int(os.getenv("GROQ_QA_CTX_MAX_CHARS", "20000")))
    user_small = _truncate(user_text, int(os.getenv("GROQ_QA_USER_MAX_CHARS", "6000")))

    system = (
        "Tu es un assistant économétrie senior pour le projet AnthroDem Lab (Croissance Naturelle 1975-2025). "
        "Tu réponds de façon professionnelle, structurée, concise. "
        "Si le contexte run contient des résultats, tu les utilises. "
        "Si l'utilisateur demande une action (exécuter une étape), dis explicitement quelle commande taper (stepX/export)."
    )

    user = (
        f"Contexte (notes du run):\n{ctx_small}\n\n"
        f"Question utilisateur:\n{user_small}\n\n"
        "Réponds en français."
    )

    data = _groq_post(
        api_key,
        {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": int(os.getenv("GROQ_QA_MAX_TOKENS", "900")),
        },
        timeout=60,
    )

    return data["choices"][0]["message"]["content"].strip()

def help_text() -> str:
    return (
        "Je peux exécuter une étape à partir d'une phrase libre.\n\n"
        "**Exemples**:\n"
        "- `Nettoie les données et fais le profil` → step1\n"
        "- `Fais la décomposition STL` → step2\n"
        "- `Lance les tests ADF/PP et TS vs DS` → step3\n"
        "- `Estime un ARIMA et diagnostics` → step4\n"
        "- `VAR + IRF/FEVD` → step5\n"
        "- `Cointégration Johansen + VECM` → step6\n"
        "- `Analyse Todd` → step7\n"
        "- `Lance tout` → RUN_ALL\n"
        "- `Export PDF` → export\n"
        "- `Export PDF vers prenom.nom@domaine.com` → export + envoi mail\n\n"
        "Paramètres d’envoi mail configurables dans la sidebar."
    )

# ---------------------------
# UI
# ---------------------------
st.title("AnthroDem Lab — Chatbot économétrique (Croissance naturelle, 1975–2025)")
st.caption("Contrat: variable cible unique (Croissance_Naturelle). Les pages lisent via manifest.lookup (zéro recalcul).")

runs = list_runs()

with st.sidebar:
    st.header("Runs")

    options = ["(Aucun run actif)"] + runs
    current = state.selected_run_id
    current_label = current if current else "(Aucun run actif)"

    # position dans la liste
    try:
        idx = options.index(current_label)
    except ValueError:
        idx = 0  # aucun par défaut

    chosen = st.selectbox("Run actif", options, index=idx)

    if chosen == "(Aucun run actif)":
        state.selected_run_id = None
        st.caption("Run: Aucun")
    else:
        state.selected_run_id = chosen
        st.caption(f"Run: {chosen}")

    st.divider()

    # ---------------------------
    # Export PDF + Email settings
    # ---------------------------
    st.header("Export PDF")

    if "mail_send_enabled" not in st.session_state:
        st.session_state.mail_send_enabled = False
    if "mail_to_email" not in st.session_state:
        st.session_state.mail_to_email = ""
    if "mail_subject" not in st.session_state:
        st.session_state.mail_subject = "Rapport économétrique – Croissance naturelle"
    if "mail_body" not in st.session_state:
        st.session_state.mail_body = "Veuillez trouver le rapport en pièce jointe."

    st.session_state.mail_send_enabled = st.checkbox(
        "Envoyer par mail après export",
        value=bool(st.session_state.mail_send_enabled),
    )
    st.session_state.mail_to_email = st.text_input(
        "Email destinataire",
        value=st.session_state.mail_to_email,
        placeholder="prenom.nom@domaine.com",
        disabled=not st.session_state.mail_send_enabled,
    )
    st.session_state.mail_subject = st.text_input(
        "Objet",
        value=st.session_state.mail_subject,
        disabled=not st.session_state.mail_send_enabled,
    )
    st.session_state.mail_body = st.text_area(
        "Message",
        value=st.session_state.mail_body,
        height=90,
        disabled=not st.session_state.mail_send_enabled,
    )

    if st.session_state.mail_send_enabled and not _is_email(st.session_state.mail_to_email):
        st.warning("Email invalide: l’envoi sera bloqué tant qu’il n’est pas valide.")

st.divider()
st.subheader("Chat")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Décris ce que tu veux faire (ex: « fais la décomposition STL » ou « lance tout »)."}
    ]

for m in st.session_state.chat_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ta demande...")
if user_msg:
    st.session_state.chat_messages.append({"role": "user", "content": user_msg})

    route = llm_route(user_msg)
    action = route.get("action")
    conf = route.get("confidence", 0.0)
    reason = route.get("reason", "")

    if action == "HELP":
        append_assistant(help_text())
        st.rerun()

    if action is None:
        answer = llm_answer(user_msg, run_id=state.selected_run_id)
        append_assistant(answer)
        st.rerun()

    # Exécution
    if action == "RUN_ALL":
        # séquence 1→7
        rid = run_step("step1_load_and_profile", {"y": Y})
        run_step("step2_descriptive", {"y": Y})
        run_step("step3_stationarity", {"y": Y, "lags": 24})
        run_step("step4_univariate", {"y": Y})
        run_step("step5_var", {"y": Y, "vars_mode": "decomp"})
        run_step("step6_cointegration", {"y": Y, "vars_mode": "decomp"})
        run_step("step7_anthropology", {"y": Y})

        append_assistant(f"RUN_ALL terminé. run_id={state.selected_run_id} (route={reason}, confidence={conf:.2f}).")
        show_note(state.selected_run_id, "m.note.step1")
        show_note(state.selected_run_id, "m.note.step2")
        append_assistant("Tu peux maintenant aller sur les pages 2→6 pour voir tables/figures, puis taper `export pdf`.")
        st.rerun()

    if action == "export_latex_pdf":
        rid = state.selected_run_id or RunManager.get_latest_run_id()
        if not rid:
            append_assistant("Export impossible: aucun run disponible. Exécute d’abord une étape (ex: « step1 »).")
            st.rerun()

        # --- email: (1) email dans le message (override), sinon (2) sidebar settings ---
        email_in_msg = _extract_email(user_msg)
        send_enabled = bool(st.session_state.get("mail_send_enabled", False))
        to_email = (email_in_msg or st.session_state.get("mail_to_email", "")).strip()

        params = {"run_id": rid}

        if email_in_msg:
            # si l'utilisateur a tapé un email dans le chat, on force l'envoi
            send_enabled = True

        if send_enabled:
            if not _is_email(to_email):
                append_assistant(
                    "Envoi mail demandé mais adresse invalide. "
                    "Renseigne un email valide dans la sidebar ou tape: `export pdf vers prenom.nom@domaine.com`."
                )
                st.rerun()

            params["to_email"] = to_email
            params["mail_subject"] = (st.session_state.get("mail_subject") or "Rapport économétrique (PDF)").strip()
            params["mail_body"] = (st.session_state.get("mail_body") or "Veuillez trouver le rapport en pièce jointe.").strip()

        rid2 = run_step("export_latex_pdf", params)

        if send_enabled:
            append_assistant(
                f"Export + envoi mail demandé sur run_id={rid}. Destinataire={to_email}. "
                f"(route={reason}, confidence={conf:.2f})"
            )
        else:
            append_assistant(f"Export demandé sur run_id={rid}. (route={reason}, confidence={conf:.2f})")

        st.rerun()

    # Étapes 1..7
    params = {"y": Y}
    if action == "step3_stationarity":
        params["lags"] = 24
    if action in {"step5_var", "step6_cointegration"}:
        params["vars_mode"] = "decomp"

    rid = run_step(action, params)
    append_assistant(f"Étape exécutée: `{action}`. run_id={rid}. (route={reason}, confidence={conf:.2f})")

    # afficher note persistée de l'étape
    note_label = None
    for s_name, s_note, _ in PLAN_STEPS:
        if s_name == action:
            note_label = s_note
            break
    if note_label:
        show_note(rid, note_label)

    # proposer next step
    nxt = next_step_hint(action)
    if nxt:
        append_assistant(f"Étape suivante suggérée: `{nxt}`.")

    st.rerun()
