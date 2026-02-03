from __future__ import annotations
import os
import streamlit as st

from src.utils import settings, SessionStore
from src.data_api import CachedHttpClient, FredClient, InseeClient, VARIABLES
from src.agent import make_plan, execute_plan, build_narrative_packet
from src.agent.tools import ToolContext
from src.narrative import render_constrained_narrative

st.set_page_config(page_title=settings.app_name, layout="wide")

def init_state():
    if "store" not in st.session_state:
        st.session_state.store = SessionStore()
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_plan" not in st.session_state:
        st.session_state.last_plan = None
    if "last_packet" not in st.session_state:
        st.session_state.last_packet = None
    if "last_state" not in st.session_state:
        st.session_state.last_state = {}

init_state()

# Clients API (singletons par session)
if "http" not in st.session_state:
    st.session_state.http = CachedHttpClient()
if "fred" not in st.session_state:
    st.session_state.fred = FredClient(st.session_state.http)
if "insee" not in st.session_state:
    st.session_state.insee = InseeClient(st.session_state.http)

ctx = ToolContext(
    http=st.session_state.http,
    fred=st.session_state.fred,
    insee=st.session_state.insee,
)

st.title("AnthroDem Lab — Accueil / Agent")

with st.sidebar:
    st.subheader("Variables disponibles")
    st.caption("Le catalogue est extensible. L’agent peut matcher par identifiant ou mots-clés.")
    for vid, spec in VARIABLES.items():
        st.write(f"- `{vid}` — {spec.label} ({spec.provider})")

    st.divider()
    st.subheader("Mode exécution")
    st.write(f"Offline: `{settings.offline}`")
    st.caption("Offline=1 => lecture cache uniquement. Sinon online avec fallback cache.")
    st.write(f"Cache HTTP: `{settings.http_cache_path}`")

# Historique chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Demande: exploration / comparaison / synthèse… (ex: 'compare unrate_us vs cpi_us')")
if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Planification
    plan = make_plan(user_text)
    st.session_state.last_plan = plan

    with st.chat_message("assistant"):
        st.markdown("**Plan d’actions (JSON)**")
        st.json(plan.model_dump())

        # Exécution
        try:
            state = execute_plan(plan, ctx, st.session_state.store)
            st.session_state.last_state = state

            packet = build_narrative_packet(plan, st.session_state.store)
            st.session_state.last_packet = packet

            st.markdown("**Artefacts produits**")
            for a in st.session_state.store.list()[-6:]:
                st.write(f"- `{a.artefact_id}` [{a.type}] {a.title}")

            st.markdown("**Synthèse narrative (auditée, basée sur artefacts)**")
            narrative = render_constrained_narrative(st.session_state.store)
            st.markdown(narrative)

            # Affichage immédiat (sans navigation)
            plot_id = state.get("plot_artefact_id")
            if plot_id:
                fig = st.session_state.store.get(plot_id).payload
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Exécution interrompue: {e}")
            st.info("Causes fréquentes: clés INSEE manquantes, mode offline sans cache préalable, série inconnue.")
