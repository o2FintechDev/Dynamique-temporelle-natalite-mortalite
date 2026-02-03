from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

from src.utils.settings import settings
from src.utils.session_state import get_session

st.set_page_config(page_title="Analyse Anthropologique", layout="wide")
st.title("Analyse Anthropologique — Hypothèses cadrées (preuves vs interprétation)")

sess = get_session(st)

if not sess.current_run_id:
    st.info("Aucun run actif. Lance un run depuis Accueil/Agent.")
    st.stop()

run_dir = settings.outputs_dir / "runs" / sess.current_run_id
manifest = run_dir / "manifest.json"
if not manifest.exists():
    st.warning("manifest.json absent.")
    st.stop()

man = json.loads(manifest.read_text(encoding="utf-8"))
artefacts = man.get("artefacts", [])

st.header("Ruptures observées (uniquement si artefacts existent)")
st.info(
    "Cette page est volontairement prudente: aucune affirmation sans artefact. "
    "Les ruptures (breaks/regimes) seront intégrées Jour 6/7. "
    "À ce stade, on s’appuie sur coverage/tests/graphiques déjà produits."
)

st.subheader("Preuves disponibles (inventaire)")
for a in artefacts:
    st.write(f"- {a['artefact_id']} | {a['kind']} | {a['name']}")

st.header("Lecture anthropologique (hypothèses)")
st.markdown(
    """
**Cadre**  
- Les résultats économétriques décrivent des dynamiques (tendance/saisonnalité/stationnarité/cointégration) sur séries agrégées.  
- Une lecture anthropologique propose des *hypothèses* sur les mécanismes sociaux, sans prétendre à la causalité.

**Référence prudente à Emmanuel Todd**  
- Todd mobilise des structures familiales, normes et trajectoires socio-historiques.  
- Ici, toute mise en perspective doit rester *non assertive* tant que les ruptures/événements ne sont pas documentés par artefacts datés.

**Hypothèses (à tester ensuite)**  
- H1 : les inflexions de natalité/mariages coïncident avec des chocs institutionnels/économiques (à documenter).  
- H2 : la mortalité présente des ruptures exogènes (ex: crises sanitaires), puis retour au régime (à estimer).  
- H3 : l’IPC / masse monétaire peuvent modifier le calendrier des comportements démographiques via contraintes budgétaires (à tester par VAR/VECM + IRF).
"""
)

st.caption("La séparation stricte preuves/hypothèses est imposée et sera durcie quand les ruptures seront modélisées.")
