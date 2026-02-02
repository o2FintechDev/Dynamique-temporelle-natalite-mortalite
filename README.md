# AnthroDem Lab

Application Streamlit avec agent IA interne (offline par défaut) pour explorer, visualiser et commenter des données démographiques françaises et indicateurs macro mensuels via APIs officielles (INSEE, FRED).

## Objectifs (phase démarrage)
- Architecture modulaire, reproductible, orientée “produit”
- Accès données **exclusivement via APIs** (pas de fichiers locaux utilisateur)
- Cache mémoire + disque, avec fallback offline (sur cache)
- Agent IA interne:
  - interprétation intention (exploration / comparaison / synthèse)
  - plan d’actions structuré (outils à appeler)
  - exécution + artefacts (figures, tables, métriques)
  - narration factuelle **auditée** (aucune phrase sans artefact)

## Configuration (secrets / env)
Créer un `.env` (ou configurer Secrets Streamlit Cloud) :
- `FRED_API_KEY` (optionnel mais recommandé)
- `INSEE_CLIENT_ID`, `INSEE_CLIENT_SECRET` (requis pour INSEE)
- `ANTHRODEM_OFFLINE=1` pour forcer le mode offline (lecture cache uniquement)

Voir `.env.example`.

## Lancement local
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
