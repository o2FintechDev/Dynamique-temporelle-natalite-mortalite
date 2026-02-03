@echo off
REM Lancer l'application Streamlit AnthroDem Lab

REM Se déplacer dans le dossier du projet
cd /d "C:\Users\audeb\Google Drive\M2\Econométrie_IA"

REM Activer l'environnement virtuel (si vous en avez un)
REM call venv\Scripts\activate

REM Lancer Streamlit
streamlit run app\streamlit_app.py

REM Garder la fenêtre ouverte en cas d'erreur
pause