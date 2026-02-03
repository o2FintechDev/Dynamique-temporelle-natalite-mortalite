# AnthroDem Lab

Application Streamlit + agent IA offline pour exploration/diagnostic/modélisation de séries mensuelles (natalité/mortalité + macro FR) avec traçabilité totale et export LaTeX/PDF.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
