@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "src\" (
  echo [ERREUR] "src" introuvable. Ce .bat doit etre a la racine du projet.
  pause
  exit /b 1
)

set "PYTHONPATH=%cd%"

if exist "venv\Scripts\activate.bat" (
  call "venv\Scripts\activate.bat"
)

python -m streamlit run "app\streamlit_app.py"

if errorlevel 1 (
  echo.
  echo [ERREUR] Streamlit KO (code=%errorlevel%).
  pause
  exit /b %errorlevel%
)

endlocal