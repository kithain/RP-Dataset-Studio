@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo L'application n'est pas installee. Lance install.bat d'abord.
  pause
  exit /b 1
)
echo Ouverture locale : http://127.0.0.1:8501
echo Pour fermer : CTRL+C dans cette fenetre.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --server.maxUploadSize 1024 --browser.gatherUsageStats false
if errorlevel 1 pause
