@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=.runtimes\python\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo L'application n'est pas installee. Lance install.bat d'abord.
  pause
  exit /b 1
)
"%APP_PYTHON%" launch_local.py
if errorlevel 1 pause
