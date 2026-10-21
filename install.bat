@echo off
setlocal
cd /d "%~dp0"
echo =================================================
echo   RP Dataset Studio - Installation locale Windows
echo =================================================
set "PYCMD="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 set "PYCMD=py -3.12"
)
if not defined PYCMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 --version >nul 2>nul
        if not errorlevel 1 set "PYCMD=py -3.11"
    )
)
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    echo Python est introuvable. Installe Python 3.12 puis reessaie.
    echo https://www.python.org/downloads/windows/
    pause
    exit /b 1
)
%PYCMD% -m venv .venv
if errorlevel 1 (
    echo Echec de creation de l'environnement Python.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Installation terminee. Double-clique sur lancer.bat
echo.
pause
exit /b 0
:error
echo Erreur pendant l'installation. Consulte les messages ci-dessus.
pause
exit /b 1
