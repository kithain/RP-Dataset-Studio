@echo off
setlocal
cd /d "%~dp0"
echo =================================================
echo   RP Dataset Studio - Installation autonome
echo   Python portable + application + Qwen 2B Heretic
echo =================================================
echo Internet requis pour la premiere installation.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
set "INSTALL_RESULT=%ERRORLEVEL%"
if not "%RPDS_NO_PAUSE%"=="1" pause
exit /b %INSTALL_RESULT%
