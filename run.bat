@echo off
title AI Face ID Security System - XGBoost & Random Forest
cd /d "%~dp0"

echo ===============================================================
echo     AI FACE ID BIOMETRIC SECURITY SYSTEM
echo     Dual Engine: XGBoost + Random Forest
echo ===============================================================
echo.
echo Starting Application...

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" run.py
) else (
    python run.py
)

pause
