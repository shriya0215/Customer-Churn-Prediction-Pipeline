@echo off
title Customer Churn Prediction - REST API Server
echo =====================================================================
echo           Starting Customer Churn Prediction REST API Server
echo =====================================================================
echo.

:: Detect Python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=py
    ) else (
        echo [ERROR] Python was not found in PATH!
        echo Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
)

echo Starting FastAPI server at http://127.0.0.1:8000 ...
echo Swagger UI documentation will open in your browser shortly.
echo Press Ctrl+C in this window anytime to stop the server.
echo.

:: Launch browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000/docs"

:: Start Uvicorn Server
%PY_CMD% run_demo.py --serve --port 8000

pause
