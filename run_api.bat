@echo off
title Customer Churn Prediction - REST API Server
cd /d "%~dp0"

echo =====================================================================
echo           Starting Customer Churn Prediction REST API Server
echo =====================================================================
echo.

:: Detect Python
if exist "C:\Users\Rutuja\AppData\Local\Programs\Python\Python314\python.exe" (
    set "PY_CMD=C:\Users\Rutuja\AppData\Local\Programs\Python\Python314\python.exe"
    goto found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    goto found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto found_python
)
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
    goto found_python
)
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py"
    goto found_python
)

echo [ERROR] Python was not found in PATH or standard directories!
pause
exit /b 1

:found_python
echo Using Python at: "%PY_CMD%"
echo Starting FastAPI server at http://127.0.0.1:8000 ...
echo Swagger UI documentation will open in your browser shortly.
echo (Keep this window open while using the API. Press Ctrl+C to stop).
echo.

:: Launch browser after 3 seconds in background
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000/docs"

:: Start Uvicorn Server
"%PY_CMD%" run_demo.py --serve --port 8000

echo.
echo Server stopped.
pause
