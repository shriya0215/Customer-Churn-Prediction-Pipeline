@echo off
title Customer Churn Prediction Pipeline
cd /d "%~dp0"

:detect_python
if exist "C:\Users\Rutuja\AppData\Local\Programs\Python\Python314\python.exe" (
    set "PY_CMD=C:\Users\Rutuja\AppData\Local\Programs\Python\Python314\python.exe"
    goto menu
)
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    goto menu
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto menu
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto menu
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto menu
)
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
    goto menu
)
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py"
    goto menu
)

echo [ERROR] Python not found!
pause
exit /b 1

:menu
cls
echo =====================================================================
echo           CUSTOMER CHURN PREDICTION PIPELINE - CONTROLLER
echo =====================================================================
echo.
echo   [1] Start REST API Server & Open Swagger UI in Browser
echo   [2] Run Verification Demo (Single & Batch Predictions)
echo   [3] Run Automated Test Suite (pytest)
echo   [4] Retrain ML Models & Generate Visualizations (Pipeline)
echo   [5] Install / Verify Dependencies (requirements.txt)
echo   [6] Exit
echo.
echo =====================================================================
set /p choice="Enter choice [1-6]: "

if "%choice%"=="1" goto opt_api
if "%choice%"=="2" goto opt_demo
if "%choice%"=="3" goto opt_test
if "%choice%"=="4" goto opt_train
if "%choice%"=="5" goto opt_install
if "%choice%"=="6" goto opt_exit

echo Invalid option. Please try again.
pause
goto menu

:opt_api
cls
echo Starting FastAPI REST Server...
echo Address: http://127.0.0.1:8000/docs
echo.
echo Starting server... Browser will open in 3 seconds.
echo (Keep this terminal window open while using the API)
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000/docs"
"%PY_CMD%" run_demo.py --serve --port 8000
echo.
echo Server stopped.
pause
goto menu

:opt_demo
cls
"%PY_CMD%" run_demo.py
echo.
pause
goto menu

:opt_test
cls
echo Running test suite...
"%PY_CMD%" -m pytest -v
echo.
pause
goto menu

:opt_train
cls
echo Retraining ML Pipeline and Generating Plots...
"%PY_CMD%" -m src.pipeline
echo.
pause
goto menu

:opt_install
cls
echo Installing dependencies from requirements.txt...
"%PY_CMD%" -m pip install -r requirements.txt
echo.
pause
goto menu

:opt_exit
exit /b 0
