@echo off
title Customer Churn Prediction - Demo Run
cd /d "%~dp0"

echo =====================================================================
echo           Customer Churn Prediction - Verification Demo
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
"%PY_CMD%" run_demo.py

echo.
echo =====================================================================
echo Demo completed. Press any key to exit.
echo =====================================================================
pause >nul
