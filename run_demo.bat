@echo off
title Customer Churn Prediction - Demo Run
echo =====================================================================
echo           Customer Churn Prediction - Verification Demo
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
        pause
        exit /b 1
    )
)

%PY_CMD% run_demo.py

echo.
echo =====================================================================
echo Demo completed. Press any key to exit.
echo =====================================================================
pause >nul
