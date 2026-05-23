@echo off
chcp 65001 >nul
title PaperDiag

:: First run: install dependencies
pip show paperdiag >nul 2>&1
if %errorlevel% neq 0 (
    echo [PaperDiag] Installing dependencies...
    pip install -e .
    echo.
)

echo   PaperDiag
echo   Opening http://localhost:5000
echo.

start http://localhost:5000
python -m paperdiag.cli webui
pause
