@echo off
title EV Intelligence Platform
echo.
echo  ===================================================
echo   EV Intelligence Platform - Starting...
echo  ===================================================
echo.
cd /d "%~dp0.."
python -m streamlit run ev_intelligence/app.py --server.port 8501
pause
