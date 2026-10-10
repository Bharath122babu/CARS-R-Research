@echo off
cd /d "%~dp0"
python -m pytest tests --basetemp=runs/test-temp
if errorlevel 1 exit /b %errorlevel%
python scripts/verify_repository.py
