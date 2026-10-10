@echo off
cd /d "%~dp0"
python -u cars_epoch.py --epochs 1
if errorlevel 1 exit /b %errorlevel%
