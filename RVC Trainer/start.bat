@echo off
cd /d "%~dp0"
where pythonw >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw rvc_trainer.py
) else (
    start "" python rvc_trainer.py
)