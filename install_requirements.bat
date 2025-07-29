@echo off
REM This script installs required Python packages for the Fake News Detection app
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
