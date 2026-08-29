@echo off

echo ==========================================
echo Antarctic Navigation AI
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3.12 -m venv .venv
)

echo.
echo Installing/checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Starting FastAPI backend...

start "Antarctic Navigation API" cmd /k ^
".venv\Scripts\python.exe -m uvicorn api.main:app --reload"

echo.
echo Waiting for FastAPI to start...
timeout /t 6 /nobreak >nul

echo.
echo Opening frontend...
start "" index.html
echo.
echo ==========================================
echo Application started.
echo Keep the API terminal open.
echo ==========================================
pause