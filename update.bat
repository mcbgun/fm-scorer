@echo off
setlocal
cd /d "%~dp0"

for /f "delims=" %%A in ('git status --porcelain') do set "DIRTY=1"
if defined DIRTY (
    echo Update stopped: local changes would be overwritten.
    echo Commit, stash, or remove them, then run update.bat again.
    exit /b 1
)

echo Checking for updates...
git pull --ff-only
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 exit /b 1
)

echo Installing pinned dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Update complete.
echo Start the app with: .venv\Scripts\uvicorn.exe main:app --reload
pause
