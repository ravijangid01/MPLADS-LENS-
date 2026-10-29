@echo off
title MPLADS-Lens Launcher (SIH26102 - MoSPI)
echo ========================================================
echo        STARTING MPLADS-LENS PROTOTYPE (SIH26102)
echo ========================================================
echo.
cd /d "C:\Users\Ravi Jangid\.gemini\antigravity\scratch\mplads-lens"

echo [1/3] Clearing any old processes on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo [2/3] Opening browser at http://localhost:8501 ...
start "" "http://localhost:8501"

echo [3/3] Launching Streamlit application...
echo --------------------------------------------------------
echo NOTE: Please KEEP THIS WINDOW OPEN while using the app.
echo To stop the app, press Ctrl + C or close this window.
echo --------------------------------------------------------
echo.

"C:\Users\Ravi Jangid\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py --server.port 8501 --server.address localhost

pause
