@echo off
setlocal

REM Ensure any previous background processes are killed on exit
trap "TASKKILL /F /IM python.exe /T" EXIT

REM Start the backend server from the root directory
echo Starting backend server...
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
cd ..
start /B python backend\app.py
set BACKEND_PID=%!

REM Start the frontend server
echo Starting frontend server...
npm install
start /B npm run dev
set FRONTEND_PID=%!

REM Wait for user input to close the script
echo Press any key to stop the servers...
pause >nul

REM Kill the background processes
TASKKILL /F /IM python.exe /T
TASKKILL /F /IM node.exe /T

endlocal
