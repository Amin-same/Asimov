@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\Meine Projekte\Asimov\pyasim\pyAsim\Scripts\pythonw.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=D:\Meine Projekte\Asimov\pyasim\pyAsim\Scripts\python.exe"

"%PYTHON_EXE%" "%~dp0train_asimov_to_walk_GUI_.py"

if errorlevel 1 pause
endlocal
