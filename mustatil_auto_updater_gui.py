@echo off
setlocal EnableExtensions EnableDelayedExpansion

title Mustatil Qt Workspace Launcher

set "APP_DIR=%~dp0"
set "LOG=%APP_DIR%launcher_install_log.txt"
set "REQ=%APP_DIR%requirements.txt"
set "MAIN=%APP_DIR%mustatil_qt_workspace.py"
set "VENV_DIR=%APP_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

set "PY_VERSION=3.12.10"
set "PY_INSTALL_DIR=%LocalAppData%\Programs\Python\Python312"
set "PY_INSTALLER=%TEMP%\python-%PY_VERSION%-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe"

echo ============================================================
echo Mustatil Qt Workspace Launcher
echo Folder: "%APP_DIR%"
echo Log: "%LOG%"
echo ============================================================

> "%LOG%" echo ============================================================
>> "%LOG%" echo Mustatil Qt Workspace Launcher
>> "%LOG%" echo Started: %DATE% %TIME%
>> "%LOG%" echo APP_DIR=%APP_DIR%
>> "%LOG%" echo LOG=%LOG%
>> "%LOG%" echo ============================================================

if not exist "%MAIN%" (
    echo ERROR: Main script not found: "%MAIN%"
    echo ERROR: Main script not found: "%MAIN%" >> "%LOG%"
    pause
    exit /b 1
)

if not exist "%REQ%" (
    echo ERROR: requirements.txt not found: "%REQ%"
    echo ERROR: requirements.txt not found: "%REQ%" >> "%LOG%"
    pause
    exit /b 1
)

goto :find_python

:find_python
set "PYTHON_EXE="

if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if "%PYTHON_EXE%"=="" if exist "%ProgramFiles%\Python312\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python312\python.exe"
if "%PYTHON_EXE%"=="" if exist "%ProgramFiles(x86)%\Python312\python.exe" set "PYTHON_EXE=%ProgramFiles(x86)%\Python312\python.exe"

if "%PYTHON_EXE%"=="" (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_EXE=py -3.12"
    )
)

if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_EXE=python"
    )
)

if not "%PYTHON_EXE%"=="" goto :python_found

echo.
echo Python 3.12 not found. Downloading and installing Python %PY_VERSION%...
echo Python 3.12 not found. Downloading and installing Python %PY_VERSION%... >> "%LOG%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'"
if errorlevel 1 (
    echo ERROR: Python download failed.
    echo ERROR: Python download failed. >> "%LOG%"
    echo URL: %PY_URL%
    pause
    exit /b 1
)

if not exist "%PY_INSTALLER%" (
    echo ERROR: Python installer was not downloaded: "%PY_INSTALLER%"
    echo ERROR: Python installer was not downloaded: "%PY_INSTALLER%" >> "%LOG%"
    pause
    exit /b 1
)

echo Installing Python silently...
echo Installing Python silently... >> "%LOG%"
"%PY_INSTALLER%" /quiet InstallAllUsers=0 TargetDir="%PY_INSTALL_DIR%" PrependPath=1 Include_pip=1 Include_launcher=1 AssociateFiles=0 Shortcuts=0
if errorlevel 1 (
    echo ERROR: Python installation failed.
    echo ERROR: Python installation failed. >> "%LOG%"
    pause
    exit /b 1
)

rem Refresh PATH for this command window
set "PATH=%PY_INSTALL_DIR%;%PY_INSTALL_DIR%\Scripts;%PATH%"

goto :find_python

:python_found
echo Using Python 3.12: %PYTHON_EXE%
echo Using Python 3.12: %PYTHON_EXE% >> "%LOG%"

if not exist "%VENV_PY%" (
    echo.
    echo [1/5] Creating virtual environment...
    echo [1/5] Creating virtual environment... >> "%LOG%"
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: venv creation failed.
        echo ERROR: venv creation failed. >> "%LOG%"
        pause
        exit /b 1
    )
)

echo.
echo [2/5] VENV diagnostics...
"%VENV_PY%" -c "import sys; print('executable=', sys.executable); print('version=', sys.version)"
"%VENV_PY%" -c "import sys; print('executable=', sys.executable); print('version=', sys.version)" >> "%LOG%" 2>&1

echo.
echo [3/5] Upgrading pip/setuptools/wheel...
echo [3/5] Upgrading pip/setuptools/wheel... >> "%LOG%"
"%VENV_PY%" -m pip install --upgrade pip wheel "setuptools<82"
if errorlevel 1 (
    echo WARNING: pip upgrade failed. Continuing...
    echo WARNING: pip upgrade failed. Continuing... >> "%LOG%"
)

echo.
echo [4/5] Installing dependencies.
echo This can take a long time for torch, geopandas, rasterio, fiona and PySide6.
echo Output is shown live below.
echo.
echo [4/5] Installing dependencies... >> "%LOG%"

"%VENV_PY%" -m pip install --prefer-binary --no-cache-dir -r "%REQ%"
set "PIP_EXIT=%ERRORLEVEL%"

echo pip exit code: %PIP_EXIT%
echo pip exit code: %PIP_EXIT% >> "%LOG%"

if not "%PIP_EXIT%"=="0" (
    echo.
    echo ERROR: Dependency installation failed.
    echo Now running a diagnostic pip check...
    "%VENV_PY%" -m pip check
    echo.
    echo ERROR: Dependency installation failed. >> "%LOG%"
    "%VENV_PY%" -m pip check >> "%LOG%" 2>&1
    pause
    exit /b 1
)

echo.
echo [5/5] Running dependency import check...
echo [5/5] Running dependency import check... >> "%LOG%"

if exist "%APP_DIR%dependency_check.py" (
    "%VENV_PY%" "%APP_DIR%dependency_check.py"
    set "CHECK_EXIT=%ERRORLEVEL%"
    "%VENV_PY%" "%APP_DIR%dependency_check.py" >> "%LOG%" 2>&1
    if not "%CHECK_EXIT%"=="0" (
        echo.
        echo ERROR: Dependency check failed.
        echo Check failed. >> "%LOG%"
        pause
        exit /b 1
    )
) else (
    echo dependency_check.py not found. Skipping import check.
    echo dependency_check.py not found. Skipping import check. >> "%LOG%"
)

echo.
echo Starting Mustatil...
echo Starting Mustatil... >> "%LOG%"
cd /d "%APP_DIR%"
"%VENV_PY%" "%MAIN%"
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ============================================================
echo Program closed with exit code %EXITCODE%
echo ============================================================
echo Program closed with exit code %EXITCODE% >> "%LOG%"

pause
exit /b %EXITCODE%
