@echo off
setlocal EnableExtensions
set "CRABRAG_DIR=%~dp0"
set "CRABRAG_ROOT=%CRABRAG_DIR:~0,-1%"
if not defined ELCQA_ROOT set "ELCQA_ROOT=%CRABRAG_ROOT%"
if not defined CRABRAG_ENV_FILE set "CRABRAG_ENV_FILE=%CRABRAG_ROOT%\config\.env"
if not defined ELCQA_ENV_FILE set "ELCQA_ENV_FILE=%CRABRAG_ENV_FILE%"
set "CRABRAG_PYTHON="
if exist "%CRABRAG_ROOT%\.venv\Scripts\python.exe" set "CRABRAG_PYTHON=%CRABRAG_ROOT%\.venv\Scripts\python.exe"
if not defined CRABRAG_PYTHON if exist "%CRABRAG_ROOT%\runtime\python\python.exe" set "CRABRAG_PYTHON=%CRABRAG_ROOT%\runtime\python\python.exe"
if not defined CRABRAG_PYTHON (
  where python >nul 2>nul
  if not errorlevel 1 set "CRABRAG_PYTHON=python"
)
if not defined CRABRAG_PYTHON (
  echo CrabRAG Python runtime not found. Run install.ps1 first.
  exit /b 1
)
pushd "%CRABRAG_ROOT%" >nul
"%CRABRAG_PYTHON%" -m services.rag_api.cli.evidence %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
