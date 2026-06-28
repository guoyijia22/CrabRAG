@echo off
setlocal EnableExtensions
set "CRABRAG_DIR=%~dp0"
set "CRABRAG_ROOT=%CRABRAG_DIR:~0,-1%"
if not defined ELCQA_ROOT set "ELCQA_ROOT=%CRABRAG_ROOT%"
pushd "%CRABRAG_ROOT%" >nul
"%CRABRAG_ROOT%\runtime\python\python.exe" -m services.rag_api.cli.evidence %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
