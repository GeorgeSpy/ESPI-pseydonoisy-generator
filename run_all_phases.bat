@echo off
REM Run the v3.2 reproducibility validators in order; stop on first failure.
REM Configure paths first (edit these or set them in your environment):
if "%ESPI_REPO%"=="" set ESPI_REPO=%~dp0
if "%ESPI_DATA%"=="" set ESPI_DATA=C:\ESPI\data
if "%ESPI_OUT%"=="" set ESPI_OUT=%~dp0_validation_out

echo ESPI_REPO=%ESPI_REPO%
echo ESPI_DATA=%ESPI_DATA%
echo ESPI_OUT=%ESPI_OUT%

python "%ESPI_REPO%phase1_backward_compat_validation.py" || goto :fail
python "%ESPI_REPO%phase2_order_invariance_validation.py" || goto :fail
python "%ESPI_REPO%phase3_replayability_validation.py" || goto :fail
python "%ESPI_REPO%phase4_calibration_validation.py" || goto :fail
python "%ESPI_REPO%phase5_utility_validation.py" || goto :fail
echo All phases completed.
goto :eof

:fail
echo A phase FAILED. Stopping.
exit /b 1
