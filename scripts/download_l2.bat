@echo off
title Legio Cluster - Download Files

cd /d "%USERPROFILE%\Documents\GitHub\tfm_smishing-detection"

echo ===================================================
echo DOWNLOAD FILES OR FOLDERS FROM LEGIO2
echo ===================================================
echo.
echo Enter the path of the file or folder you want to download.
echo - To download an entire folder (e.g. models): models
echo - To download a file (e.g. results/metrics.json): results/metrics.json
echo.
set /p TARGET="Path in legio2 (inside tfm_smishing-detection/): "

if "%TARGET%"=="" (
    echo Operation cancelled.
    pause
    exit /b
)

echo.
echo ===================================================
echo DOWNLOADING: %TARGET%
echo ===================================================
:: Download the file/folder to the root of the local project
scp -r "legio2:~/tfm_smishing-detection/%TARGET%" .

echo.
echo ===================================================
echo DOWNLOAD FINISHED
echo The requested files have been downloaded.
echo ===================================================
pause
