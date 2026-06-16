@echo off
chcp 65001 > nul
title Legio Cluster - Sync and Connect

echo ===================================================
echo 1. PACKAGING THE PROJECT (IGNORING JUNK)...
echo ===================================================
cd /d "%USERPROFILE%\Documents\GitHub\tfm_smishing-detection"

:: Compress everything excluding heavy/useless folders
tar -czf envio_cluster.tar.gz --exclude="envio_cluster.tar.gz" --exclude=".git" --exclude=".venv" --exclude="__pycache__" --exclude=".mypy_cache" --exclude="data/raw" --exclude="output" --exclude="docs" .

echo.
echo ===================================================
echo 2. UPLOADING A SINGLE FILE TO THE CLUSTER...
echo ===================================================
:: Delete the old folder on the server to avoid orphan files and recreate it
ssh legio2 "rm -rf ~/tfm_smishing-detection && mkdir -p ~/tfm_smishing-detection"
scp envio_cluster.tar.gz legio2:~/tfm_smishing-detection/

:: Delete the package from your PC to save space
del envio_cluster.tar.gz

echo.
echo ===================================================
echo 3. UNZIPPING IN LEGIO2 AND CONNECTING...
echo ===================================================
:: Tell the cluster to extract it there and delete the packaged file
ssh legio2 "cd ~/tfm_smishing-detection && tar -xzf envio_cluster.tar.gz && rm envio_cluster.tar.gz"

:: Finally, open the normal terminal
ssh legio2