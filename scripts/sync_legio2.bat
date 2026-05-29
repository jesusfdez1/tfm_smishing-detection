@echo off
title Legio Cluster - Sync & Connect

:: 1. CONFIGURA TUS RUTAS AQUI
:: OJO: Si usas rsync en Windows (a traves de Git Bash o WSL), la ruta C:\ suele escribirse como /c/ o /mnt/c/
set RUTA_LOCAL="/c/Users/Ámbito académico/Documents/GitHub/tfm_smishing-detection/"
set RUTA_REMOTA="legio2:~/tfm_smishing-detection/"

echo ===================================================
echo INICIANDO SINCRONIZACION CON EL CLUSTER...
echo ===================================================
echo Sincronizando desde: %RUTA_LOCAL%
echo Hacia: %RUTA_REMOTA%
echo.

:: 2. COMANDO DE SINCRONIZACION
:: Usamos rsync para enviar solo los archivos que hayan cambiado [cite: 97]
rsync -avz --progress --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.mypy_cache/' %RUTA_LOCAL% %RUTA_REMOTA%

echo.
echo Sincronizacion finalizada.
echo.
echo ===================================================
echo CONECTANDO AL PORTAL LEGIO2...
echo ===================================================

:: 3. ACCESO AL CLUSTER
ssh legio2