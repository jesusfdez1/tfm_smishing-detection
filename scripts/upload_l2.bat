@echo off
chcp 65001 > nul
title Legio Cluster - Sync and Connect

echo ===================================================
echo 1. EMPAQUETANDO EL PROYECTO (IGNORANDO BASURA)...
echo ===================================================
cd /d "%USERPROFILE%\Documents\GitHub\tfm_smishing-detection"

:: Comprime todo excluyendo las carpetas pesadas/inútiles
tar -czf envio_cluster.tar.gz --exclude=".git" --exclude=".venv" --exclude="__pycache__" --exclude=".mypy_cache" .

echo.
echo ===================================================
echo 2. SUBIENDO UN UNICO ARCHIVO AL CLUSTER...
echo ===================================================
:: Borramos la carpeta antigua en el servidor para evitar archivos huerfanos y la creamos de nuevo
ssh legio2 "rm -rf ~/tfm_smishing-detection && mkdir -p ~/tfm_smishing-detection"
scp envio_cluster.tar.gz legio2:~/tfm_smishing-detection/

:: Borramos el empaquetado de tu PC para no ocupar espacio tontamente
del envio_cluster.tar.gz

echo.
echo ===================================================
echo 3. DESCOMPRIMIENDO EN LEGIO2 Y CONECTANDO...
echo ===================================================
:: Le decimos al cluster que lo extraiga allí y borre el archivo empaquetado
ssh legio2 "cd ~/tfm_smishing-detection && tar -xzf envio_cluster.tar.gz && rm envio_cluster.tar.gz"

:: Finalmente, te abre la terminal normal
ssh legio2