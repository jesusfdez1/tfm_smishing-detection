@echo off
title Legio Cluster - Download Files

cd /d "%USERPROFILE%\Documents\GitHub\tfm_smishing-detection"

echo ===================================================
echo DESCARGAR ARCHIVOS O CARPETAS DESDE LEGIO2
echo ===================================================
echo.
echo Introduce la ruta del archivo o carpeta que quieres descargar.
echo - Para descargar una carpeta entera (ej. models): models
echo - Para descargar un archivo (ej. results/metrics.json): results/metrics.json
echo.
set /p TARGET="Ruta en legio2 (dentro de tfm_smishing-detection/): "

if "%TARGET%"=="" (
    echo Operacion cancelada.
    pause
    exit /b
)

echo.
echo ===================================================
echo DESCARGANDO: %TARGET%
echo ===================================================
:: Descargamos el archivo/carpeta a la raiz del proyecto local
scp -r "legio2:~/tfm_smishing-detection/%TARGET%" .

echo.
echo ===================================================
echo DESCARGA FINALIZADA
echo Los archivos solicitados se han descargado.
echo ===================================================
pause
