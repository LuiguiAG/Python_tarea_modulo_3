@echo off

REM crear entorno virtual
python -m venv entorno_modulo_3

REM Activar entorno virtual
call entorno_modulo_3\Scripts\activate

REM Instalar paquetes necesarios
pip install -r requerimientos_mod_3.txt

echo Proyecto configurado exitosamente.
