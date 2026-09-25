# Resonador Halbach para MRI de ultra bajo campo

La imagen de la convocatoria plantea optimizar un resonador Halbach de imanes
permanentes para maximizar la homogeneidad del campo. Este repositorio comienza
con un caso **cilíndrico** y una pregunta medible: ¿cuánto mejoran pequeñas
correcciones de orientación un arreglo con errores de montaje conocidos?

## Primer experimento

El script representa cada imán rectangular con una cuadratura de dipolos 3D,
calcula el campo en una región cilíndrica y usa `torch.autograd` + Adam para
ajustar un ángulo por imán. Compara tres casos: arreglo ideal, arreglo con
errores simulados y arreglo corregido. Reporta `Bx` medio, RMS y rango
pico a pico en ppm, más las correcciones angulares.

```bash
python -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
.venv/bin/python halbach.py --device cuda --steps 250 --output resultado.json
```

Ejemplo para explorar otra geometría (todas las longitudes en **metros**):

```bash
.venv/bin/python halbach.py --magnets 12 --ring-radius 0.11 \
  --radial-width 0.025 --tangential-width 0.020 --length 0.12 \
  --roi-radius 0.04 --roi-half-length 0.025 --output resultado_12.json
```

El número de imanes es discreto: se compara entre corridas. La amplitud de
los errores se cambia con `--error-deg` y la corrección permitida con
`--max-correction-deg`. Con `--error-deg 0` se obtiene la referencia nominal;
la optimización puede no mejorar un arreglo perfectamente simétrico.
Los parámetros por defecto son solo un ejemplo numérico: producen alrededor
de 78 mT y no definen el campo objetivo de un MRI de ultra bajo campo.

Se usa PyTorch con CUDA 13.0 para la RTX 3070 Ti. Antes de ejecutar,
`nvidia-smi` debe mostrar la GPU. Comprueba también:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin CUDA')"
```

`--device auto` selecciona CUDA si está disponible y CPU en caso contrario;
`--device cuda` falla de forma explícita cuando la GPU no está accesible.

## Límites y ruta de investigación

Este **no es todavía un simulador de diseño final**. La cuadratura de dipolos
aproxima bloques de magnetización uniforme. Omite permeabilidad, desmagnetización,
variación de `Br`, hierro, temperatura y tolerancias mecánicas. Los ppm calculados
son del componente `Bx` dentro de la ROI, no una garantía de calidad MRI. La
optimización angular solo sirve como primera demostración diferenciable.

1. Medir o fijar requisitos: diámetro y longitud de la ROI, `B0` objetivo,
   homogeneidad requerida, tamaño/costo de los imanes y tolerancias.
2. Comparar número de imanes, radio, longitud y sección usando el mismo volumen
   de ROI y restricciones de espacio/costo. Registrar además campo de fuga.
3. Sustituir el modelo aproximado por bloques prismáticos analíticos o FEM y
   contrastar el mapa `B(x,y,z)` con mediciones de imanes reales.
4. Diseñar bobinas de RF y, si corresponde, bobinas de shim con restricciones de
   corriente, calentamiento y acceso a la muestra. Validar el conjunto completo.

El modelo permite reproducir el experimento con `--seed` y guarda parámetros y
resultados en JSON para comparar corridas.

## Referencias iniciales

- [Conceptos prácticos de imanes Halbach para resonancia magnética](https://link.springer.com/article/10.1007/s00723-023-01602-2).
- [Diferenciación automática de PyTorch](https://docs.pytorch.org/docs/stable/notes/autograd).
- [Instalación oficial de PyTorch](https://pytorch.org/get-started/locally/).
