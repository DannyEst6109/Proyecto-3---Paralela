# Dataset ASL Alphabet

Fuente: [ASL Alphabet de Akash Nagaraj en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

DOI de la version publicada: https://doi.org/10.34740/KAGGLE/DSV/29550

Licencia indicada por la fuente: GPL 2.

El dataset se descarga desde el endpoint publico de Kaggle:

```text
https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet
```

## Archivos locales

- `data/downloads/asl-alphabet.zip`: archivo original descargado.
- `data/downloads/verification.json`: conteos y SHA-256 producidos al completar la verificacion local.
- `data/raw/asl_alphabet_train/asl_alphabet_train/`: imagenes de entrenamiento, separadas por clase.
- `data/raw/asl_alphabet_test/asl_alphabet_test/`: imagenes de prueba oficiales.

Los datos permanecen excluidos de Git mediante `.gitignore`. Cada integrante debe obtener su propia copia; se comparte la configuracion y el manifiesto de particiones.

## Validacion esperada

El conjunto de entrenamiento debe contener 29 carpetas, con 3,000 imagenes JPEG por clase y 87,000 en total. Las clases son A-Z, del, nothing y space. El archivo de verificacion registra el numero real de imagenes oficiales de prueba; no debe suponerse que cubre todas las clases.

La extraccion comprueba la integridad CRC de las entradas ZIP. El SHA-256 identifica la copia descargada; no constituye una comparacion con un checksum publicado por Kaggle.

## Particion del laboratorio base

Se conservara como referencia la seleccion original de 600 imagenes por clase: 420 para entrenamiento, 90 para validacion y 90 para prueba. Se utilizan bloques de 100 fotogramas, margenes de 25 en cada extremo y semilla 42. El manifiesto de las particiones debe ser identico para CPU y GPU.

Estas particiones se generaran durante la implementacion; descargar y extraer el dataset no ejecuta el preprocesamiento ni el entrenamiento.
