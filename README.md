# FNN Implementation

Este repositorio contiene la implementación completa de una red neuronal artificial desde cero, utilizando únicamente la librería NumPy para todas las operaciones numéricas. No se emplean frameworks de alto nivel como TensorFlow o PyTorch. El objetivo del proyecto es comprender el funcionamiento interno de una red neuronal, incluyendo propagación hacia adelante, retropropagación, actualización de pesos y optimización.

Se han desarrollado distintos módulos que permiten:
- Definir capas densas y capas convolutivas.
- Aplicar funciones de activación como Sigmoid, Tanh, ReLU y Leaky ReLU.
- Usar diferentes funciones de pérdida, como error cuadrático medio y entropía cruzada.
- Entrenar con optimizadores como SGD, Momentum, RMSProp, Adagrad y Adam.

## Estructura del proyecto

- Clasificación supervisada:
  - Conjunto de datos Iris: clasificación de tres especies de flores a partir de cuatro características medidas.
  - Conjunto de datos MNIST: clasificación de imágenes de dígitos escritos a mano, tanto mediante redes densas como mediante una red con capa convolutiva básica.

- Problemas lógicos:
  - Generalización del problema XOR a n dimensiones, utilizado para evaluar la capacidad de la red para aprender patrones no lineales.

- Regresión:
  - Aproximación de un plano en n dimensiones con ruido añadido, con el objetivo de analizar la capacidad del modelo para ajustarse a datos continuos.

## Instalación

Para ejecutar el proyecto es necesario instalar las dependencias incluidas en el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```
