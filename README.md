# FNN

## Requisitos del Proyecto y Guía de Implementación

## 1. Estructura de Datos
- **Entrada (x)** → X = (x₁, x₂, x₃, …): Usar mini-lotes (mini-batches) de tamaño fijo para procesar múltiples muestras de entrenamiento en paralelo.  
- **Salida (y → y’)** → La salida debe ser un vector, permitiendo que la red maneje múltiples clases de salida simultáneamente (por ejemplo, en tareas de clasificación como MNIST).

---

## 2. Arquitectura de la Red
- El modelo debe soportar cualquier número de capas (**n**), con configuraciones flexibles de neuronas por capa.  
- El constructor de la clase `NeuralNetwork` debe recibir el número de neuronas en cada capa como argumento.  
  **Ejemplo:**

  ```bash
  model = NeuralNetwork(layers=[784, 128, 64, 10])
  ```

- El diseño debe permitir una extensión modular, de manera que las capas, optimizadores y funciones de pérdida puedan reemplazarse o ampliarse fácilmente.

---

## 3. Modularización del Código

La implementación debe dividirse en clases y módulos bien definidos.

**Clases requeridas**
- `SGD`: Implementa el algoritmo de **Stochastic Gradient Descent**.  
- `Adam`: Implementa el algoritmo de optimización **Adam** (requisito obligatorio).  
- `NeuralNetwork`: Gestiona la propagación hacia adelante, la retropropagación y la actualización de pesos.  
- `Scheduler`: Define un esquema de adaptación del learning rate para SGD.  

  **Ejemplo:**
  ```bash
  def schedule(alpha, epoch):
      return alpha / (1 + decay_rate * epoch)
  ```

**Función de Pérdida**
- Dado que MNIST es un problema de clasificación, se debe usar **Cross-Entropy Loss**.

---

## 4. Entrenamiento, Validación y Prueba
- Dividir el conjunto de datos en tres subconjuntos:
  - `train_set` → Usado para actualizar los pesos.  
  - `validate_set` → Usado para ajustar hiperparámetros y evitar sobreajuste.  
  - `test_set` → Usado para evaluar el rendimiento final.  

  **Ejemplo de flujo de trabajo:**
  ```bash
  train(model, X_train, y_train)
  validate(model, X_val, y_val)
  test(model, X_test, y_test)
  ```

---

## 5. Experimentación y Pruebas en Notebook
- Realizar los experimentos en un **Jupyter Notebook**, importando y probando los módulos creados.  
- Cada componente (optimizador, red, scheduler, etc.) debe poder probarse de forma independiente.  
- Mostrar claramente:
  - Creación del modelo  
  - Entrenamiento con mini-batches  
  - Esquema de aprendizaje del alpha  
  - Evolución de la precisión y la pérdida  

---

## 6. Paso de Parámetros y Configuración
- Todas las funciones de actualización deben permitir parámetros configurables.  
  **Ejemplo:**
  ```bash
  adam.run(alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
  ```
- Evitar valores codificados dentro de las funciones. Los parámetros deben ser flexibles y configurables externamente.

---

## Resumen de Puntos Clave de Implementación

| Componente | Requisito |
|-------------|------------|
| **Entrada** | Procesamiento por mini-batches |
| **Salida** | Vector de salida (multiclase) |
| **Capas** | Número configurable de capas |
| **Optimizadores** | Clases separadas para SGD y Adam |
| **Función de pérdida** | Cross-Entropy |
| **Learning Rate** | Esquema adaptativo para SGD |
| **Validación** | Incluir validación tras el entrenamiento |
| **Código** | Totalmente modular y reutilizable |
| **Pruebas** | Implementadas y verificadas en Jupyter Notebook |
