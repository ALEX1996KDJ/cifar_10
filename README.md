# Proyecto de Visión por Computadora - Clasificador CIFAR-10

Este proyecto implementa una red neuronal convolucional (CNN) para clasificar imágenes del dataset CIFAR-10.

## Dataset
El dataset CIFAR-10 contiene 60,000 imágenes a color de 32x32 pixels en 10 clases:
- Avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión
# Editar el README.md para incluir información sobre las visualizaciones
cat >> README.md << 'EOF'

## 📊 Visualización de Resultados

El proyecto incluye un script completo de visualización (`visualize_training.py`) que genera:

- `class_distribution.png`: Distribución de clases en el dataset
- `sample_images.png`: Ejemplos de imágenes del CIFAR-10
- `training_history_detailed.png`: Gráficas de precisión y pérdida
- `confusion_matrix.png`: Matriz de confusión del modelo
- `prediction_examples.png`: Ejemplos de predicciones correctas e incorrectas
- `training_history.npy`: Datos del historial de entrenamiento
- `cifar10_cnn_model.keras`: Modelo entrenado

## 🚀 Uso

```bash
# Ejecutar el script de visualización completa
python visualize_training.py

# Los archivos se generarán automáticamente
