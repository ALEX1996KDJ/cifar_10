📖 Descripción
Este proyecto implementa una red neuronal convolucional (CNN) desde cero para clasificar imágenes del dataset CIFAR-10. El modelo alcanza hasta 69% de precisión en la clasificación de 10 categorías diferentes de objetos.

🎨 Dataset CIFAR-10
El dataset contiene 60,000 imágenes a color de 32x32 píxeles divididas en 10 clases:

✈️ Avión	🚗 Automóvil	🐦 Pájaro
🐱 Gato	🦌 Ciervo	🐕 Perro
🐸 Rana	🐎 Caballo	⛵ Barco
🚚 Camión		


🏗️ Arquitectura del Modelo
Model: "Sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D)  (None, 15, 15, 32)       0         
                                                                 
 conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling2D)  (None, 6, 6, 64)         0         
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
 flatten (Flatten)           (None, 1024)              0         
                                                                 
 dense (Dense)               (None, 64)                65600     
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________
📊 Resultados
📈 Métricas de Rendimiento
Precisión en entrenamiento: 69.0%

Precisión en validación: 69.0%

Pérdida: 0.9054

🎯 Reporte de Clasificación
              precision    recall  f1-score   support

      avión       0.75      0.65      0.70      1000
  automóvil       0.85      0.80      0.83      1000
     pájaro       0.60      0.53      0.57      1000
       gato       0.52      0.46      0.48      1000
     ciervo       0.64      0.62      0.63      1000
       perro       0.56      0.65      0.60      1000
       rana       0.80      0.74      0.77      1000
    caballo       0.73      0.74      0.74      1000
      barco       0.72      0.88      0.79      1000
     camión       0.74      0.82      0.77      1000

    accuracy                           0.69     10000
   macro avg       0.69      0.69      0.69     10000
weighted avg       0.69      0.69      0.69     10000

🚀 Instalación Rápida
Prerrequisitos
Python 3.8+

pip

Git

⚡ Comandos de Instalación
# Clonar el repositorio
git clone https://github.com/ALEX1996KDJ/cifar_10.git
cd cifar_10

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

💻 Uso
🏃‍♂️ Entrenamiento del Modelo
# Ejecutar entrenamiento completo
python src/models/cifar10_cnn.py


📊 Visualización de Resultados
# Ejecutar visualización completa
python visualize_training.py


🧪 Notebook Interactivo
# Abrir Jupyter Notebook
jupyter notebook notebooks/explore_cifar10.ipynb

📈 Visualizaciones Generadas
El proyecto genera automáticamente:

📊 training_history.png - Gráficas de precisión y pérdida

🎯 confusion_matrix.png - Matriz de confusión

🖼️ sample_images.png - Ejemplos del dataset

📋 class_distribution.png - Distribución de clases

🔍 prediction_examples.png - Predicciones correctas/incorrectas

👨‍💻 Autor
Alex - ALEX1996KDJ

🙏 Agradecimientos
Dataset proporcionado por CIFAR-10

Comunidad de TensorFlow y Keras