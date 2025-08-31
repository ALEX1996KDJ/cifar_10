# src/models/cifar10_cnn.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

class CIFAR10CNN:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_data(self):
        """Cargar dataset CIFAR-10 automáticamente desde Keras"""
        print("Cargando dataset CIFAR-10...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalizar imágenes
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convertir labels a one-hot encoding
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """Construir modelo CNN"""
        model = keras.Sequential([
            # Capa convolucional 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # Capa convolucional 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Capa convolucional 3
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten y capas densas
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        """Entrenar el modelo"""
        print("Entrenando modelo...")
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """Evaluar el modelo"""
        print("Evaluando modelo...")
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Pérdida: {loss:.4f}")
        print(f"Precisión: {accuracy:.4f}")
        
        # Predecir y mostrar classification report
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nReporte de clasificación:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        return loss, accuracy
    
    def plot_training_history(self):
        """Graficar historial de entrenamiento"""
        if self.history is None:
            print("No hay historial de entrenamiento disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Gráfico de precisión
        ax1.plot(self.history.history['accuracy'], label='Precisión entrenamiento')
        ax1.plot(self.history.history['val_accuracy'], label='Precisión validación')
        ax1.set_title('Precisión del modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        
        # Gráfico de pérdida
        ax2.plot(self.history.history['loss'], label='Pérdida entrenamiento')
        ax2.plot(self.history.history['val_loss'], label='Pérdida validación')
        ax2.set_title('Pérdida del modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_model(self, filepath='cifar10_cnn_model.h5'):
        """Guardar modelo entrenado"""
        self.model.save(filepath)
        print(f"Modelo guardado como {filepath}")
    
    def load_model(self, filepath='cifar10_cnn_model.h5'):
        """Cargar modelo pre-entrenado"""
        self.model = keras.models.load_model(filepath)
        print(f"Modelo cargado desde {filepath}")

def main():
    # Crear y entrenar modelo
    cnn = CIFAR10CNN()
    
    # Cargar datos (se descargan automáticamente)
    (x_train, y_train), (x_test, y_test) = cnn.load_data()
    
    # Construir modelo
    model = cnn.build_model()
    print(model.summary())
    
    # Entrenar
    history = cnn.train(x_train, y_train, x_test, y_test, epochs=10)
    
    # Evaluar
    cnn.evaluate(x_test, y_test)
    
    # Graficar resultados
    cnn.plot_training_history()
    
    # Guardar modelo
    cnn.save_model()

if __name__ == "__main__":
    main()