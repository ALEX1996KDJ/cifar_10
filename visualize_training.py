#!/usr/bin/env python3
"""
Script completo para visualizar datos y entrenamiento de la red CIFAR-10
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import os

# Configurar matplotlib para mejor visualización
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['image.cmap'] = 'viridis'

class TrainingVisualizer:
    def __init__(self):
        self.history = None
        self.model = None
        self.class_names = [
            'avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
            'perro', 'rana', 'caballo', 'barco', 'camión'
        ]
    
    def load_and_explore_data(self):
        """Cargar y explorar el dataset CIFAR-10"""
        print("🔍 Cargando y explorando dataset CIFAR-10...")
        
        # Cargar datos
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalizar
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convertir a one-hot encoding
        y_train_categorical = keras.utils.to_categorical(y_train, 10)
        y_test_categorical = keras.utils.to_categorical(y_test, 10)
        
        # Mostrar información del dataset
        print(f"📊 Forma de los datos de entrenamiento: {x_train.shape}")
        print(f"📊 Forma de las etiquetas de entrenamiento: {y_train.shape}")
        print(f"📊 Forma de los datos de prueba: {x_test.shape}")
        print(f"📊 Forma de las etiquetas de prueba: {y_test.shape}")
        
        # Mostrar distribución de clases
        self._plot_class_distribution(y_train, "Distribución de Clases en Entrenamiento")
        
        # Mostrar ejemplos de imágenes
        self._plot_sample_images(x_train, y_train, "Ejemplos de Imágenes del Dataset")
        
        return (x_train, y_train_categorical), (x_test, y_test_categorical), (y_train, y_test)
    
    def _plot_class_distribution(self, y_data, title):
        """Graficar distribución de clases"""
        unique, counts = np.unique(y_data, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Clase', fontsize=14)
        plt.ylabel('Cantidad de Imágenes', fontsize=14)
        plt.xticks(unique, [self.class_names[i] for i in unique], rotation=45)
        
        # Agregar valores en las barras
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sample_images(self, x_data, y_data, title, num_samples=25):
        """Mostrar muestras de imágenes"""
        plt.figure(figsize=(12, 12))
        
        # Calcular grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        for i in range(num_samples):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(x_data[i])
            plt.title(f'{self.class_names[y_data[i][0]]}', fontsize=10)
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def build_model(self):
        """Construir el modelo CNN con visualización"""
        print("🛠️ Construyendo modelo CNN...")
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            
            layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
            
            layers.Flatten(name='flatten'),
            layers.Dense(64, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout'),
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Mostrar arquitectura del modelo
        print("📋 Resumen del modelo:")
        model.summary()
        
        self.model = model
        return model
    
    def train_with_progress(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        """Entrenar el modelo con visualización de progreso"""
        print("🚀 Iniciando entrenamiento...")
        
        # Callback para visualizar progreso
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                acc = logs['accuracy']
                val_acc = logs['val_accuracy']
                loss = logs['loss']
                val_loss = logs['val_loss']
                print(f"Época {epoch+1}: accuracy={acc:.4f}, val_accuracy={val_acc:.4f}, loss={loss:.4f}, val_loss={val_loss:.4f}")
        
        # Entrenar
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[ProgressCallback()],
            verbose=0  # Usamos nuestro propio callback para verbose
        )
        
        return self.history
    
    def plot_training_history(self):
        """Graficar historial de entrenamiento detallado"""
        if self.history is None:
            print("❌ No hay historial de entrenamiento disponible")
            return
        
        history = self.history.history
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precisión
        ax1.plot(history['accuracy'], label='Entrenamiento', linewidth=2)
        ax1.plot(history['val_accuracy'], label='Validación', linewidth=2)
        ax1.set_title('Precisión durante el Entrenamiento', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pérdida
        ax2.plot(history['loss'], label='Entrenamiento', linewidth=2)
        ax2.plot(history['val_loss'], label='Validación', linewidth=2)
        ax2.set_title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precisión por época
        ax3.bar(range(len(history['accuracy'])), history['accuracy'], 
                alpha=0.7, label='Entrenamiento')
        ax3.bar(range(len(history['val_accuracy'])), history['val_accuracy'], 
                alpha=0.7, label='Validación')
        ax3.set_title('Precisión por Época', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Precisión')
        ax3.legend()
        
        # Diferencia entre train y validation
        diff = np.array(history['accuracy']) - np.array(history['val_accuracy'])
        ax4.plot(diff, color='red', linewidth=2, marker='o')
        ax4.axhline(y=0, color='black', linestyle='--')
        ax4.set_title('Diferencia: Train Accuracy - Val Accuracy', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Diferencia')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_and_visualize(self, x_test, y_test, y_test_original):
        """Evaluar el modelo y visualizar resultados"""
        print("📊 Evaluando modelo...")
        
        # Evaluación básica
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"✅ Precisión final: {accuracy:.4f}")
        print(f"✅ Pérdida final: {loss:.4f}")
        
        # Predicciones
        y_pred = self.model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Matriz de confusión
        self._plot_confusion_matrix(y_test_original, y_pred_classes)
        
        # Ejemplos de predicciones
        self._plot_prediction_examples(x_test, y_test_original, y_pred_classes, num_examples=12)
        
        # Reporte de clasificación
        self._print_classification_report(y_test_original, y_pred_classes)
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Graficar matriz de confusión"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.xlabel('Predicción', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_examples(self, x_data, y_true, y_pred, num_examples=12):
        """Mostrar ejemplos de predicciones"""
        # Encontrar ejemplos correctos e incorrectos
        correct_indices = np.where(y_true.flatten() == y_pred)[0]
        incorrect_indices = np.where(y_true.flatten() != y_pred)[0]
        
        # Seleccionar ejemplos
        correct_examples = correct_indices[:min(6, len(correct_indices))]
        incorrect_examples = incorrect_indices[:min(6, len(incorrect_indices))]
        
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        fig.suptitle('Ejemplos de Predicciones\n(Verde: Correcto, Rojo: Incorrecto)', 
                    fontsize=16, fontweight='bold')
        
        # Mostrar predicciones correctas
        for i, idx in enumerate(correct_examples):
            ax = axes[0, i]
            ax.imshow(x_data[idx])
            ax.set_title(f'Real: {self.class_names[y_true[idx][0]]}\nPred: {self.class_names[y_pred[idx]]}', 
                        fontsize=10, color='green')
            ax.axis('off')
            # Marco verde para correctos
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        # Mostrar predicciones incorrectas
        for i, idx in enumerate(incorrect_examples):
            ax = axes[1, i]
            ax.imshow(x_data[idx])
            ax.set_title(f'Real: {self.class_names[y_true[idx][0]]}\nPred: {self.class_names[y_pred[idx]]}', 
                        fontsize=10, color='red')
            ax.axis('off')
            # Marco rojo para incorrectos
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        
        # Ocultar ejes vacíos
        for i in range(len(correct_examples), 6):
            axes[0, i].axis('off')
        for i in range(len(incorrect_examples), 6):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_classification_report(self, y_true, y_pred):
        """Imprimir reporte de clasificación detallado"""
        from sklearn.metrics import classification_report
        
        print("\n📋 Reporte de Clasificación Detallado:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
    
    def save_results(self):
        """Guardar todos los resultados"""
        if self.history:
            # Guardar historial como numpy array
            np.save('training_history.npy', self.history.history)
            print("💾 Historial de entrenamiento guardado como 'training_history.npy'")
        
        if self.model:
            self.model.save('cifar10_cnn_model.keras')
            print("💾 Modelo guardado como 'cifar10_cnn_model.keras'")

def main():
    """Función principal"""
    print("=" * 60)
    print("🎯 VISUALIZADOR DE ENTRENAMIENTO CIFAR-10 CNN")
    print("=" * 60)
    
    # Crear visualizador
    visualizer = TrainingVisualizer()
    
    try:
        # 1. Cargar y explorar datos
        (x_train, y_train), (x_test, y_test), (y_train_orig, y_test_orig) = visualizer.load_and_explore_data()
        
        # 2. Construir modelo
        model = visualizer.build_model()
        
        # 3. Entrenar con visualización
        print("\n" + "=" * 60)
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        visualizer.train_with_progress(x_train, y_train, x_test, y_test, epochs=10)
        
        # 4. Visualizar resultados del entrenamiento
        print("\n" + "=" * 60)
        print("📊 VISUALIZANDO RESULTADOS DEL ENTRENAMIENTO")
        print("=" * 60)
        visualizer.plot_training_history()
        
        # 5. Evaluar y visualizar predicciones
        print("\n" + "=" * 60)
        print("🔍 EVALUANDO MODELO Y PREDICCIONES")
        print("=" * 60)
        visualizer.evaluate_and_visualize(x_test, y_test, y_test_orig)
        
        # 6. Guardar resultados
        visualizer.save_results()
        
        print("\n" + "=" * 60)
        print("✅ ¡ENTRENAMIENTO Y VISUALIZACIÓN COMPLETADOS!")
        print("=" * 60)
        print("📁 Archivos generados:")
        print("  - class_distribution.png (Distribución de clases)")
        print("  - sample_images.png (Ejemplos de imágenes)")
        print("  - training_history_detailed.png (Gráficas de entrenamiento)")
        print("  - confusion_matrix.png (Matriz de confusión)")
        print("  - prediction_examples.png (Ejemplos de predicciones)")
        print("  - training_history.npy (Datos de entrenamiento)")
        print("  - cifar10_cnn_model.keras (Modelo entrenado)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()