# src/utils/data_utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_sample_images(x_data, y_data, class_names, num_samples=10):
    """Mostrar muestras de imágenes del dataset"""
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_data[i])
        plt.xlabel(class_names[np.argmax(y_data[i])])
    plt.tight_layout()
    plt.show()

def get_class_names():
    """Nombres de las clases de CIFAR-10"""
    return [
        'avión', 'automóvil', 'pájaro', 'gato', 'ciervo',
        'perro', 'rana', 'caballo', 'barco', 'camión'
    ]