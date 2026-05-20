"""
============================================
Predicción - Skin Cancer Classifier
============================================
Módulo para realizar predicciones sobre nuevas imágenes.
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


# Nombres de clases
CLASS_NAMES = ['benign', 'malignant']
CLASS_LABELS_ES = {
    'benign': 'Benigno (No cáncer)',
    'malignant': 'Maligno (Posible cáncer de piel)'
}


def load_model(model_path: str) -> tf.keras.Model:
    """
    Carga el modelo entrenado.
    
    Args:
        model_path: Ruta al archivo del modelo (.h5 o SavedModel)
        
    Returns:
        Modelo cargado
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"  [OK] Modelo cargado desde: {model_path}")
    return model


def preprocess_image(image_path: str = None, image_array: np.ndarray = None,
                     target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocesa una imagen para predicción.
    
    Args:
        image_path: Ruta a la imagen (opcional)
        image_array: Array numpy de la imagen (opcional)
        target_size: Tamaño objetivo
        
    Returns:
        Imagen preprocesada lista para predicción
    """
    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
    elif image_array is not None:
        img = Image.fromarray(image_array).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
    else:
        raise ValueError("Debe proporcionar image_path o image_array")
    
    # Normalizar
    img_array = img_array.astype('float32') / 255.0
    
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_single(model, image_path: str = None, 
                   image_array: np.ndarray = None) -> dict:
    """
    Realiza predicción sobre una sola imagen.
    
    Args:
        model: Modelo entrenado
        image_path: Ruta a la imagen
        image_array: Array de la imagen
        
    Returns:
        Diccionario con resultado de predicción
    """
    # Preprocesar
    img_processed = preprocess_image(
        image_path=image_path, 
        image_array=image_array
    )
    
    # Predecir
    predictions = model.predict(img_processed, verbose=0)
    
    # Interpretar resultado
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    result = {
        'class': predicted_class,
        'class_label': CLASS_LABELS_ES[predicted_class],
        'confidence': confidence,
        'confidence_percentage': f"{confidence * 100:.1f}%",
        'probabilities': {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_NAMES))
        },
        'is_malignant': predicted_class == 'malignant'
    }
    
    return result


def predict_batch(model, image_dir: str) -> list:
    """
    Realiza predicciones sobre un directorio de imágenes.
    
    Args:
        model: Modelo entrenado
        image_dir: Directorio con imágenes
        
    Returns:
        Lista de resultados
    """
    results = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_dir, filename)
            try:
                result = predict_single(model, image_path=image_path)
                result['filename'] = filename
                results.append(result)
            except Exception as e:
                print(f"  [ERROR] Error procesando {filename}: {e}")
    
    return results


if __name__ == "__main__":
    print("Módulo de predicción - Skin Cancer Classifier")
    print("Uso: from src.predict import predict_single, load_model")
