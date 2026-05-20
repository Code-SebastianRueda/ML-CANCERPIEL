"""
============================================
Preprocesamiento de Datos - Skin Cancer Classifier
============================================
Módulo para descarga, limpieza y preparación del dataset HAM10000.
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import splitfolders
import cv2


# ============================================
# Mapeo de diagnósticos a categorías binarias
# ============================================
# HAM10000 tiene 7 clases de lesiones:
# - akiec: Actinic keratoses (precanceroso)
# - bcc: Basal cell carcinoma (maligno)
# - bkl: Benign keratosis (benigno)
# - df: Dermatofibroma (benigno)
# - mel: Melanoma (maligno)
# - nv: Melanocytic nevi (benigno)
# - vasc: Vascular lesions (benigno)

MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']
BENIGN_CLASSES = ['nv', 'bkl', 'df', 'vasc']

LABEL_MAP = {
    'nv': 'benign',
    'bkl': 'benign',
    'df': 'benign',
    'vasc': 'benign',
    'mel': 'malignant',
    'bcc': 'malignant',
    'akiec': 'malignant',
}

CLASS_NAMES = ['benign', 'malignant']


def download_dataset(data_dir: str) -> str:
    """
    Descarga el dataset HAM10000 desde Kaggle usando kagglehub.
    
    Args:
        data_dir: Directorio donde almacenar los datos
        
    Returns:
        Ruta al dataset descargado
    """
    raw_dir = os.path.join(data_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    # Verificar si ya existe
    if os.path.exists(os.path.join(raw_dir, 'HAM10000_metadata.csv')):
        print("  [INFO] Dataset ya descargado previamente.")
        return raw_dir
    
    print("  [INFO] Descargando dataset HAM10000 desde Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print(f"  [OK] Dataset descargado en: {path}")
        
        # Copiar archivos al directorio raw
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join(raw_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        
        return raw_dir
        
    except Exception as e:
        print(f"  [ERROR] Error descargando dataset: {e}")
        print("  [INFO] Por favor descarga manualmente desde:")
        print("         https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"         y coloca los archivos en: {raw_dir}")
        raise


def validate_images(image_dir: str, valid_extensions: tuple = ('.jpg', '.jpeg', '.png')) -> tuple:
    """
    Valida imágenes y elimina las corruptas.
    
    Args:
        image_dir: Directorio con imágenes
        valid_extensions: Extensiones válidas
        
    Returns:
        Tupla (total_images, corrupted_images, valid_images)
    """
    total = 0
    corrupted = 0
    valid = 0
    corrupted_files = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                total += 1
                filepath = os.path.join(root, file)
                try:
                    img = Image.open(filepath)
                    img.verify()
                    valid += 1
                except Exception:
                    corrupted += 1
                    corrupted_files.append(filepath)
                    os.remove(filepath)
    
    print(f"  [INFO] Total imágenes encontradas: {total}")
    print(f"  [OK]   Imágenes válidas: {valid}")
    print(f"  [WARN] Imágenes corruptas eliminadas: {corrupted}")
    
    return total, corrupted, valid


def organize_dataset(raw_dir: str, processed_dir: str) -> pd.DataFrame:
    """
    Organiza el dataset en carpetas benign/malignant.
    
    Args:
        raw_dir: Directorio con datos crudos
        processed_dir: Directorio de salida organizado
        
    Returns:
        DataFrame con metadata del dataset
    """
    print("  [INFO] Organizando dataset...")
    
    # Buscar archivo de metadata
    metadata_path = None
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if 'metadata' in f.lower() and f.endswith('.csv'):
                metadata_path = os.path.join(root, f)
                break
    
    if metadata_path is None:
        raise FileNotFoundError("No se encontró el archivo de metadata CSV")
    
    print(f"  [INFO] Metadata encontrada: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Agregar columna de clasificación binaria
    df['binary_label'] = df['dx'].map(LABEL_MAP)
    
    # Crear directorios de salida
    for label in CLASS_NAMES:
        os.makedirs(os.path.join(processed_dir, label), exist_ok=True)
    
    # Buscar directorios con imágenes
    image_dirs = []
    for root, dirs, files in os.walk(raw_dir):
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(jpg_files) > 10:
            image_dirs.append(root)
    
    print(f"  [INFO] Directorios con imágenes: {image_dirs}")
    
    # Copiar imágenes organizadas
    copied = 0
    not_found = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Organizando"):
        image_id = row['image_id']
        label = row['binary_label']
        
        # Buscar la imagen
        found = False
        for img_dir in image_dirs:
            for ext in ['.jpg', '.jpeg', '.png']:
                src_path = os.path.join(img_dir, f"{image_id}{ext}")
                if os.path.exists(src_path):
                    dst_path = os.path.join(processed_dir, label, f"{image_id}{ext}")
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                    copied += 1
                    found = True
                    break
            if found:
                break
        
        if not found:
            not_found += 1
    
    print(f"\n  [OK] Imágenes copiadas: {copied}")
    print(f"  [WARN] Imágenes no encontradas: {not_found}")
    
    return df


def split_dataset(processed_dir: str, output_dir: str, 
                  ratio: tuple = (0.7, 0.15, 0.15), seed: int = 42) -> None:
    """
    Divide el dataset en train/validation/test.
    
    Args:
        processed_dir: Directorio con datos organizados
        output_dir: Directorio de salida
        ratio: Proporción (train, val, test)
        seed: Semilla para reproducibilidad
    """
    print(f"  [INFO] Dividiendo dataset: train={ratio[0]}, val={ratio[1]}, test={ratio[2]}")
    
    splitfolders.ratio(
        processed_dir,
        output=output_dir,
        seed=seed,
        ratio=ratio,
        group_prefix=None,
        move=False
    )
    
    # Contar imágenes por split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)
        if os.path.exists(split_path):
            total = 0
            for label in CLASS_NAMES:
                label_path = os.path.join(split_path, label)
                if os.path.exists(label_path):
                    count = len(os.listdir(label_path))
                    total += count
                    print(f"    {split}/{label}: {count} imágenes")
            print(f"    {split} TOTAL: {total} imágenes")
    
    print("  [OK] Dataset dividido exitosamente.")


def create_data_generators(train_dir: str, val_dir: str, test_dir: str,
                           img_size: tuple = (224, 224), batch_size: int = 32):
    """
    Crea generadores de datos con Data Augmentation.
    
    Args:
        train_dir: Directorio de entrenamiento
        val_dir: Directorio de validación
        test_dir: Directorio de test
        img_size: Tamaño de imagen objetivo
        batch_size: Tamaño de batch
        
    Returns:
        Tupla (train_generator, val_generator, test_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Data Augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )
    
    # Solo normalización para validación y test
    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    
    # Generadores
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"  [OK] Train samples: {train_generator.samples}")
    print(f"  [OK] Validation samples: {val_generator.samples}")
    print(f"  [OK] Test samples: {test_generator.samples}")
    print(f"  [OK] Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    print("Módulo de preprocesamiento - Skin Cancer Classifier")
    print("Ejecutar desde el notebook principal para el flujo completo.")
