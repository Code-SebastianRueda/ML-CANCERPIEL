"""
============================================
ENTRENAMIENTO OPTIMIZADO - Skin Cancer Classifier
============================================
Ejecutar: .\venv\Scripts\python.exe train_model.py

Este script entrena el modelo con la configuración óptima
para obtener las mejores métricas posibles en HAM10000 binario.
"""

import os
import numpy as np
import pandas as pd
import shutil
from PIL import Image
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import splitfolders

# ============================================
# CONFIGURACIÓN
# ============================================
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASS_NAMES = ['benign', 'malignant']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SPLIT_DIR = os.path.join(DATA_DIR, 'split')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

for d in [MODELS_DIR, os.path.join(RESULTS_DIR, 'metrics')]:
    os.makedirs(d, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

LABEL_MAP = {
    'nv': 'benign', 'bkl': 'benign', 'df': 'benign', 'vasc': 'benign',
    'mel': 'malignant', 'bcc': 'malignant', 'akiec': 'malignant',
}


# ============================================
# PASO 1: DESCARGAR Y ORGANIZAR DATASET
# ============================================
def setup_dataset():
    """Descarga y organiza el dataset si no existe."""
    train_dir = os.path.join(SPLIT_DIR, 'train')
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print("[OK] Dataset ya preparado.")
        return
    
    # Descargar
    print("[INFO] Descargando dataset HAM10000...")
    import kagglehub
    dataset_path = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
    print(f"[OK] Descargado en: {dataset_path}")
    
    # Buscar metadata
    metadata_path = None
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if 'metadata' in f.lower() and f.endswith('.csv'):
                metadata_path = os.path.join(root, f)
                break
    
    df = pd.read_csv(metadata_path)
    df['binary_label'] = df['dx'].map(LABEL_MAP)
    
    # Buscar imágenes
    image_dirs = []
    for root, dirs, files in os.walk(dataset_path):
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(jpg_files) > 10:
            image_dirs.append(root)
    
    # Organizar
    for label in CLASS_NAMES:
        os.makedirs(os.path.join(PROCESSED_DIR, label), exist_ok=True)
    
    print("[INFO] Organizando imágenes...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['image_id']
        label = row['binary_label']
        for img_dir in image_dirs:
            for ext in ['.jpg', '.jpeg', '.png']:
                src = os.path.join(img_dir, f'{image_id}{ext}')
                if os.path.exists(src):
                    dst = os.path.join(PROCESSED_DIR, label, f'{image_id}{ext}')
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    break
    
    # Split 70/15/15
    print("[INFO] Dividiendo dataset...")
    splitfolders.ratio(PROCESSED_DIR, output=SPLIT_DIR, seed=SEED, ratio=(0.7, 0.15, 0.15))
    print("[OK] Dataset listo.")


# ============================================
# PASO 2: CREAR GENERADORES
# ============================================
def create_generators():
    """Crea generadores con preprocesamiento correcto."""
    train_dir = os.path.join(SPLIT_DIR, 'train')
    val_dir = os.path.join(SPLIT_DIR, 'val')
    test_dir = os.path.join(SPLIT_DIR, 'test')
    
    # Data augmentation + preprocesamiento EfficientNet
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=SEED
    )
    
    val_gen = val_test_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    
    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"[OK] Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")
    print(f"[OK] Class weights: {class_weight_dict}")
    
    return train_gen, val_gen, test_gen, class_weight_dict


# ============================================
# PASO 3: CONSTRUIR Y ENTRENAR MODELO
# ============================================
def build_and_train(train_gen, val_gen, class_weight_dict):
    """Construye y entrena el modelo en 2 fases."""
    
    model_path = os.path.join(MODELS_DIR, 'best_model.h5')
    
    # --- CONSTRUIR MODELO ---
    print("\n" + "=" * 60)
    print("  CONSTRUYENDO MODELO")
    print("=" * 60)
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Congelar para fase 1
    
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # --- FASE 1: ENTRENAR CLASIFICADOR ---
    print("\n" + "=" * 60)
    print("  FASE 1: Entrenando clasificador (base congelada)")
    print("=" * 60)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_p1 = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]
    
    history1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print(f"\n[OK] Fase 1 - val_accuracy: {max(history1.history['val_accuracy']):.4f}")
    
    # --- FASE 2: FINE-TUNING ---
    print("\n" + "=" * 60)
    print("  FASE 2: Fine-Tuning (descongelando modelo base)")
    print("=" * 60)
    
    base_model.trainable = True
    
    # Recompilar con lr bajo
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_p2 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]
    
    history2 = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print(f"\n[OK] Fase 2 - val_accuracy: {max(history2.history['val_accuracy']):.4f}")
    
    # Guardar modelo final
    model.save(os.path.join(MODELS_DIR, 'skin_cancer_model.h5'))
    
    # Cargar mejor modelo
    model = tf.keras.models.load_model(model_path)
    print(f"[OK] Mejor modelo cargado desde: {model_path}")
    
    return model


# ============================================
# PASO 4: EVALUAR
# ============================================
def evaluate_model(model, train_gen, val_gen, test_gen):
    """Evalúa el modelo en los 3 splits."""
    
    print("\n" + "=" * 60)
    print("  EVALUACIÓN FINAL")
    print("=" * 60)
    
    results = {}
    
    for name, gen in [('TRAIN', train_gen), ('VALIDATION', val_gen), ('TEST', test_gen)]:
        gen.reset()
        y_pred_proba = model.predict(gen, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = gen.classes
        
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        results[name] = {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'auc': auc
        }
        
        print(f"\n  --- {name} ---")
        print(f"  Accuracy:  {report['accuracy']:.4f}")
        print(f"  Precision: {report['weighted avg']['precision']:.4f}")
        print(f"  Recall:    {report['weighted avg']['recall']:.4f}")
        print(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"\n  {classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")
    
    # Guardar resultados
    df_results = pd.DataFrame(results).T
    csv_path = os.path.join(RESULTS_DIR, 'metrics', 'final_metrics.csv')
    df_results.to_csv(csv_path)
    print(f"\n[OK] Métricas guardadas en: {csv_path}")
    
    return results


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SKIN CANCER CLASSIFIER - ENTRENAMIENTO OPTIMIZADO")
    print("=" * 60)
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  GPU: {tf.config.list_physical_devices('GPU')}")
    print()
    
    # 1. Dataset
    setup_dataset()
    
    # 2. Generadores
    train_gen, val_gen, test_gen, class_weight_dict = create_generators()
    
    # 3. Entrenar
    model = build_and_train(train_gen, val_gen, class_weight_dict)
    
    # 4. Evaluar
    evaluate_model(model, train_gen, val_gen, test_gen)
    
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("  Modelo guardado en: models/best_model.h5")
    print("  Para usar la app: streamlit run src/app.py")
    print("=" * 60)
