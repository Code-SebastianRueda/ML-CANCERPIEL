"""
============================================
Entrenamiento del Modelo - Skin Cancer Classifier
============================================
Módulo para construir y entrenar la CNN con Transfer Learning.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)


def build_model(input_shape: tuple = (224, 224, 3), num_classes: int = 2,
                learning_rate: float = 0.001) -> Model:
    """
    Construye el modelo CNN usando Transfer Learning con EfficientNetB0.
    
    Arquitectura:
    - Base: EfficientNetB0 (pre-entrenado en ImageNet)
    - GlobalAveragePooling2D
    - Dropout(0.3)
    - Dense(128, relu)
    - Dense(num_classes, softmax)
    
    Args:
        input_shape: Forma de entrada (height, width, channels)
        num_classes: Número de clases de salida
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Modelo compilado
    """
    print("  [INFO] Construyendo modelo con EfficientNetB0...")
    
    # Modelo base pre-entrenado
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar capas del modelo base
    base_model.trainable = False
    
    # Construir modelo completo
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"  [OK] Modelo construido exitosamente.")
    print(f"  [INFO] Total parámetros: {model.count_params():,}")
    print(f"  [INFO] Parámetros entrenables: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"  [INFO] Parámetros no entrenables: {sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}")
    
    return model


def get_callbacks(model_save_path: str, patience: int = 5) -> list:
    """
    Configura callbacks para el entrenamiento.
    
    Args:
        model_save_path: Ruta para guardar el mejor modelo
        patience: Paciencia para EarlyStopping
        
    Returns:
        Lista de callbacks
    """
    callbacks = [
        # Detener si no mejora
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # Reducir learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Guardar mejor modelo
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"  [OK] Callbacks configurados:")
    print(f"       - EarlyStopping (patience={patience})")
    print(f"       - ReduceLROnPlateau (factor=0.2, patience=3)")
    print(f"       - ModelCheckpoint -> {model_save_path}")
    
    return callbacks


def train_model(model, train_generator, val_generator, 
                callbacks: list, epochs: int = 20) -> tf.keras.callbacks.History:
    """
    Entrena el modelo CNN.
    
    Args:
        model: Modelo compilado
        train_generator: Generador de datos de entrenamiento
        val_generator: Generador de datos de validación
        callbacks: Lista de callbacks
        epochs: Número máximo de épocas
        
    Returns:
        Historial de entrenamiento
    """
    print(f"\n  [INFO] Iniciando entrenamiento...")
    print(f"  [INFO] Épocas máximas: {epochs}")
    print(f"  [INFO] Batch size: {train_generator.batch_size}")
    print(f"  [INFO] Steps per epoch: {len(train_generator)}")
    print(f"  [INFO] Validation steps: {len(val_generator)}")
    print("-" * 50)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n  [OK] Entrenamiento completado.")
    print(f"  [INFO] Épocas ejecutadas: {len(history.history['loss'])}")
    print(f"  [INFO] Mejor val_accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  [INFO] Mejor val_loss: {min(history.history['val_loss']):.4f}")
    
    return history


def fine_tune_model(model, train_generator, val_generator,
                    callbacks: list, epochs: int = 10, 
                    unfreeze_layers: int = 20,
                    learning_rate: float = 1e-5) -> tf.keras.callbacks.History:
    """
    Fine-tuning: descongelar últimas capas del modelo base.
    
    Args:
        model: Modelo pre-entrenado
        train_generator: Generador de entrenamiento
        val_generator: Generador de validación
        callbacks: Callbacks
        epochs: Épocas adicionales
        unfreeze_layers: Número de capas a descongelar
        learning_rate: Learning rate reducido para fine-tuning
        
    Returns:
        Historial de fine-tuning
    """
    print(f"\n  [INFO] Iniciando Fine-Tuning...")
    print(f"  [INFO] Descongelando últimas {unfreeze_layers} capas...")
    
    # Descongelar últimas capas
    base_model = model.layers[1]  # EfficientNetB0
    base_model.trainable = True
    
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    # Recompilar con learning rate bajo
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  [INFO] Parámetros entrenables después de fine-tuning: {trainable_params:,}")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n  [OK] Fine-tuning completado.")
    
    return history


if __name__ == "__main__":
    print("Módulo de entrenamiento - Skin Cancer Classifier")
    print("Ejecutar desde el notebook principal para el flujo completo.")
