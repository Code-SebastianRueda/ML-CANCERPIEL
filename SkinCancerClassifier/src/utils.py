"""
============================================
Utilidades Generales - Skin Cancer Classifier
============================================
Módulo con funciones auxiliares para el proyecto.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def create_project_dirs(base_path: str) -> dict:
    """
    Crea la estructura de directorios del proyecto.
    
    Args:
        base_path: Ruta base del proyecto
        
    Returns:
        Diccionario con las rutas creadas
    """
    dirs = {
        'data_raw': os.path.join(base_path, 'data', 'raw'),
        'data_processed': os.path.join(base_path, 'data', 'processed'),
        'data_train': os.path.join(base_path, 'data', 'train'),
        'data_val': os.path.join(base_path, 'data', 'val'),
        'data_test': os.path.join(base_path, 'data', 'test'),
        'models': os.path.join(base_path, 'models'),
        'results_cm': os.path.join(base_path, 'results', 'confusion_matrix'),
        'results_roc': os.path.join(base_path, 'results', 'roc_curves'),
        'results_metrics': os.path.join(base_path, 'results', 'metrics'),
        'results_training': os.path.join(base_path, 'results', 'training'),
        'results_predictions': os.path.join(base_path, 'results', 'predictions'),
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"  [OK] {name}: {path}")
    
    return dirs


def save_metrics_to_csv(metrics: dict, filepath: str) -> None:
    """
    Guarda métricas en formato CSV.
    
    Args:
        metrics: Diccionario con métricas
        filepath: Ruta del archivo CSV
    """
    import pandas as pd
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"  [OK] Métricas guardadas en: {filepath}")


def save_metrics_to_json(metrics: dict, filepath: str) -> None:
    """
    Guarda métricas en formato JSON.
    
    Args:
        metrics: Diccionario con métricas
        filepath: Ruta del archivo JSON
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4, default=str)
    print(f"  [OK] Métricas guardadas en: {filepath}")


def plot_training_history(history, save_path: str = None) -> None:
    """
    Genera gráficas de accuracy y loss durante el entrenamiento.
    
    Args:
        history: Historial de entrenamiento de Keras
        save_path: Ruta para guardar la gráfica
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Gráfica guardada en: {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names: list, title: str = "Confusion Matrix",
                          save_path: str = None) -> None:
    """
    Genera una matriz de confusión profesional tipo heatmap.
    
    Args:
        cm: Matriz de confusión (numpy array)
        class_names: Nombres de las clases
        title: Título de la gráfica
        save_path: Ruta para guardar
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 16},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score: float, title: str = "ROC Curve",
                   save_path: str = None) -> None:
    """
    Genera una curva ROC profesional.
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        auc_score: Área bajo la curva
        title: Título de la gráfica
        save_path: Ruta para guardar
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='#2196F3', linewidth=3,
             label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linewidth=1.5, 
             linestyle='--', label='Random Classifier')
    
    plt.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Curva ROC guardada en: {save_path}")
    
    plt.show()


def get_timestamp() -> str:
    """Retorna timestamp actual formateado."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title: str) -> None:
    """Imprime un separador de sección formateado."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")
