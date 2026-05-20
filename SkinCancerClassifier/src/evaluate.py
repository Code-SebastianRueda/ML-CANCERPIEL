"""
============================================
Evaluación del Modelo - Skin Cancer Classifier
============================================
Módulo para evaluar el modelo en train, validation y test.
Genera métricas completas, matrices de confusión y curvas ROC.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from src.utils import plot_confusion_matrix, plot_roc_curve, save_metrics_to_csv


def get_predictions(model, generator) -> tuple:
    """
    Obtiene predicciones del modelo para un generador de datos.
    
    Args:
        model: Modelo entrenado
        generator: Generador de datos
        
    Returns:
        Tupla (y_true, y_pred, y_pred_proba)
    """
    generator.reset()
    
    # Predicciones
    y_pred_proba = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = generator.classes
    
    return y_true, y_pred, y_pred_proba


def compute_metrics(y_true, y_pred, y_pred_proba, class_names: list) -> dict:
    """
    Calcula métricas completas de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_pred_proba: Probabilidades de predicción
        class_names: Nombres de las clases
        
    Returns:
        Diccionario con todas las métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # ROC-AUC (para clasificación binaria)
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    # Classification Report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names
    )
    
    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def evaluate_split(model, generator, split_name: str, class_names: list,
                   results_dir: str) -> dict:
    """
    Evaluación completa de un split (train/val/test).
    
    Args:
        model: Modelo entrenado
        generator: Generador de datos del split
        split_name: Nombre del split ('train', 'validation', 'test')
        class_names: Nombres de las clases
        results_dir: Directorio para guardar resultados
        
    Returns:
        Diccionario con métricas
    """
    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN: {split_name.upper()}")
    print(f"{'='*60}")
    
    # Obtener predicciones
    y_true, y_pred, y_pred_proba = get_predictions(model, generator)
    
    # Calcular métricas
    metrics = compute_metrics(y_true, y_pred, y_pred_proba, class_names)
    
    # Mostrar métricas
    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n  Classification Report:")
    print(metrics['classification_report'])
    
    # Guardar Matriz de Confusión
    cm_path = os.path.join(results_dir, 'confusion_matrix', 
                           f'confusion_matrix_{split_name}.png')
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        title=f'Confusion Matrix - {split_name.upper()}',
        save_path=cm_path
    )
    
    # Guardar Curva ROC
    if 'roc_auc' in metrics:
        roc_path = os.path.join(results_dir, 'roc_curves',
                                f'roc_curve_{split_name}.png')
        plot_roc_curve(
            metrics['fpr'],
            metrics['tpr'],
            metrics['roc_auc'],
            title=f'ROC Curve - {split_name.upper()}',
            save_path=roc_path
        )
    
    # Guardar métricas en CSV
    metrics_csv = {
        'split': split_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics.get('roc_auc', None)
    }
    csv_path = os.path.join(results_dir, 'metrics', f'metrics_{split_name}.csv')
    save_metrics_to_csv(metrics_csv, csv_path)
    
    return metrics


def evaluate_all(model, train_generator, val_generator, test_generator,
                 class_names: list, results_dir: str) -> dict:
    """
    Evaluación completa en los tres splits.
    
    Args:
        model: Modelo entrenado
        train_generator: Generador de entrenamiento
        val_generator: Generador de validación
        test_generator: Generador de test
        class_names: Nombres de las clases
        results_dir: Directorio de resultados
        
    Returns:
        Diccionario con métricas de todos los splits
    """
    all_metrics = {}
    
    # Evaluar cada split
    all_metrics['train'] = evaluate_split(
        model, train_generator, 'train', class_names, results_dir
    )
    all_metrics['validation'] = evaluate_split(
        model, val_generator, 'validation', class_names, results_dir
    )
    all_metrics['test'] = evaluate_split(
        model, test_generator, 'test', class_names, results_dir
    )
    
    # Comparación final
    print_comparison(all_metrics)
    save_comparison(all_metrics, results_dir)
    
    return all_metrics


def print_comparison(all_metrics: dict) -> None:
    """Imprime tabla comparativa de métricas."""
    print(f"\n{'='*60}")
    print(f"  COMPARACIÓN FINAL DE MÉTRICAS")
    print(f"{'='*60}")
    print(f"\n  {'Métrica':<12} {'Train':<10} {'Validation':<12} {'Test':<10}")
    print(f"  {'-'*44}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        train_val = all_metrics['train'].get(metric, 'N/A')
        val_val = all_metrics['validation'].get(metric, 'N/A')
        test_val = all_metrics['test'].get(metric, 'N/A')
        
        if isinstance(train_val, (int, float)):
            print(f"  {metric:<12} {train_val:<10.4f} {val_val:<12.4f} {test_val:<10.4f}")


def save_comparison(all_metrics: dict, results_dir: str) -> None:
    """Guarda tabla comparativa en CSV."""
    comparison = []
    for split_name, metrics in all_metrics.items():
        comparison.append({
            'split': split_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics.get('roc_auc', None)
        })
    
    df = pd.DataFrame(comparison)
    csv_path = os.path.join(results_dir, 'metrics', 'comparison_all_splits.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Comparación guardada en: {csv_path}")


def plot_metrics_comparison(all_metrics: dict, save_path: str = None) -> None:
    """
    Genera gráfica comparativa de métricas entre splits.
    
    Args:
        all_metrics: Diccionario con métricas de todos los splits
        save_path: Ruta para guardar
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    splits = ['train', 'validation', 'test']
    
    data = []
    for split in splits:
        for metric in metrics_names:
            val = all_metrics[split].get(metric, None)
            if val is not None and isinstance(val, (int, float)):
                data.append({'Split': split, 'Metric': metric, 'Value': val})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df, x='Metric', y='Value', hue='Split', 
                     palette='viridis')
    
    plt.title('Comparación de Métricas: Train vs Validation vs Test',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Métrica', fontsize=13)
    plt.ylabel('Valor', fontsize=13)
    plt.ylim([0, 1.05])
    plt.legend(fontsize=12, title='Split')
    plt.grid(axis='y', alpha=0.3)
    
    # Agregar valores sobre las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [OK] Comparación guardada en: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Módulo de evaluación - Skin Cancer Classifier")
    print("Ejecutar desde el notebook principal para el flujo completo.")
