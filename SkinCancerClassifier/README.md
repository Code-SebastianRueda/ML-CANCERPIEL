# 🔬 Clasificación Inteligente de Cáncer de Piel usando Deep Learning

## Skin Cancer Classification with Convolutional Neural Networks (CNN)

---

## 📋 Descripción del Proyecto

Sistema de **Deep Learning** y **Computer Vision** para la clasificación automática de lesiones cutáneas en imágenes dermatoscópicas. El modelo determina si una lesión es:

- **Benigna** — No presenta características cancerígenas
- **Maligna** — Posible cáncer de piel, requiere evaluación médica

El proyecto implementa un pipeline completo: descarga automática del dataset, preprocesamiento, balanceo de clases, entrenamiento con Transfer Learning, evaluación exhaustiva y despliegue web con Docker.

---

## 🎯 Objetivos

### Objetivo General
Desarrollar una solución basada en Deep Learning que permita analizar imágenes de lesiones cutáneas y clasificarlas automáticamente como benignas o malignas.

### Objetivos Específicos
1. Implementar Transfer Learning con MobileNetV2 pre-entrenado en ImageNet
2. Aplicar técnicas de balanceo de clases (oversampling) para mejorar la detección de lesiones malignas
3. Evaluar el modelo con métricas completas en Train, Validation y Test
4. Desplegar la solución en una aplicación web interactiva con Docker

---

## 🏗️ Arquitectura del Modelo CNN

```
Input (224 x 224 x 3)
        │
        ▼
┌───────────────────────────┐
│      MobileNetV2          │  ← Pre-entrenado en ImageNet
│   (Feature Extractor)     │     1.3M parámetros
│   include_top = False     │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  GlobalAveragePooling2D   │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│    Dense(256, ReLU)       │
│    Dropout(0.4)           │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│    Dense(128, ReLU)       │
│    Dropout(0.3)           │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│   Dense(2, Softmax)       │  → [Benigno, Maligno]
└───────────────────────────┘
```

### Estrategia de Entrenamiento (2 Fases)

| Fase | Descripción | Learning Rate | Épocas |
|------|-------------|---------------|--------|
| **Fase 1** | Feature Extraction (base congelada) | 1e-3 | 10 |
| **Fase 2** | Fine-Tuning (últimas 50 capas descongeladas) | 1e-4 | 10 |

### Callbacks
- **EarlyStopping** — Detiene si val_loss no mejora (patience=5)
- **ReduceLROnPlateau** — Reduce lr si val_loss se estanca (factor=0.5)
- **ModelCheckpoint** — Guarda el mejor modelo según val_accuracy

---

## 📊 Dataset

### Skin Cancer MNIST: HAM10000
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Total:** 10,015 imágenes dermatoscópicas
- **Categorías originales:** 7 tipos de lesiones
- **Clasificación binaria:** Benigno vs Maligno

### Mapeo de Clases

| Clase Original | Descripción | Clasificación |
|---|---|---|
| nv | Melanocytic nevi | **Benigno** |
| bkl | Benign keratosis | **Benigno** |
| df | Dermatofibroma | **Benigno** |
| vasc | Vascular lesions | **Benigno** |
| mel | Melanoma | **Maligno** |
| bcc | Basal cell carcinoma | **Maligno** |
| akiec | Actinic keratoses | **Maligno** |

### División del Dataset
- **Train:** 70% (con oversampling para balancear clases)
- **Validation:** 15%
- **Test:** 15%

### Balanceo de Clases (Oversampling)
El dataset original tiene ~80% benigno y ~20% maligno. Se aplica **oversampling** en el conjunto de entrenamiento duplicando imágenes de la clase maligna hasta igualar la cantidad de benignas. Esto permite que el modelo aprenda ambas clases de forma equitativa.

---

## 📁 Estructura del Proyecto

```
SkinCancerClassifier/
│
├── data/
│   ├── raw/                  # Dataset original descargado
│   ├── processed/            # Imágenes organizadas por clase
│   └── split/
│       ├── train/            # Entrenamiento (balanceado)
│       ├── val/              # Validación
│       └── test/             # Test
│
├── notebooks/
│   └── skin_cancer_classifier.ipynb   # Notebook principal (completo)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Descarga y preprocesamiento
│   ├── train.py              # Entrenamiento del modelo
│   ├── evaluate.py           # Evaluación y métricas
│   ├── predict.py            # Predicción sobre nuevas imágenes
│   ├── app.py                # Aplicación web (Streamlit)
│   └── utils.py              # Funciones auxiliares
│
├── models/
│   └── best_model.h5         # Modelo entrenado
│
├── results/
│   ├── confusion_matrix/     # Matrices de confusión (Train/Val/Test)
│   ├── roc_curves/           # Curvas ROC con AUC
│   ├── metrics/              # Métricas en CSV
│   ├── training/             # Gráficas de entrenamiento
│   └── predictions/          # Predicciones de ejemplo
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── train_model.py            # Script de entrenamiento directo
└── README.md
```

---

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Python 3.11
- pip
- Docker (para despliegue)
- ~4 GB de RAM disponible

### 1. Crear entorno virtual e instalar dependencias

```bash
cd SkinCancerClassifier
py -m venv CancerPiel
.\CancerPiel\Scripts\activate
pip install -r requirements.txt
```

### 2. Registrar kernel de Jupyter

```bash
python -m ipykernel install --user --name=CancerPiel --display-name "CancerPiel"
```

### 3. Obtener el dataset y entrenar el modelo

> **⚠️ IMPORTANTE:** El dataset NO está incluido en el repositorio por su tamaño (~2.5 GB). Se descarga automáticamente al ejecutar el notebook.

```bash
jupyter notebook notebooks/skin_cancer_classifier.ipynb
```

Seleccionar kernel **"CancerPiel"** y ejecutar todas las celdas en orden. El notebook:
1. Descarga automáticamente el dataset HAM10000 desde Kaggle (requiere cuenta de Kaggle)
2. Organiza las imágenes por clase (benigno/maligno)
3. Aplica oversampling para balancear clases
4. Entrena el modelo en 2 fases
5. Evalúa y genera todas las métricas y gráficas
6. Guarda el modelo en `models/best_model.h5`

### 4. Ejecutar la aplicación web (local)

```bash
streamlit run src/app.py
```

Acceder en: http://localhost:8501

### 5. Ejecutar con Docker

```bash
cd docker
docker-compose up --build
```

Acceder en: http://localhost:8501

---

## 📈 Métricas de Evaluación

El modelo se evalúa en tres conjuntos con las siguientes métricas:

| Métrica | Descripción |
|---|---|
| **Accuracy** | Proporción de predicciones correctas |
| **Precision** | De las predicciones positivas, cuántas son correctas |
| **Recall** | De los casos reales positivos, cuántos se detectaron |
| **F1-Score** | Media armónica de Precision y Recall |
| **ROC-AUC** | Capacidad discriminativa del modelo (1.0 = perfecto) |

### Resultados generados automáticamente:
- `results/metrics/` — Archivos CSV con métricas por split
- `results/confusion_matrix/` — Matrices de confusión (heatmap)
- `results/roc_curves/` — Curvas ROC con AUC
- `results/training/` — Gráficas de accuracy/loss y balanceo de clases

---

## 🔧 Preprocesamiento

| Paso | Descripción |
|---|---|
| Resize | Todas las imágenes a 224×224 px |
| Normalización | Escalar pixeles a rango [0, 1] con `rescale=1./255` |
| Validación | Eliminación automática de imágenes corruptas |
| Oversampling | Duplicar clase minoritaria (maligno) hasta igualar mayoría |
| Data Augmentation | Rotation ±30°, zoom ±20%, flip H/V, shift ±15%, shear |

---

## 🖥️ Aplicación Web (Frontend)

La aplicación web permite:
- Subir una imagen dermatoscópica
- Ver preview de la imagen cargada
- Ejecutar la predicción con un clic
- Ver resultado: **Benigno** o **Maligno** con porcentaje de confianza
- Ver probabilidades detalladas de cada clase
- Barra visual de índice de malignidad

**Diseño:** Moderno, responsivo, minimalista con estética médica/tecnológica.

**Tecnología:** Python + Streamlit

---

## 🐳 Docker

El contenedor ejecuta automáticamente la aplicación Streamlit:

```yaml
# docker-compose.yml
services:
  skin-cancer-app:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ../models:/app/models:ro
```

**Puerto:** 8501  
**Comando:** `streamlit run src/app.py`

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.11 | Lenguaje principal |
| TensorFlow | 2.15.1 | Framework de Deep Learning |
| Keras | 2.15.0 | API de alto nivel para redes neuronales |
| MobileNetV2 | — | Modelo base (Transfer Learning) |
| NumPy | 1.26.4 | Procesamiento numérico |
| Pandas | 3.x | Manipulación de datos |
| Scikit-Learn | 1.8 | Métricas de evaluación |
| OpenCV | 4.11 | Procesamiento de imágenes |
| Matplotlib | 3.10 | Visualización |
| Seaborn | 0.13 | Gráficas estadísticas |
| Streamlit | 1.57 | Aplicación web |
| Docker | — | Contenedorización |
| split-folders | 0.6 | División de dataset |
| kagglehub | 1.0 | Descarga de dataset |

---

## 📓 Notebook Principal

El archivo `notebooks/skin_cancer_classifier.ipynb` contiene el proyecto completo:

1. Introducción y contexto médico
2. Importación de librerías y configuración
3. Descarga y exploración del dataset (EDA)
4. Distribución de clases y análisis exploratorio
5. Preprocesamiento, oversampling y data augmentation
6. Construcción del modelo CNN (MobileNetV2)
7. Entrenamiento en 2 fases (Feature Extraction + Fine-Tuning)
8. Evaluación completa (Train / Validation / Test)
9. Matrices de confusión
10. Curvas ROC
11. Comparación final de métricas
12. Interpretación de resultados
13. Predicción sobre nuevas imágenes
14. Conclusiones

---

## ⚠️ Disclaimer

Este proyecto es de carácter **educativo y académico**. NO debe utilizarse como herramienta de diagnóstico médico real. Ante cualquier lesión cutánea sospechosa, consulte siempre con un dermatólogo certificado.

---

## 📚 Referencias

- Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018).
- [HAM10000 Dataset - Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [ISIC Archive](https://www.isic-archive.com/)
- [Transfer Learning Guide - TensorFlow](https://www.tensorflow.org/guide/keras/transfer_learning)

---

*Proyecto desarrollado para sustentación académica de Machine Learning y Deep Learning.*
