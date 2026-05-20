"""
============================================
Aplicación Web - Skin Cancer Classifier
============================================
Frontend con Streamlit para clasificación de lesiones cutáneas.
Ejecutar: streamlit run src/app.py
"""

import os
import sys
import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf


# ============================================
# CONFIGURACIÓN DE PÁGINA
# ============================================
st.set_page_config(
    page_title="Skin Cancer AI Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    
    .hero {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .hero p {
        color: #e0e0e0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #ffffff;
        padding: 0.3rem 0.9rem;
        border-radius: 16px;
        font-size: 0.75rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .card h3 {
        color: #1a1a1a;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .card p {
        color: #333333;
        font-size: 0.9rem;
        margin: 0.3rem 0;
    }
    .card .label {
        color: #555555;
        font-weight: 600;
    }
    
    .result-box {
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-benign {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .result-benign h2 { color: #1b5e20; font-size: 1.6rem; font-weight: 800; margin: 0; }
    .result-benign .conf { color: #2e7d32; font-size: 2.2rem; font-weight: 800; margin: 0.5rem 0; }
    .result-benign .desc { color: #333333; font-size: 0.95rem; }
    
    .result-malignant {
        background-color: #fce4ec;
        border: 2px solid #e53935;
    }
    .result-malignant h2 { color: #b71c1c; font-size: 1.6rem; font-weight: 800; margin: 0; }
    .result-malignant .conf { color: #c62828; font-size: 2.2rem; font-weight: 800; margin: 0.5rem 0; }
    .result-malignant .desc { color: #333333; font-size: 0.95rem; }
    
    .prob-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #e0e0e0;
    }
    .prob-box .prob-label { color: #555555; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
    .prob-box .prob-value { font-size: 1.8rem; font-weight: 800; margin-top: 0.2rem; }
    .prob-benign { color: #2e7d32; }
    .prob-malignant { color: #c62828; }
    
    .disclaimer {
        background: #fff8e1;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin-top: 2rem;
        color: #333333;
        font-size: 0.85rem;
    }
    .disclaimer strong { color: #e65100; }
    
    .footer {
        text-align: center;
        color: #666666;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
        margin-top: 2rem;
    }
    
    .status-ok {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .empty-state {
        background: #ffffff;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 2px dashed #ccc;
    }
    .empty-state p { color: #666666; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ============================================
# FUNCIONES
# ============================================
@st.cache_resource
def load_trained_model():
    """Carga el modelo entrenado."""
    model_paths = [
        os.path.join('models', 'best_model.h5'),
        os.path.join('models', 'skin_cancer_model.h5'),
        os.path.join('models', 'best_model.keras'),
    ]
    for path in model_paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    return None


def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Preprocesa imagen para el modelo."""
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img, img_batch


def predict(model, img_batch):
    """Realiza predicción."""
    preds = model.predict(img_batch, verbose=0)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx])
    labels = ['Benigno', 'Maligno']
    return labels[idx], confidence, preds[0]


# ============================================
# INTERFAZ
# ============================================
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🔬 Skin Cancer Classification</h1>
        <p>Sistema de inteligencia artificial para clasificación de lesiones cutáneas</p>
        <span class="badge">EfficientNetB0 · Transfer Learning · HAM10000</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar modelo
    model = load_trained_model()
    
    if model is None:
        st.error("**Modelo no encontrado.** Entrena el modelo ejecutando el notebook primero.")
        return
    
    st.markdown('<div class="status-ok">✅ Modelo cargado y listo</div>', unsafe_allow_html=True)
    
    # Layout
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("""
        <div class="card">
            <h3>📤 Subir Imagen Dermatológica</h3>
            <p>Selecciona una imagen de la lesión cutánea para analizar</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Selecciona imagen",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            img, img_batch = preprocess_image(uploaded_file)
            st.image(img, caption="Imagen cargada", use_container_width=True)
            
            st.markdown(f"""
            <div class="card">
                <h3>📋 Detalles de la imagen</h3>
                <p><span class="label">Archivo:</span> {uploaded_file.name}</p>
                <p><span class="label">Dimensiones:</span> {img.size[0]} × {img.size[1]} px</p>
                <p><span class="label">Tamaño:</span> {uploaded_file.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_right:
        if uploaded_file is not None:
            st.markdown("")
            
            analyze = st.button("🔍  Analizar Imagen", use_container_width=True, type="primary")
            
            if analyze:
                with st.spinner("Procesando con el modelo de IA..."):
                    result, confidence, probabilities = predict(model, img_batch)
                
                # Resultado
                if result == "Benigno":
                    st.markdown(f"""
                    <div class="result-box result-benign">
                        <h2>✅ BENIGNO</h2>
                        <div class="conf">{confidence*100:.1f}%</div>
                        <p class="desc">La lesión no presenta características malignas según el modelo.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-malignant">
                        <h2>⚠️ MALIGNO</h2>
                        <div class="conf">{confidence*100:.1f}%</div>
                        <p class="desc">La lesión presenta características sospechosas. Consulte un especialista.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probabilidades
                st.markdown("")
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(f"""
                    <div class="prob-box">
                        <div class="prob-label">Benigno</div>
                        <div class="prob-value prob-benign">{probabilities[0]*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with p2:
                    st.markdown(f"""
                    <div class="prob-box">
                        <div class="prob-label">Maligno</div>
                        <div class="prob-value prob-malignant">{probabilities[1]*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown("**Índice de malignidad:**")
                st.progress(float(probabilities[1]))
                
        else:
            st.markdown("")
            st.markdown("")
            st.markdown("""
            <div class="empty-state">
                <p>🖼️ Sube una imagen para obtener el análisis</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Aviso:</strong> Este sistema es educativo y NO reemplaza el diagnóstico médico profesional. 
        Ante cualquier lesión sospechosa, consulte con un dermatólogo certificado.
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Skin Cancer Classification · EfficientNetB0 · Transfer Learning · HAM10000
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
