import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Configuration page ---
st.set_page_config(
    page_title="Détection du Paludisme - CNN",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Charger le modèle ---
@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model("malaria_detector_final.h5")

model = charger_modele()

# --- CSS design ---
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .positive {
        background-color: #fdecea;
        border-left: 6px solid #dc3545;
    }
    .negative {
        background-color: #e0f7e9;
        border-left: 6px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🦠 Détection du Paludisme avec CNN</h1>
    <p>Analysez une image de cellule sanguine pour détecter la présence de parasites</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
# --- Sidebar ---
with st.sidebar:
    st.markdown("## ℹ️ À propos")
    st.success("""
    🧬 Cet outil repose sur un **réseau de neurones convolutionnel (CNN)**  
    entraîné sur des images microscopiques de cellules sanguines.
    """)

    st.markdown("""
    **⚙️ Fonctionnement :**
    - 📥 **Entrée** : Image d'une cellule (JPG/PNG)  
    - 📊 **Sortie** : Probabilité d'infection
    """)

    st.info("📈 **Précision du modèle : ~98%**")

    st.warning("""
    ⚠️ **Important :**  
    Cet outil est à objectif **éducatif** et **ne remplace pas** un diagnostic médical professionnel.
    """)

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Choisissez une image de cellule...", type=["jpg", "jpeg", "png"])

# --- Container pour les résultats ---
result_container = st.container()

if uploaded_file:
    result_container.empty()
    
    # Affichage image
    img = Image.open(uploaded_file).resize((130, 130))
    st.image(img, caption="Image chargée", use_container_width=True)

    # Prétraitement
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array)
    proba = prediction[0][0]

    # Déterminer si la cellule est infectée
    is_infected = proba < 0.5

    # Ajuster le pourcentage pour affichage
    if is_infected:
        proba_affiche = 1 - proba  # pourcentage d’infection
    else:
        proba_affiche = proba      # pourcentage de santé

    # Définir label et recommandations
    if is_infected:
        label = "Cellule infectée"
        conseil = "⚠️ Recommandation : consulter un professionnel de santé pour analyse complémentaire."
        style_box = "positive"
        couleur_barre = "#dc3545"
    else:
        label = "Cellule saine"
        conseil = "✅ Recommandation : faible risque détecté."
        style_box = "negative"
        couleur_barre = "#28a745"

    # Affichage du résultat
    with result_container:
        st.markdown(f"""
        <div class="result-box {style_box}">
            <h2>{'✅ ' + label if is_infected else '🩺 ' + label}</h2>
            <p><strong>Probabilité :</strong> {proba_affiche:.2%}</p>
            <div style="background-color: #e9ecef; border-radius: 25px; height: 20px; overflow: hidden; margin-top: 1rem;">
                <div style="width: {proba_affiche*100}%; height: 100%; background-color: {couleur_barre}; border-radius: 25px;"></div>
            </div>
            <p style="margin-top: 1rem; font-weight: bold;">{conseil}</p>
            <br>
            <br>
        </div>
        """, unsafe_allow_html=True)


# --- Footer chaleureux ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🧬 Votre Assistant Détection Paludisme</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous aider à analyser les cellules sanguines.
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour améliorer la précision
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil est éducatif et ne remplace pas un diagnostic médical professionnel.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

