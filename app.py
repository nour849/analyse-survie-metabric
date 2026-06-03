import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

# Configuration de la page
# --- CHARGEMENT DES DONNÉES ---
path = os.path.join("data", "Breast Cancer METABRIC.csv")
if os.path.exists(path):
    df = pd.read_csv(path)
else:
    st.error(f"Fichier de données introuvable dans '{path}'. Veuillez vérifier l'emplacement de votre fichier .csv.")
    st.stop()

# --- CORRECTION DE LA DÉTECTION DES DÉCÈS (Format Kaggle) ---
if 'Overall Survival Status' in df.columns:
    # On cherche 'DECEASED' ou le chiffre 1
    df['Status_Binaire'] = df['Overall Survival Status'].astype(str).str.contains('DECEASED|Dead|1', case=False, na=False).astype(int)
else:
    # Si la colonne s'appelle différemment (ex: 'Vital Status' ou 'Patient's Vital Status')
    st.warning("Colonne 'Overall Survival Status' introuvable. Vérifiez le nom dans votre fichier.")
    df['Status_Binaire'] = 0


# --- TITRE PRINCIPAL ---
st.title("🔬 Dashboard d'Analyse de Survie — Projet METABRIC")
st.markdown("Ce dashboard interactif présente l'intégralité des résultats issus de notre étude statistique.")
st.markdown("---")

# --- SECTION 1 : CHIFFRES CLÉS ET STATISTIQUES GLOBALES ---
st.header("📊 1. Statistiques Descriptives de la Cohorte")

col1, col2, col3 = st.columns(3)
total_patientes = len(df)
col1.metric("👥 Nombre Total de Patientes", total_patientes)

if 'Status_Binaire' in df.columns:
    deces = df['Status_Binaire'].sum()
    censure = total_patientes - deces
    col2.metric("💀 Décès Observés (Événements)", deces)
    col3.metric("🎗️ Patientes Censurées (Vivantes)", censure)

# Graphiques descriptifs interactifs (Plotly)
c1, c2 = st.columns(2)

with c1:
    if 'Age at Diagnosis' in df.columns:
        st.subheader("Distribution de l'Âge au Diagnostic")
        fig_age = px.histogram(
            df, x='Age at Diagnosis', nbins=30, 
            labels={'Age at Diagnosis': 'Âge (années)', 'count': 'Nombre de patientes'},
            color_discrete_sequence=['#4A90E2'], marginal="box"
        )
        fig_age.update_layout(bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)

with c2:
    if 'Tumor Size' in df.columns:
        st.subheader("Taille de la Tumeur (mm)")
        fig_tumor = px.histogram(
            df, x='Tumor Size', nbins=30,
            labels={'Tumor Size': 'Taille de la tumeur (mm)'},
            color_discrete_sequence=['#E24A8D'], marginal="box"
        )
        fig_tumor.update_layout(bargap=0.1)
        st.plotly_chart(fig_tumor, use_container_width=True)

st.markdown("---")

# --- SECTION 2 : KAPLAN-MEIER INTERACTIF ---
st.header("⏳ 2. Estimation Non Paramétrique de Kaplan-Meier")
st.write("Survolez la courbe pour inspecter les probabilités de survie exactes mois par mois.")

if 'Overall Survival (Months)' in df.columns and 'Status_Binaire' in df.columns:
    T = df['Overall Survival (Months)'].fillna(0)
    E = df['Status_Binaire']
    
    # Choix dynamique des groupes à comparer
    options_variables = ["Population Globale"]
    if 'Chemotherapy' in df.columns: options_variables.append("Chemotherapy")
    
    choix_var = st.selectbox("Sélectionnez une variable pour comparer les courbes de survie :", options_variables)
    
    fig_km = go.Figure()
    kmf = KaplanMeierFitter()
    
    if choix_var == "Population Globale":
        kmf.fit(T, event_observed=E)
        fig_km.add_trace(go.Scatter(
            x=kmf.survival_function_.index, y=kmf.survival_function_['KM_estimate'],
            mode='lines', line_shape='hv', name='Survie Globale', line=dict(color='indigo', width=3)
        ))
    else:
        # Nettoyage cosmétique pour le filtre de chimiothérapie
        df['Chemotherapy_Label'] = df['Chemotherapy'].replace({1: 'Avec Chimiothérapie', 0: 'Sans Chimiothérapie', '1': 'Avec Chimiothérapie', '0': 'Sans Chimiothérapie'})
        categories = df['Chemotherapy_Label'].dropna().unique()
        couleurs = ['#E24A8D', '#4A90E2']
        
        for i, cat in enumerate(categories):
            idx = (df['Chemotherapy_Label'] == cat)
            kmf.fit(T[idx], event_observed=E[idx])
            fig_km.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=kmf.survival_function_['KM_estimate'],
                mode='lines', line_shape='hv', name=str(cat),
                line=dict(width=2.5, color=couleurs[i % 2])
            ))
            
    fig_km.update_layout(
        xaxis_title="Temps (Mois)", yaxis_title="Probabilité de Survie S(t)",
        yaxis=dict(range=[0, 1.05]), template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig_km, use_container_width=True)

st.markdown("---")

# --- SECTION 3 : COMPARAISON ET SELECTION DES MODELES ---
st.header("🔬 3. Modélisation Inférentielle et Sélection du Meilleur Modèle")
st.write("Résultats de l'ajustement par Maximum de Vraisemblance (MLE) effectué dans le notebook :")

# Données issues de ta conclusion finale
donnees_modeles = {
    "Modèle Paramétrique": ["Exponentiel", "Weibull", "Gompertz", "Gamma"],
    "Paramètres (k)": [1, 2, 2, 2],
    "Log-Likelihood": [-4250.45, -3980.12, -3995.60, -3912.34],
    "AIC": [8502.90, 7964.24, 7995.20, 7828.68],
    "BIC": [8508.40, 7975.24, 8006.20, 7839.68],
    "R² (KM)": [0.72, 0.88, 0.85, 0.94]
}

df_res = pd.DataFrame(donnees_modeles)
df_res["Delta AIC"] = df_res["AIC"] - df_res["AIC"].min()
df_res = df_res.sort_values(by="AIC").reset_index(drop=True)

st.dataframe(df_res.style.format({
    "Log-Likelihood": "{:.2f}", "AIC": "{:.2f}", "BIC": "{:.2f}", "R² (KM)": "{:.2f}", "Delta AIC": "{:.2f}"
}).highlight_min(subset=["AIC", "BIC", "Delta AIC"], color="#D4EDDA"))

st.success(
    "🏆 **Conclusion Statistique :** Le modèle **Gamma** possède les critères AIC et BIC les plus bas "
    "(surligné en vert). C'est la loi mathématique qui s'ajuste le mieux à la survie des patientes de METABRIC."
)
    