#streamlit run streamlit_brut.py


#biblios
import pandas as pd 
import numpy as np
import streamlit as st
import os, sys, json
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Le module utilitaire généré
import importlib
if 'ptb_import' not in sys.modules:
    sys.path.append('.')
ptb_import = importlib.import_module('ptb_import')

# barre laterale de selection
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Accéder au dataset PTB(données brutes) :", 
    ("Exploration du DataFrame", "Sélection Patient", "Modèle ML"))


#PAGE 1
if page == "Exploration du DataFrame":
    #titre page :
    st.title('Données brutes du dataset PTB diag-ecg-database-1.0.0')

    # recuperer csv metadatas et records
    meta_df = pd.read_csv('ptb_metadata.csv')
    st.write('Exploration du dataframe')
    st.write(meta_df.head())
    st.write('Taille du dataset : ' , meta_df.shape)
    
    st.subheader(' Repartition pathologies')
    #visualisation des pathologies principales :
    label_counts = meta_df['primary_pathology'].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(10,5))
    label_counts.head(5).plot(kind='bar')
    plt.title('Principales pathologies (primary_pathology)sur les patients du dataset')
    plt.ylabel('Nombre d"enregistrements')
    plt.xlabel('Pathologie')
    plt.tight_layout()
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt) #affichage sur streamlit 
    st.write(label_counts.head())
    
        
    # Créer deux colonnes
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Repartition patients')  #(avec: 1 normal, 0 anormal)
        class_counts = meta_df['is_normal'].map({1:'Normal',0:'Anormal'}).value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            class_counts.values,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#EE5A5AA4", "#BCF7A6"],  # Vert / Rouge
            explode=[0.05, 0.05]            # léger zoom pour le style
        )

        ax.set_title("Répartition Normal vs Anormal")
        ax.axis('equal')  # découpe circulaire propre
        st.pyplot(fig)
        #st.write('Rappel : normal=1, anormal=0 ')

    with col2 :
        st.subheader('Repartition sexe ') #(avec: "female": 1, "male": 0)
        class_counts = meta_df['sex'].map({1:'female',0:'male'}).value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            class_counts.values,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#A8C7F3A2", "#EA9ED8D6"],  # Vert / Rouge
            explode=[0.05, 0.05]            # léger zoom pour le style
        )

        ax.set_title("Répartition Normal vs Anormal")
        ax.axis('equal')  # découpe circulaire propre
        st.pyplot(fig)
    
    
    # Plot repartition normal / anormale par sexe
    df = meta_df.copy()
    df['sex'] = df['sex'].map({1:'female', 0 : 'male'})
    df['is_normal'] = df['is_normal'].map({1:'Normal', 0:'Anormale'})

    st.subheader('Répartition Normal / Anormal par sexe')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df, x="sex", hue="is_normal",
                palette=["#EE5A5A", "#8DEB8D"], ax=ax)

    for bars in ax.containers:
        for bar in bars:
            sex = bar.get_x() + bar.get_width()/2
            sex_name = ax.get_xticklabels()[round(sex)].get_text()
            pct = bar.get_height() / df[df['sex']==sex_name].shape[0] * 100
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f"{pct:.1f}%", ha='center', va='bottom')
    ax.set_title("Répartition Normal / Anormal par sexe")
    ax.set_ylabel("Nombre d'enregistrements")

    st.pyplot(fig)



#PAGE 2
if page == "Sélection Patient":
    st.title('Visualisation pour un patient')
    # Sélection du patient
    meta_df = pd.read_csv('ptb_metadata.csv')
    patients = sorted(meta_df["patient_id"].unique())
    selected_patient = st.selectbox("Choisissez un patient", patients)

    # Filtrage des enregistrements de ce patient
    patient_records = meta_df[meta_df["patient_id"] == selected_patient]

    # Extraire les enregistrements disponibles
    record_stem = patient_records["record_stem"].tolist() 
    selected_record = st.selectbox("Choisissez un enregistrement", record_stem)

    # Afficher l’enregistrement sélectionné
    st.write("Données de l'enregistrement sélectionné :")
    st.write(patient_records[patient_records["record_stem"] == selected_record])


    # AFFICHER METADATA 
    # --- État persistant pour afficher/masquer car description metadata longue
    if "show_diag" not in st.session_state:
        st.session_state.show_diag = False  # initialement masqué

    # afficher metadata du patient associé à l’enregistrement sélectionné :
    if st.button("Afficher/Masquer le diagnostic"):
        st.session_state.show_diag = not st.session_state.show_diag
        
    if st.session_state.show_diag:
        def linebreak(text):
            # Mot commençant par une majuscule et ayant plus de 4 lettres
            pattern = r'\b([A-Z][a-zA-Z]{4,})'
            return re.sub(pattern, r'\n\1', text)
        
        #st.write(patient_records['diagnoses'][patient_records["record_stem"] == selected_record])
        diag = patient_records.loc[patient_records["record_stem"] == selected_record, "diagnoses"].iloc[0]
        diag_clean = linebreak(diag)
        st.text(f"Diagnostic : {diag_clean}")

    #AFFICHER ECG:
    # --- État persistant pour afficher/masquer car description metadata longue
    if "show_ecg" not in st.session_state:
        st.session_state.show_ecg = False  # initialement masqué

    # afficher metadata du patient associé à l’enregistrement sélectionné :
    if st.button("Afficher/Masquer l'ECG"):
        st.session_state.show_ecg = not st.session_state.show_ecg
        
    if st.session_state.show_ecg:
        # Sélection d'un enregistrement (ex: le premier)
        row = patient_records[patient_records["record_stem"] == selected_record].iloc[0]
        record_path = row['record_path']
        print('Record:', record_path)

        sig, meta = ptb_import.read_signal(record_path)
        fs = meta['fs']
        lead_names = meta['sig_name']

        # Visualiser les 3 premiers leads sur 10 secondes
        sec = 10
        stop = int(sec * fs)
        plt.figure(figsize=(12,6))
        
        for i in range(min(3, sig.shape[1])):
            plt.subplot(3,1,i+1)
            plt.plot(sig[:stop, i])
            plt.title(f"Lead {lead_names[i]} — {sec}s")
            plt.xlabel('échantillon')
            plt.ylabel('mV')
        plt.tight_layout()
        st.pyplot(plt)
        st.write(f"Visualisation de l'enregistrement ({selected_record}) des 3 premières dérivations pour le {selected_patient} , sur 10sec")
        
        
#PAGE 3
if page == "Modèle ML":
    st.title('Mise en place Machine Learning ')

