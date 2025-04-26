import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.customize_running import center_running
import pandas as pd
import numpy as np
import random 
import ydata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
import requests
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap, to_hex
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots
import warnings

import reg_age_couleur
st.set_page_config(
    page_title="Analyse du profil des clients Aramisauto",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


switch_voiture = st.sidebar.toggle("Activer le mode voiture")

st.sidebar.markdown("##### Choisissez les couleurs pour les graphiques :")

color1 = st.sidebar.color_picker("Couleur 1", "#4169E1")
color2 = st.sidebar.color_picker("Couleur 2", "#B19CD9") 


st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #999; padding-top: 1rem;">
       Développé par BRAULT Juliette, CAUSEUR Léna et PRUSIEWICZ Louis.
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_data(show_spinner=True)
def load_data():
    table_full = pd.read_csv("table_full.csv")
    table_vehicule = pd.read_csv("table_vehicule.csv")
    
    import json
    list_options = set()
    for value in table_vehicule['OFFER_EQUIPMENTS_MAIN_LIST'].dropna():
        try:
            for option in json.loads(value):
                list_options.add(option['label'])
        except:
            continue

    for option in list_options:
        clean_label = option.lower().replace(" ", "_").replace("/", "_").replace('é', 'e').replace('è', 'e').replace('à', 'a').replace('ù', 'u')
        flag_name = f'flag_{clean_label}'
        table_vehicule[flag_name] = table_vehicule['OFFER_EQUIPMENTS_MAIN_LIST'].apply(
            lambda x: 1 if isinstance(x, str) and option in x else 0
        )

    return table_full, table_vehicule

table_full, table_vehicule = load_data()
def bienvenue():
    st.subheader("Bienvenue sur l'application d'analyse du profil des clients Aramisauto")

    st.markdown("<br>", unsafe_allow_html=True)

    # Ajout logo
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("logo_aramisauto.jpg", width=300)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contexte et objectifs
    st.markdown(
        """
        <div style="font-size: 20px; line-height:1.4;">

        **Contexte :**  

        Aramisauto souhaite mieux comprendre le profil de ses clients pour optimiser sa stratégie marketing.  
        Nous analysons des données internes, enrichies avec des sources externes, pour obtenir une vue complète du comportement de nos clients.



        **Objectifs du projet :**

        - **Analyser les données internes** pour identifier les comportements et les préférences des clients.
        - **Enrichir les données** avec des informations externes afin d'avoir une compréhension globale.
        - **Développer une appli interactive** pour visualiser et interpréter ces données.
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Instructions de navigation
    st.markdown("""
        <div style="font-size: 20px; line-height:1.4;">

        **Navigation :**  
        Nous avons basé notre analyse des clients d'Aramisauto autour de 4 hypothèses principales.
        - **Hypothèse 1 :**  Est ce que le prix d'un véhicule est influencé par le nombre d'options disponibles ?
        - **Hypothèse 2 :**  Est ce que les clients ayant consulté intensément une fiche véhicule présentent un comportement et un profil distincts des autres clients ?
        - **Hypothèse 3 :**  Est ce que l'age d'un client va influencé le choix de couleur de sa futur voiture ?
        - **Hypothèse 4 :**  Est ce que la localisation d'un client va avoir une influencé sur le choix de sa voiture ?
        </div>
        """,
    unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Information complémentaire
    st.info("Cette application a été développée dans le cadre du projet Datamining pour l'année 2024/2025.")
    
    
    # st.subheader("Statistiques descriptives")
    # st.write(table_full.describe(include='all'))
    
    
    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3, animation_length=600)

bienvenue.__name__ = "Accueil" # change le nom dans le sidebar


    
def ensemble():
    st.title("Statistiques descriptives")

    #metrics clients
    st.markdown("<h1 style='text-decoration: underline;'>Statistiques clients</h1>", unsafe_allow_html=True)
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    
    col1_1.metric("Âge moyen des clients", f"{table_full['age_client'].mean():.1f} ans")
    col1_2.metric("Ancienneté moyenne", f"{table_full['anciennete'].mean():.1f} ans")
    col1_3.metric("Nombre de femmes", table_full['GENDER'].value_counts().get('F', 0))
    col1_4.metric("Nombre d'hommes", table_full['GENDER'].value_counts().get('H', 0))

    # Graphiques clients
    col_client_1, col_client_2 = st.columns(2)

    with col_client_1.container():
        st.subheader("Répartition par tranches d'âge")
        fig1, ax1 = plt.subplots()
        ax1.hist(table_full['age_client'].dropna(), bins=10, color=color1, edgecolor='black')
        ax1.set_xlabel("Âge")
        ax1.set_ylabel("Nombre de clients")
        ax1.set_title("Distribution de l'âge des clients")
        st.pyplot(fig1)

    with col_client_2.container():
        st.subheader("Répartition par genre")
        fig2, ax2 = plt.subplots()
        gender_counts = table_full['GENDER'].value_counts()
        ax2.pie(gender_counts, labels=["Hommes", "Femmes"], autopct='%1.1f%%', startangle=90, 
            colors=[color1, color2], wedgeprops={'edgecolor': 'black'})
        ax2.axis('equal')
        st.pyplot(fig2)

    
    #metrics voitures
    st.markdown("<h1 style='text-decoration: underline;'>Statistiques voitures</h1>", unsafe_allow_html=True)
    
    col2_1, col2_2, col2_3 = st.columns(3)

    col2_1.metric("Nombre de voitures disponibles :", table_vehicule.shape[0])
    col2_2.metric("Nombre de modèles de voitures :", table_vehicule['VEHICULE_MODELE'].nunique())
    col2_3.metric("Nombre de marques différentes disponibles :", table_vehicule["VEHICULE_MARQUE"].nunique())
    
    col3_1, col3_2, col3_3 = st.columns(3)
    
    col3_1.metric("Taux de reprise", f"{(table_full['FLAG_REPRISE'].sum() / table_full['FLAG_COMMANDE'].sum() * 100):.1f} %")
    col3_2.metric("Kilométrage moyen", f"{table_vehicule['VEHICULE_KM'].mean():,.0f} km")    
    col3_3.metric("Prix moyen TTC", f"{table_full['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")

    
    col_voitures_1, col_voitures_2 = st.columns(2)
    #repartition par marque
    with col_voitures_1.container():
        st.subheader("Nombre de voitures par marque")
        qtte_marque = st.number_input("Nombre de marques à représenter", min_value=1, max_value=table_vehicule['VEHICULE_MARQUE'].nunique(), value=10)
        fig5, ax5 = plt.subplots()
        top_marques = table_vehicule['VEHICULE_MARQUE'].value_counts().head(qtte_marque)
        ax5.barh(top_marques.index[::-1], top_marques.values[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
        ax5.set_xlabel("Nombre de voitures")
        ax5.set_ylabel("Marque")
        st.pyplot(fig5)
        
    #repartion par modele
    with col_voitures_2.container():
        st.subheader("Nombre de voitures par modèle")
        qtte_modele = st.number_input("Nombre de modèles à représenter", min_value=1, max_value=table_vehicule['VEHICULE_MODELE'].nunique(), value=10)
        fig7, ax7 = plt.subplots()
        top_modeles = table_vehicule['VEHICULE_MODELE'].value_counts().head(qtte_modele)
        ax7.barh(top_modeles.index[::-1], top_modeles.values[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
        ax7.set_xlabel("Nombre de voitures")
        ax7.set_ylabel("Modèle")
        st.pyplot(fig7)
    

    col_voitures_2_1, col_voitures_2_2 = st.columns(2)
    # VO vs VN
    with col_voitures_2_1.container():
        st.subheader("Répartition des voitures neuves / d'occasions")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        table_vehicule['VEHICULE_TYPE'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax6, colors=[color1, color2],wedgeprops={'edgecolor': 'black'})
        ax6.set_ylabel('')
        st.pyplot(fig6)

    with col_voitures_2_2.container():
        st.subheader("Prix de vente des véhicules")
        nb_bins = st.slider("Nombre de bins", 0, 30, 20)
        fig4, ax4 = plt.subplots()
        prix_vente = table_full['PRIX_VENTE_TTC_COMMANDE'].dropna()
        ax4.hist(prix_vente, bins=nb_bins, color=color1,edgecolor='black')
        ax4.set_xlabel("Prix de vente TTC")
        ax4.set_ylabel("Quantité")
        st.pyplot(fig4)
        
    col_voitures_3_1, col_voitures_3_2 = st.columns(2)
    #repartition par carburant
    with col_voitures_3_1.container():
        energy_colors = {'Essence': "#2ecc71",'Diesel': "#FDDB15",'Hybride': "#9900D0",'Electrique': "#2980b9",'Autre': "#A4A4A4"}
        st.subheader("Répartition des carburants")
        fig3, ax3 = plt.subplots()
        top_energies = table_vehicule['VEHICULE_ENERGIE'].value_counts()
        colors = [energy_colors.get(energy, "#bdc3c7") for energy in top_energies.index]
        ax3.barh(top_energies.index[::-1], top_energies.values[::-1], color=plt.cm.Purples(np.linspace(0.2, 0.9, 10)))
        ax3.set_xlabel("Nombre de voitures")
        ax3.set_ylabel("Carburant")
        st.pyplot(fig3)

    
    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3.5, animation_length=600)
        
ensemble.__name__ = "Statistiques descriptives"       
        
def page_1():
    st.title("Hypothèse Prix x Options")
    st.write("Dans cette hypothèse, nous cherchons à déterminer si le nombre d’options ou d’équipements disponibles sur un modèle de véhicule a un impact significatif sur son prix de vente.")
    st.write("L'objectif est d'étudier s'il existe une corrélation entre le niveau d’équipement d'un véhicule et son positionnement tarifaire. En d’autres termes, plus un véhicule possède d’options (comme la climatisation, le GPS, le radar de recul, etc.), plus son prix final est susceptible d'être élevé. Cette analyse permettrait de mieux comprendre le rôle des équipements dans la valorisation commerciale des véhicules et d’identifier dans quelle mesure ils influencent la stratégie de tarification.")
    #tout pour pouvoir travailler comme il faut
    EQUIPEMENTS = ['flag_jantes_alliage','flag_toit_ouvrant_panoramique','flag_climatisation','flag_regulateur_de_vitesse','flag_radar_de_recul','flag_gps','flag_camera_de_recul','flag_interieur_cuir','flag_bluetooth','flag_apple_car_play','flag_android_auto']
    table_vehicule['NB_EQUIPEMENTS'] = table_vehicule[EQUIPEMENTS].sum(axis=1)
    
    table_merged = table_full.merge(table_vehicule,how="left",left_on="VEHICULE_ID_COMMANDE",right_on="VEHICULE_ID")    
    
    st.subheader("Prix moyen selon le nombre d’équipements")
    col_1, col_2, col_3 = st.columns(3)
    nb_equip = col_3.slider("Choisissez le nombre d'équipements voulu (pour le prix moyen):", 0, table_merged['NB_EQUIPEMENTS'].max(), 5)
    
    #metric
    col_1,col_2, col_3 = st.columns(3)
    with col_1.container():
        st.metric("Prix moyen (0 équipement)",f"{table_merged[table_merged['NB_EQUIPEMENTS'] == 0]['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")
    with col_2.container():
        st.metric("Prix moyen (11 équipements)",f"{table_merged[table_merged['NB_EQUIPEMENTS'] == table_merged['NB_EQUIPEMENTS'].max()]['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")
    with col_3.container():
        st.metric(f"Prix moyen ({nb_equip} équipement(s))",f"{table_merged[table_merged['NB_EQUIPEMENTS'] == nb_equip]['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")
        
    
    col_1, col_2 = st.columns(2)
    #prix en fonction du nombre d'equip / scatterplot
    with col_1.container():
        fig8, ax8 = plt.subplots()
        ax8.scatter(table_merged["NB_EQUIPEMENTS"],table_merged["PRIX_VENTE_TTC_COMMANDE"],alpha=0.5, color=color1, edgecolors='black')
        ax8.set_xlabel("Nombre d'équipements")
        ax8.set_ylabel("Prix TTC en €")
        ax8.set_title("Corrélation entre prix et nombre d'équipements")
        st.pyplot(fig8)
    
    #prix moyen en fonction du nombre d'equip
    with col_2.container():
        mean_prix = table_merged.groupby("NB_EQUIPEMENTS")["PRIX_VENTE_TTC_COMMANDE"].mean().reset_index()
        fig9, ax9 = plt.subplots()
        ax9.plot(mean_prix["NB_EQUIPEMENTS"],mean_prix["PRIX_VENTE_TTC_COMMANDE"],marker='o',color=color1)
        ax9.set_xlabel("Nombre d'équipements")
        ax9.set_ylabel("Prix moyen TTC en €")
        ax9.set_title("Prix moyen par nombre d'équipements")
        st.pyplot(fig9)
        
    with st.container():
        col_1, col_2 = st.columns(2)    
        col_1.write('Ce graphique montre la relation entre le nombre d’équipements présents dans un véhicule et son prix TTC. On observe une tendance générale où le prix augmente avec le nombre d’équipements. Les points sont dispersés, mais on remarque clairement des "paliers" pour chaque nombre d’équipements, indiquant que les voitures mieux équipées tendent à être plus chères')
        col_2.write("Ce graphique présente l'évolution du prix moyen TTC en fonction du nombre d’équipements. La courbe montre une progression régulière, avec une accélération notable à partir de 8 équipements. Cela confirme que plus un véhicule dispose d’équipements, plus son prix moyen est élevé, avec un effet particulièrement marqué pour les véhicules très bien équipés.")
    col_1, col_2 = st.columns(2)
    
    
    #prix par marque en fct du nmobre d'équipement
    with col_1.container():
        st.subheader("Prix moyen selon le nombre d’équipements pour chaque marque")
        #choix marque 
        marques = table_merged['VEHICULE_MARQUE_x'].dropna().unique()
        selected_marque = st.selectbox("Marque : ", sorted(marques))

        table_merged_marque = table_merged[table_merged['VEHICULE_MARQUE_x'] == selected_marque]
        prix_equip = table_merged_marque.groupby("NB_EQUIPEMENTS")["PRIX_VENTE_TTC_COMMANDE"].mean().reset_index()

        fig10, ax10 = plt.subplots()
        ax10.plot(prix_equip["NB_EQUIPEMENTS"],prix_equip["PRIX_VENTE_TTC_COMMANDE"],marker='o',linestyle='-',color=color1)
        ax10.set_xlabel("Nombre d'équipements")
        ax10.set_ylabel("Prix moyen TTC en €")
        ax10.set_title(f"Prix moyen par nombre d’équipements de la marque {selected_marque}")
        st.pyplot(fig10)
        
        
        
    #prix par modele en fct du nmobre d'équipement
    with col_2.container():
        st.subheader("Prix moyen selon le nombre d’équipements pour chaque modèle")
        #choix modele
        col1, col2 = st.columns(2)
        marques = table_merged['VEHICULE_MARQUE_x'].dropna().unique()
        selected_marque = col1.selectbox("Marque :", sorted(marques))
        modeles = table_merged[table_merged['VEHICULE_MARQUE_x'] == selected_marque]['VEHICULE_MODELE_x'].dropna().unique()
        selected_modele = col2.selectbox("Modèle :", sorted(modeles))

        #filtre df
        table_merged_modele = table_merged[(table_merged['VEHICULE_MARQUE_x'] == selected_marque) &(table_merged['VEHICULE_MODELE_x'] == selected_modele)]
        prix_equip = table_merged_modele.groupby("NB_EQUIPEMENTS")["PRIX_VENTE_TTC_COMMANDE"].mean().reset_index()


        fig11, ax11 = plt.subplots()
        ax11.plot(prix_equip["NB_EQUIPEMENTS"],prix_equip["PRIX_VENTE_TTC_COMMANDE"],marker='o',linestyle='-',color=color1)
        ax11.set_xlabel("Nombre d'équipements")
        ax11.set_ylabel("Prix moyen TTC en €")
        ax11.set_title(f"Prix moyen par nombre d’équipements du modele {selected_modele}")
        st.pyplot(fig11)

        
    st.subheader("Analyse en Composantes Principales des Équipements")
    
    flag_cols = [col for col in table_merged.columns if col.startswith("flag_") and col.endswith("_y")]
    X = table_merged[flag_cols].copy()
    X.columns = [col.replace("_y", "") for col in X.columns]
    df = table_merged.dropna(subset=["PRIX_VENTE_TTC_COMMANDE"])
    X = X.loc[df.index]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["PRIX"] = df["PRIX_VENTE_TTC_COMMANDE"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["PRIX"], cmap="viridis", alpha=0.6)
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Prix TTC")
    st.pyplot(fig)
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)
    top_features_pc1 = loadings.abs().sort_values("PC1", ascending=False)

    st.subheader("Equipements les plus déterminants dans la fixation du prix")
    st.dataframe(top_features_pc1)
    correlation = np.corrcoef(df_pca["PC1"], df_pca["PRIX"])[0, 1]
    st.markdown(f"**Corrélation entre PC1 et le prix :** `{correlation:.2f}`")
    st.write("La corrélation de 0,47 établie entre la première composante de l’ACP et le prix démontre qu’en général, plus une voiture est équipée, plus son prix augmente. Bien que cette relation ne soit pas absolue, elle indique clairement que les équipements exercent une influence notable sur la valeur du véhicule.")

    st.title("Conclusion de l'hypothèse")
    st.write("On constate que plus une voiture est dotée d’équipements, plus son prix tend à augmenter. Cependant, l’impact de chaque option n’est pas équivalent : des équipements comme le Bluetooth ou la climatisation, devenus très répandus, influencent moins aujourd’hui le prix que par le passé. En revanche, des options telles qu’Android Auto, Apple CarPlay ou le radar de recul contribuent de manière plus significative à la valorisation du véhicule. Ainsi, même si certaines fonctionnalités se généralisent, le nombre total d’équipements demeure un indicateur fiable de l’évolution du prix d’une voiture.")

    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3.5, animation_length=600)
    
        
page_1.__name__ = "Hypothèse Prix x Options"
    
    
def page_2():
    st.title("Hypothèse Consultation x Profil")
    
    st.subheader("On cherche à savoir ici si les clients ayant consulté intensément une fiche véhicule présentent un comportement et un profil distincts des autres clients (temps de réflexion, caractéristiques du véhicule, âge...).")

    st.markdown("### Méthodologie et processing des données")
    
    st.write("""
    1. **Correspondance Véhicule Consulté vs Véhicule Acheté :**  
       Pour garantir la pertinence des données, nous avons filtré les consultations pour ne retenir que celles concernant un véhicule effectivement acheté. Cela est réalisé en comparant la marque, le modèle et l'énergie du véhicule entre la page produit et la fiche d'achat.

    2. **Identification des clients 'extrêmes' :**  
       Nous avons isolé les clients ayant réalisé un nombre particulièrement élevé de consultations (>= 20), afin d'examiner de plus près leurs comportements spécifiques.

    3. **Calcul du temps de réflexion :**  
       Pour chaque client extrême, nous avons calculé le temps de réflexion, c'est-à-dire l'intervalle en jours entre la première consultation de la fiche véhicule et la date de commande. Ce paramètre nous permet de jauger le délai dans le processus décisionnel.
    """)

    st.markdown("<h2 style='text-decoration: underline;'>Statistiques descriptives</h2>", unsafe_allow_html=True)

    profil_client = pd.read_csv("profil_client.csv")

    # Définir les intervalles de consultations (adaptés selon vos données)
    max_consult = profil_client['Nb_Consultations'].max()
    bins = [20, 30, 40, 60, 80, 100, max_consult + 1]
    labels = ["20-30", "31-40","41-60", "61-80", "81-100", "101 et plus"]
    
    # Créer la colonne "Tranche" avec pd.cut()
    profil_client['Tranche'] = pd.cut(profil_client['Nb_Consultations'], bins=bins, labels=labels, right=False)
    
    # Calculer la répartition des consultations par tranche
    repartition = profil_client['Tranche'].value_counts().sort_index().reset_index()
    repartition.columns = ['Intervalle des consultations', 'Nombre de clients']

    repartition.set_index("Intervalle des consultations", inplace=True)

    # Calcul des statistiques de consultations
    max_consult = profil_client['Nb_Consultations'].max()
    mean_consult = profil_client['Nb_Consultations'].mean()
    total_consult = profil_client['Nb_Consultations'].sum()

    # Affichage des métriques pour les consultations
    col_consult_1, col_consult_2, col_consult_3 = st.columns(3)
    col_consult_1.metric("Consultations maximales effectué par un client", f"{max_consult}")
    col_consult_2.metric("Nombre de consultations moyennes", f"{mean_consult:.0f}")
    col_consult_3.metric("Nombre total de consultations", f"{total_consult}")

    # --- Graphique 1 : Histogramme du nombre de consultations ---
    # Création de l'histogramme avec le label personnalisé pour l'axe des X
    fig_hist = px.histogram(
        profil_client,
        x='Nb_Consultations',
        nbins=30,
        title="Histogramme du nombre de consultations",
        labels={"Nb_Consultations": "Nombre de consultations", "y":"Nombre de clients"},
        color_discrete_sequence=[color1]
    )
    
    # Personnalisation des informations affichées au survol
    fig_hist.update_traces(
        hovertemplate="Nombre de clients : %{y}<extra></extra>"
    )

    fig_hist.update_traces(
    marker_line_color='black',  # définit la couleur de la bordure
    marker_line_width=1         # définit l'épaisseur de la bordure
    )

    fig_hist.update_layout(
    title={
        'text': "Histogramme du nombre de consultations",
        'x': 0.5,           # Centre le titre horizontalement (0.5 = 50%)
        'xanchor': 'center',
        'font': {
            'size': 24    # Définit la taille de la police du titre
        }
    })

    fig_hist.update_layout(
        yaxis_title="Nombre de clients"
    )


    # Affichage de l'histogramme sur la moitié de la page (avec colonnes)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; font-size: 24px;"><strong>Répartition par tranches de consultations</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.table(repartition)

    st.write("La majorité des clients consulte une fiche véhicule entre 20 et 30 fois, ce qui indique une recherche ciblée et efficace d’informations. Toutefois, on observe également une minorité de clients qui revient 101 fois ou plus sur une même fiche. Ce comportement intensif peut traduire une hésitation marquée ou alors un besoin d'approfondissement avant de prendre une décision d’achat, suggérant la présence d'un profil distinct. Nous allons maintenant étudier plus en détail si ces consultations répétées s'accompagnent d’un temps de réflexion plus long, ou encore si elles se corrèlent avec des caractéristiques spécifiques du véhicule ou des tranches d’âge particulières.")

    st.markdown("<h2 style='text-decoration: underline;'>Comportement et profil des clients</h2>", unsafe_allow_html=True)

    st.write(
    "Nous avons défini quatre niveaux d'intensité en fonction du nombre de consultations : 'Faible' pour moins de 40 consultations, 'Moyen' pour un nombre compris entre 40 et 60, 'Elevé' pour entre 60 et 80, et 'Très élevé' pour plus de 80 consultations. Cette classification facilite l'analyse du comportement des clients ainsi que leur profil."
    )


    # --- Graphique 2 : Bar chart du temps de réflexion moyen par intensité de consultation ---
    order_list = ["Faible", "Moyen", "Elevé", "Très élevé"]

    # Calcul du temps de réflexion moyen par intensité de consultation
    temps_moyen = (
        profil_client
        .groupby('Consultation_Intensité')['Temps_Reflexion']
        .mean()
        .reset_index()
    )

    # Convertir la colonne en catégorie ordonnée
    temps_moyen['Consultation_Intensité'] = pd.Categorical(
        temps_moyen['Consultation_Intensité'],
        categories=order_list,
        ordered=True
    )

    # Trier le DataFrame selon l'ordre défini
    temps_moyen = temps_moyen.sort_values('Consultation_Intensité')

    # Création du bar chart
    fig_bar = px.bar(
        temps_moyen,
        x='Consultation_Intensité',
        y='Temps_Reflexion',
        title="Temps de réflexion moyen par intensité de consultation",
        labels={
            'Temps_Reflexion': "Temps de réflexion (jours)",
            'Consultation_Intensité': "Intensité de consultation"
        },
        text_auto=".1f",
        color_discrete_sequence=[color1]
    )

    # Personnalisation du graphique si besoin
    fig_bar.update_traces(
        marker_line_color='black',
        marker_line_width=1,
        hovertemplate="Temps de réflexion : %{y} jours"
    )
    fig_bar.update_layout(
        title={
            'x': 0.5,   
            'xanchor': 'center',
            'font': {'size': 24}
        },
        yaxis_title="Temps de réflexion (jours)"
    )

    # --- Graphique 3 : Heatmap de la répartition (%) de l'intensité des consultations par tranche d'âge ---
    heat_data = pd.crosstab(
        profil_client['Tranche_Age'],
        profil_client['Consultation_Intensité'], normalize = 'index'
    ) * 100

    # Définition de l'ordre souhaité pour les lignes et les colonnes
    ordre_lignes = ["Moins de 30", "30-44", "45-59", "60+"]
    ordre_colonnes = ["Faible", "Moyen", "Elevé", "Très élevé"]

    # Réindexer le DataFrame heat_data
    heat_data = heat_data.reindex(index=ordre_lignes, columns=ordre_colonnes)

    fig_heat = px.imshow(
        heat_data,
        text_auto=".2f",
        title="Pourcentage de niveaux de consultation par tranche d'âge",
        labels={'x': "Intensité de consultation", 'y': "Tranche d'âge", 'color': "Pourcentage (%)"}
    )

    fig_heat.update_layout(
    title={
        'x': 0.5,       
        'xanchor': 'center',
        'font': {'size': 24}  
    })

    fig_heat.update_coloraxes(showscale=False)

    # Désactivation complète du survol pour chaque trace
    for trace in fig_heat.data:
        trace.hoverinfo = 'skip'       # Supprime l'info de survol
        trace.hovertemplate = ''       # Définit un template vide

    # Désactivation du mode hover dans la mise en page
    fig_heat.update_layout(hovermode=False)


    # Affichage dans une colonne de 50% de la page
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
        st.write("En moyenne, plus un client consulte intensément une fiche véhicule, plus son temps de réflexion tend à être élevé. Ainsi, les clients avec une intensité de consultations faible ont en moyenne un temps de réflexion d'environ 9,33 jours, tandis que ceux dont l'intensité est moyenne ou élevée ont respectivement un temps de réflexion d'environ 10,90 et 12,50 jours. Il est également intéressant de noter que la différence entre les groupes 'Elevé' (12,50) et 'Très Elevé' (12,58) est minime. Ce constat suggère qu’après un certain seuil de consultations, l’augmentation du nombre de visites ne correspond pas à une prolongation significative du temps de réflexion.")

    with col2:
        st.plotly_chart(fig_heat, use_container_width=True)
        st.write("Pour la tranche d’âge Moins de 30, on observe une absence totale de consultations dans la catégorie 'Elevé'. La répartition indique que 69,23 % des consultations se situent en 'Faible', 20,51 % en 'Moyen' et 10,26 % en 'Très élevé'. Ce profil suggère que les plus jeunes adoptent une approche polarisée, oscillant entre une consultation limitée et un engagement marqué, sans passer par un niveau intermédiaire. Pour la tranche 30-44, la grande majorité des consultations se positionne également dans la catégorie 'Faible'(64,44 %), suivie de 21,11 % en 'Moyen'. Les niveaux 'Elevé' (6,67 %) et 'Très élevé' (7,78 %) représentent des parts moindres, indiquant une recherche généralement rapide et ciblée, avec quelques cas d’engagement plus approfondi. La tranche 45-59 présente un schéma similaire, avec 65 % des consultations en 'Faible', 20 % en 'Moyen', et 7,50 % dans chacune des catégories 'Elevé' et 'Très élevé'. Cette répartition homogène laisse penser que le comportement de navigation est assez équilibré dans ce groupe. Enfin, pour la tranche 60+, même si la majorité des consultations reste en 'Faible' (71,70 %), on note une proportion plus élevée dans le niveau 'Elevé' (11,32 %) accompagnée de 13,21 % en 'Moyen' et seulement 3,77 % en 'Très élevé'. Ce constat suggère que, malgré une tendance globale à une consultation moins intensive, une partie des clients de ce groupe s’engage de façon plus approfondie. En résumé, bien que tous les groupes semblent majoritairement se contenter d’un niveau de consultation faible, des nuances importantes apparaissent selon les tranches d’âge. Les moins de 30 adoptent une stratégie polarisée, les 30-44 et 45-59 affichent un comportement de navigation plutôt équilibré, tandis que les 60+ se distinguent par une petite proportion d’utilisateurs manifestant un engagement plus intense.")

    # --- Graphique 4 : Scatter Plot : Nombre de consultations vs Temps de réflexion ---
    # Affiche la relation entre le nombre de consultations et le temps de réflexion, avec la tranche d'âge en couleur.
    fig_scatter = px.scatter(
        profil_client,
        x='Nb_Consultations',
        y='Temps_Reflexion',
        color='Tranche_Age',
        size='Nb_Consultations',   # pour donner un indice visuel sur le nombre de consultations
        title="Nombre de consultations vs Temps de réflexion par tranche d'âge",
        hover_data=['ID_Client'],  # ajustez en fonction de vos colonnes disponibles
        labels={"Tranche_Age": "Tranches d'âge"}
    )

    fig_scatter.update_layout(
        title={
            'x': 0.5,   
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title ="Nombre de consultations",
        yaxis_title="Temps de réflexion (jours)"
    )

    fig_scatter.update_traces(
        hovertemplate=(
        "Nombre de consultations : %{x}<br>"
        "Temps de réflexion : %{y} jours")
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write(" Les clients des tranches 45-59 et Moins de 30 tendent à consulter la fiche véhicule plus fréquemment et présentent également un temps de réflexion relativement long, suggérant un engagement approfondi dans leur processus décisionnel. En revanche, la tranche 60+ adopte une approche moins intensive, avec des points situés principalement entre 20 et 80 consultations, indiquant une recherche d'informations plus rapide ou minimale. Enfin, bien que la tranche 30-44 affiche dans l'ensemble un comportement de consultation modéré (majoritairement entre 20 et 100 consultations), quelques cas extrêmes avec plus de 140 consultations viennent souligner la présence de profils tout aussi intéressés.")

    st.markdown("<h2 style='text-decoration: underline;'>Préférence du véhicule</h2>", unsafe_allow_html=True)

    # --- Graphique 5 : Sunburst Chart de la répartition des véhicules achetés par intensité de consultation ---
    # Ce graphique permet de visualiser, dans chaque groupe d'intensité, la distribution des véhicules achetés.
    vehicules_par_groupe = (
        profil_client
        .groupby(['Consultation_Intensité', 'Véhicule_Achete'])
        .size()
        .reset_index(name='Count')
    )

    fig_sunburst = px.sunburst(
        vehicules_par_groupe,
        path=['Consultation_Intensité', 'Véhicule_Achete'],
        values='Count',
        title="Répartition des véhicules achetés par intensité de consultation",
        height=900, 
        width=700 
    )
    fig_sunburst.update_layout(
        title={
            'x': 0.5,   
            'xanchor': 'center',
            'font': {'size': 24}
        }
    )

    # Mise à jour du survol (hovertext)
    # On récupère le trace de la figure (le sunburst est généralement un seul trace)
    sun_trace = fig_sunburst.data[0]
    # Crée une nouvelle liste de hovertemplate pour chaque point
    new_hovertemplates = []
    # Pour chaque point, on vérifie si c'est un grand cercle (niveau d'intensité)
    # Pour les points de niveau parent, "parent" est une chaîne vide.
    for label, parent, val in zip(sun_trace.labels, sun_trace.parents, sun_trace.values):
        if parent == "":  
            # Pour le grand cercle, on affiche uniquement le nombre (Count)
            new_hovertemplates.append(f"Nombre: {val:.0f}<extra></extra>")
        else:
            # Pour les niveaux enfants (les voitures), affiche le nom et le nombre
            new_hovertemplates.append(f"Véhicule: {label}<br>Nombre: {val:.0f}<extra></extra>")

    # Appliquer cette liste de hovertemplate sur le trace
    sun_trace.hovertemplate = new_hovertemplates

    st.plotly_chart(fig_sunburst, use_container_width=True)
    st.write("*Cliquez sur un niveau d'intensité de consultation pour plus de détails.*")

    st.write("On observe que certains modèles apparaissent dans plusieurs niveaux d'intensité. Par exemple, le modèle Peugeot 2008 Essence figure dans la catégorie 'Elevé' (3 fois), apparaît massivement dans 'Faible' (20 fois), se présente également dans 'Moyen' (11 fois) et dans 'Très élevé' (5 fois). Cela peut suggérer que ce modèle intéresse un large éventail de clients, qu'ils consultent peu ou beaucoup, ce qui indique peut-être une forte attractivité globale. Pour la catégorie 'Faible', la diversité est importante et les effectifs sont souvent plus élevés (par exemple, Peugeot 2008 Essence avec 20 consultations, Peugeot 208 Essence avec 19). Cela pourrait indiquer que, pour ces modèles populaires, un grand nombre de clients consulte la fiche véhicule de manière ponctuelle avant d'acheter, ce qui peut traduire une décision d'achat rapide ou une confiance préalable dans la marque. Dans la catégorie 'Moyen', le comportement est intermédiaire. Des modèles comme Peugeot 2008 Essence (11 consultations) et d'autres modèles Peugeot ou Renault apparaissent avec des effectifs modérés, suggérant que ces véhicules intéressent des clients qui hésitent ou cherchent à comparer avant de prendre leur décision. Dans la catégorie 'Elevé', on note la présence de modèles comme la Peugeot 3008 Diesel/Essence et la Renault Clio 5 Essence, mais avec des effectifs relativement faibles (entre 1 et 3). Cela pourrait signifier qu'un nombre restreint de clients, manifestant un fort engagement en termes de consultations, porte un intérêt soutenu à un ensemble spécifique de modèles. La catégorie 'Très élevé' regroupe quelques modèles (notamment certains Peugeot et Renault) avec de faibles effectifs également. Enfin, par exmeple, le Volkswagen Tiguan Diesel est recensé uniquement dans la catégorie 'Élevé', tandis que le Hyundai i30 Fastback Essence et le Seat Ateca Diesel se retrouvent exclusivement dans la catégorie 'Très élevé'. Ces observations suggèrent que ces modèles mobilisent l'attention d'un segment spécifique de clients particulièrement exigeants.")


    # ----- Tableau 1 : Prix moyen par intensité de consultation -----
    # Création du DataFrame avec les données fournies
    df_prix = pd.DataFrame({
        "Consultation_Intensité": ["Faible", "Moyen", "Elevé","Très élevé"],
        "PRIX_VENTE_TTC_COMMANDE": [ 20528.468571, 21255.000000, 23227.222222, 23040.473684]
    })


    # ----- Tableau 2 : Distribution (%) des types de carburant par intensité de consultation -----
    # Création du DataFrame avec les données fournies
    df_carburant = pd.DataFrame({
        "Consultation_Intensité": ["Faible", "Moyen", "Elevé", "Très élevé"],
        "Diesel": [26.3, 26.0, 33.3, 15.8],
       "Essence": [73.7, 74.0, 66.7, 84.2]
    })

    

    df_prix.rename(columns={
        "Consultation_Intensité": "Niveau de consultation",
        "PRIX_VENTE_TTC_COMMANDE": "Prix de vente moyen TTC (€)"
    }, inplace=True)

    # Pour le DataFrame des carburants
    df_carburant.rename(columns={
        "Consultation_Intensité": "Niveau de consultation",
        "Diesel": "Diesel (%)",
        "Essence": "Essence (%)"
    }, inplace=True)

    # On définit "Consultation_Intensité" comme index pour une présentation assimilable au tableau fourni
    df_prix.set_index("Niveau de consultation", inplace=True)
    df_carburant.set_index("Niveau de consultation", inplace=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Prix moyen par intensité de consultation")
        st.write(" ")
        st.write(" ")
        st.table(df_prix.style.format({"Prix de vente moyen TTC (€)": "{:,.2f}"}))
        st.write("On constate qu’une tendance générale s’établit entre le niveau d’intensité de consultation et le prix de vente moyen TTC. Les véhicules dont la fiche est consultée de manière « Faible » affichent un prix moyen de 20 528,47 €, tandis que ce prix passe à 21 255,00 € dans la catégorie « Moyen ». La montée se poursuit de façon plus marquée pour la catégorie « Elevé », où le prix moyen atteint 23 227,22 €, suggérant ainsi que les véhicules plus coûteux sont susceptibles de susciter un intérêt plus poussé et un examen plus attentif. Cependant, le groupe « Très élevé » montre une légère baisse avec un prix moyen de 23 040,47 €, indiquant qu’au-delà d’un seuil d’intensité, le prix ne continue pas nécessairement d’augmenter.")

    with col2:
        st.write("### Distribution des types de carburant par intensité de consultation")
        st.table(df_carburant.style.format({"Diesel (%)": "{:.1f}", "Essence (%)": "{:.1f}"}))
        st.write("Ces chiffres suggèrent que, pour les consultations modérées (Faible et Moyen), les comportements de recherche sont similaires pour les deux types de motorisation. L'intérêt pour le Diesel augmente légèrement lorsque les consultations s'intensifient (Elevé), ce qui pourrait indiquer une préférence ou une recherche plus poussée pour certains modèles Diesel. Cependant, dans le groupe des consultations « Très élevées », l'intérêt se focalise quasiment exclusivement sur les véhicules à Essence, signalant peut-être des caractéristiques ou des avantages distinctifs perçus sur ces modèles par les clients les plus engagés.")

    st.title("Conclusion de l'hypothèse")

    st.write("L'analyse menée confirme que les clients qui consultent intensément une fiche véhicule présentent des comportements et des profils distincts de ceux qui la consultent moins fréquemment.")
    st.write("En effet, en moyenne, plus le nombre de consultations est élevé, plus le temps de réflexion s’allonge. Ainsi, les clients affichant une forte intensité de consultation s’engagent plus longuement dans leur réflexion. Par ailleurs, l'analyse par tranche d'âge révèle des comportements distincts, avec les clients de moins de 30 ans adoptant une approche polarisée, oscillant entre une consultation limitée et un engagement marqué sans passer par la catégorie 'Élevé', alors que les tranches 30-44 et 45-59 présentent un comportement de navigation relativement équilibré et que la tranche 60+ se distingue par une prédominance de consultations en 'Faible' accompagnée d'une proportion légèrement supérieure dans la catégorie 'Élevé'. Ainsi, les tranches d'âge 'Moins de 30' et 45-59 ans semblent regarder la fiche véhicule le plus intensivement.")
    st.write("De plus, certains modèles, tels que le Peugeot 2008 Essence, apparaissent dans plusieurs catégories d'intensité, indiquant une attractivité globale qui suscite à la fois des consultations ponctuelles et des recherches plus approfondies. Par ailleurs, d'autres véhicules se distinguent en n'apparaissant que dans les catégories 'Élevé' et 'Très élevé'. Ainsi, le Volkswagen Tiguan Diesel est recensé uniquement dans la catégorie 'Élevé', tandis que le Hyundai i30 Fastback Essence et le Seat Ateca Diesel se retrouvent exclusivement dans la catégorie 'Très élevé'. Ces observations renforcent l'hypothèse selon laquelle les clients ayant consulté intensément une fiche véhicule présentent un profil distinct.")
    st.write("Ensuite, une analyse des prix révèle que les véhicules dont la fiche est consultée intensément tendent à être plus onéreux, suggérant qu’un investissement financier supérieur implique une recherche d'information plus approfondie. Ainsi, on constate que les clients qui explorent davantage les fiches véhicules sont surtout attirés par des modèles affichant un prix de vente moyen plus élevé.")
    st.write("Enfin, l'analyse de la motorisation renforce l'hypothèse en montrant que les comportements varient selon l'intensité de consultation. Pour des niveaux faibles et moyens, la répartition Diesel/Essence reste stable, tandis qu'une préférence pour le Diesel se manifeste dans la catégorie 'Élevé' avant de disparaître en faveur de l'Essence dans la catégorie 'Très élevé'.")

    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3.5, animation_length=600)

page_2.__name__ = "Hypothèse Consultation x Profil"
            
def page_3():
    warnings.filterwarnings("ignore")
    st.title("Analyse des préférences de couleurs selon l'âge")

    # Introduction
    st.markdown("""
    Dans cette section, nous analysons les préférences de couleurs des véhicules en fonction de l'âge des clients.  
    Nous allons examiner :
    - La répartition des couleurs par tranche d'âge.
    - L'évolution continue des préférences de couleurs avec l'âge.
    - Les résultats d'une analyse de régression logistique pour comprendre l'impact de l'âge sur le choix des couleurs.
    """)

    # Charger les données
    df = pd.read_csv('table_full.csv')

    # Standardiser les couleurs
    df = reg_age_couleur.standardize_colors(df)

    # Créer les tranches d'âge
    df = reg_age_couleur.create_age_groups(df)

    # Section 1 : Répartition des couleurs par tranche d'âge
    st.subheader("1. Répartition des couleurs par tranche d'âge")
    st.write("Nous analysons ici la répartition des couleurs standardisées pour chaque tranche d'âge.")

    fig_heatmap, fig_barplots, crosstab_pct = reg_age_couleur.analyze_colors_by_age(df)

    # Afficher la heatmap
    st.write("**Heatmap des pourcentages de couleurs par tranche d'âge :**")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.write("""
    **Interprétation :**  
    La heatmap montre les pourcentages de chaque couleur pour différentes tranches d'âge.  
    On observe que certaines couleurs, comme le gris et le noir, sont populaires dans presque toutes les tranches d'âge, tandis que d'autres, comme le rouge ou le bleu, varient davantage selon l'âge. En efet le rouge semble être plus populaire chez les personnes âgées que chez les jeunes, alors que le bleu lui est plus populaire chez les jeunes que chez les personnes agées.
    """)

    # Afficher les barplots
    st.write("**Barplots des couleurs par tranche d'âge :**")
    st.plotly_chart(fig_barplots, use_container_width=True)
    st.write("""
    **Interprétation :**  
    Les barplots permettent de visualiser les proportions exactes de chaque couleur pour chaque tranche d'âge.  
    On continue à observer la préférence pour le gris noir et blanc à travers les tranches d'âge. On note également que les 18-25 ans achètent beaucoup moins de voiture que les autres tranches d'âges. Les autres tranches d'âges achètent environ dix fois plus que les 18-25 ans.
    """)

    # Section 2 : Évolution continue des préférences de couleurs avec l'âge
    st.subheader("2. Évolution continue des préférences de couleurs avec l'âge")
    st.write("Nous visualisons ici l'évolution des préférences de couleurs en fonction de l'âge des clients, avec des tranches d'âge plus fines.")

    color_evolution = reg_age_couleur.plot_color_preferences_by_age_continuous(df)

    st.write("**Graphique de l'évolution continue des préférences de couleurs :**")
    st.plotly_chart(color_evolution, use_container_width=True)
    st.write("""
    **Interprétation :**  
    Ce graphique montre une évolution plus détaillée des préférences de couleurs avec l'âge.  
    On remarque que certaines couleurs, comme le blanc, augmentent en popularité avec l'âge, tandis que d'autres, comme le rouge, diminuent progressivement.
    """)


    # Section 3 : Analyse de régression logistique
    st.subheader("3. Analyse de régression logistique")
    st.write("""
    Nous utilisons une régression logistique pour comprendre l'impact de l'âge sur le choix des couleurs.  
    Les résultats incluent :
    - Les coefficients de régression (log-odds).
    - Les odds ratios pour interpréter l'effet de l'âge sur chaque couleur.
    """)

    # Obtenir les résultats et les graphiques interactifs
    # La fonction `perform_regression_analysis` retourne :
    # - Un DataFrame contenant les coefficients, odds ratios, p-values et leur significativité
    # - Deux graphiques interactifs : un pour les coefficients et un pour les odds ratios
    regression_results, fig_coefficients, fig_odds_ratios = reg_age_couleur.perform_regression_analysis(df)

    # Afficher le graphique interactif des coefficients de régression
    st.write("**Graphique des coefficients de régression :**")
    
    st.plotly_chart(fig_coefficients, use_container_width=True)
    st.write("Ce graphique montre l'effet de l'âge sur la probabilité de choisir chaque couleur. Les barres positives indiquent que l'âge augmente la probabilité de choisir une couleur, tandis que les barres négatives indiquent que l'âge diminue cette probabilité. Les barres plus foncées sont celles qui sont statistiquement significatives ainsi plus on viellit plus on a de chance lorsqu'on achète une voiture d'en choisir une blanche et on a moins de chance d'en choisir une noir ou bleu.")
    
    
    st.write("**Graphique des odds ratios :**")
    st.plotly_chart(fig_odds_ratios, use_container_width=True)
    st.write("Ce graphique montre l'impact relatif de l'âge sur les chances de choisir chaque couleur. Un odds ratio supérieur à 1 indique que l'âge augmente les chances de choisir une couleur, tandis qu'un odds ratio inférieur à 1 indique que l'âge diminue ces chances. Ici les barres foncés représentent les couleurs qui sont significativement influencées par l'âge. Ainsi, on confirme le constat fait précédement qui est que l'age influence positivement le choix de la couleur blanche et négativement le choix des couleurs noire et bleue. Ainsi plus on viellit plus on a de chance de choisir une voiture blanche et moins on a de chance de choisir une voiture noir ou bleu.")
    
    # Afficher les résultats sous forme de tableau interactif
    st.write("**Résumé des résultats de régression :**")
    # Le tableau affiche les coefficients, odds ratios, p-values et si l'effet est significatif
    st.dataframe(regression_results)
    st.write("On voit ici résumé les résultats de la régression logistique. On voit que les couleurs noire et bleu sont significativement influencées par l'âge, tandis que les autres couleurs ne le sont pas. En effet on voit que les p-values des couleurs noir et bleu sont inférieures à 0.05, ce qui signifie qu'il y a une relation significative entre l'âge et le choix de ces couleurs. La colonne 'Significatif' indique si l'effet est significatif (1) ou non (0).La colonne 'coefficient' indique l'effet de l'âge sur le choix de la couleur, et la colonne 'odds_ratio' indique l'impact relatif de l'âge sur les chances de choisir cette couleur. Par exemple, un odds ratio de 0.5 pour le noir signifie qu'avec chaque année d'âge supplémentaire, les chances de choisir une voiture noire diminuuent de 50%. De même un odds ratio de 1.5 pour le blanc signifie qu'avec chaque année d'âge supplémentaire, les chances de choisir une voiture blanche augmentent de 50%.")
    # Conclusion de la section
    st.subheader("Conclusion")
    st.write("""
    En conclusion, nous avons observé des tendances intéressantes dans les préférences de couleurs selon l'âge des clients. Nous avons vu des tendances générales, comme la popularité du blanc et du noir, mais aussi des variations plus fines selon les tranches d'âge. 
    Les résultats de la régression logistique montrent que certaines couleurs sont significativement influencées par l'âge comme le blanc, le noir et le bleu tandis que d'autres ne le sont pas.  
    Ces informations peuvent être utilisées pour mieux cibler les clients en fonction de leur profil. 
    """)

page_3.__name__ = "Hypothèse Age x Couleur"

def page_4():
    
    
    st.title("La zone géographique a t'elle un impact sur le choix du véhicule ?")

    ##Explication de l'hypothèse
    st.write("Nous allons chercher à savoir ici si la zone géographique d'un client a un impact sur le type de voiture qu'il choisit. Pour cela, nous allons séparer les clients en 5 clusters et analyser la répartition des véhicules par zone.")
    st.write("Nous allons utiliser les données de localisation des clients et les croiser avec les données de véhicules pour voir s'il y a des tendances géographiques dans le choix des véhicules. Ainsi, nous avons déterminé trois principales caractéristiques de voitures : la marque, le type de voiture et le type d'énergie utilisé.")
    st.write("Nous allons également utiliser des tests statistiques pour vérifier si les différences observées sont significatives.")
    
    st.image("Carte.PNG", caption="Carte de France avec les clusters géographiques", use_container_width=True)


    # Display the Plotly map in Streamlit
   
    st.write("Cette carte montre la répartition géographique des 5 clusters.")
    st.write("Les clusters sont élaborés grâce à la méthode KMeans, qui regroupe les clients en fonction de leurs coordonnées géographiques. Chaque cluster est représenté par une couleur différente sur la carte.")
    st.write("En survolant les points, vous pouvez voir le nom de la commune associée à chaque point. Cela permet d'identifier facilement les zones géographiques où se trouvent les clients.")
    st.write("Nous allons maintenant analyser la répartition des véhicules par cluster.")
    
    # Energy distribution by cluster
    st.subheader("Répartition des énergies par cluster")
    st.image("énergiesPNG.PNG", caption="Répartition des énergies/carburant par cluster", use_container_width=True)
    

    st.write("Ce graphique montre la répartition des types d'énergie par cluster. On voit que les tendances dans chaque cluster suivent la tendance générale des ventes vues dans la partie statistique descriptives. Ainsi dans chaque cluster, on voit qu'il y a une majorité de voitures essence et les voitures électriques représentent une petite fraction des ventes. On remarque également que les clusters Nord+Paris et Sud-Est ont tout les deux beaucoup de ventes d'hybrides alors que les autres clusters ont tendance à avoir plus de voitures diesel que d'hybrides.")
    

    # Top marques by cluster
    st.subheader("Répartition des marques par cluster")

    st.image("marque.PNG", caption="Répartition des marques par cluster", use_container_width=True)
    st.write("On représente ici la distribution des 5 marques les plus populaires de chaque clusters. On voit que les tendances dans chaque cluster suivent également la tendance générale des ventes vues dans la partie statistique descriptives. Ainsi dans chaque cluster, il y a une majorité de voitures Citroën et Peugeot. On peut l'expliquer par le fait que ce sont des marques françaises et qu'elles sont donc plus populaire dans l'ensemble de la France. On remarque que le Sud-Est est la seule région avec des Volkswagen dans son Top 5. De même le Nord-Est est la seule région avec la marque Opel et le Sud-Ouest est le seul à avoir Nissan dans son top 5. Ainsi dans chaque cluster on retrouve toujours les marques Citroën, Peugeot et Dacia, puis 2 autres marques qui varient selon le cluster dans lequel on se trouve. ")

    # Category distribution by cluster
    st.subheader("Répartition des catégories par cluster")
    st.image("catégorie.PNG", caption="Répartition des catégories par cluster", use_container_width=True)
    st.write("Cette visualisation montre la répartition des catégories par cluster. Dans chaque cluster, on voit une majorité de 4x4 SUV et citadine puis en plus faible quantité on trouve des monospaces, des berlines compactes et des breaks. On remarque et c'est le cas pour l'ensemble des graphiques que le Nord et Paris représentent une grande partie des ventes, puis on a le Sud-Est, ensuite le Nord-Est,le Sud-Ouest et le Nord-Ouest qui représentent moins de ventes. ")

# 5. Tests statistiques
    # Display p-values in scientific notation
    st.subheader("Résultats des tests d'indépendance (p-value):")

    st.write("Dans cette partie nous allons déterminer si la zone géographaphique a une influence sur la marque et le type d'énergie de la voiture à l'aide des résultats de test de chi2. Le test de chi2 nous permet de déterminer si deux variables qualitatives sont indépendantes ou non.")
    
    st.markdown("**Marque** :  ")
    st.write("Nous avons donc testé l'hypothèse nulle : La variable marque est indépendante du cluster géographique auquel le client appartient. Autrement dit on teste si la distribution des différentes marque acheté par les clients est la même aux quatre coin de la France. L'hypothèse alternative est donc la suivante : La variable marque n'est pas indépendante du cluster géographique et donc la distribution des marques de voitures achetés par les clients n'est pas la même selon où se trouve en France.")
    st.markdown("""Ici la pvalue vaut **2.27e-249**, elle est donc inférieur à 5%. En statistique cela signifie que l'on peut rejeter l'hypothèse nulle à un niveau de confiance de 95%. Ainsi nous avançons qu'il est possible que la localisation du client a un impact sur la marque de voiture qu'il va choisir.""")
    
    st.markdown("**Énergie :**  ")
    st.write("""Nous avons donc testé l'hypothèse nulle : La variable énergie est indépendante du cluster géographique auquel le client appartient. Autrement dit on teste si la distribution des différentes énergies acheté par les clients est la même aux quatre coin de la France. L'hypothèse alternative est donc la suivante : La variable énergie n'est pas indépendante du cluster géographique et donc la distribution des énergies de voitures achetés par les clients n'est pas la même selon où on se trouve en France.""")
    st.write("""Ici la pvalue vaut **1.23e-63**, elle est donc inférieur à 5%. En statistique cela signifie que l'on peut rejeter l'hypothèse nulle à un niveau de confiance de 95%. Ainsi nous avançons qu'il est possible que la localisation du client a un impact sur le type d'énergie de voiture qu'il va choisir.""")

    st.markdown("**Catégorie** :  ")
    st.write("Nous avons donc testé l'hypothèse nulle : La variable catégorie est indépendante du cluster géographique auquel le client appartient. Autrement dit on teste si la distribution des différentes catégories acheté par les clients est la même aux quatre coin de la France. L'hypothèse alternative est donc la suivante : La variable catégorie n'est pas indépendante du cluster géographique et donc la distribution des catégories de voitures achetés par les clients n'est pas la même selon où on se trouve en France.")
    st.markdown("""Ici la pvalue vaut **1.41e-32**, elle est donc inférieur à 5%. En statistique cela signifie que l'on peut rejeter l'hypothèse nulle à un niveau de confiance de 95%. Ainsi nous avançons qu'il est possible que la localisation du client a un impact sur la catégorie de voiture qu'il va choisir.""") 

    st.markdown("**Conclusion** :")
    st.write("En conclusion, nous avons pu voir que la localisation du client a un impact sur la marque, le type et le type d'énergie consommé de la voiture qu'il va acheter. Cependant, il est important de noter que ces résultats sont basés sur des données historiques et qu'ils ne garantissent pas que ces tendances se poursuivront à l'avenir. Il est donc important de continuer à surveiller les tendances du marché et d'adapter les stratégies en conséquence.")

page_4.__name__ = "Hypothèse Localisation x type voiture"

def page_recommandations():
    st.title("Recommandations et Communication")

    # Introduction
    st.markdown("""
    Cette section propose des recommandations stratégiques basées sur les analyses précédentes et les statistiques descriptives.  
    Ces recommandations visent à optimiser les ventes, améliorer l'expérience client et renforcer la communication marketing d'Aramisauto.
    """)

    # Bilan des analyses
    st.subheader("Bilan des analyses")
    st.write("""
    Les analyses effectuées ont permis de mettre en évidence plusieurs tendances et comportements clés :
    - **Statistiques descriptives** :
        - Les SUV/4x4 et citadines dominent les ventes, suivis par les monospaces et berlines compactes.
        - Les véhicules essence sont majoritaires, mais les hybrides gagnent en popularité dans certaines régions.
        - Les marques françaises (Peugeot, Citroën) sont les plus vendues, avec des variations régionales.
        - Les couleurs neutres (gris, blanc, noir) sont les plus populaires, mais des nuances apparaissent selon l'âge.
    - **Hypothèse Prix x Options** :
        - Plus un véhicule est équipé, plus son prix est élevé. Certains équipements (Apple CarPlay, radar de recul) influencent davantage le prix.
    - **Hypothèse Consultation x Profil** :
        - Les clients qui consultent intensément une fiche véhicule tendent à acheter des modèles plus chers et à prendre plus de temps pour décider.
        - Les jeunes adoptent une approche polarisée (consultations faibles ou très élevées), tandis que les tranches 30-59 ans montrent un comportement plus équilibré.
    - **Hypothèse Age x Couleur** :
        - Les préférences de couleurs évoluent avec l'âge : les jeunes préfèrent le bleu et le noir, tandis que les seniors privilégient le blanc.
    - **Hypothèse Localisation x Type de voiture** :
        - La localisation influence les choix de marques, types de véhicules et énergies. Par exemple, les hybrides sont populaires dans le Sud-Est, tandis que le diesel reste dominant dans d'autres régions.
    """)

    # Recommandations stratégiques
    st.subheader("Recommandations stratégiques")
    st.write("""
    Sur la base des résultats obtenus, voici nos recommandations pour optimiser les ventes et la communication :
    """)

    st.markdown("### 1. Personnalisation de l'offre")
    st.write("""
    - **Adapter les stocks aux préférences locales** :
        - Augmenter la disponibilité des hybrides dans le Sud-Est et des diesels dans le Nord.
        - Proposer davantage de SUV/4x4 et citadines, qui dominent les ventes.
    - **Cibler les jeunes clients** :
        - Mettre en avant des modèles noirs ou bleus pour les 18-25 ans.
        - Proposer des véhicules équipés de technologies modernes (Apple CarPlay, radar de recul).
    - **Offrir des options personnalisées** :
        - Permettre aux clients de configurer leur véhicule (couleur, équipements) pour répondre à leurs attentes spécifiques.
    """)

    st.markdown("### 2. Communication ciblée")
    st.write("""
    - **Campagnes marketing régionales** :
        - Promouvoir les hybrides dans le Sud-Est et les diesels dans le Nord.
        - Adapter les messages publicitaires aux préférences locales (ex. : SUV dans les zones rurales, citadines en milieu urbain).
    - **Mise en avant des marques françaises** :
        - Capitaliser sur la popularité de Peugeot et Citroën pour renforcer la confiance des clients.
    """)

    st.markdown("### 3. Optimisation de l'expérience utilisateur")
    st.write("""
    - **Améliorer le moteur de recherche du site** :
        - Ajouter des filtres pour trier les véhicules par couleur, type d'énergie ou catégorie.
    - **Proposer des recommandations personnalisées** :
        - Utiliser les données des clients pour suggérer des véhicules adaptés à leur profil (âge, localisation, comportement de consultation).
    """)

    st.markdown("### 4. Fidélisation des clients")
    st.write("""
    - **Programmes de fidélité** :
        - Offrir des réductions ou des avantages aux clients fidèles (ex. : options gratuites, entretien offert).
    - **Suivi post-achat** :
        - Envoyer des enquêtes de satisfaction et des offres personnalisées pour inciter à un nouvel achat et avoir plus de données pour construire de nouvelles hypothèses et affiner le profiling.
    """)

    st.markdown("### 5. Stratégie de tarification")
    st.write("""
    - **Valoriser les équipements premium** :
        - Mettre en avant les équipement d'un véhicule, qui justifient un prix plus élevé.
    - **Offrir des promotions ciblées** :
        - Réductions sur les modèles moins populaires dans certaines régions pour stimuler les ventes.
    """)

    # Conclusion
    st.subheader("Conclusion")
    st.write("""
    En mettant en œuvre ces recommandations, Aramisauto peut mieux répondre aux attentes de ses clients, augmenter ses ventes et renforcer sa position sur le marché.  
    Une stratégie basée sur les données permet non seulement d'optimiser les performances actuelles, mais aussi de s'adapter aux évolutions futures des préférences des consommateurs.
    """)
page_recommandations.__name__ = "Recommandations et Communication"


pg = st.navigation({
    "Analyse du profil des clients Aramisauto" : [bienvenue, ensemble, page_1, page_2, page_3, page_4, page_recommandations],
})


pg.run()


