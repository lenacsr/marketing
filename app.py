import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.customize_running import center_running
import pandas as pd
import numpy as np
import random 
import ydata
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

warnings.filterwarnings("ignore")
table_full = pd.read_csv("table_full.csv")
table_vehicle = pd.read_csv("table_vehicule.csv")

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
        attendre que ça soit fini mais en gros on parle des hypothèses
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


    
def vue_d_ensemble():
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

    col2_1.metric("Nombre de voitures disponibles :", table_vehicle.shape[0])
    col2_2.metric("Nombre de modèles de voitures :", table_vehicle['VEHICULE_MODELE'].nunique())
    col2_3.metric("Nombre de marques différentes disponibles :", table_vehicle["VEHICULE_MARQUE"].nunique())
    
    col3_1, col3_2, col3_3 = st.columns(3)
    
    col3_1.metric("Taux de reprise", f"{(table_full['FLAG_REPRISE'].sum() / table_full['FLAG_COMMANDE'].sum() * 100):.1f} %")
    col3_2.metric("Kilométrage moyen", f"{table_vehicle['VEHICULE_KM'].mean():,.0f} km")    
    col3_3.metric("Prix moyen TTC", f"{table_full['PRIX_VENTE_TTC_COMMANDE'].mean():,.0f} €")

    
    col_voitures_1, col_voitures_2 = st.columns(2)
    #repartition par marque
    with col_voitures_1.container():
        st.subheader("Nombre de voitures par marque")
        qtte_marque = st.slider("Choisissez le nombre de marques à afficher", 1, table_vehicle['VEHICULE_MARQUE'].nunique(), 10)
        fig5, ax5 = plt.subplots()
        top_marques = table_vehicle['VEHICULE_MARQUE'].value_counts().head(qtte_marque)
        ax5.barh(top_marques.index[::-1], top_marques.values[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
        ax5.set_xlabel("Nombre de voitures")
        ax5.set_ylabel("Marque")
        st.pyplot(fig5)
        
    #repartion par modele
    with col_voitures_2.container():
        st.subheader("Nombre de voitures par modèle")
        qtte_modele = st.slider("Choisissez le nombre de modèles à afficher", 1, table_vehicle['VEHICULE_MODELE'].nunique(), 10)
        fig7, ax7 = plt.subplots()
        top_modeles = table_vehicle['VEHICULE_MODELE'].value_counts().head(qtte_modele)
        ax7.barh(top_modeles.index[::-1], top_modeles.values[::-1], color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
        ax7.set_xlabel("Nombre de voitures")
        ax7.set_ylabel("Modèle")
        st.pyplot(fig7)
    

    col_voitures_2_1, col_voitures_2_2 = st.columns(2)
    # VO vs VN
    with col_voitures_2_1.container():
        st.subheader("Répartition des voitures neuves / d'occasions")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        table_vehicle['VEHICULE_TYPE'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax6, colors=[color1, color2],wedgeprops={'edgecolor': 'black'})
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
        top_energies = table_vehicle['VEHICULE_ENERGIE'].value_counts()
        colors = [energy_colors.get(energy, "#bdc3c7") for energy in top_energies.index]
        ax3.barh(top_energies.index[::-1], top_energies.values[::-1], color=plt.cm.Purples(np.linspace(0.2, 0.9, 10)))
        ax3.set_xlabel("Nombre de voitures")
        ax3.set_ylabel("Carburant")
        st.pyplot(fig3)

    
    if switch_voiture:
        rain(emoji="🚗", font_size=70, falling_speed=3.5, animation_length=600)
        
vue_d_ensemble.__name__ = "Statistiques descriptives"       
        
def page_1():
    st.title("Nom Hypothèse")
    st.write("description de l'hypothèse vite fait")
    st.write("graphiques/tableaux + interprétations")
    st.write("conclu de l'hypothèse")
        
page_1.__name__ = "Hypothèse 1"
    
    
def page_2():
    st.title("Nom Hypothèse")
    st.write("description de l'hypothèse vite fait")
    st.write("graphiques/tableaux + interprétations")
    st.write("conclu de l'hypothèse")

page_2.__name__ = "Hypothèse 2"
            
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
    On observe que certaines couleurs, comme le gris et le noir, sont populaires dans presque toutes les tranches d'âge, tandis que d'autres, comme le rouge ou le bleu, varient davantage selon l'âge. En efet le rouge semble être plus populaire chez les personnes agées que chez les jeunes, alors que le bleu lui est plus populaire chez les jeunes que chez les personnes agées.
    """)

    # Afficher les barplots
    st.write("**Barplots des couleurs par tranche d'âge :**")
    st.plotly_chart(fig_barplots, use_container_width=True)
    st.write("""
    **Interprétation :**  
    Les barplots permettent de visualiser les proportions exactes de chaque couleur pour chaque tranche d'âge.  
    On continue a observer la préférence pour le gris noir et blanc à travers les tranches d'âges. On note également que les 18-25 ans achètent beaucoup moins de voiture que les autres tranches d'âges. Les autres tranches d'âges achètent environ dix fois plus que les 18-25 ans.
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
    On remarque que certaines couleurs, comme le noir, augmentent en popularité avec l'âge, tandis que d'autres, comme le rouge, diminuent progressivement.
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
    st.write("Ce graphique montre l'effet de l'âge sur la probabilité de choisir chaque couleur. Les barres positives indiquent que l'âge augmente la probabilité de choisir une couleur, tandis que les barres négatives indiquent que l'âge diminue cette probabilité. Les barres plus foncées sont celles qui sont statistiquement significatives ainsi plus on viellit plus on a de chance lorsqu'on achete une voiture d'en choisir une blanche et on a moins de chance d'en choisir une noir ou bleu.")
    
    
    st.write("**Graphique des odds ratios :**")
    st.plotly_chart(fig_odds_ratios, use_container_width=True)
    st.write("Ce graphique montre l'impact relatif de l'âge sur les chances de choisir chaque couleur. Un odds ratio supérieur à 1 indique que l'âge augmente les chances de choisir une couleur, tandis qu'un odds ratio inférieur à 1 indique que l'âge diminue ces chances. Ici les barres foncés représentent les couleurs qui sont significativement influencées par l'âge. Ainsi on confirme le constat fait précédement qui est que l'age influence positivement le choix de la couleur blanche et négativement le choix des couleurs noir et bleu. Ainsi plus on viellit plus on a de chance de choisir une voiture blanche et moins on a de chance de choisir une voiture noir ou bleu.")
    
    # Afficher les résultats sous forme de tableau interactif
    st.write("**Résumé des résultats de régression :**")
    # Le tableau affiche les coefficients, odds ratios, p-values et si l'effet est significatif
    st.dataframe(regression_results)
    st.write("On voit ici résumé les résultats de la régression logistique. On voit que les couleurs noir et bleu sont significativement influencées par l'âge, tandis que les autres couleurs ne le sont pas. En effet on voit que les p-values des couleurs noir et bleu sont inférieures à 0.05, ce qui signifie qu'il y a une relation significative entre l'âge et le choix de ces couleurs. La colonne 'Significatif' indique si l'effet est significatif (1) ou non (0).La colonne 'coefficient' indique l'effet de l'âge sur le choix de la couleur, et la colonne 'odds_ratio' indique l'impact relatif de l'âge sur les chances de choisir cette couleur. Par exemple, un odds ratio de 0.5 pour le noir signifie qu'avec chaque année d'âge supplémentaire, les chances de choisir une voiture noire diminuuent de 50%. De même un odds ratio de 1.5 pour le blanc signifie qu'avec chaque année d'âge supplémentaire, les chances de choisir une voiture blanche augmentent de 50%.")
    # Conclusion de la section
    st.subheader("Conclusion")
    st.write("""
    En conclusion, nous avons observé des tendances intéressantes dans les préférences de couleurs selon l'âge des clients. Nous avons vu des tendances générales, comme la popularité du blanc et du noir, mais aussi des variations plus fines selon les tranches d'âge. 
    Les résultats de la régression logistique montrent que certaines couleurs sont significativement influencées par l'âge comme le blanc, le noir et le bleu tandis que d'autres ne le sont pas.  
    Ces informations peuvent être utilisées pour mieux cibler les clients en fonction de leur profil. 
    """)

page_3.__name__ = "Hypothèse 3"

page_3.__name__ = "Hypothèse 3"

def page_4():
    
    
    st.title("La zone géographique a t'elle un impact sur le choix du véhicule ?")

    ##Explication de l'hypothèse
    st.write("Nous allons chercher à savoir ici si la zone géographique d'un client a un impact sur le type de voiture qu'il choisit. Pour cela, nous allons séparer les clients en 5 clusters et analyser la répartition des véhicules par zone.")
    st.write("Nous allons utiliser les données de localisation des clients et les croiser avec les données de véhicules pour voir s'il y a des tendances géographiques dans le choix des véhicules. Ainsi, nous avons déterminé trois principales caractéristiques de voitures : la marque, le type de voiture et le type d'énergie utilisé.")
    st.write("Nous allons également utiliser des tests statistiques pour vérifier si les différences observées sont significatives.")
    
    # Chargement des données

    df = pd.read_csv('table_full.csv', parse_dates=['DATE_CREATION', 'DATE_COMMANDE'])



# 1. Prétraitement des données géographiques
# Conversion des codes postaux en coordonnées (nécessite une API ou fichier de référence)
# Exemple avec fichier des codes postaux français :

    def get_geo_data(code_postal):
        """Récupère les coordonnées géographiques pour un code postal donné"""
        url = "https://geo.api.gouv.fr/communes"
        params = {
            "codePostal": code_postal,
            "fields": "codesPostaux,nom,centre",
            "format": "json",
            "geometry": "centre"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            communes = response.json()
            
            results = []
            for commune in communes:
                # Extraction des coordonnées depuis l'objet 'centre'
                if 'centre' in commune:
                    coordinates = commune['centre']['coordinates']
                    for cp in commune['codesPostaux']:
                        results.append({
                            'CODE_POSTAL': cp,
                            'commune': commune['nom'],
                            'longitude': coordinates[0],
                            'latitude': coordinates[1]
                        })
            return results
        
        except requests.exceptions.RequestException as e:
            print(f"Erreur pour le code postal {code_postal}: {str(e)}")
            return []

    # Récupération de tous les codes postaux uniques du dataframe
    codes_postaux_uniques = df['CODE_POSTAL'].unique()

    # Création du dataframe géographique
    geo_data = []
    for cp in codes_postaux_uniques:
        geo_data.extend(get_geo_data(cp))

    cp_geo = pd.DataFrame(geo_data).drop_duplicates()
    # Avant la fusion, convertir explicitement les types
    # Harmonisation des formats
    df['CODE_POSTAL'] = df['CODE_POSTAL'].astype(str).str.zfill(5).str.strip()
    cp_geo['CODE_POSTAL'] = cp_geo['CODE_POSTAL'].astype(str).str.zfill(5).str.strip()

    # Merge
    df = df.merge(cp_geo, on='CODE_POSTAL', how='left')
    df = df.dropna(subset=['latitude', 'longitude'])


    # Vérification des types pour le plot
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')


    # 2. Clustering géographique
    coords = df[['latitude', 'longitude']].dropna()
    kmeans = KMeans(n_clusters= 5, random_state=42)
    df['geo_cluster'] = kmeans.fit_predict(coords)


    # 3. Analyse des véhicules par cluster
    features_vehicule = ['VEHICULE_MARQUE', 'VEHICULE_ENERGIE', 'VEHICULE_CATEGORIE']
    results = {}

    for feature in features_vehicule:
        contingency_table = pd.crosstab(df['geo_cluster'], df[feature])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        results[feature] = p

    # 4. Visualisation
    # Define a custom discrete color scale for the clusters


    df['geo_cluster'] = df['geo_cluster'].astype(str)

    cluster_labels = {
    "0": "Nord-Est",
    "1": "Nord+Paris",
    "2": "Nord-Ouest",
    "3": "Sud-Est",
    "4": "Sud-Ouest"
}

# Map the cluster numbers to region names
    df['geo_cluster_label'] = df['geo_cluster'].map(cluster_labels)

    custom_color = ["#ffadad",  "#ffd6a5",  "#fdffb6",  "#caffbf",  "#9bf6ff",  "#a0c4ff",  "#bdb2ff","#ffc6ff"]

    fig_map = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="geo_cluster_label",  # Colorer par cluster
        hover_name="commune",
        hover_data={"geo_cluster_label": True, "latitude": False, "longitude": False},
        text="geo_cluster_label",
        zoom=3,
        title="Répartition géographique des clusters",
        color_discrete_sequence=custom_color
    )

    fig_map.update_traces(
        marker=dict(size=10, opacity=0.8),  # Taille et opacité des points
        textfont=dict(size=10, color="black"),
        textposition="top center"
    )

    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": df['latitude'].mean(), "lon": df['longitude'].mean()},
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)


    # Display the Plotly map in Streamlit
   
    st.write("Cette carte interactive montre la répartition géographique des 5 clusters.")
    st.write("Les clusters sont élaborés grâce à la méthode KMeans, qui regroupe les clients en fonction de leurs coordonnées géographiques. Chaque cluster est représenté par une couleur différente sur la carte.")
    st.write("En survolant les points, vous pouvez voir le nom de la commune associée à chaque point. Cela permet d'identifier facilement les zones géographiques où se trouvent les clients.")
    st.write("Nous allons maintenant analyser la répartition des véhicules par cluster.")
    
    # Energy distribution by cluster
    st.subheader("Répartition des énergies par cluster")
    energy_dist = df.groupby(['geo_cluster_label', 'VEHICULE_ENERGIE']).size().unstack()
    fig_energy, ax_energy = plt.subplots()
    colors = custom_color
    energy_dist.plot(kind='bar', stacked=True, ax=ax_energy, color=colors)
    ax_energy.set_title('Répartition des énergies par cluster')
    ax_energy.set_ylabel('Nombre de véhicules')
    ax_energy.set_xlabel('Cluster')
    ax_energy.set_xticklabels(ax_energy.get_xticklabels(), rotation=0, ha='center')
    
    # Move the legend outside the plot
    ax_energy.legend(
        title="Energies",
        bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot (to the right)
        loc='upper left',  # Anchor the legend to the upper-left corner of the bounding box
        fontsize=10  # Adjust font size for readability
    )

    st.pyplot(fig_energy)

    st.write("Ce graphique montre la répartition des types d'énergie par cluster. On voit que les tendances dans chaque cluster suivent la tendance générale des ventes vues dans la partie statistique descriptives. Ainsi dans chaque cluster, on voit qu'il y a une majorité de voitures essence et les voitures électriques représentent une petite fraction des ventes. On remarque également que les clusters Nord+Paris et Sud-Est ont tout les deux beaucoup de ventes d'hybrides alors que les autres clusters ont tendance à avoir plus de voitures diesel que d'hybrides.")
    

    # Top marques by cluster
    st.subheader("Répartition des marques par cluster")

    # Filter to include only the top 5 brands for each cluster
    top_marques_counts = (
        df.groupby(['geo_cluster_label', 'VEHICULE_MARQUE'])
        .size()
        .reset_index(name='count')  # Reset index to make 'count' a column
    )

    # Get the top 5 brands for each cluster
    top_marques_filtered = (
        top_marques_counts.groupby('geo_cluster_label', group_keys=False)
        .apply(lambda x: x.nlargest(5, 'count'))
    )

    # Pivot the filtered data for plotting
    top_marques_pivot = top_marques_filtered.pivot(
        index='geo_cluster_label', columns='VEHICULE_MARQUE', values='count'
    ).fillna(0)

    
    # Plot the filtered data
    fig_marques, ax_marques = plt.subplots(figsize=(10, 6))  # Adjust figure size for better readability
    top_marques_pivot.plot(kind='bar', stacked=True, ax=ax_marques, color=custom_color[:top_marques_pivot.shape[1]])

    # Set titles and labels
    ax_marques.set_title('Répartition des marques par cluster (Top 5)', fontsize=14)
    ax_marques.set_ylabel('Nombre de véhicules', fontsize=12)
    ax_marques.set_xlabel('Cluster', fontsize=12)

    ax_marques.set_xticklabels(ax_marques.get_xticklabels(), rotation=0, ha='center')  # Set rotation to 0 and align center

    # Move the legend outside the plot
    ax_marques.legend(
        title="Marques",
        bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot (to the right)
        loc='upper left',  # Anchor the legend to the upper-left corner of the bounding box
        fontsize=10  # Adjust font size for readability
    )

    # Display the plot in Streamlit
    st.pyplot(fig_marques)
    st.write("On représente ici la distribution des 5 marques les plus populaires de chaque clusters. On voit que les tendances dans chaque cluster suivent également la tendance générale des ventes vues dans la partie statistique descriptives. Ainsi dans chaque cluster, il y a une majorité de voitures Citroën et Peugeot. On peut l'expliquer par le fait que ce sont des marques françaises et qu'elles sont donc plus populaire dans l'ensemble de la France. On remarque que le Sud-Est est la seule région avec des Volkswagen dans son Top 5. De même le Nord-Est est la seule région avec la marque Opel et le Sud-Ouest est le seul à avoir Nissan dans son top 5. Ainsi dans chaque cluster on retrouve toujours les marques Citroën, Peugeot et Dacia, puis 2 autres marques qui varient selon le cluster dans lequel on se trouve. ")

    # Category distribution by cluster
    st.subheader("Répartition des catégories par cluster")
    category_dist = df.groupby(['geo_cluster_label', 'VEHICULE_CATEGORIE']).size().unstack()
    fig_category, ax_category = plt.subplots()
    colors = custom_color
    category_dist.plot(kind='bar', stacked=True, ax=ax_category, color=colors)
    ax_category.set_title('Répartition des catégories par cluster')
    ax_category.set_ylabel('Nombre de véhicules')
    ax_category.set_xlabel('Cluster')
    ax_category.set_xticklabels(ax_category.get_xticklabels(), rotation=0, ha='center')
    
    # Move the legend outside the plot
    ax_category.legend(
        title="Catégories",
        bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot (to the right)
        loc='upper left',  # Anchor the legend to the upper-left corner of the bounding box
        fontsize=10  # Adjust font size for readability
    )

    st.pyplot(fig_category)
    st.write("Cette visualisation montre la répartition des catégories par cluster. Dans chaque cluster, on voit une majorité de 4x4 SUV et citadine puis en plus faible quantité on trouve des monospaces, des berlines compactes et des breaks. On remarque et c'est le cas pour l'ensemble des graphiques que le Nord et Paris représentent une grande partie des ventes, puis on a le Sud-Est, ensuite le Nord-Est,le Sud-Ouest et le Nord-Ouest qui représentent moins de ventes. ")

# 5. Tests statistiques
    # Display p-values in scientific notation
    st.subheader("Résultats des tests d'indépendance (p-value):")

    st.write("Dans cette partie nous allons déterminer si la zone géographaphique a une influence sur la marque et le type d'énergie de la voiture à l'aide des résultats de test de chi2. Le test de chi2 nous permet de déterminer si deux variables qualitatives sont indépendantes ou non.")
    
    st.markdown("**Marque** :  ")
    st.write("Nous avons donc tester l'hypothèse nulle : La variable marque est indépendante du cluster géographique auquel le client appartient. Autrement dit on teste si la distribution des différentes marque acheté par les clients est la même aux quatre coin de la France. L'hypothèse alternative est donc la suivante : La variable marque n'est pas indépendante du cluster géographique et donc la distribution des marques de voitures achetés par les clients n'est pas la même selon ou on se trouve en France.")
    st.markdown(f"Ici la pvalue vaut **{results['VEHICULE_MARQUE']:.2e}**, elle est donc inférieur à 5%. En statistique cela signifie que l'on peut rejeter l'hypothèse nulle à un niveau de confiance de 95%. Ainsi nous avançons qu'il est possible que la localisation du client a un impact sur la marque de voiture qu'il va choisir.")
    
    st.markdown("**Énergie :**  ")
    st.write("Nous avons donc tester l'hypothèse nulle : La variable énergie est indépendante du cluster géographique auquel le client appartient. Autrement dit on teste si la distribution des différentes énergies acheté par les clients est la même aux quatre coin de la France. L'hypothèse alternative est donc la suivante : La variable énergie n'est pas indépendante du cluster géographique et donc la distribution des énergies de voitures achetés par les clients n'est pas la même selon ou on se trouve en France.")
    st.write(f"Ici la pvalue vaut **{results['VEHICULE_ENERGIE']:.2e}**, elle est donc inférieur à 5%. En statistique cela signifie que l'on peut rejeter l'hypothèse nulle à un niveau de confiance de 95%. Ainsi nous avançons qu'il est possible que la localisation du client a un impact sur le type d'énergie de voiture qu'il va choisir.")

    st.markdown("**Catégorie** :  ")
    st.write("Nous avons donc tester l'hypothèse nulle : La variable catégorie est indépendante du cluster géographique auquel le client appartient. Autrement dit on teste si la distribution des différentes catégories acheté par les clients est la même aux quatre coin de la France. L'hypothèse alternative est donc la suivante : La variable catégorie n'est pas indépendante du cluster géographique et donc la distribution des catégories de voitures achetés par les clients n'est pas la même selon ou on se trouve en France.")
    st.markdown(f"Ici la pvalue vaut **{results['VEHICULE_CATEGORIE']:.2e}**, elle est donc inférieur à 5%. En statistique cela signifie que l'on peut rejeter l'hypothèse nulle à un niveau de confiance de 95%. Ainsi nous avançons qu'il est possible que la localisation du client a un impact sur la catégorie de voiture qu'il va choisir.") 

    st.markdown("**Conclusion** :")
    st.write("En conclusion, nous avons pu voir que la localisation du client a un impact sur la marque, le type et le type d'énergie consommé de la voiture qu'il va acheter. Cependant, il est important de noter que ces résultats sont basés sur des données historiques et qu'ils ne garantissent pas que ces tendances se poursuivront à l'avenir. Il est donc important de continuer à surveiller les tendances du marché et d'adapter les stratégies en conséquence.")

page_4.__name__ = "Hypothèse 4"

st.set_page_config(
    page_title="Analyse du profil des clients Aramisauto",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation({
    "Analyse du profil des clients Aramisauto" : [bienvenue, vue_d_ensemble, page_1, page_2, page_3, page_4]
})

switch_voiture = st.sidebar.toggle("Activer le mode voiture")

color1 = st.sidebar.color_picker("Couleur 1", "#89CFF0")
color2 = st.sidebar.color_picker("Couleur 2", "#B19CD9") 

st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #999; padding-top: 1rem;">
       Développé par BRAULT Juliette, CAUSEUR Léna et PRUSIEWICZ Louis.
    </div>
    """,
    unsafe_allow_html=True
)

pg.run()


