import streamlit as st
from datetime import datetime, timedelta
from price_data import PriceData
from strategies.BuyAndHold import BuyAndHold
from strategies.Momentum import Momentum
from Indicateurs import Indicateurs
from strategies.data import DataDownloader
import pandas as pd
import plotly.express as px
import logging
from strategies.MinVariance import MinimumVariance
from strategies.ML import MLInvestmentStrategy
import docx
from PIL import Image
import os
from strategies.DL_V2 import DLInvestmentStrategy
import os
import docx
import io
import streamlit as st
from PIL import Image
import base64
import mammoth
from bs4 import BeautifulSoup


# Initialisation des outils
price_lib = PriceData()
indicateurs = Indicateurs()
data_downloader = DataDownloader()

# Charger les tickers et entreprises
TICKERS_DICT = price_lib.universe()

# Interface Streamlit
st.title("📈 Dashboard des Stratégies d'Investissement")

# Paramètres de la stratégie
st.sidebar.header("Paramètres de la stratégie")
strategie_choisie = st.sidebar.selectbox("Sélectionner une Stratégie", ["BuyAndHold", "Momentum", "MinimumVariance", "MachineLearning", "DeepLearning"])

# Paramètres d'entrée
montant_initial = st.sidebar.number_input("Montant initial (€)", min_value=100, max_value=100000, value=1000)

data = price_lib.prices(list(TICKERS_DICT.keys()))
date_min = data.index.min()

# Vérifier si date_min est valide (NaT)
if pd.isna(date_min):
    date_min = datetime(2023, 1, 1).date()
else:
    date_min = date_min.date()


# Sélection des dates d'investissement
date_investissement = pd.to_datetime(
    st.sidebar.date_input("Date d'investissement", value=datetime(2023, 1, 1), min_value=date_min)
)
date_fin_investissement = pd.to_datetime(
    st.sidebar.date_input("Date de fin d'investissement", value=datetime.now().date(), min_value=date_investissement)
)

# Paramètres spécifiques à la stratégie Machine Learning et PCA
if strategie_choisie == "MachineLearning":
    lookback_period = st.sidebar.number_input("Période d'historique (jours)", min_value=1, max_value=365, value=10)
    
    # Nouveau paramètre pour activer/désactiver le PCA
    use_pca = st.sidebar.checkbox("Appliquer PCA avant Machine Learning ?", value=False)
    
    # Si PCA activé, demander le nombre de composantes
    if use_pca:
        n_components = st.sidebar.number_input("Nombre de composantes PCA", min_value=1, max_value=10, value=2)
    else:
        n_components = None
elif strategie_choisie == "DeepLearning":
    lookback_period = st.sidebar.number_input("Période d'historique (jours)", min_value=1, max_value=365, value=10)
else:
    lookback_period = None



# Validation explicite des dates
if date_investissement is None or date_fin_investissement is None:
    st.error("Veuillez sélectionner des dates valides pour la stratégie.")
    st.stop()

# Sélection de plusieurs entreprises par leurs noms
entreprises_selectionnees = st.sidebar.multiselect("Sélectionner les Entreprises", list(TICKERS_DICT.values()))

# Convertir les noms d'entreprises sélectionnés en tickers correspondants
tickers_selectionnes = [ticker for ticker, name in TICKERS_DICT.items() if name in entreprises_selectionnees]

if not tickers_selectionnes:
    st.warning("⚠️ Veuillez sélectionner au moins une entreprise pour exécuter la stratégie.")
    st.stop()

# Paramètres supplémentaires pour la stratégie Momentum et MinimumVariance
if strategie_choisie in ["Momentum", "MinimumVariance"]:
    max_assets = len(tickers_selectionnes)
    nombre_actifs = st.sidebar.number_input("Nombre d'actifs à sélectionner", min_value=1, max_value=max_assets, value=min(3, max_assets))
    periode_reroll = st.sidebar.number_input("Période de réévaluation (jours)", min_value=1, max_value=365, value=30)
    periode_historique = st.sidebar.number_input("Période historique (jours)", min_value=1, max_value=365, value=90)
else:
    nombre_actifs, periode_reroll, periode_historique = None, None, None

# Instancier la stratégie sélectionnée
strategie = None


def _charger_donnees(tickers, date_debut, date_fin):
    """Télécharge les données pour les tickers sélectionnés entre deux dates."""
    data_dict = {}
    for ticker in tickers:
        try:
            print(00000000000000)
            data = data_downloader.download_data(ticker, date_debut, date_fin)
            if not data.empty:
                data_dict[ticker] = data.reset_index()
            else:
                st.warning(f"Aucune donnée disponible pour {ticker}.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des données pour {ticker}: {e}")
    return data_dict

# Charger les données en fonction des dates choisies
data_dict = _charger_donnees(tickers_selectionnes, date_investissement, date_fin_investissement)

if st.sidebar.button("Lancer l'analyse"):
    if strategie_choisie == "BuyAndHold" and tickers_selectionnes:
        logging.info(
            f"Lancement de la stratégie 'BuyAndHold' : {tickers_selectionnes} | {montant_initial} | {date_investissement} | {date_fin_investissement}"
        )
        strategie = BuyAndHold(montant_initial, date_investissement, tickers_selectionnes, date_fin_investissement)
    elif strategie_choisie == "Momentum" and tickers_selectionnes and nombre_actifs and periode_reroll and periode_historique:
        logging.info(
            f"Lancement de la stratégie 'Momentum' : {tickers_selectionnes} | {montant_initial} | {nombre_actifs} | {periode_reroll} | {periode_historique} | {date_investissement} | {date_fin_investissement}"
        )
        strategie = Momentum(
            montant_initial, tickers_selectionnes, nombre_actifs, periode_reroll, periode_historique, date_investissement, date_fin_investissement
        )
    elif strategie_choisie == "MinimumVariance" and tickers_selectionnes:
        logging.info(
            f"Lancement de la stratégie 'MinimumVariance' : {tickers_selectionnes} | {montant_initial} | {date_investissement} | {date_fin_investissement}"
        )
        strategie = MinimumVariance(
            montant_initial, tickers_selectionnes, nombre_actifs, periode_reroll, periode_historique, date_investissement, date_fin_investissement
        )
    elif strategie_choisie == "MachineLearning" and tickers_selectionnes:
        logging.info(
            f"Lancement de la stratégie 'MachineLearning' : {tickers_selectionnes} | {montant_initial} | {lookback_period} | {date_investissement} | {date_fin_investissement}"
        )
        strategie = MLInvestmentStrategy(
            tickers=tickers_selectionnes,
            start_date=date_investissement,
            end_date=date_fin_investissement,
            initial_capital=montant_initial,
            lookback_period=lookback_period,
            use_pca=use_pca,
            n_components=n_components if use_pca else 2  # valeur par défaut en cas de PCA désactivé
        )
    elif strategie_choisie == "DeepLearning" and tickers_selectionnes:
        logging.info(
            f"Lancement de la stratégie 'DeepLearning' : {tickers_selectionnes} | {montant_initial} | {lookback_period} | {date_investissement} | {date_fin_investissement}"
        )
        strategie = DLInvestmentStrategy(
            tickers=tickers_selectionnes,
            start_date=date_investissement,
            end_date=date_fin_investissement,
            initial_capital=montant_initial,
            lookback_period=lookback_period
        )
    try:
        # Exécution de la stratégie
        performance_results = strategie.execute()
        # Vérification que le résultat est un dictionnaire structuré
        if not isinstance(performance_results, dict):
            st.error("Les résultats de la stratégie ne sont pas structurés correctement.")
        else:
            tab_names = [
                "Résumé des Indicateurs",
                "Explication",
                "Graphique des Prix",
                "Valeur du Portefeuille",
                "Proportions",
                "Matrice de Corrélation"
            ]
            # Ajouter "Tableau Récapitulatif" seulement si la stratégie le requiert
            if strategie_choisie in ["Momentum", "MinimumVariance"]:
                tab_names.append("Tableau Récapitulatif")

            # Création des onglets en filtrant les None
            tabs = st.tabs(tab_names)

        # Ajout du contenu explicatif dans l'onglet "Explication"
        with tabs[1]:
            with tabs[1]:
                def afficher_latex_strategie(strategie_choisie):
                    """Affiche un fichier .latex ligne par ligne dans Streamlit, en formatant correctement math et texte."""
                    import os

                    base_dir = os.getcwd()
                    chemin_fichier = os.path.join(f"{strategie_choisie}.latex")

                    if not os.path.isfile(chemin_fichier):
                        st.warning(f"❌ Fichier introuvable : {chemin_fichier}")
                        return

                    with open(chemin_fichier, "r", encoding="utf-8") as f:
                        lignes = f.readlines()

                    st.markdown(f"### 📄 Formule mathématique pour {strategie_choisie}")

                    for ligne in lignes:
                        ligne = ligne.strip()
                        if not ligne:
                            continue  # ignorer les lignes vides

                        # Si la ligne contient une formule (commence par \ ou contient =), afficher en LaTeX
                        if ligne.startswith("\\") or "=" in ligne:
                            st.latex(ligne)
                        else:
                            st.markdown(ligne)


                # Exécuter la fonction
                afficher_latex_strategie(strategie_choisie)



            with tabs[0]:
                    st.markdown("# 📊 Tableau de Bord des Performances")
                    
                    # Création d'un conteneur avec bordure pour le résumé
                    with st.container():
                        st.markdown("""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:20px; background-color:#f8f9fa">
                            <h3 style="text-align:center; color:#0366d6;">💰 Aperçu des Performances</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Métriques principales dans des colonnes avec plus de style
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div style="background-color:#e6f3ff; padding:10px; border-radius:5px; text-align:center;">', unsafe_allow_html=True)
                            st.metric("🔹 Gain Total", f"{performance_results.get('gain_total', 0):,.2f} €", 
                                    delta=None, delta_color="normal")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div style="background-color:#e6ffe6; padding:10px; border-radius:5px; text-align:center;">', unsafe_allow_html=True)
                            st.metric("📈 Gain Total (%)", f"{performance_results.get('pourcentage_gain_total', 0):.2f}%",
                                    delta=None, delta_color="normal")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div style="background-color:#fff5e6; padding:10px; border-radius:5px; text-align:center;">', unsafe_allow_html=True)
                            st.metric("📊 Performance Annualisée", f"{performance_results.get('performance_annualisee', 0):.2f}%", 
                                    delta=None, delta_color="normal")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Section Volatilité avec visualisation améliorée
                    st.markdown("""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin:20px 0; background-color:#f8f9fa">
                        <h3 style="text-align:center; color:#0366d6;">📉 Mesures de Volatilité</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        volatilite_historique = performance_results.get('volatilite_historique', 0) * 100
                        st.markdown('<div style="background-color:#f0f0f0; padding:15px; border-radius:8px; text-align:center;">', unsafe_allow_html=True)
                        st.metric("📊 Volatilité Historique", f"{volatilite_historique:.2f}%")
                        st.progress(min(volatilite_historique/50, 1.0))  # Une barre de progression pour visualiser
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        ewma_volatility = performance_results.get('ewma_volatility', 0) * 100
                        st.markdown('<div style="background-color:#f0f0f0; padding:15px; border-radius:8px; text-align:center;">', unsafe_allow_html=True)
                        st.metric("📊 Volatilité EWMA", f"{ewma_volatility:.2f}%")
                        st.progress(min(ewma_volatility/50, 1.0))  # Une barre de progression pour visualiser
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Section VaR organisée en onglets pour plus de clarté
                    st.markdown("""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin:20px 0; background-color:#f8f9fa">
                        <h3 style="text-align:center; color:#0366d6;">🚨 Indicateurs de Risque</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    var_tabs = st.tabs(["📉 Value at Risk (VaR)", "🔥 Conditional VaR (CVaR)"])
                    
                    with var_tabs[0]:
                        st.markdown('<div style="padding:10px; border-radius:5px; background-color:#ffeded;">', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            var_param = abs(performance_results.get('VaR Paramétrique', 0)) * 100
                            st.metric("⚠️ VaR Paramétrique", f"{var_param:.2f}%")
                            st.progress(min(var_param/30, 1.0), "danger")
                        with col2:
                            var_hist = abs(performance_results.get('VaR Historique', 0)) * 100
                            st.metric("⚠️ VaR Historique", f"{var_hist:.2f}%")
                            st.progress(min(var_hist/30, 1.0), "danger")
                        with col3:
                            var_cf = abs(performance_results.get('VaR Cornish-Fisher', 0)) * 100
                            st.metric("⚠️ VaR Cornish-Fisher", f"{var_cf:.2f}%")
                            st.progress(min(var_cf/30, 1.0), "danger")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Informations sur l'interprétation
                        with st.expander("📚 Comment interpréter la VaR ?"):
                            st.info("""
                            La Value-at-Risk (VaR) représente la perte maximale potentielle sur un horizon de temps spécifique 
                            avec un niveau de confiance donné (généralement 95% ou 99%).
                            
                            - **VaR Paramétrique**: Calcul basé sur la distribution normale
                            - **VaR Historique**: Calcul basé sur les rendements historiques réels
                            - **VaR Cornish-Fisher**: Calcul tenant compte des moments d'ordre supérieur (asymétrie, kurtosis)
                            """)
                    
                    with var_tabs[1]:
                        st.markdown('<div style="padding:10px; border-radius:5px; background-color:#ffeded;">', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            cvar_param = abs(performance_results.get('CVaR Paramétrique', 0)) * 100
                            st.metric("🔥 CVaR Paramétrique", f"{cvar_param:.2f}%")
                            st.progress(min(cvar_param/40, 1.0), "danger")
                        with col2:
                            cvar_hist = abs(performance_results.get('CVaR Historique', 0)) * 100
                            st.metric("🔥 CVaR Historique", f"{cvar_hist:.2f}%")
                            st.progress(min(cvar_hist/40, 1.0), "danger")
                        with col3:
                            cvar_cf = abs(performance_results.get('CVaR Cornish-Fisher', 0)) * 100
                            st.metric("🔥 CVaR Cornish-Fisher", f"{cvar_cf:.2f}%")
                            st.progress(min(cvar_cf/40, 1.0), "danger")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Informations sur l'interprétation
                        with st.expander("📚 Comment interpréter la CVaR ?"):
                            st.info("""
                            La Conditional Value-at-Risk (CVaR), aussi appelée Expected Shortfall, représente la perte moyenne 
                            attendue dans les scénarios qui dépassent la VaR. C'est une mesure de risque plus prudente que la VaR.
                            
                            - **CVaR Paramétrique**: Calcul basé sur la distribution normale
                            - **CVaR Historique**: Calcul basé sur les rendements historiques réels
                            - **CVaR Cornish-Fisher**: Calcul tenant compte des moments d'ordre supérieur
                            """)
                    
                    # Section comparative en bas de page
                    st.markdown("""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin:20px 0; background-color:#f8f9fa">
                        <h3 style="text-align:center; color:#0366d6;">📊 Comparaison des Indicateurs de Risque</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Création d'un graphique comparatif
                    st.bar_chart({
                        'VaR Paramétrique': var_param,
                        'VaR Historique': var_hist,
                        'VaR CF': var_cf,
                        'CVaR Paramétrique': cvar_param,
                        'CVaR Historique': cvar_hist,
                        'CVaR CF': cvar_cf
                    })
            # Graphique des Prix
            with tabs[2]:
                st.subheader("📈 Évolution des Prix Normalisés des Tickers")
                try:
                    fig_prices = indicateurs._graphique_evolution_prix(
                        data_dict=data_dict,
                        date_investissement=date_investissement,
                        date_fin=date_fin_investissement,
                    )
                    st.plotly_chart(fig_prices)
                except Exception as e:
                    st.error(f"Erreur lors de la création du graphique : {e}")

            # Valeur du Portefeuille 
            with tabs[3]:
                if strategie_choisie == "ACP":
                    # 📉 Evolution du capital
                    if not performance_results["capital_evolution"].empty:
                        st.subheader("💰 Évolution du Capital")
                        fig = px.line(performance_results["capital_evolution"], x="Date", y="Total Capital", title="📈 Évolution du Capital")
                        st.plotly_chart(fig)
                if strategie_choisie == "MachineLearning":
                    st.subheader("💰 Évolution de la Valeur du Portefeuille")
                    # Récupération de l'évolution du capital
                    capital_evolution_df = performance_results["capital_evolution"]
                    fig = indicateurs.plot_capital_evolution_plotly(performance_results.get("capital_evolution",[]))
                    st.plotly_chart(fig)
                if strategie_choisie == "DeepLearning":
                    st.subheader("💰 Évolution de la Valeur du Portefeuille")
                    # Récupération de l'évolution du capital
                    capital_evolution_df = performance_results["capital_evolution"]
                    fig = indicateurs.plot_capital_evolution_plotly(performance_results.get("capital_evolution",[]))
                    st.plotly_chart(fig)
                if strategie_choisie=="BuyAndHold":
                    st.subheader("💰 Évolution de la Valeur du Portefeuille")
                    try:
                        fig = indicateurs._graphique_valeur_portefeuille(data_dict, montant_initial,tickers_selectionnes,date_investissement,date_fin_investissement)
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage de l'image : {e}")
                elif strategie_choisie == "Momentum"or strategie_choisie=="MinimumVariance":
                    st.subheader("💰 Évolution de la Valeur du Portefeuille")
                    try:
                        dates = performance_results.get("dates", [])
                        valeurs_portefeuille = performance_results.get("valeurs_portefeuille", [])
                        if dates and valeurs_portefeuille:
                            fig = Indicateurs.evolution_valeur_portefeuille(dates, valeurs_portefeuille)
                            st.plotly_chart(fig)
                        else:
                            st.error("Données insuffisantes pour afficher l'évolution de la valeur du portefeuille.")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

            # Proportions des Entreprises
            with tabs[4]:
                if strategie_choisie == "MachineLearning":
                    summary_data = performance_results

                    if "error" in summary_data:
                        st.error("❌ " + summary_data["error"])
                    else:
                        detailed_results = summary_data  # Résultats détaillés issus de la fonction aggregate_portfolio_results

                        if not isinstance(detailed_results, dict) or not detailed_results:
                            st.error("⚠️ Les résultats détaillés de la stratégie sont vides ou mal formatés.")
                        else:
                            # 📈 Affichage des statistiques de Trading
                            st.subheader("📈 Statistiques de Trading")

                            # 🔹 Affichage du détail des transactions par actif
                            if "trade_summary" in summary_data and isinstance(summary_data["trade_summary"], pd.DataFrame):
                                st.subheader("📌 Détails des Transactions par Actif")
                                st.dataframe(summary_data["trade_summary"])  # ✅ Affiche directement le DataFrame
                            else:
                                st.warning("⚠️ Aucune donnée de transaction disponible.")
                                        # 🔹 PCA - Scree Plot (Variance expliquée)
                    if use_pca==True:
                        if strategie.use_pca and performance_results.get("pca") is not None:
                            st.subheader("📊 Scree Plot - Variance expliquée par composante principale")
                            pca_fig = strategie.scree_plot(performance_results["pca"],n_components)
                            st.plotly_chart(pca_fig)

                            st.subheader("🔥 Heatmap des Coefficients de Contribution des Actifs")
                            loadings_fig = strategie.plot_loadings(performance_results["pca"], strategie.feature_cols,n_components)
                            st.plotly_chart(loadings_fig)

                if strategie_choisie == "DeepLearning":
                    st.subheader("📊 Résumé des trades")

                    try:
                        trade_summary = performance_results.get('trade_summary', pd.DataFrame())

                        fig_gain = strategie.graphique_gain_total_par_ticker(trade_summary)
                        fig_trades = strategie.graphique_trades_achat_vente(trade_summary)

                        if fig_gain:
                            st.plotly_chart(fig_gain, use_container_width=True)
                        else:
                            st.warning("Impossible d'afficher le graphique de gains.")

                        if fig_trades:
                            st.plotly_chart(fig_trades, use_container_width=True)
                        else:
                            st.warning("Impossible d'afficher le graphique des trades.")
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des graphiques interactifs : {e}")





                if strategie_choisie=="BuyAndHold":
                    st.subheader("📊 Proportions des Entreprises dans le Portefeuille")
                    try:
                        fig_prices = indicateurs._graphique_proportions(
                            data_dict=data_dict,
                            montant_initial=montant_initial,
                            tickers_selectionnes=tickers_selectionnes,
                            date_investissement=date_investissement,
                            date_fin=date_fin_investissement,
                        )
                        st.plotly_chart(fig_prices)
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des proportions : {e}")
                elif strategie_choisie == "Momentum"or strategie_choisie=="MinimumVariance":
                    st.subheader("📊 Taux d'Apparition des Entreprises dans le Portefeuille")
                    try:
                        # Récupérer la répartition des actifs et calculer le taux d'apparition
                        repartition = performance_results.get("repartition")
                        if repartition is not None and not repartition.empty:
                            # Exclure la colonne "Date" si elle est présente
                            if "Date" in repartition.columns:
                                repartition_numeric = repartition.drop(columns=["Date"])
                            else:
                                repartition_numeric = repartition

                            # Remplacer les NaN par 0
                            repartition_numeric.fillna(0, inplace=True)

                            # Calculer le taux d'apparition
                            total_periods = repartition_numeric.shape[0]
                            taux_apparition = (repartition_numeric > 0).sum(axis=0) / total_periods * 100

                            # Tracer le graphique des taux d'apparition
                            fig = px.bar(
                                x=taux_apparition.index,
                                y=taux_apparition.values,
                                title="Taux d'Apparition des Actifs dans le Portefeuille",
                                labels={"x": "Actifs", "y": "Taux d'Apparition (%)"},
                                text_auto=".2f"
                            )
                            fig.update_layout(
                                xaxis_title="Actifs",
                                yaxis_title="Taux d'Apparition (%)",
                                showlegend=False
                            )
                            st.plotly_chart(fig)
                        else:
                            st.error("Données insuffisantes pour afficher le taux d'apparition.")
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse des proportions : {e}")

                    st.subheader("📈 Répartition Dynamique du Portefeuille au Fil du Temps")
                    try:
                        if repartition is not None and not repartition.empty:
                            # Vérifier et ajouter la colonne "Date" si elle n'existe pas
                            if "Date" not in repartition.columns:
                                repartition.reset_index(inplace=True)
                                repartition.rename(columns={"index": "Date"}, inplace=True)

                            # Générer le graphique interactif de répartition
                            fig_repartition = px.area(
                                repartition.melt(id_vars="Date", var_name="Actif", value_name="Proportion"),
                                x="Date",
                                y="Proportion",
                                color="Actif",
                                title="Répartition Dynamique du Portefeuille",
                                labels={"Proportion": "Proportion (%)", "Date": "Date", "Actif": "Actifs"}
                            )
                            fig_repartition.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Proportion (%)",
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig_repartition)
                        else:
                            st.error("Données insuffisantes pour afficher la répartition dynamique.")
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage de la répartition dynamique : {e}")


            # Matrice de Corrélation
            with tabs[5]:
                st.subheader("🔗 Matrice de Corrélation des Entreprises")
                try:
                    # Calcul de la matrice de corrélation
                    correlation_matrix = indicateurs.matrice_correlation(data_dict,tickers_selectionnes,date_investissement,date_fin_investissement)
                    # Création de la visualisation avec Plotly
                    fig_corr = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Matrice de Corrélation des Rendements Journaliers",
                    )
                    st.plotly_chart(fig_corr)
                except Exception as e:
                    st.error(f"Erreur lors de la création de la matrice de corrélation : {e}")

                        
            if (strategie_choisie == "Momentum" or strategie_choisie=="MinimumVariance" ) and len(tabs) > 5:
                # Remplacer le code existant dans la section "Tableau Récapitulatif" par celui-ci
                with tabs[6]:
                    st.subheader("📝 Tableau Récapitulatif des Investissements")
                    try:
                        # Initialisation du tableau récapitulatif
                        investment_summary = []

                        # Récupération des données
                        dates = performance_results.get("dates", [])
                        valeurs_portefeuille = performance_results.get("valeurs_portefeuille", [])
                        actifs_selectionnes = performance_results.get("actifs_selectionnes", [])
                        valeurs_momentum = performance_results.get("valeurs_momentum", [])
                        performances_buy_and_hold = performance_results.get("performances_buy_and_hold", [])

                        # Vérification de la présence des données
                        if dates and valeurs_portefeuille and actifs_selectionnes:
                            for i in range(len(dates) - 1):
                                # Dates de début et de fin
                                start_date = dates[i].date()
                                end_date = dates[i + 1].date()
                                portefeuille_value = valeurs_portefeuille[i + 1]
                                valeur_momentum = valeurs_momentum[i + 1] if i + 1 < len(valeurs_momentum) else None
                                performance_buy_and_hold = performances_buy_and_hold[i] if i < len(performances_buy_and_hold) else None

                                # Extraction des actifs sélectionnés pour cette période
                                actifs = actifs_selectionnes[i]["Actifs sélectionnés"] if i < len(actifs_selectionnes) else []

                                # Ajout au tableau récapitulatif
                                investment_summary.append({
                                    "Date de début": start_date,
                                    "Date de fin": end_date,
                                    "Valeur du portefeuille (€)": round(portefeuille_value, 2),
                                    "Performance (%)": round(performance_buy_and_hold, 2) if performance_buy_and_hold is not None else None,
                                    "Actifs sélectionnés": ", ".join(actifs)
                                })

                            # Création d'un DataFrame
                            df_summary = pd.DataFrame(investment_summary)
                            
                            # Formatage des colonnes
                            df_summary["Date de début"] = pd.to_datetime(df_summary["Date de début"]).dt.strftime('%d/%m/%Y')
                            df_summary["Date de fin"] = pd.to_datetime(df_summary["Date de fin"]).dt.strftime('%d/%m/%Y')
                            
                            # Formater la valeur du portefeuille avec le séparateur de milliers et symbole €
                            df_summary["Valeur du portefeuille (€)"] = df_summary["Valeur du portefeuille (€)"].apply(
                                lambda x: f"{x:,.2f} €".replace(",", " ").replace(".", ",")
                            )
                            
                            # Formater la performance avec le symbole %
                            df_summary["Performance (%)"] = df_summary["Performance (%)"].apply(
                                lambda x: f"{x:+.2f} %".replace(".", ",") if pd.notnull(x) else "N/A"
                            )
                            
                            # Créer une version stylisée du DataFrame
                            def highlight_performance(df):
                                # Créer une copie du DataFrame
                                styled = df.copy()
                                
                                # Extraire les valeurs numériques des performances (retirer % et convertir en float)
                                performance_values = []
                                for val in df["Performance (%)"]:
                                    try:
                                        # Extraire la valeur numérique (supprimer le +, %, et remplacer , par .)
                                        cleaned = val.replace("%", "").replace("+", "").replace(",", ".").strip()
                                        performance_values.append(float(cleaned) if cleaned != "N/A" else 0)
                                    except (ValueError, AttributeError):
                                        performance_values.append(0)
                                
                                # Appliquer la coloration conditionnelle
                                background_colors = []
                                text_colors = []
                                
                                for val in performance_values:
                                    if val > 0:
                                        background_colors.append('rgba(0, 255, 0, 0.1)')  # Vert très clair
                                        text_colors.append('green')
                                    elif val < 0:
                                        background_colors.append('rgba(255, 0, 0, 0.1)')  # Rouge très clair
                                        text_colors.append('red')
                                    else:
                                        background_colors.append('transparent')
                                        text_colors.append('black')
                                
                                # Créer le style pour chaque ligne
                                for i in range(len(styled)):
                                    styled.iloc[i, styled.columns.get_loc("Performance (%)")] = f'<span style="color:{text_colors[i]}; background-color:{background_colors[i]}; padding:2px 5px; border-radius:3px;">{df.iloc[i, df.columns.get_loc("Performance (%)")]} </span>'
                                
                                return styled
                            
                            # Appliquer le style
                            styled_df = highlight_performance(df_summary)
                            
                            # Afficher le tableau avec style dans Streamlit
                            st.markdown("### Récapitulatif des périodes d'investissement")
                            
                            # Utiliser un conteneur avec une hauteur maximale et scrolling
                            container = st.container()
                            with container:
                                # Afficher le tableau formaté
                                st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                                
                                # Ajouter du CSS pour le style du tableau
                                st.markdown("""
                                <style>
                                table {
                                    width: 100%;
                                    border-collapse: collapse;
                                    margin-bottom: 20px;
                                }
                                
                                th {
                                    background-color: #f2f2f2;
                                    padding: 10px;
                                    text-align: left;
                                    border-bottom: 2px solid #ddd;
                                    position: sticky;
                                    top: 0;
                                    z-index: 1;
                                }
                                
                                td {
                                    padding: 8px;
                                    border-bottom: 1px solid #ddd;
                                }
                                
                                tr:hover {
                                    background-color: #f5f5f5;
                                }
                                
                                /* Alternance de couleurs pour les lignes */
                                tr:nth-child(even) {
                                    background-color: #f9f9f9;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                            
                            # Afficher des statistiques supplémentaires
                            if len(df_summary) > 0:
                                st.markdown("### Statistiques des périodes")
                                
                                # Extraire les performances numériques pour les statistiques
                                performances = []
                                for val in df_summary["Performance (%)"]:
                                    try:
                                        # Extraire la valeur numérique
                                        cleaned = val.replace("%", "").replace("+", "").replace(",", ".").strip()
                                        performances.append(float(cleaned) if cleaned != "N/A" else None)
                                    except (ValueError, AttributeError):
                                        performances.append(None)
                                
                                performances = [p for p in performances if p is not None]
                                
                                if performances:
                                    avg_performance = sum(performances) / len(performances)
                                    best_performance = max(performances)
                                    worst_performance = min(performances)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Nombre de périodes", len(df_summary))
                                    
                                    with col2:
                                        st.metric("Performance moyenne", f"{avg_performance:.2f}%".replace(".", ","), 
                                                delta=f"{avg_performance:.2f}%".replace(".", ","))
                                    
                                    with col3:
                                        # Trouver l'indice de la meilleure performance
                                        best_idx = performances.index(best_performance)
                                        st.metric("Meilleure performance", 
                                                f"{best_performance:.2f}%".replace(".", ","))
                                
                                    # Ajouter un graphique de performance
                                    st.subheader("Graphique des performances par période")
                                    
                                    # Créer les données pour le graphique
                                    chart_data = pd.DataFrame({
                                        "Période": [f"P{i+1}" for i in range(len(performances))],
                                        "Performance (%)": performances
                                    })
                                    
                                    # Créer le graphique avec plotly express
                                    import plotly.express as px
                                    
                                    fig = px.bar(
                                        chart_data, 
                                        x="Période", 
                                        y="Performance (%)",
                                        color="Performance (%)",
                                        color_continuous_scale=["red", "yellow", "green"],
                                        labels={"Performance (%)": "Performance (%)", "Période": "Période d'investissement"},
                                        title="Performance par période d'investissement"
                                    )
                                    
                                    fig.update_layout(
                                        xaxis_title="Période d'investissement",
                                        yaxis_title="Performance (%)",
                                        coloraxis_showscale=False,
                                        height=400
                                    )
                                    
                                    # Ajouter les valeurs sur les barres
                                    fig.update_traces(
                                        texttemplate='%{y:.2f}%', 
                                        textposition='outside'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.warning("⚠️ Données insuffisantes pour afficher le tableau récapitulatif.")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la création du tableau récapitulatif : {e}")
                        import traceback
                        st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la stratégie : {e}")
