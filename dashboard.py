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
from strategies.PCA import ACPInvestmentStrategy

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
strategie_choisie = st.sidebar.selectbox("Sélectionner une Stratégie", ["BuyAndHold", "Momentum", "MinimumVariance", "MachineLearning", "ACP"])

# Paramètres d'entrée
montant_initial = st.sidebar.number_input("Montant initial (€)", min_value=100, max_value=100000, value=1000)

# Télécharger les données historiques pour déterminer la date minimale
data = price_lib.prices(list(TICKERS_DICT.keys()))
date_min = data.index.min().date()

# Sélection des dates d'investissement
date_investissement = pd.to_datetime(
    st.sidebar.date_input("Date d'investissement", value=datetime(2023, 1, 1), min_value=date_min)
)
date_fin_investissement = pd.to_datetime(
    st.sidebar.date_input("Date de fin d'investissement", value=datetime.now().date(), min_value=date_investissement)
)

# Paramètres spécifiques aux stratégies Machine Learning et PCA
if strategie_choisie == "MachineLearning":
    lookback_period = st.sidebar.number_input("Période d'historique (jours)", min_value=1, max_value=365, value=10)
else:
    lookback_period = None

if strategie_choisie == "ACP":
    n_components = st.sidebar.number_input("Nombre de composantes principales", min_value=1, max_value=10, value=2)
else:
    n_components = None

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
        )
    elif strategie_choisie == "ACP" and tickers_selectionnes:
        logging.info(
            f"Lancement de la stratégie 'PCA' : {tickers_selectionnes} | {montant_initial} | {n_components} | {date_investissement} | {date_fin_investissement}"
        )
        strategie = ACPInvestmentStrategy(
            tickers=tickers_selectionnes,
            start_date=date_investissement,
            end_date=date_fin_investissement,
            initial_capital=montant_initial,
            n_components=n_components,
        )

    try:
        # Exécution de la stratégie
        performance_results = strategie.execute()
        # Vérification que le résultat est un dictionnaire structuré
        if not isinstance(performance_results, dict):
            st.error("Les résultats de la stratégie ne sont pas structurés correctement.")
        else:
            # Création des onglets pour afficher les résultats
            if strategie_choisie == "Momentum" or strategie_choisie=="MinimumVariance":
                tabs = st.tabs([
                    "Résumé des Indicateurs",
                    "Graphique des Prix",
                    "Valeur du Portefeuille",
                    "Proportions",
                    "Matrice de Corrélation",
                    "Tableau Récapitulatif",
                ])
            else:
                tabs = st.tabs([
                    "Résumé des Indicateurs",
                    "Graphique des Prix",
                    "Valeur du Portefeuille",
                    "Proportions",
                    "Matrice de Corrélation",
                ])

        with tabs[0]:
            st.subheader("📊 Résumé des Indicateurs")

            # 🎯 Section Gains et Performance
            st.markdown("## 💰 Gains et Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔹 Gain Total", f"{performance_results.get('gain_total', 0):,.2f} €")
            with col2:
                st.metric("📈 Pourcentage Gain Total", f"{performance_results.get('pourcentage_gain_total', 0):.2f} %")
            with col3:
                st.metric("📊 Performance Annualisée", f"{performance_results.get('performance_annualisee', 0):.2f} %")

            # 🎭 Section Volatilité
            st.markdown("## 📉 Volatilité et Risque")
            with st.expander("🔍 **Détails de la Volatilité**", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Volatilité Historique", f"{performance_results.get('volatilite_historique', 0) * 100:.2f} %")
                with col2:
                    st.metric("📊 Volatilité EWMA", f"{performance_results.get('ewma_volatility', 0) * 100:.2f} %")

            # ⚠️ Section VaR (Value at Risk)
            st.markdown("## 🚨 Value at Risk (VaR)")
            with st.expander("📉 **Détails des Risques VaR**", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⚠️ VaR Paramétrique", f"{abs(performance_results.get('VaR Paramétrique', 0)) * 100:.2f} %")
                with col2:
                    st.metric("⚠️ VaR Historique", f"{abs(performance_results.get('VaR Historique', 0)) * 100:.2f} %")
                with col3:
                    st.metric("⚠️ VaR Cornish-Fisher", f"{abs(performance_results.get('VaR Cornish-Fisher', 0)) * 100:.2f} %")

            # 🚨 Section CVaR (Conditional VaR)
            st.markdown("## 🔥 Conditional Value at Risk (CVaR)")
            with st.expander("⚠️ **Détails des Risques CVaR**", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔥 CVaR Paramétrique", f"{abs(performance_results.get('CVaR Paramétrique', 0)) * 100:.2f} %")
                with col2:
                    st.metric("🔥 CVaR Historique", f"{abs(performance_results.get('CVaR Historique', 0)) * 100:.2f} %")
                with col3:
                    st.metric("🔥 CVaR Cornish-Fisher", f"{abs(performance_results.get('CVaR Cornish-Fisher', 0)) * 100:.2f} %")

            # Graphique des Prix
            with tabs[1]:
                st.subheader("📈 Évolution des Prix Normalisés des Tickers")
                try:
                    fig_prices = indicateurs.afficher_graphique_interactif(
                        data_dict=data_dict,
                        tickers_selectionnes=tickers_selectionnes,
                        montant_initial=montant_initial,
                        date_investissement=date_investissement,
                        date_fin=date_fin_investissement,
                    )
                    st.plotly_chart(fig_prices)
                except Exception as e:
                    st.error(f"Erreur lors de la création du graphique : {e}")

            # Valeur du Portefeuille 
            with tabs[2]:
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
            with tabs[3]:
                if strategie_choisie == "ACP":
                    # 🔹 PCA - Scree Plot (Variance expliquée)
                    st.subheader("📊 Scree Plot - Variance expliquée par composante principale")
                    pca_fig = strategie.scree_plot(performance_results["pca"])  # Appel de la fonction avec l'objet PCA
                    st.plotly_chart(pca_fig)

                    # 🔥 Heatmap des Coefficients de Contribution (Loadings)
                    st.subheader("🔥 Heatmap des Coefficients de Contribution des Actifs")
                    loadings_fig = strategie.plot_loadings(performance_results["pca"], strategie.tickers)
                    st.plotly_chart(loadings_fig)
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
            with tabs[4]:
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
                # Tableau Récapitulatif
                with tabs[5]:
                    st.subheader("📝 Tableau Récapitulatif des Investissements")
                    try:
                        # Initialisation du tableau récapitulatif
                        investment_summary = []

                        # Récupération des données
                        dates = performance_results.get("dates", [])
                        valeurs_portefeuille = performance_results.get("valeurs_portefeuille", [])
                        actifs_selectionnes = performance_results.get("actifs_selectionnes", [])
                        valeurs_momentum = performance_results.get("valeurs_momentum", [])  # Assurez-vous que cette clé existe
                        performances_buy_and_hold = performance_results.get("performances_buy_and_hold", [])  # Performances en %

                        # Vérification de la présence des données
                        if dates and valeurs_portefeuille and actifs_selectionnes and valeurs_momentum and performances_buy_and_hold:
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
                                    "Valeur cumulative du portefeuille (€)": portefeuille_value,
                                    "Performance Buy and Hold (%)": f"{performance_buy_and_hold:.2f} %" if performance_buy_and_hold is not None else "N/A",
                                    "Actifs sélectionnés": ", ".join(actifs)
                                })

                            # Création d'un DataFrame
                            df_summary = pd.DataFrame(investment_summary)

                            # Affichage du tableau dans Streamlit
                            st.dataframe(df_summary)
                        
                        else:
                            st.error("Données insuffisantes pour afficher le tableau récapitulatif.")
                    except Exception as e:
                        st.error(f"Erreur lors de la création du tableau récapitulatif : {e}")

    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la stratégie : {e}")
