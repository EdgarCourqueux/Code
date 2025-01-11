import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from price_data import PriceData
from strategies.BuyAndHold import BuyAndHold
from strategies.Momentum import Momentum
from Indicateurs import Indicateurs
import pandas as pd
import plotly.express as px
import logging

# Exemple de log
price_lib = PriceData()

# Charger le fichier des tickers et entreprises
TICKERS_DICT = price_lib.universe()

# Interface Streamlit
st.title("📈 Dashboard des Stratégies d'Investissement")

# Paramètres de la stratégie
st.sidebar.header("Paramètres de la stratégie")
strategie_choisie = st.sidebar.selectbox("Sélectionner une Stratégie", ["BuyAndHold", "Momentum"])

# Paramètres d'entrée
montant_initial = st.sidebar.number_input("Montant initial (€)", min_value=100, max_value=100000, value=1000)

# Télécharger les données historiques pour déterminer la date minimale
data = price_lib.prices(list(TICKERS_DICT.keys()))
date_min = data.index.min().date()

# Paramètres spécifiques aux stratégies
if strategie_choisie == "BuyAndHold":
    date_investissement = pd.to_datetime(
        st.sidebar.date_input("Date d'investissement", value=datetime(2023, 1, 1), min_value=date_min)
    )
    date_fin_investissement = pd.to_datetime(
        st.sidebar.date_input("Date de fin d'investissement", value=datetime.now().date(), min_value=date_investissement)
    )
else:  # Momentum
    date_investissement = pd.to_datetime(
        st.sidebar.date_input("Date de début de la stratégie", value=datetime(2023, 1, 1), min_value=date_min)
    )
    date_fin_investissement = pd.to_datetime(
        st.sidebar.date_input("Date de fin de la stratégie", value=datetime.now().date(), min_value=date_investissement)
    )

# Validation explicite des dates
if date_investissement is None or date_fin_investissement is None:
    st.error("Veuillez sélectionner des dates valides pour la stratégie.")
    st.stop()

# Sélection de plusieurs entreprises par leurs noms
entreprises_selectionnees = st.sidebar.multiselect(
    "Sélectionner les Entreprises", list(TICKERS_DICT.values())
)

# Convertir les noms d'entreprises sélectionnés en tickers correspondants
tickers_selectionnes = [ticker for ticker, name in TICKERS_DICT.items() if name in entreprises_selectionnees]

# Paramètres supplémentaires pour la stratégie Momentum
if strategie_choisie == "Momentum":
    if tickers_selectionnes:
        max_assets = len(tickers_selectionnes)
        nombre_actifs = st.sidebar.number_input(
            "Nombre d'actifs à sélectionner",
            min_value=1,
            max_value=max_assets,
            value=min(3, max_assets),  # La valeur par défaut ne dépasse pas max_assets
        )
        periode_reroll = st.sidebar.number_input("Période de réévaluation (jours)", min_value=1, max_value=365, value=30)
        periode_historique = st.sidebar.number_input("Période historique (jours)", min_value=1, max_value=365, value=90)
    else:
        st.sidebar.warning("Veuillez sélectionner au moins une entreprise pour configurer les paramètres.")
else:
    nombre_actifs = None
    periode_reroll = None
    periode_historique = None

# Instancier la stratégie sélectionnée
strategie = None
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

# Exécuter la stratégie
if st.sidebar.button("Lancer l'analyse") and strategie:
    try:
        # Exécution de la stratégie
        performance_results = strategie.execute()

        # Vérification que le résultat est un dictionnaire structuré
        if not isinstance(performance_results, dict):
            st.error("Les résultats de la stratégie ne sont pas structurés correctement.")
        else:
            # Création des onglets pour afficher les résultats
            tabs = st.tabs(
                ["Résumé des Indicateurs", "Graphique des Prix", "Valeur du Portefeuille", "Proportions", "Matrice de Corrélation"]
            )

            with tabs[0]:
                st.subheader("📊 Résumé des Indicateurs")

                # Gains et Performance : Affichage direct
                st.markdown("### Gains et Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gain Total (€)", f"{performance_results.get('gain_total', 0):.2f}€")
                with col2:
                    st.metric("Pourcentage Gain Total (%)", f"{performance_results.get('pourcentage_gain_total', 0):.2f}%")
                with col3:
                    st.metric("Performance Annualisée (%)", f"{performance_results.get('performance_annualisee', 0):.2f}%")

                # Volatilité
                with st.expander("Volatilité"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Volatilité Historique (%)", f"{performance_results.get('volatilite_historique', 0)*100:.2f}%")
                    with col2:
                        st.metric("Volatilité EWMA (%)", f"{performance_results.get('ewma_volatility', 0)*100:.2f}%")

                # Value at Risk (VaR)
                with st.expander("Value at Risk (VaR)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VaR Paramétrique (%)", f"{abs(performance_results.get('VaR Paramétrique', 0))*100:.2f}%")
                    with col2:
                        st.metric("VaR Historique (%)", f"{abs(performance_results.get('VaR Historique', 0))*100:.2f}%")
                    with col3:
                        st.metric("VaR Cornish-Fisher (%)", f"{abs(performance_results.get('VaR Cornish-Fisher', 0))*100:.2f}%")

                # Conditional VaR (CVaR)
                with st.expander("Conditional VaR (CVaR)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CVaR Paramétrique (%)", f"{abs(performance_results.get('CVaR Paramétrique', 0))*100:.2f}%")
                    with col2:
                        st.metric("CVaR Historique (%)", f"{abs(performance_results.get('CVaR Historique', 0))*100:.2f}%")
                    with col3:
                        st.metric("CVaR Cornish-Fisher (%)", f"{abs(performance_results.get('CVaR Cornish-Fisher', 0))*100:.2f}%")

            # Graphique des Prix
            with tabs[1]:
                st.subheader("📈 Évolution des Prix Normalisés des Tickers")
                try:
                    fig_prices = Indicateurs.afficher_graphique_interactif(
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
                if strategie_choisie=="BuyAndHold":
                    st.subheader("💰 Évolution de la Valeur du Portefeuille")
                    try:
                        from PIL import Image

                        image = Image.open("graph_multi_tickers.png")
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage de l'image : {e}")
                elif strategie_choisie == "Momentum":
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
                if strategie_choisie=="BuyAndHold":
                    st.subheader("📊 Proportions des Entreprises dans le Portefeuille")
                    try:
                        image = Image.open("graph_proportions_tickers.png")
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des proportions : {e}")
                elif strategie_choisie == "Momentum":
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
                    if len(tickers_selectionnes) == 0:
                        st.error("Veuillez sélectionner au moins une entreprise pour calculer la matrice de corrélation.")
                    else:
                        correlation_matrix = Indicateurs.matrice_correlation(
                            tickers_selectionnes,
                            start_date=date_investissement.strftime('%Y-%m-%d'),
                            end_date=date_fin_investissement.strftime('%Y-%m-%d')
                        )
                        fig_corr = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title="Matrice de Corrélation des Rendements Journaliers",
                        )
                        st.plotly_chart(fig_corr)
                except Exception as e:
                    st.error(f"Erreur lors de la création de la matrice de corrélation : {e}")


    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la stratégie : {e}")