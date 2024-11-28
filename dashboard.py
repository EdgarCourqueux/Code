import streamlit as st
import yfinance as yf
from datetime import datetime
from BuyAndHold import BuyAndHold
import pandas as pd
import plotly.express as px

# Charger les tickers et entreprises à partir d'un fichier texte
def load_tickers(file_path):
    tickers_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ',' in line:
                ticker, company = line.strip().split(',', 1)
                tickers_dict[ticker] = company
    return tickers_dict

# Charger le fichier des tickers et entreprises
TICKERS_DICT = load_tickers("tickers.txt")

# Interface Streamlit
st.title("📈 Dashboard des Stratégies d'Investissement")

# Paramètres de la stratégie
st.sidebar.header("Paramètres de la stratégie")
strategie_choisie = st.sidebar.selectbox("Sélectionner une Stratégie", ["BuyAndHold", "AutreStrategie"])

# Paramètres d'entrée pour BuyAndHold
montant_initial = st.sidebar.number_input("Montant initial (€)", min_value=100, max_value=100000, value=1000)

# Sélection de plusieurs entreprises par leurs noms
entreprises_selectionnees = st.sidebar.multiselect(
    "Sélectionner les Entreprises",
    list(TICKERS_DICT.values())
)

# Convertir les noms d'entreprises sélectionnés en tickers correspondants
tickers_selectionnes = [ticker for ticker, name in TICKERS_DICT.items() if name in entreprises_selectionnees]

# Télécharger les données historiques pour déterminer la date minimale
if tickers_selectionnes:
    # Télécharge les données pour le premier ticker sélectionné afin d'obtenir la date minimale
    data = yf.download(tickers_selectionnes[0], start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    data.index = pd.to_datetime(data.index)  # S'assurer que les dates sont au format datetime64
    date_min = data.index.min().date()
    date_investissement = pd.to_datetime(st.sidebar.date_input("Date d'investissement", value=datetime(2023, 1, 1), min_value=date_min))

# Instancier la stratégie sélectionnée
if strategie_choisie == "BuyAndHold" and tickers_selectionnes:
    strategie = BuyAndHold(montant_initial, date_investissement, tickers_selectionnes)
else:
    strategie = None

# Fonction pour calculer la matrice de corrélation
def calculer_matrice_correlation(tickers, start_date="2010-01-01"):
    data_dict = {
        ticker: yf.download(ticker, start=start_date, end=datetime.now().strftime('%Y-%m-%d'))['Adj Close']
        for ticker in tickers
    }
    df_prices = pd.DataFrame(data_dict)
    df_returns = df_prices.pct_change().dropna()
    return df_returns.corr()

# Exécuter la stratégie
if st.sidebar.button("Lancer l'analyse") and strategie:
    try:
        # Exécution de la stratégie
        performance_results = strategie.execute()

        # Création des onglets pour afficher les résultats
        tabs = st.tabs(["Résumé des Indicateurs", "Graphique des Prix", "Valeur du Portefeuille", "Proportions", "Matrice de Corrélation"])
        
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
            fig_prices = strategie.afficher_graphique_interactif(tickers_selectionnes=tickers_selectionnes, montant_initial=montant_initial, date_investissement=date_investissement)
            st.plotly_chart(fig_prices)
        
        # Valeur du Portefeuille
        with tabs[2]:
            st.subheader("💰 Évolution de la Valeur du Portefeuille")
            from PIL import Image
            image = Image.open("graph_multi_tickers.png")
            st.image(image, use_column_width=True)
        
        # Proportions des Entreprises
        with tabs[3]:
            st.subheader("📊 Proportions des Entreprises dans le Portefeuille")
            image = Image.open("graph_proportions_tickers.png")
            st.image(image, use_column_width=True)
        
        # Matrice de Corrélation
        with tabs[4]:
            st.subheader("🔗 Matrice de Corrélation des Entreprises")
            correlation_matrix = calculer_matrice_correlation(tickers_selectionnes)
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Matrice de Corrélation des Rendements Journaliers",
            )
            st.plotly_chart(fig_corr)

    except Exception as e:
        st.error(f"Erreur lors de l'exécution de la stratégie : {e}")
