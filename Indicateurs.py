import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import plotly.express as px
import matplotlib.dates as mdates
import os
import pickle

class Indicateurs:

    ######################PERFORMANCE#########################################
    def performance(self, df, montant_initial, date_initiale, date_finale):
        # Vérifier que le DataFrame contient une colonne 'Adj Close'
        if 'Adj Close' not in df.columns:
            raise ValueError(f"La colonne 'Adj Close' est absente ou mal formatée.")

        # Vérifier si le DataFrame est vide après filtrage
        df['Date'] = pd.to_datetime(df['Date'])
        df_filtered = df[(df['Date'] >= pd.to_datetime(date_initiale)) & (df['Date'] <= pd.to_datetime(date_finale))]

        if df_filtered.empty:
            raise ValueError("Aucune donnée disponible pour la période spécifiée.")

        # Calculer les métriques
        prix_initial = df_filtered.iloc[0]['Adj Close']
        prix_final = df_filtered.iloc[-1]['Adj Close']

        if prix_initial is None or prix_final is None:
            raise ValueError("Prix initial ou final non disponible.")

        pourcentage_gain_total = (prix_final - prix_initial) / prix_initial * 100
        gain_total = montant_initial * (prix_final / prix_initial - 1)
        performance_annualisee = ((prix_final / prix_initial) ** (365 / len(df_filtered)) - 1) * 100

        return {
            "pourcentage_gain_total": pourcentage_gain_total,
            "gain_total": gain_total,
            "prix_initial": prix_initial,
            "prix_final": prix_final,
            "performance_annualisee": performance_annualisee
        }


    ######################Graphique#########################################
    def matrice_correlation(self,data_dict,tickers_selectionnes, date_debut, date_fin):
        """
        Calcule la matrice de corrélation des rendements journaliers pour une liste de tickers.

        :param tickers_selectionnes: list, Liste des tickers à analyser.
        :param date_debut: str, La date de début (format 'YYYY-MM-DD').
        :param date_fin: str, La date de fin (format 'YYYY-MM-DD').
        :return: DataFrame, Matrice de corrélation des rendements journaliers.
        """
        try:
            # Charger les données des tickers sélectionnés


            # Créer un DataFrame pour contenir les rendements journaliers de chaque entreprise
            rendement_df = pd.DataFrame()

            for ticker, df in data_dict.items():
                # Vérifier si la colonne 'Adj Close' existe
                if 'Adj Close' not in df.columns:
                    raise ValueError(f"La colonne 'Adj Close' est absente pour {ticker}.")

                # Calculer les rendements journaliers (logarithmiques pour la stabilité)
                df = df.sort_values(by='Date')
                df['Rendement'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

                # Ajouter les rendements au DataFrame principal
                rendement_df[ticker] = df['Rendement']

            # Supprimer les lignes contenant des NaN
            rendement_df.dropna(inplace=True)

            # Calculer la matrice de corrélation
            correlation_matrix = rendement_df.corr()

            return correlation_matrix

        except Exception as e:
            raise ValueError(f"Erreur lors du calcul de la matrice de corrélation : {e}")

    def plot_evolution(self):
        """
        Génère un graphique interactif de l'évolution du portefeuille dans le temps.
        """
        if not hasattr(self, 'history') or not self.history:
            print("❌ Aucune donnée disponible pour tracer le graphique.")
            return None

        # Conversion en DataFrame
        df = pd.DataFrame(self.history, columns=["Date", "Valeur"])
        df.sort_values(by="Date", inplace=True)

        # Création du graphique interactif
        fig = px.line(
            df,
            x="Date",
            y="Valeur",
            title="📈 Évolution en Temps Réel du Portefeuille ML",
            labels={"Date": "Date", "Valeur": "Valeur du Portefeuille (€)"},
            line_shape="linear"
        )

        fig.update_traces(line=dict(width=2, color="blue"))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Valeur du Portefeuille (€)",
            hovermode="x unified"
        )

        return fig

    def plot_capital_evolution_plotly(self,capital_evolution_df):
        """Génère un graphique interactif avec Plotly"""
        if capital_evolution_df.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=capital_evolution_df["Date"], y=capital_evolution_df["Capital"],
                                mode="lines+markers", name="Capital"))

        fig.update_layout(title="Évolution du Capital",
                        xaxis_title="Date",
                        yaxis_title="Capital (€)",
                        template="plotly_dark")
        return fig

    def afficher_graphique_interactif(self, data_dict,tickers_selectionnes, montant_initial, date_investissement, date_fin):
        fig_prices = self._graphique_evolution_prix(data_dict, date_investissement, date_fin)
        # Graphique des proportions des tickers dans le portefeuille
        self._graphique_proportions(data_dict, montant_initial, tickers_selectionnes, date_investissement, date_fin)
        return fig_prices



    def _graphique_evolution_prix(self, data_dict, date_investissement, date_fin):
        # Fixer la date minimale à 2010
        date_debut = pd.to_datetime("2010-01-01")
        date_investissement = pd.to_datetime(date_investissement)
        date_fin = pd.to_datetime(date_fin)

        # S'assurer que la date d'investissement ne précède pas 2010
        if date_investissement < date_debut:
            date_investissement = date_debut

        fig_prices = go.Figure()
        for ticker, data in data_dict.items():
            if not data.empty:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convertir les dates
                data = data.sort_values(by='Date')

                # Filtrer les données pour commencer à partir de 2010
                data = data[data['Date'] >= date_debut]

                # Tracé des lignes de prix de clôture
                fig_prices.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Adj Close'],
                    mode='lines',
                    name=f"{ticker} (Prix de Clôture)"
                ))

                # Filtrage des données pour les points d'investissement
                data_invest = data[
                    (data['Date'] >= date_investissement) &
                    (data['Date'] <= date_fin)
                ]
                if not data_invest.empty:
                    prix_investissement = data_invest['Adj Close'].iloc[0]
                    fig_prices.add_trace(go.Scatter(
                        x=[date_investissement],
                        y=[prix_investissement],
                        mode='markers',
                        marker=dict(color='green', size=10),
                        name=f"Début Investissement {ticker}"
                    ))

                    prix_fin_investissement = data_invest['Adj Close'].iloc[-1]
                    fig_prices.add_trace(go.Scatter(
                        x=[date_fin],
                        y=[prix_fin_investissement],
                        mode='markers',
                        marker=dict(color='red', size=10),
                        name=f"Fin Investissement {ticker}"
                    ))

        fig_prices.update_layout(
            title=f"📈 Évolution des Prix des Tickers ({date_investissement.date()} - {date_fin.date()})",
            xaxis_title="Date",
            yaxis_title="Prix de Clôture",
            hovermode="x unified"
        )
        return fig_prices



    def _graphique_valeur_portefeuille(self, data_dict, montant_initial, tickers_selectionnes, date_investissement, date_fin):
        date_investissement = pd.to_datetime(date_investissement)
        date_fin = pd.to_datetime(date_fin)

        portfolio_values = []
        fig = go.Figure()

        # Ajout des courbes des actions
        for ticker, data in data_dict.items():
            if not data.empty:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data_invest = data[
                    (data['Date'] >= date_investissement) &
                    (data['Date'] <= date_fin)
                ]
                if not data_invest.empty:
                    prix_investissement = data_invest['Adj Close'].iloc[0]
                    montant_investissement_ticker = montant_initial / len(tickers_selectionnes) * (
                        data_invest['Adj Close'] / prix_investissement
                    )
                    montant_investissement_ticker.index = data_invest['Date']
                    portfolio_values.append(montant_investissement_ticker)

                    # Ajouter la courbe du cours de l'action
                    fig.add_trace(go.Scatter(
                        x=data_invest['Date'],
                        y=data_invest['Adj Close'],
                        mode='lines',
                        name=f"Cours {ticker} (échelle secondaire)",
                        yaxis='y2'  # Associer à l'axe secondaire
                    ))

        # Calculer la valeur totale normalisée du portefeuille
        if portfolio_values:
            df_portfolio = pd.concat(portfolio_values, axis=1).ffill()
            df_portfolio['Montant_Total'] = df_portfolio.sum(axis=1)

            # Ajouter la courbe de la valeur totale du portefeuille
            fig.add_trace(go.Scatter(
                x=df_portfolio.index,
                y=df_portfolio['Montant_Total'],
                mode='lines',
                name="Valeur totale du portefeuille",
                line=dict(width=2, dash='solid')
            ))

            # Mettre à jour la mise en page pour inclure un axe secondaire
            fig.update_layout(
                title="📈 Évolution de la Valeur Normalisée du Portefeuille et des Cours des Actions",
                xaxis=dict(title="Date"),
                yaxis=dict(
                    title="Valeur totale du portefeuille (€)",
                    side='left'
                ),
                yaxis2=dict(
                    title="Cours des actions (€)",
                    overlaying='y',
                    side='right'
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                hovermode="x unified"
            )

            return fig
        else:
            print("Aucune donnée pour la valeur du portefeuille.")
            return None



    def _graphique_proportions(self, data_dict, montant_initial, tickers_selectionnes, date_investissement, date_fin):
        portfolio_values = []
        for ticker, data in data_dict.items():
            if not data.empty:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Assurer la conversion des dates
                data_invest = data[(data['Date'] >= pd.Timestamp(date_investissement)) & (data['Date'] <= pd.Timestamp(date_fin))]
                if not data_invest.empty:
                    prix_investissement = data_invest['Adj Close'].iloc[0]
                    montant_investissement_ticker = montant_initial / len(tickers_selectionnes) * (
                            data_invest['Adj Close'] / prix_investissement)
                    montant_investissement_ticker.index = data_invest['Date']
                    montant_investissement_ticker = montant_investissement_ticker.rename(f"Montant_{ticker}")
                    portfolio_values.append(montant_investissement_ticker)

        if portfolio_values:
            df_portfolio = pd.concat(portfolio_values, axis=1).ffill()
            df_portfolio['Montant_Total'] = df_portfolio.sum(axis=1)

            # Calculer les proportions
            proportions = {}
            for ticker in tickers_selectionnes:
                if f"Montant_{ticker}" in df_portfolio.columns:
                    df_portfolio[f"{ticker}_Proportion"] = (
                        df_portfolio[f"Montant_{ticker}"] / df_portfolio['Montant_Total']
                    ) * 100
                    proportions[ticker] = df_portfolio[f"{ticker}_Proportion"]

            # Préparer les données pour le graphique interactif
            df_proportions = pd.DataFrame(proportions)
            df_proportions['Date'] = df_portfolio.index
            df_proportions = df_proportions.melt(id_vars="Date", var_name="Entreprise", value_name="Proportion")

            # Créer le graphique interactif
            fig = px.area(
                df_proportions,
                x="Date",
                y="Proportion",
                color="Entreprise",
                title="Proportions des Entreprises dans le Portefeuille au Fil du Temps",
                labels={"Proportion": "Proportion (%)", "Date": "Date", "Entreprise": "Entreprises"},
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Proportion (%)",
                hovermode="x unified",
                legend_title="Entreprises",
            )

            return fig
        else:
            print("Aucune donnée pour les proportions.")
            return None



    def evolution_valeur_portefeuille(dates, valeurs_portefeuille):
        """
        Graphique de l'évolution de la valeur du portefeuille au fil du temps.

        :param dates: Liste des dates correspondant aux périodes.
        :param valeurs_portefeuille: Liste des valeurs du portefeuille à ces dates.
        :return: Figure Plotly à afficher.
        """
        # Vérifier et aligner les longueurs des données
        min_length = min(len(dates), len(valeurs_portefeuille))
        dates = dates[:min_length]
        valeurs_portefeuille = valeurs_portefeuille[:min_length]

        # Créer le graphique
        fig = px.line(
            x=dates,
            y=valeurs_portefeuille,
            title="Évolution de la Valeur du Portefeuille",
            labels={"x": "Date", "y": "Valeur du Portefeuille (€)"},
        )
        fig.update_traces(line=dict(width=2))
        return fig

    def graphique_taux_apparition(taux_apparition):
        """
        Génère un graphique montrant le taux d'apparition des actifs dans le portefeuille.

        :param taux_apparition: Dictionnaire avec les tickers comme clés et les taux d'apparition comme valeurs.
        :return: Figure Plotly à afficher.
        """
        # Préparer les données pour le graphique
        data = pd.DataFrame({
            "Ticker": list(taux_apparition.keys()),
            "Taux d'Apparition (%)": [val * 100 for val in taux_apparition.values()]
        })

        # Créer le graphique
        fig = px.bar(
            data,
            x="Ticker",
            y="Taux d'Apparition (%)",
            title="Taux d'Apparition des Actifs dans le Portefeuille",
            text_auto=".2f",
            labels={"Ticker": "Actif", "Taux d'Apparition (%)": "Taux (%)"}
        )
        fig.update_layout(
            xaxis_title="Actifs",
            yaxis_title="Taux d'Apparition (%)",
            showlegend=False
        )
        return fig

    def evolution_rendements_actifs(dates, rendements_cumules):
        """
        Graphique de l'évolution des rendements cumulés des actifs sélectionnés.

        :param dates: Liste des dates correspondant aux périodes.
        :param rendements_cumules: DataFrame où chaque colonne représente un actif, avec les rendements cumulés.
        :return: Figure Plotly à afficher.
        """
        rendements_cumules['Date'] = dates
        df = rendements_cumules.melt(id_vars="Date", var_name="Actif", value_name="Rendement Cumulé (%)")

        fig = px.line(
            df,
            x="Date",
            y="Rendement Cumulé (%)",
            color="Actif",
            title="Évolution des Rendements Cumulés des Actifs",
        )
        fig.update_traces(line=dict(width=2))
        return fig
    
    def afficher_repartition_dynamique(self, repartition):
        """
        Génère un graphique interactif de la répartition du portefeuille au fil du temps.

        :param repartition: DataFrame avec les colonnes représentant les actifs et les lignes représentant les périodes.
        """
        if repartition.empty:
            print("Aucune donnée disponible pour la répartition.")
            return

        # Remplacer les valeurs NaN par 0 pour les calculs
        repartition = repartition.fillna(0)

        # Préparer les données pour le graphique
        repartition.reset_index(inplace=True)
        repartition = repartition.melt(id_vars="index", var_name="Actif", value_name="Proportion")
        repartition.rename(columns={"index": "Date"}, inplace=True)

        # Créer le graphique d'aire empilée
        fig = px.area(
            repartition,
            x="Date",
            y="Proportion",
            color="Actif",
            title="Répartition Dynamique du Portefeuille au Fil du Temps",
            labels={"Proportion": "Proportion (%)", "Date": "Date", "Actif": "Actifs"}
        )
        fig.update_traces(mode="lines")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Proportion (%)",
            hovermode="x unified"
        )

        return fig


    ######################Indicateurs#########################################
    def volatilite_historique(self, df, lambda_ewma=0.94):
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values(by='Date')
        df['Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        volatilite_historique = df['Returns'].std() * np.sqrt(252)
        return {
            "volatilite_historique": volatilite_historique
        }

    def calculate_ewma_volatility(self, prices):
        # Vérification des données d'entrée
        if not isinstance(prices, pd.Series) or prices.empty or len(prices) < 2:
            raise ValueError("Les données des prix doivent être un Series non vide avec au moins 2 valeurs pour calculer la volatilité EWMA.")
        
        # Calcul des rendements logarithmiques
        log_rets = np.log(prices / prices.shift(1)).dropna()
        if log_rets.empty:
            raise ValueError("Les rendements calculés sont vides.")

        # Paramètres pour EWMA
        lmb = 0.94
        rolling_win = min(252, len(log_rets))  # Utiliser une fenêtre de 252 jours ou moins si les données sont insuffisantes

        # Coefficients de pondération pour la fenêtre
        weights = [(1 - lmb) * (lmb ** i) for i in range(rolling_win - 1, -1, -1)]
        weights = np.array(weights)

        # Vérification des dimensions
        if len(weights) != rolling_win:
            raise ValueError("Les dimensions des pondérations ne correspondent pas à la fenêtre de calcul.")

        # Calcul de la volatilité EWMA
        recent_rets = log_rets[-rolling_win:]  # Derniers rendements pour la fenêtre
        ewma_volatility = np.sqrt(np.sum(weights * (recent_rets ** 2)) * 252)  # Annualisation
        return ewma_volatility

    
    def calculate_var(self, df, alpha, method):
        # Calcul des rendements
        returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna()
        
        if method == "parametric":
            # Méthode Paramétrique Normale
            mean = returns.mean()
            std_dev = returns.std()
            var = stats.norm.ppf(alpha) * std_dev - mean
        
        elif method == "historical":
            # Méthode Historique
            var = returns.quantile(alpha)
        
        elif method == "cornish-fisher":
            # Méthode Cornish-Fisher
            mean = returns.mean()
            std_dev = returns.std()
            skew = stats.skew(returns)
            kurtosis = stats.kurtosis(returns, fisher=True)
            
            z = stats.norm.ppf(alpha)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                    (z**3 - 3 * z) * (kurtosis - 3) / 24 - 
                    (2 * z**3 - 5 * z) * (skew**2) / 36)
            var = z_cf * std_dev - mean
        
        else:
            raise ValueError("Méthode non reconnue pour le calcul de la VaR.")
        
        return var

    def calculate_cvar(self, df, alpha, method):
        # Calcul de la VaR pour utiliser son seuil de perte
        var = self.calculate_var(df, alpha, method)
        returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna()
        if method == "parametric":
            # CVaR Paramétrique sous distribution normale
            mean = returns.mean()
            std_dev = returns.std()
            conditional_loss = stats.norm.pdf(stats.norm.ppf(alpha)) * std_dev / alpha
            cvar = mean - conditional_loss
        
        elif method == "historical":
            # CVaR Historique : moyenne des pertes au-delà de la VaR
            cvar = returns[returns <= var].mean()
        
        elif method == "cornish-fisher":
            # CVaR Cornish-Fisher
            mean = returns.mean()
            std_dev = returns.std()
            skew = stats.skew(returns)
            kurtosis = stats.kurtosis(returns, fisher=True)
            
            z = stats.norm.ppf(alpha)
            z_cf = (z + (z**2 - 1) * skew / 6 + 
                    (z**3 - 3 * z) * (kurtosis - 3) / 24 - 
                    (2 * z**3 - 5 * z) * (skew**2) / 36)
            conditional_loss_cf = stats.norm.pdf(z_cf) * std_dev / alpha
            cvar = mean - conditional_loss_cf
        else:
            raise ValueError("Méthode non reconnue pour le calcul de la CVaR.")
        return cvar
