import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import plotly.express as px
import matplotlib.dates as mdates

class Indicateurs:

    ######################PERFORMANCE#########################################
    def performance(self, df, montant_initial, date_initiale, date_finale):
        # Convertir les colonnes de dates en format datetime sans spécifier de format explicite
        df['Date'] = pd.to_datetime(df['Date'])
        date_initiale = pd.to_datetime(date_initiale)
        date_finale = pd.to_datetime(date_finale)
        
        # Filtrer les données entre les dates spécifiées
        df_filtered = df[(df['Date'] >= date_initiale) & (df['Date'] <= date_finale)].copy()
        if df_filtered.empty:
            raise ValueError("Aucune donnée disponible pour la période spécifiée.")
        
        # Calcul des prix et des performances
        prix_initial = df_filtered.iloc[0]['Adj Close']
        prix_final = df_filtered.iloc[-1]['Adj Close']
        nb_jours = (df_filtered['Date'].iloc[-1] - df_filtered['Date'].iloc[0]).days
        pourcentage_gain_total = (prix_final - prix_initial) / prix_initial * 100
        gain_total = montant_initial * (prix_final / prix_initial - 1)
        performance_annualisee = ((prix_final / prix_initial) ** (365 / nb_jours) - 1) * 100
        
        # Performance journalière
        df_filtered['Daily_Performance'] = df_filtered['Adj Close'].pct_change() * 100

        # Résumé par semaine et par mois
        df_resampled_weekly = df_filtered.resample('W-WED', on='Date').last()
        df_resampled_weekly['Weekly_Performance'] = df_resampled_weekly['Adj Close'].pct_change() * 100
        df_resampled_monthly = df_filtered.resample('ME', on='Date').last()
        df_resampled_monthly['Monthly_Performance'] = df_resampled_monthly['Adj Close'].pct_change() * 100

        return {
            "pourcentage_gain_total": pourcentage_gain_total,
            "gain_total": gain_total,
            "prix_initial": prix_initial,
            "prix_final": prix_final,
            "performance_annualisee": performance_annualisee,
            "daily_performance": df_filtered[['Date', 'Adj Close', 'Daily_Performance']],
            "weekly_performance": df_resampled_weekly[['Adj Close', 'Weekly_Performance']],
            "monthly_performance": df_resampled_monthly[['Adj Close', 'Monthly_Performance']]
        }
    ######################Graphique#########################################
    def matrice_correlation(tickers_selectionnes, start_date="2010-01-01", end_date=None):
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not isinstance(tickers_selectionnes, list) or len(tickers_selectionnes) == 0:
            raise ValueError("tickers_selectionnes doit être une liste non vide.")

        # Télécharger les données pour chaque ticker
        data_dict = {}
        for ticker in tickers_selectionnes:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                if not data.empty:
                    data_dict[ticker] = data
            except Exception as e:
                print(f"Erreur lors du téléchargement des données pour {ticker}: {e}")
        
        # Vérifier si des données valides ont été collectées
        if len(data_dict) == 0:
            raise ValueError("Aucune donnée valide n'a été collectée pour les tickers sélectionnés.")

        # Convertir les données en DataFrame
        df_prices = pd.DataFrame(data_dict)
        
        # Calculer les rendements journaliers
        df_returns = df_prices.pct_change().dropna()

        # Calculer la matrice de corrélation
        correlation_matrix = df_returns.corr()

        return correlation_matrix

    def afficher_graphique_interactif(tickers_selectionnes, montant_initial, date_investissement, date_fin):
        data_dict = {
            ticker: yf.download(ticker, start="2010-01-01", end=date_fin.strftime('%Y-%m-%d')).reset_index()
            for ticker in tickers_selectionnes
        }

        fig_prices = go.Figure()
        for ticker, data in data_dict.items():
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(by='Date')

            # Tracé des lignes de prix de clôture
            fig_prices.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Adj Close'],
                mode='lines',
                name=f"{ticker} (Prix de Clôture)"
            ))

            # Ajouter un point vert pour le début de l'investissement
            data_invest = data[(data['Date'] >= date_investissement) & (data['Date'] <= date_fin)]
            if not data_invest.empty:
                prix_investissement = data_invest['Adj Close'].iloc[0]
                fig_prices.add_trace(go.Scatter(
                    x=[date_investissement],
                    y=[prix_investissement],
                    mode='markers',
                    marker=dict(color='green', size=10),
                    name=f"Début Investissement {ticker}"
                ))

                # Ajouter un point rouge pour la fin de l'investissement
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

        # Normalisation des valeurs des tickers pour calculer la proportion du portefeuille
        portfolio_values = []
        for ticker, data in data_dict.items():
            data['Date'] = pd.to_datetime(data['Date'])
            data_invest = data[(data['Date'] >= date_investissement) & (data['Date'] <= date_fin)].copy()
            prix_investissement = data_invest['Adj Close'].iloc[0]
            montant_investissement_ticker = montant_initial / len(tickers_selectionnes) * (data_invest['Adj Close'] / prix_investissement)
            montant_investissement_ticker.index = data_invest['Date']
            montant_investissement_ticker = montant_investissement_ticker.rename(f"Montant_{ticker}")
            portfolio_values.append(montant_investissement_ticker)

        df_portfolio = pd.concat(portfolio_values, axis=1).ffill()
        df_portfolio['Montant_Total'] = df_portfolio.sum(axis=1)

        for ticker in tickers_selectionnes:
            df_portfolio[f'{ticker}_Proportion'] = (df_portfolio[f"Montant_{ticker}"] / df_portfolio['Montant_Total']) * 100

        fig, ax1 = plt.subplots(figsize=(10, 8))

        normalised_values = {}
        for ticker in tickers_selectionnes:
            if f"Montant_{ticker}" in df_portfolio:
                normalised_values[ticker] = (
                    df_portfolio[f"Montant_{ticker}"] - df_portfolio[f"Montant_{ticker}"].iloc[0]
                )
                ax1.plot(
                    df_portfolio.index,
                    normalised_values[ticker],
                    label=f"Valeur normalisée de {ticker}",
                    linestyle="--"
                )

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Valeur Normalisée des Entreprises (€)")
        ax1.grid(True)

        ax2 = ax1.twinx()
        df_portfolio["Montant_Total_Normalise"] = (
            df_portfolio["Montant_Total"] - df_portfolio["Montant_Total"].iloc[0]
        )
        ax2.plot(
            df_portfolio.index,
            df_portfolio["Montant_Total_Normalise"],
            label="Valeur normalisée du Portefeuille Total",
            linewidth=2.5,
            color="black"
        )
        ax2.set_ylabel("Valeur Normalisée du Portefeuille (€)", color="black")
        ax2.tick_params(axis='y', labelcolor="black")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        fig.suptitle(
            f"Évolution Normalisée de la Valeur du Portefeuille ({date_investissement.date()} - {date_fin.date()})",
            fontsize=14
        )

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.savefig("graph_multi_tickers.png")
        plt.show()

        plt.figure(figsize=(12, 8))
        bottom = pd.Series(0, index=df_portfolio.index)

        for ticker in tickers_selectionnes:
            if f"{ticker}_Proportion" in df_portfolio.columns:
                plt.fill_between(
                    df_portfolio.index,
                    bottom,
                    bottom + df_portfolio[f"{ticker}_Proportion"],
                    label=f"Proportion de {ticker}",
                    alpha=0.5
                )
                bottom += df_portfolio[f"{ticker}_Proportion"]

        plt.title("Proportions des Entreprises dans le Portefeuille au Fil du Temps (Aire Empilée)", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Proportion (%)", fontsize=14)
        plt.legend(title="Entreprises", fontsize=12)
        plt.grid(True)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.savefig("graph_proportions_tickers.png")
        plt.show()

        return fig_prices


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
        log_rets = np.log(prices / prices.shift(1))
        log_rets.columns=[prices[1]]
        log_rets.dropna(inplace=True)
        log_rets
        # Settings
        lmb=0.94
        rolling_win=252
        # Coeff Data -> Pond vector
        pond=[[(1-lmb)*lmb**i for i in range(rolling_win-1,-1,-1)]]*len(log_rets.columns)
        pond=np.transpose(pond)
        # Calcul EWMA
        x=(log_rets.iloc[-rolling_win:]**2.0).to_frame()*pond
        ewma_volatility=(x.sum()*252)**0.5
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