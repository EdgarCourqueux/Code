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
    def matrice_correlation(self, tickers_selectionnes, start_date="2010-01-01", end_date=None):
        """
        Calcule et affiche une matrice de corrélation pour les entreprises sélectionnées.
        
        Arguments:
            tickers_selectionnes (list): Liste des tickers des entreprises.
            start_date (str): Date de début pour les données (format 'YYYY-MM-DD').
            end_date (str): Date de fin pour les données (par défaut : aujourd'hui).
            
        Retourne:
            DataFrame: Matrice de corrélation entre les rendements journaliers des entreprises.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Télécharger les données pour chaque ticker
        data_dict = {
            ticker: yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            for ticker in tickers_selectionnes
        }
        
        # Créer un DataFrame combiné avec les prix ajustés
        df_prices = pd.DataFrame(data_dict)
        
        # Calcul des rendements journaliers
        df_returns = df_prices.pct_change().dropna()
        
        # Calcul de la matrice de corrélation
        correlation_matrix = df_returns.corr()
        
        # Afficher un graphique de la matrice de corrélation
        plt.figure(figsize=(10, 8))
        plt.matshow(correlation_matrix, cmap="coolwarm", fignum=1)
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.title("Matrice de Corrélation des Rendements Journaliers", pad=20)
        plt.tight_layout()
        plt.show()
        
        return correlation_matrix

    def afficher_graphique_interactif(self, tickers_selectionnes, montant_initial, date_investissement):
        # Télécharger les données pour chaque ticker depuis 2010
        data_dict = {
            ticker: yf.download(ticker, start="2010-01-01", end=datetime.now().strftime('%Y-%m-%d')).reset_index()
            for ticker in tickers_selectionnes
        }

        # Créer le graphique interactif pour l'évolution des prix (données complètes depuis 2010)
        fig_prices = go.Figure()
        for ticker, data in data_dict.items():
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(by='Date')  # Ensure data is sorted by date

            # Ajouter la courbe du prix de clôture depuis 2010
            fig_prices.add_trace(go.Scatter(
                x=data['Date'],
                y=data['Adj Close'],
                mode='lines',
                name=f"{ticker} (Prix de Clôture)"
            ))

            # Extraire le prix à la date d'investissement pour le point
            data_invest = data[data['Date'] >= date_investissement]
            if not data_invest.empty:
                prix_investissement = data_invest['Adj Close'].iloc[0]
                # Ajouter un point pour marquer la date d'investissement sans normalisation
                fig_prices.add_trace(go.Scatter(
                    x=[date_investissement],
                    y=[prix_investissement],
                    mode='markers',
                    marker=dict(color='green', size=10),
                    name=f"Investissement {ticker}"
                ))

        fig_prices.update_layout(
            title="📈 Évolution des Prix des Tickers depuis 2010 avec Date d'Investissement",
            xaxis_title="Date",
            yaxis_title="Prix de Clôture",
            hovermode="x unified"
        )

        # Calculer la valeur du portefeuille pour chaque ticker et sa valeur totale (seulement depuis la date d'investissement)
        portfolio_values = []
        for ticker, data in data_dict.items():
            data['Date'] = pd.to_datetime(data['Date'])
            data_invest = data[data['Date'] >= date_investissement].copy()
            prix_investissement = data_invest['Adj Close'].iloc[0]
            montant_investissement_ticker = montant_initial / len(tickers_selectionnes) * (data_invest['Adj Close'] / prix_investissement)
            montant_investissement_ticker.index = data_invest['Date']
            montant_investissement_ticker = montant_investissement_ticker.rename(f"Montant_{ticker}")
            portfolio_values.append(montant_investissement_ticker)

        # Concaténer les valeurs dans un DataFrame unique
        df_portfolio = pd.concat(portfolio_values, axis=1).ffill()
        df_portfolio['Montant_Total'] = df_portfolio.sum(axis=1)

        # Calculer les proportions
        for ticker in tickers_selectionnes:
            df_portfolio[f'{ticker}_Proportion'] = (df_portfolio[f"Montant_{ticker}"] / df_portfolio['Montant_Total']) * 100
        # Deuxième figure : Valeur du portefeuille normalisée pour superposition avec double axe y
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # Axe de gauche : Courbes des entreprises normalisées
        normalised_values = {}
        for ticker in tickers_selectionnes:
            if f"Montant_{ticker}" in df_portfolio:
                # Normaliser pour que chaque courbe commence à 0
                normalised_values[ticker] = (
                    df_portfolio[f"Montant_{ticker}"] - df_portfolio[f"Montant_{ticker}"].iloc[0]
                )
                ax1.plot(
                    df_portfolio.index,
                    normalised_values[ticker],
                    label=f"Valeur normalisée de {ticker}",
                    linestyle="--"
                )
            else:
                print(f"Données pour {ticker} manquantes dans df_portfolio")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Valeur Normalisée des Entreprises (€)")
        ax1.tick_params(axis='y')
        ax1.grid(True)

        # Axe de droite : Courbe du portefeuille total (échelle ajustée)
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

        # Ajouter une légende combinée pour les deux axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        # Configurer le format de date pour l'axe x
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        # Ajouter un titre
        fig.suptitle(
            f"Évolution Normalisée de la Valeur du Portefeuille depuis {date_investissement.date()} (initialement {montant_initial} €)",
            fontsize=14
        )

        # Ajuster l'affichage et sauvegarder
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)  # Ajuster pour ne pas couper le titre
        plt.savefig("graph_multi_tickers.png")
        plt.show()


        # Troisième graphique : Proportions des entreprises (aire empilée)
        plt.figure(figsize=(12, 8))
        bottom = pd.Series(0, index=df_portfolio.index)  # Pour construire l'aire empilée

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

        # Tracer les lignes horizontales pointillées
        num_tickers = len(tickers_selectionnes)
        for i in range(1, num_tickers):  # Éviter les lignes à 0 % et 100 %
            proportion_line = (100 / num_tickers) * i
            plt.axhline(
                y=proportion_line,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                label=f"Ligne {proportion_line:.1f} %"
            )

        # Ajouter des détails au graphique
        plt.title("Proportions des Entreprises dans le Portefeuille au Fil du Temps (Aire Empilée)", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Proportion (%)", fontsize=14)
        plt.legend(title="Entreprises", fontsize=12)
        plt.grid(True)

        # Configurer les dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

        # Ajuster l'affichage et sauvegarder
        plt.tight_layout()
        plt.savefig("graph_proportions_tickers.png")
        plt.show()

        return fig_prices



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