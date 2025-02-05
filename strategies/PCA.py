import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from strategies.data import DataDownloader
from Indicateurs import Indicateurs  # ✅ Importation des fonctions de calcul de risque

class ACPInvestmentStrategy(Indicateurs):
    def __init__(self, tickers, start_date, end_date, initial_capital=1000, n_components=2):
        """
        Stratégie basée sur l'ACP pour l'investissement sur plusieurs tickers.

        :param tickers: Liste des tickers d'actions.
        :param start_date: Date de début de l'analyse.
        :param end_date: Date de fin de l'analyse.
        :param initial_capital: Capital initial réparti équitablement.
        :param n_components: Nombre de composantes principales à analyser.
        """
        super().__init__()  # Héritage des méthodes de la classe Indicateurs
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital / len(tickers)
        self.n_components = n_components
        self.data_downloader = DataDownloader()
        self.pca = None

    def download_data(self):
        """Télécharge les données des tickers."""
        all_data = {}

        for ticker in self.tickers:
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"⚠️ Aucune donnée disponible pour {ticker}.")
                continue

            data = data[['Close']].rename(columns={'Close': ticker})
            all_data[ticker] = data

        # Fusionner les prix des différents tickers
        price_data = pd.concat(all_data.values(), axis=1, join='inner')
        return price_data

    def compute_pca(self, returns):
        """Effectue une ACP sur les rendements et retourne les scores des composantes principales."""
        returns_clean = returns.dropna()  # Supprime les NaN AVANT d'appliquer l'ACP
        self.pca = PCA(n_components=self.n_components)
        principal_components = self.pca.fit_transform(returns_clean)

        # Convertir les scores en DataFrame avec le bon index
        scores_df = pd.DataFrame(principal_components, index=returns_clean.index, columns=[f'PC{i+1}' for i in range(self.n_components)])

        # Coefficients de contribution des actifs aux composantes
        loadings = pd.DataFrame(self.pca.components_.T, index=returns.columns, columns=[f'PC{i+1}' for i in range(self.n_components)])
        
        return scores_df, loadings

    def execute(self):
        """Exécute la stratégie basée sur l'ACP et retourne les performances du portefeuille."""
        price_data = self.download_data()
        print(price_data)
        if price_data.empty:
            print("❌ Impossible d'exécuter la stratégie : aucune donnée disponible.")
            return None

        # Calcul des rendements logarithmiques
        returns = np.log(price_data / price_data.shift(1))

        # Effectuer l'ACP sur les rendements
        pca_scores, loadings = self.compute_pca(returns)

        # Initialisation du capital par actif
        capital_per_ticker = {ticker: self.initial_capital for ticker in self.tickers}
        capital_history = []
        portfolio_results = {}

        # Backtest de la stratégie
        for date in pca_scores.index:
            current_scores = pca_scores.loc[date]
            if pd.isna(current_scores).any():
                continue  # Ignorer les dates avec des valeurs manquantes

            # Prendre une décision basée sur la première composante principale
            for ticker in self.tickers:
                score = loadings.loc[ticker, 'PC1']
                open_price = price_data.loc[date, ticker]

                if pd.isna(open_price) or open_price == 0:
                    continue  # Éviter les erreurs avec des valeurs manquantes

                if score > 0:  # 📈 Acheter si contribution positive
                    capital_per_ticker[ticker] *= (1 + returns.loc[date, ticker])
                elif score < 0:  # 📉 Vendre si contribution négative
                    capital_per_ticker[ticker] *= (1 - returns.loc[date, ticker])

            # Stocker l'évolution du capital
            total_capital = sum(capital_per_ticker.values())
            capital_history.append({"Date": date, "Total Capital": total_capital})

        # Transformer l'historique du capital en DataFrame
        capital_evolution_df = pd.DataFrame(capital_history)

        # Calcul des indicateurs de risque et de performance pour chaque ticker
        for ticker in self.tickers:
            data = self.data_downloader.download_data(ticker, self.start_date, self.end_date)
            if data.empty:
                print(f"⚠️ Pas de données pour {ticker}, métriques de risque ignorées.")
                continue
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])
            try:
                # 📊 Calcul des métriques financières
                volatilite_historique = self.volatilite_historique(data).get("volatilite_historique", 0)
                var_parametric = self.calculate_var(data, alpha=0.05, method="parametric")
                var_historical = self.calculate_var(data, alpha=0.05, method="historical")
                var_cornish_fisher = self.calculate_var(data, alpha=0.05, method="cornish-fisher")
                cvar_parametric = self.calculate_cvar(data, alpha=0.05, method="parametric")
                cvar_historical = self.calculate_cvar(data, alpha=0.05, method="historical")
                cvar_cornish_fisher = self.calculate_cvar(data, alpha=0.05, method="cornish-fisher")
            except Exception as e:
                print(f"⚠️ Erreur lors du calcul des métriques pour {ticker} : {e}")
                continue

            # 📈 Calcul du rendement total et annualisé
            final_capital = capital_per_ticker[ticker]
            gain_total = final_capital - self.initial_capital
            pourcentage_gain_total = ((final_capital / self.initial_capital) - 1) * 100
            days_invested = len(data)

            if days_invested > 0:
                performance_annualisee = ((final_capital / self.initial_capital) ** (365 / max(days_invested, 1)) - 1) * 100
            else:
                performance_annualisee = 0

            portfolio_results[ticker] = {
                "gain_total": gain_total,
                "pourcentage_gain_total": pourcentage_gain_total,
                "volatilite_historique": volatilite_historique,
                "VaR Paramétrique": var_parametric,
                "VaR Historique": var_historical,
                "VaR Cornish-Fisher": var_cornish_fisher,
                "CVaR Paramétrique": cvar_parametric,
                "CVaR Historique": cvar_historical,
                "CVaR Cornish-Fisher": cvar_cornish_fisher,
                "performance_annualisee": performance_annualisee
            }
        return {
            "gain_total": sum(res["gain_total"] for res in portfolio_results.values()),
            "pourcentage_gain_total": np.mean([res["pourcentage_gain_total"] for res in portfolio_results.values()]),
            "performance_annualisee": np.mean([res["performance_annualisee"] for res in portfolio_results.values()]),
            "volatilite_historique": np.mean([res["volatilite_historique"] for res in portfolio_results.values()]),
            "VaR Paramétrique": np.mean([res["VaR Paramétrique"] for res in portfolio_results.values()]),
            "VaR Historique": np.mean([res["VaR Historique"] for res in portfolio_results.values()]),
            "VaR Cornish-Fisher": np.mean([res["VaR Cornish-Fisher"] for res in portfolio_results.values()]),
            "CVaR Paramétrique": np.mean([res["CVaR Paramétrique"] for res in portfolio_results.values()]),
            "CVaR Historique": np.mean([res["CVaR Historique"] for res in portfolio_results.values()]),
            "CVaR Cornish-Fisher": np.mean([res["CVaR Cornish-Fisher"] for res in portfolio_results.values()]),
            "capital_evolution": capital_evolution_df,
            "pca": self.pca
        }
