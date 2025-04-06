import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import plotly.express as px
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.dates as mdates
import os
import pickle
from sklearn.decomposition import PCA

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
    def plot_model_prediction_accuracy(y_true, y_pred, title='Model Prediction Accuracy'):
        """
        Affiche la matrice de confusion et les métriques de classification pour évaluer 
        la précision du modèle.
        
        Args:
            y_true: Labels réels
            y_pred: Prédictions du modèle
            title: Titre du graphique
        """
        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Configuration de la figure
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        
        # Matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
        ax[0].set_xlabel('Prédictions')
        ax[0].set_ylabel('Valeurs réelles')
        ax[0].set_title('Matrice de confusion')
        ax[0].set_xticklabels(['Baisse', 'Hausse'])
        ax[0].set_yticklabels(['Baisse', 'Hausse'])
        
        # Calcul des métriques
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Affichage des métriques sous forme de barres
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        
        bars = ax[1].bar(metrics, values, color='royalblue')
        ax[1].set_ylim(0, 1.0)
        ax[1].set_title('Métriques de performance')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax[1].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points de décalage vertical
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    def plot_feature_importance(model, feature_names, top_n=15, title='Feature Importance'):
        """
        Visualise l'importance des features pour le modèle de deep learning.
        
        Args:
            model: Modèle entraîné (MLP ou autre)
            feature_names: Liste des noms des features
            top_n: Nombre de features à afficher
            title: Titre du graphique
        """
        if hasattr(model, 'feature_importances_'):
            # Pour les modèles qui ont l'attribut feature_importances_
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Pour les modèles linéaires
            importances = np.abs(model.coef_[0])
        else:
            # Pour les modèles sans importance directe (comme MLP)
            # On utilise la première couche de poids comme approximation
            if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
                importances = np.abs(model.coefs_[0]).mean(axis=1)
            else:
                print("Ce modèle ne prend pas en charge l'importance des features.")
                return None
        
        # Créer un DataFrame avec les importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Trier par importance et prendre les top_n features
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis', ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Importance (absolue)', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        return fig

    def plot_roc_and_precision_recall(model, X, y, title='Courbes ROC et Precision-Recall'):
        """
        Affiche les courbes ROC et Precision-Recall pour évaluer les performances du modèle.
        
        Args:
            model: Modèle entraîné
            X: Features pour l'évaluation
            y: Labels réels
            title: Titre du graphique
        """
        # Obtenir les probabilités de prédiction
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            # Utiliser decision_function si predict_proba n'est pas disponible
            y_proba = model.decision_function(X) if hasattr(model, 'decision_function') else None
        
        if y_proba is None:
            print("Ce modèle ne prend pas en charge les probabilités de prédiction.")
            return None
        
        # Préparer la figure
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        
        ax[0].plot(fpr, tpr, color='royalblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_ylim([0.0, 1.05])
        ax[0].set_xlabel('Taux de faux positifs')
        ax[0].set_ylabel('Taux de vrais positifs')
        ax[0].set_title('Courbe ROC')
        ax[0].legend(loc="lower right")
        
        # Courbe Precision-Recall
        precision, recall, _ = precision_recall_curve(y, y_proba)
        
        ax[1].step(recall, precision, color='royalblue', where='post', lw=2)
        ax[1].fill_between(recall, precision, alpha=0.2, color='royalblue', step='post')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_title('Courbe Precision-Recall')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig

    def plot_capital_evolution_with_predictions(capital_history, predictions_history, ticker):
        """
        Affiche l'évolution du capital avec les prédictions du modèle.
        
        Args:
            capital_history: Liste de tuples (date, capital)
            predictions_history: Liste de tuples (date, prédiction, action)
            ticker: Le symbole de l'actif
        """
        # Convertir les données en DataFrame
        capital_df = pd.DataFrame(capital_history, columns=['Date', 'Capital'])
        capital_df.set_index('Date', inplace=True)
        
        # Convertir les prédictions en DataFrame
        pred_df = pd.DataFrame(predictions_history, columns=['Date', 'Prediction', 'Action'])
        pred_df.set_index('Date', inplace=True)
        
        # Fusionner les DataFrames
        combined_df = capital_df.join(pred_df, how='left')
        
        # Créer la figure
        fig, ax1 = plt.subplots(figsize=(15, 7))
        
        # Tracer l'évolution du capital
        ax1.plot(combined_df.index, combined_df['Capital'], 'b-', label='Capital')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital (€)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Formater l'axe des x pour les dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Formater l'axe des y pour montrer les valeurs avec le symbole €
        def euros(x, pos):
            return f'{x:.0f} €'
        
        ax1.yaxis.set_major_formatter(FuncFormatter(euros))
        
        # Marquer les positions d'achat et de vente
        buy_dates = combined_df[combined_df['Action'] == 'Buy'].index
        sell_dates = combined_df[combined_df['Action'] == 'Sell'].index
        
        # Valeurs correspondantes du capital
        buy_values = combined_df.loc[buy_dates, 'Capital']
        sell_values = combined_df.loc[sell_dates, 'Capital']
        
        # Ajouter les marqueurs pour les décisions d'achat/vente
        ax1.scatter(buy_dates, buy_values, color='green', marker='^', s=50, label='Achat')
        ax1.scatter(sell_dates, sell_values, color='red', marker='v', s=50, label='Vente')
        
        # Légende et titre
        ax1.legend(loc='upper left')
        plt.title(f'Évolution du capital et décisions de trading pour {ticker}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_comparative_performance(results_dict, benchmark_data, initial_capital=1000):
        """
        Compare la performance de la stratégie avec un benchmark (par exemple, buy-and-hold).
        
        Args:
            results_dict: Dictionnaire des résultats de la stratégie
            benchmark_data: DataFrame contenant les prix de clôture du benchmark
            initial_capital: Capital initial
        """
        # Créer un DataFrame pour la stratégie
        capital_evolution = results_dict["capital_evolution"]
        strategy_df = capital_evolution.pivot(index='Date', columns='Ticker', values='Capital')
        
        # Calculer la valeur totale du portefeuille pour chaque date
        strategy_df['Total'] = strategy_df.sum(axis=1)
        
        # Calculer la performance du benchmark (buy-and-hold)
        benchmark_df = benchmark_data.copy()
        benchmark_df['Capital'] = initial_capital * (benchmark_df['Close'] / benchmark_df['Close'].iloc[0])
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Tracer la performance de la stratégie
        ax.plot(strategy_df.index, strategy_df['Total'], 'b-', label='DL Strategy')
        
        # Tracer la performance du benchmark
        ax.plot(benchmark_df.index, benchmark_df['Capital'], 'r--', label='Buy-and-Hold')
        
        # Formater l'axe des x pour les dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Formater l'axe des y pour montrer les valeurs avec le symbole €
        def euros(x, pos):
            return f'{x:.0f} €'
        
        ax.yaxis.set_major_formatter(FuncFormatter(euros))
        
        # Ajouter des lignes de grille
        ax.grid(True, alpha=0.3)
        
        # Légende et titre
        ax.legend(loc='upper left')
        plt.title('Comparaison de la performance: Stratégie DL vs Buy-and-Hold')
        plt.ylabel('Capital (€)')
        
        plt.tight_layout()
        return fig

    def plot_model_confidence_analysis(y_proba, y_true, bins=10):
        """
        Analyse la confiance du modèle en fonction de ses prédictions.
        
        Args:
            y_proba: Probabilités prédites par le modèle
            y_true: Labels réels
            bins: Nombre de bins pour regrouper les probabilités
        """
        # Créer un DataFrame avec les probabilités et les résultats réels
        df = pd.DataFrame({
            'probability': y_proba,
            'actual': y_true
        })
        
        # Créer des bins de probabilité
        df['bin'] = pd.cut(df['probability'], bins=bins, labels=False)
        
        # Calculer la précision par bin
        bin_analysis = df.groupby('bin').agg({
            'probability': 'mean',
            'actual': 'mean',
            'bin': 'count'
        }).rename(columns={
            'probability': 'avg_probability',
            'actual': 'actual_frequency',
            'bin': 'count'
        })
        
        # Calculer le pourcentage des prédictions par bin
        bin_analysis['percentage'] = 100 * bin_analysis['count'] / len(df)
        
        # Créer la figure
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Tracer la courbe de calibration
        ax1.plot(bin_analysis['avg_probability'], bin_analysis['actual_frequency'], 
                'bo-', label='Courbe de calibration')
        ax1.plot([0, 1], [0, 1], 'r--', label='Calibration parfaite')
        ax1.set_xlabel('Probabilité moyenne prédite')
        ax1.set_ylabel('Fréquence observée', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.legend(loc='upper left')
        
        # Ajouter un second axe y pour l'histogramme
        ax2 = ax1.twinx()
        ax2.bar(range(len(bin_analysis)), bin_analysis['percentage'], alpha=0.3, color='g',
                label='% des prédictions')
        ax2.set_ylabel('% des prédictions', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_xticks(range(len(bin_analysis)))
        ax2.set_xticklabels([f'{i/bins:.1f}-{(i+1)/bins:.1f}' for i in range(bins)], rotation=45)
        
        plt.title('Analyse de calibration du modèle')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_portfolio_risk_metrics(results_dict, tickers):
        """
        Affiche les métriques de risque pour chaque actif et pour l'ensemble du portefeuille.
        
        Args:
            results_dict: Dictionnaire des résultats de la stratégie
            tickers: Liste des tickers dans le portefeuille
        """
        # Collecter les métriques de risque pour chaque ticker
        metrics = ['volatilite_historique', 'VaR Paramétrique', 'CVaR Paramétrique', 
                'sharpe_ratio', 'max_drawdown']
        
        data = []
        
        # Ajouter les métriques pour chaque ticker
        for ticker in tickers:
            if ticker in results_dict:
                ticker_results = results_dict[ticker]
                data.append([
                    ticker,
                    ticker_results.get('volatilite_historique', np.nan),
                    ticker_results.get('VaR Paramétrique', np.nan),
                    ticker_results.get('CVaR Paramétrique', np.nan),
                    ticker_results.get('sharpe_ratio', np.nan),
                    ticker_results.get('max_drawdown', np.nan)
                ])
        
        # Ajouter les métriques pour l'ensemble du portefeuille
        data.append([
            'Portfolio',
            results_dict.get('volatilite_historique', np.nan),
            results_dict.get('VaR Paramétrique', np.nan),
            results_dict.get('CVaR Paramétrique', np.nan),
            results_dict.get('sharpe_ratio', np.nan),
            results_dict.get('max_drawdown', np.nan)
        ])
        
        # Créer un DataFrame
        risk_df = pd.DataFrame(data, columns=['Asset'] + metrics)
        
        # Configurer la figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Volatilité
        sns.barplot(x='Asset', y='volatilite_historique', data=risk_df, ax=axes[0, 0])
        axes[0, 0].set_title('Volatilité annualisée (%)')
        axes[0, 0].set_ylabel('Volatilité (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # VaR et CVaR
        var_cvar_df = risk_df.melt(id_vars='Asset', value_vars=['VaR Paramétrique', 'CVaR Paramétrique'],
                                var_name='Métrique', value_name='Valeur')
        sns.barplot(x='Asset', y='Valeur', hue='Métrique', data=var_cvar_df, ax=axes[0, 1])
        axes[0, 1].set_title('Value at Risk (95%) et CVaR (%)')
        axes[0, 1].set_ylabel('Valeur (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Ratio de Sharpe
        sns.barplot(x='Asset', y='sharpe_ratio', data=risk_df, ax=axes[1, 0])
        axes[1, 0].set_title('Ratio de Sharpe')
        axes[1, 0].set_ylabel('Ratio de Sharpe')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Drawdown maximal
        sns.barplot(x='Asset', y='max_drawdown', data=risk_df, ax=axes[1, 1])
        axes[1, 1].set_title('Drawdown maximal (%)')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Métriques de risque du portefeuille', fontsize=16)
        plt.tight_layout()
        
        return fig

    def plot_rolling_performance(capital_history, window=30, title='Performance glissante'):
        """
        Affiche la performance glissante de la stratégie.
        
        Args:
            capital_history: DataFrame avec l'historique du capital
            window: Taille de la fenêtre glissante (en jours)
            title: Titre du graphique
        """
        # Créer un DataFrame avec l'évolution du capital
        if isinstance(capital_history, list):
            capital_df = pd.DataFrame(capital_history, columns=['Date', 'Capital'])
            capital_df.set_index('Date', inplace=True)
        else:
            capital_df = capital_history.copy()
        
        # Calculer le rendement quotidien
        capital_df['daily_return'] = capital_df['Capital'].pct_change()
        
        # Calculer les rendements cumulés glissants
        capital_df['rolling_return'] = (1 + capital_df['daily_return']).rolling(window=window).apply(
            lambda x: x.prod() - 1, raw=True) * 100
        
        # Calculer la volatilité glissante
        capital_df['rolling_vol'] = capital_df['daily_return'].rolling(window=window).std() * np.sqrt(window) * 100
        
        # Calculer le ratio de Sharpe glissant (en supposant un taux sans risque nul)
        capital_df['rolling_sharpe'] = (
            capital_df['daily_return'].rolling(window=window).mean() * window /
            (capital_df['daily_return'].rolling(window=window).std() * np.sqrt(window))
        )
        
        # Créer la figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Rendement glissant
        capital_df['rolling_return'].plot(ax=axes[0], color='b')
        axes[0].set_ylabel(f'Rendement {window}j (%)')
        axes[0].set_title(f'Rendement glissant sur {window} jours')
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].grid(True, alpha=0.3)
        
        # Volatilité glissante
        capital_df['rolling_vol'].plot(ax=axes[1], color='orange')
        axes[1].set_ylabel(f'Volatilité {window}j (%)')
        axes[1].set_title(f'Volatilité glissante sur {window} jours')
        axes[1].grid(True, alpha=0.3)
        
        # Ratio de Sharpe glissant
        capital_df['rolling_sharpe'].plot(ax=axes[2], color='g')
        axes[2].set_ylabel(f'Sharpe {window}j')
        axes[2].set_title(f'Ratio de Sharpe glissant sur {window} jours')
        axes[2].axhline(y=0, color='r', linestyle='--')
        axes[2].grid(True, alpha=0.3)
        
        # Formater l'axe des x pour les dates
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[2].xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
#####################################################################################
    def matrice_correlation(self, data_dict, tickers_selectionnes, date_debut, date_fin):
        """
        Calcule la matrice de corrélation des rendements journaliers pour une liste de tickers.

        :param tickers_selectionnes: list, Liste des tickers à analyser.
        :param date_debut: str, La date de début (format 'YYYY-MM-DD').
        :param date_fin: str, La date de fin (format 'YYYY-MM-DD').
        :return: DataFrame, Matrice de corrélation des rendements journaliers.
        """
        try:
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
    
    
    def scree_plot(self, pca, n_components):
        explained_variance = pca.explained_variance_ratio_[:n_components] * 100
        cumulative_variance = np.cumsum(explained_variance)

        fig = go.Figure()
        
        # Couleurs plus attrayantes pour le graphique
        bar_color = "rgba(55, 83, 109, 0.7)"
        line_color = "rgba(220, 20, 60, 0.8)"  # Rouge cramoisi

        # Barres de variance avec bordures
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=explained_variance,
            name="Variance expliquée (%)",
            marker=dict(
                color=bar_color,
                line=dict(color="rgba(0, 0, 0, 0.5)", width=1)
            ),
            opacity=0.85
        ))

        # Ligne de variance cumulée avec marqueurs distincts
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=cumulative_variance,
            mode="lines+markers",
            name="Variance cumulée (%)",
            line=dict(color=line_color, width=3),
            marker=dict(size=8, symbol="diamond", line=dict(color="white", width=1))
        ))

        # Ajouter ligne de référence à 95% de variance cumulée
        fig.add_shape(
            type="line",
            x0=0,
            x1=n_components-1,
            y0=95,
            y1=95,
            line=dict(color="green", width=2, dash="dash"),
            xref="paper",
            yref="y"
        )
        
        # Annotation pour la ligne de référence
        fig.add_annotation(
            x=n_components-1,
            y=95,
            text="95% Variance expliquée",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=0,
            font=dict(size=12, color="green")
        )

        # Mise à jour de la mise en page
        fig.update_layout(
            title={
                'text': "📊 Scree Plot - Variance expliquée par composante principale",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=22, family="Arial", color="#2E4057")
            },
            xaxis_title=dict(text="Composantes principales", font=dict(size=14)),
            yaxis_title=dict(text="Variance expliquée (%)", font=dict(size=14)),
            yaxis=dict(range=[0, 110]),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="rgba(240, 240, 240, 0.5)"
        )

        return fig
    def plot_loadings(self, pca, feature_names,test):

        loadings = pd.DataFrame(
            pca.components_.T,
            index=feature_names,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)]
        )

        fig = px.imshow(
            loadings,
            labels=dict(x="Composantes principales", y="Actifs", color="Contribution"),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto"
        )

        fig.update_layout(
            title="🔥 Heatmap des Coefficients de Contribution des Actifs (Loadings)",
            width=900,
            height=600,
            template="plotly_white"
        )

        fig.update_xaxes(tickangle=-45)

        return fig


    def plot_capital_evolution_plotly(self, capital_evolution_df):
        """Génère un graphique interactif avec Plotly pour plusieurs tickers avec une courbe moyenne en évidence"""
        
        # ✅ Vérifier si le DataFrame est vide
        if capital_evolution_df.empty:
            print("⚠️ Aucune donnée disponible pour afficher l'évolution du capital.")
            return None

        # ✅ Vérifier que les colonnes essentielles sont présentes
        required_columns = {"Date", "Capital", "Ticker"}
        if not required_columns.issubset(capital_evolution_df.columns):
            print(f"⚠️ Le DataFrame ne contient pas toutes les colonnes nécessaires : {capital_evolution_df.columns}")
            return None
        
        # ✅ Conversion de la colonne 'Date' en format datetime pour éviter les erreurs
        capital_evolution_df["Date"] = pd.to_datetime(capital_evolution_df["Date"])

        # ✅ Calcul de la moyenne du capital pour chaque date (valeur du portefeuille)
        portfolio_value_df = capital_evolution_df.groupby("Date")["Capital"].mean().reset_index()
        portfolio_value_df["Ticker"] = "Portfolio"  # Pour identifier dans la légende

        # ✅ Fusionner les données des tickers avec la courbe moyenne
        combined_df = pd.concat([capital_evolution_df, portfolio_value_df], ignore_index=True)
        
        # Palette de couleurs améliorée
        colors = px.colors.qualitative.Bold
        
        # ✅ Création du graphique interactif avec Plotly
        fig = px.line(
            combined_df,
            x="Date",
            y="Capital",
            color="Ticker",  # Différencier les tickers par couleur
            title="📈 Évolution du Capital par Ticker avec Valeur Moyenne du Portefeuille",
            labels={"Date": "Date", "Capital": "Capital (€)", "Ticker": "Actif"},
            template="simple_white"  # Thème plus propre
        )

        # Personnaliser les couleurs pour chaque ligne
        for i, ticker in enumerate(combined_df["Ticker"].unique()):
            if ticker == "Portfolio":
                # Courbe du portfolio en noir et plus épaisse
                fig.update_traces(
                    selector=dict(name=ticker),
                    line=dict(color="black", width=4, dash="solid"),
                    mode="lines",
                )
            else:
                # Autres actifs avec des couleurs de la palette
                fig.update_traces(
                    selector=dict(name=ticker),
                    line=dict(color=colors[i % len(colors)], width=2),
                    mode="lines"
                )

        # ✅ Amélioration de l'affichage
        fig.update_layout(
            title={
                'text': "📈 Évolution du Capital par Actif",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=22, family="Arial", color="#2E4057")
            },
            xaxis_title=dict(text="Date", font=dict(size=14, family="Arial", color="#2E4057")),
            yaxis_title=dict(text="Capital (€)", font=dict(size=14, family="Arial", color="#2E4057")),
            legend_title=dict(text="Actifs", font=dict(size=14, family="Arial", color="#2E4057")),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            paper_bgcolor="white",
            plot_bgcolor="rgba(240, 240, 240, 0.5)"
        )
        
        # Amélioration des axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.6)',
            zeroline=False,
            tickangle=-45,
            tickformat="%d %b %Y",
            tickfont=dict(size=12, color="#505050")
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.6)',
            zeroline=False,
            tickformat=",.0f €",
            tickfont=dict(size=12, color="#505050")
        )
        
        # Ajout d'un filigrane
        fig.add_annotation(
            text="Dashboard Financier",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=50, color="rgba(200,200,200,0.1)")
        )
        
        return fig



    def _graphique_valeur_portefeuille(self, data_dict, montant_initial, tickers_selectionnes, date_investissement, date_fin):
        date_investissement = pd.to_datetime(date_investissement)
        date_fin = pd.to_datetime(date_fin)

        portfolio_values = []
        fig = go.Figure()
        
        # Palette de couleurs améliorée
        colors = px.colors.qualitative.Vivid

        # Ajout des courbes des actions
        for i, (ticker, data) in enumerate(data_dict.items()):
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

                    # Ajouter la courbe du cours de l'action avec couleur personnalisée
                    fig.add_trace(go.Scatter(
                        x=data_invest['Date'],
                        y=data_invest['Adj Close'],
                        mode='lines',
                        name=f"Cours {ticker}",
                        yaxis='y2',
                        line=dict(color=colors[i % len(colors)], width=1.5, dash='dot'),
                        hovertemplate='<b>%{x|%d %b %Y}</b><br>' +
                                    f'{ticker}: %{{y:.2f}}€<extra></extra>',
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
                line=dict(width=3, color="rgba(25, 25, 112, 0.8)"),  # Bleu foncé
                fill='tozeroy',
                fillcolor="rgba(25, 25, 112, 0.1)"
            ))

            # Mettre à jour la mise en page pour inclure un axe secondaire
            fig.update_layout(
                title={
                    'text': "📊 Évolution du Portefeuille vs Cours des Actions",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=22, family="Arial", color="#2E4057")
                },
                xaxis=dict(
                    title="Date",
                    tickangle=-45,
                    tickformat="%d %b %Y",
                    tickfont=dict(size=12, color="#505050")
                ),
                yaxis=dict(
                    title=dict(text="Valeur du portefeuille (€)", font=dict(size=14, color="#2E4057")),
                    side='left',
                    tickformat=",.0f €",
                    tickfont=dict(size=12, color="#505050"),
                    gridcolor='rgba(211, 211, 211, 0.6)'
                ),
                yaxis2=dict(
                    title=dict(text="Cours des actions (€)", font=dict(size=14, color="#2E4057")),
                    overlaying='y',
                    side='right',
                    tickformat=",.2f €",
                    tickfont=dict(size=12, color="#505050")
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1
                ),
                hovermode="x unified",
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="rgba(240, 240, 240, 0.5)"
            )
            
            # Annotations pour périodes importantes
            fig.add_annotation(
                x=date_investissement,
                y=df_portfolio['Montant_Total'].iloc[0],
                text="Début",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#2E4057",
                ax=-40,
                ay=-40,
                font=dict(size=12, color="#2E4057"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderpad=4
            )
            
            fig.add_annotation(
                x=date_fin,
                y=df_portfolio['Montant_Total'].iloc[-1],
                text="Fin",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#2E4057",
                ax=40,
                ay=-40,
                font=dict(size=12, color="#2E4057"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderpad=4
            )

            return fig
        else:
            print("Aucune donnée pour la valeur du portefeuille.")
            return None

    def graphique_gain_total_par_ticker(self,trade_summary):
        if not trade_summary.empty and 'Gain total' in trade_summary.columns:
            fig = px.bar(
                trade_summary,
                x='Ticker',
                y='Gain total',
                color='Gain total',
                color_continuous_scale='viridis',
                title='📈 Gain Total (€) par Ticker',
                labels={'Gain total': 'Gain (€)', 'Ticker': 'Action'},
                height=500
            )
            fig.update_layout(xaxis_title='Ticker', yaxis_title='Gain (€)', template='plotly_white')
            return fig
        return None
    def graphique_trades_achat_vente(self,trade_summary):
        if not trade_summary.empty and "Trades d'achat" in trade_summary.columns and "Trades de vente" in trade_summary.columns:
            tickers = trade_summary['Ticker']
            achats = trade_summary["Trades d'achat"]
            ventes = trade_summary["Trades de vente"]

            fig = go.Figure(data=[
                go.Bar(name='Achats', x=tickers, y=achats, marker_color='steelblue'),
                go.Bar(name='Ventes', x=tickers, y=ventes, marker_color='orange')
            ])

            fig.update_layout(
                barmode='group',
                title='📊 Nombre de Trades d\'Achat et de Vente par Ticker',
                xaxis_title='Ticker',
                yaxis_title='Nombre de Trades',
                template='plotly_white',
                height=500
            )
            return fig
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
        if not isinstance(prices, pd.Series):
            raise ValueError("`prices` doit être une Series.")

        log_rets = np.log(prices / prices.shift(1)).dropna()

        lmb = 0.94
        rolling_win = min(252, len(log_rets))  # Adapter à la taille des données

        if rolling_win < 2:
            raise ValueError("Pas assez de données pour le calcul EWMA.")

        # Poids décroissants (du plus ancien au plus récent)
        weights = np.array([(1 - lmb) * (lmb ** i) for i in range(rolling_win - 1, -1, -1)])

        # Rendements au carré
        squared_returns = log_rets[-rolling_win:] ** 2

        # Calcul de la variance pondérée puis racine (volatilité)
        ewma_variance = np.sum(weights * squared_returns)
        ewma_volatility = np.sqrt(ewma_variance * 252)  # annualisé
        print(1)
        return float(ewma_volatility)


    
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
