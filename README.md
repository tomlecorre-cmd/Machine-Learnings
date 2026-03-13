<div align="center">
  
# 🚀 Algorithmic Trading & Quant ML : S&P 500
**Conception d'une stratégie d'investissement systématique (Long/Cash) par Machine Learning et Deep Learning.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](#)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](#)
[![Deep Learning](https://img.shields.io/badge/Deep_Learning-LSTM_%7C_GRU-purple.svg?logo=tensorflow)](#)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg?logo=streamlit)](#)
[![Finance](https://img.shields.io/badge/Finance-Quant_Trading-success.svg)](#)



</div>

---

## 🎯 Executive Summary
Ce projet vise à modéliser et exploiter les inefficiences à court terme (J+1) du marché boursier américain (S&P 500). L'objectif n'est pas de créer une "boule de cristal", mais d'extraire un signal statistique probabiliste (Hit Ratio > 50%) pour surperformer un investissement passif (*Buy & Hold*) en optimisant le couple rendement/risque.

Nous avons développé une pipeline complète allant de l'ingénierie financière (Feature Engineering) jusqu'au déploiement d'un tableau de bord analytique, en mettant en compétition des algorithmes de Machine Learning tabulaire et des architectures de Deep Learning séquentiel.

### 💡 Pourquoi ce projet se démarque ?
- **Rigueur Temporelle :** Utilisation stricte de la *Walk-Forward Validation* pour garantir l'absence de Data Leakage.
- **Backtesting Réaliste :** Simulation d'un portefeuille dynamique intégrant les frictions de marché (frais de transaction).
- **Gestion du Risque :** La stratégie passe en "Cash" (liquidité) lors de signaux baissiers, évitant ainsi les krachs majeurs.
- **Transparence :** Utilisation des valeurs SHAP pour "dé-blackboxer" l'algorithme.

---

## 📊 Performances & Backtesting (Aperçu)

| Métrique Financière | Marché (Buy & Hold) | Stratégie Quant (XGBoost) | Amélioration |
| :--- | :---: | :---: | :---: |
| **Hit Ratio (Accuracy)** | ~ 50.0% | **62.6%** | `+ 12.6 pts` |
| **Rendement Annualisé** | *Variable* | **15.02%** | `Surperformance` |
| **Maximum Drawdown** | Élevé (Subit les krachs) | **Réduit** | `Risque lissé` |
| **Ratio de Sharpe** | Benchmark | **Supérieur** | `Alpha généré` |

> *Note : Les résultats complets et visualisations dynamiques sont accessibles via le Dashboard Streamlit.*

---

## 🤖 Modèles Mathématiques Implémentés

Le cœur quantitatif du projet repose sur la comparaison de 4 architectures distinctes :

1. **Random Forest (Baseline) :** Un ensemble d'arbres de décision robuste, idéal pour capturer les premières relations non-linéaires entre nos indicateurs techniques et la direction du prix.
2. **XGBoost (Modèle Principal) :** Algorithme de *Gradient Boosting* extrêmement performant sur les données tabulaires financières. Sa régularisation native limite l'overfitting face au "bruit" des marchés.
3. **LSTM (Long Short-Term Memory) :** Réseau de neurones récurrent (RNN) utilisé pour analyser le marché sous forme de séquences temporelles (fenêtres glissantes) et capturer la "mémoire" des dynamiques de prix.
4. **GRU (Gated Recurrent Unit) :** Alternative au LSTM, plus légère en paramètres, optimisant le temps de calcul tout en conservant une forte capacité prédictive sur les séries temporelles boursières.

---

## 🛠️ Architecture du Projet (Pipeline en 8 étapes)

La conception du système a été itérative et modulaire, répartie sur 8 volets distincts :

* 📥 `01_data_collection.ipynb` : **Acquisition des données.** Extraction des données historiques (OHLCV) via l'API Yahoo Finance et intégration de variables macro-économiques (VIX, S&P500, WTI) pour contextualiser le marché.
* 🔍 `02_eda.ipynb` : **Analyse Exploratoire (EDA).** Étude de la stationnarité des séries (Dickey-Fuller), détection de la saisonnalité et *Winsorization* pour neutraliser l'impact des outliers (krachs historiques exceptionnels).
* ⚙️ `03_feature_engineering.ipynb` : **Création du Signal.** Traduction des prix bruts en plus de 40 indicateurs techniques (RSI, MACD, Bandes de Bollinger, ATR, SMA/EMA). Création stricte de la variable cible (Target J+1).
* 🌳 `04_models_classical.ipynb` : **Machine Learning.** Entraînement de la Régression Logistique, Random Forest et XGBoost. Mise en place de la validation Walk-Forward et génération des matrices de confusion.
* 🧠 `05_models_deep_learning.ipynb` : **Deep Learning.** Tenseurisation des données pour les architectures séquentielles (LSTM/GRU). Entraînement, prédictions et fusion des résultats multi-modèles.
* 💰 `06_backtesting.ipynb` : **Moteur d'Exécution.** Application des probabilités à une stratégie de portefeuille Long-Only. Intégration des vrais rendements de marché, déduction des frais de courtage et calcul des KPIs (Drawdown, Sharpe).
* 🔦 `07_explainability.ipynb` : **Interprétabilité.** Génération des valeurs *SHAP* pour isoler le pouvoir prédictif local et global de chaque indicateur financier.
* 🖥️ `08_dashboard.py` : **Déploiement.** Application Streamlit interactive permettant à l'utilisateur de simuler des investissements, filtrer par secteur et auditer le comportement de l'IA (Focus Action).

---

## 👁️ Interface Interactive (Streamlit)

Le projet inclut une interface de visualisation de qualité production.

<div align="center">
  <img src="dashboard_accueil.png" alt="Aperçu du Dashboard" width="800"/>
  <br>
  <i>Tableau de bord : Métriques globales, jauge de sentiment algorithmique et performance sectorielle.</i>
</div>

<br>

<div align="center">
  <img src="dashboard_shap.png" alt="Analyse SHAP" width="800"/>
  <br>
  <i>Audit du modèle : Décomposition locale des forces de marché (SHAP) pour l'action NVIDIA.</i>
</div>

---

## ⚙️ Installation & Lancement Local

Pour cloner et exécuter l'environnement sur votre machine :

```bash
# 1. Cloner le repository
git clone [https://github.com/VOTRE_PSEUDO/nom-du-repo.git](https://github.com/VOTRE_PSEUDO/nom-du-repo.git)
cd nom-du-repo

# 2. Installer les dépendances Python
pip install -r requirements.txt

# 3. (Optionnel) Re-générer les données via les notebooks 01 à 07
# ...

# 4. Lancer le Terminal d'Analyse Quantitatif
streamlit run 08_dashboard.py
