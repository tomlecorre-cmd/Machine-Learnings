import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION GENERALE
# ─────────────────────────────────────────────
st.set_page_config(page_title="Quant Trading Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #1e2130; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid #2d3250;
    }
    .big-metric { font-size: 2.5rem; font-weight: bold; color: #00d4aa; }
    .sector-badge {
        background-color: #2d3250; border-radius: 5px; padding: 3px 10px; font-size: 0.8rem; color: #a0a0a0;
    }
    .signal-buy {
        background-color: rgba(0,212,170,0.1); border: 2px solid #00d4aa; border-radius: 10px; 
        padding: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; color: #00d4aa;
    }
    .signal-wait {
        background-color: rgba(255,165,0,0.1); border: 2px solid #ffa500; border-radius: 10px; 
        padding: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; color: #ffa500;
    }
    .signal-sell {
        background-color: rgba(255,68,68,0.1); border: 2px solid #ff4444; border-radius: 10px; 
        padding: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; color: #ff4444;
    }
    .info-box {
        background-color: #161b22; border-left: 4px solid #4e9af1; padding: 15px; border-radius: 5px; font-size: 0.95rem; color: #c9d1d9; margin-top: 20px; line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# CORRECTION : Fonction style_fig blindée contre les "undefined"
def style_fig(fig, hauteur=400, titre="", titre_y="", type_x='date'):
    layout_kwargs = dict(
        height=hauteur, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white',
        xaxis=dict(showgrid=(type_x!='date'), gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=20, t=50 if titre else 20, b=20)
    )
    if titre:
        layout_kwargs['title'] = dict(text=titre, font=dict(size=16, color="white"))
    if titre_y:
        layout_kwargs['yaxis']['title'] = titre_y
        
    fig.update_layout(**layout_kwargs)
    return fig

def add_explication(texte):
    st.markdown(f'<div class="info-box">📊 <b>Analyse Quantitative :</b><br><br>{texte}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNEES
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    features_df    = pd.read_parquet('data/03_features.parquet')
    predictions_df = pd.read_parquet('data/05_all_predictions.parquet')
    backtesting_df = pd.read_parquet('data/06_backtesting_results.parquet')
    macro_df       = pd.read_parquet('data/01_macro_data.parquet')
    shap_df        = pd.read_csv('data/07_shap_importance.csv')

    features_df['date']    = pd.to_datetime(features_df['date'])
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    macro_df.index         = pd.to_datetime(macro_df.index)
    return features_df, predictions_df, backtesting_df, macro_df, shap_df

features_df, predictions_df, backtesting_df, macro_df, shap_df = load_data()

ALL_SECTORS = sorted(features_df['sector'].unique().tolist())
ALL_TICKERS = sorted(features_df['ticker'].unique().tolist())
SECTOR_MAP  = features_df[['ticker', 'sector']].drop_duplicates().set_index('ticker')['sector'].to_dict()
SECTOR_COLORS = {'Technology': '#00d4aa', 'Finance': '#4e9af1', 'Healthcare': '#f77f7f', 'Energy': '#ffd166', 'Industrials': '#a78bfa'}

VOL_MARCHE = 18.0

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image('https://img.icons8.com/fluency/96/stock-market.png', width=60)
    st.title('Quant Trading System')
    st.divider()
    page = st.selectbox('📋 Tableau de bord', ['🏠 Accueil', '📊 Analyse par stock', '⚖️ Comparaison de stocks', '🏭 Vue par secteur', '💰 Backtesting', '⭐ Focus Action', '🔍 Forces de Marché (SHAP)'])
    st.divider()
    st.caption('📅 Période de test OOS')
    st.caption(f"{predictions_df['date'].min().strftime('%d/%m/%Y')} → {predictions_df['date'].max().strftime('%d/%m/%Y')}")
    st.caption('📈 Univers')
    st.caption(f'{len(ALL_TICKERS)} actions — {len(ALL_SECTORS)} secteurs')

# ─────────────────────────────────────────────
# PAGE 1 — ACCUEIL
# ─────────────────────────────────────────────
if page == '🏠 Accueil':
    st.title('📈 Stratégie Quantitative Long/Cash')
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Univers', f'{len(ALL_TICKERS)} actions', '5 secteurs')
    col2.metric('Modèle', 'XGBoost', 'Walk-forward')
    col3.metric('Hit Ratio (Test)', '62.6%', '+12.6% vs Aléatoire')
    col4.metric('Rendement', '15.02%', 'Annualisé')

    st.divider()
    st.subheader('🌡️ Pression Directionnelle du Marché')

    last_date  = predictions_df['date'].max()
    last_preds = predictions_df[predictions_df['date'] == last_date]
    pct_hausse = last_preds['xgb_pred_direction'].mean() * 100

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric('📅 Date des signaux', last_date.strftime('%d/%m/%Y'))
        st.metric('🟢 Biais Acheteur', f'{pct_hausse:.1f}%')
        st.metric('🔴 Biais Baissier', f'{100 - pct_hausse:.1f}%')

    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode='gauge+number', value=pct_hausse,
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#00d4aa'},
                   'steps': [{'range': [0, 30], 'color': 'rgba(255,68,68,0.2)'}, {'range': [30, 70], 'color': 'rgba(255,165,0,0.2)'}, {'range': [70, 100], 'color': 'rgba(0,212,170,0.2)'}],
                   'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': 50}}
        ))
        st.plotly_chart(style_fig(fig_gauge, hauteur=250, titre="Sentiment IA Global (%)"), use_container_width=True)

    with col3:
        last_vix = macro_df['VIX'].iloc[-1]
        vix_color, vix_label = ('🟢', 'Calme') if last_vix < 20 else ('🟠', 'Elevé') if last_vix < 30 else ('🔴', 'Panique')
        st.metric(f'{vix_color} Indice VIX', f'{last_vix:.1f}', vix_label, delta_color="off")
        st.metric('📈 S&P500', f'{macro_df["SP500"].iloc[-1]:,.0f}')
        st.metric('🛢️ Pétrole', f'{macro_df["OIL"].iloc[-1]:.1f}$')

    st.divider()
    
    col_graph, col_text = st.columns([3, 1])
    with col_graph:
        sector_perf = [{'Secteur': sec, 'Accuracy': (predictions_df[predictions_df['sector'] == sec]['xgb_pred_direction'] == predictions_df[predictions_df['sector'] == sec]['target_direction']).mean()} for sec in ALL_SECTORS]
        fig_sect = px.bar(pd.DataFrame(sector_perf), x='Secteur', y='Accuracy', color='Secteur', color_discrete_map=SECTOR_COLORS, text=pd.DataFrame(sector_perf)['Accuracy'].apply(lambda x: f'{x:.1%}'))
        fig_sect.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='Bruit stat. (50%)')
        fig_sect.update_layout(showlegend=False, yaxis_tickformat='.0%')
        st.plotly_chart(style_fig(fig_sect, titre="Taux de succès (Accuracy) par secteur", type_x='cat'), use_container_width=True)
    with col_text:
        add_explication("Obtenir un taux supérieur à 52% valide une espérance de gain positive. Le secteur Technologie offre souvent un momentum très net à exploiter.")

    st.divider()
    st.subheader('🏆 Top 5 et Flop 5 stocks — Rendement cumulé')
    stock_perf = []
    for ticker in ALL_TICKERS:
        s_data = predictions_df[predictions_df['ticker'] == ticker]
        if len(s_data) > 0:
            stock_perf.append({'Ticker': ticker, 'Secteur': SECTOR_MAP.get(ticker, ''), 'Rend. Cum': ((1 + s_data['target_return']).prod() - 1) * 100, 'Accuracy': (s_data['xgb_pred_direction'] == s_data['target_direction']).mean() * 100})
    stock_perf_df = pd.DataFrame(stock_perf).sort_values('Rend. Cum', ascending=False)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**🟢 Top 5 Performances**')
        st.dataframe(stock_perf_df.head(5)[['Ticker', 'Secteur', 'Rend. Cum', 'Accuracy']].style.format({'Rend. Cum': '{:.1f}%', 'Accuracy': '{:.1f}%'}), use_container_width=True, hide_index=True)
    with col2:
        st.markdown('**🔴 Flop 5 Performances**')
        st.dataframe(stock_perf_df.tail(5)[['Ticker', 'Secteur', 'Rend. Cum', 'Accuracy']].style.format({'Rend. Cum': '{:.1f}%', 'Accuracy': '{:.1f}%'}), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE 2 — ANALYSE PAR STOCK
# ─────────────────────────────────────────────
elif page == '📊 Analyse par stock':
    st.title('📊 Analyse Individuelle')
    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1: sector_filter = st.selectbox('Filtre Sectoriel', ['Tous'] + ALL_SECTORS)
    with col2: selected_ticker = st.selectbox('Sélection du sous-jacent', ALL_TICKERS if sector_filter == 'Tous' else sorted([t for t in ALL_TICKERS if SECTOR_MAP.get(t) == sector_filter]))

    st.markdown(f'### {selected_ticker} — <span class="sector-badge">{SECTOR_MAP.get(selected_ticker, "")}</span>', unsafe_allow_html=True)
    st.divider()

    stock_data  = features_df[features_df['ticker'] == selected_ticker].sort_values('date')
    stock_preds = predictions_df[predictions_df['ticker'] == selected_ticker].sort_values('date')

    col1, col2, col3, col4, col5 = st.columns(5)
    total_return = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[0] - 1) * 100
    volatility   = stock_data['return'].std() * np.sqrt(252) * 100
    acc_stock    = (stock_preds['xgb_pred_direction'] == stock_preds['target_direction']).mean() * 100
    proba_up     = stock_preds.iloc[-1]['xgb_pred_proba']

    col1.metric('Prix Clôture', f"{stock_data['close'].iloc[-1]:.2f}$", f"{stock_data['return'].iloc[-1]*100:+.2f}%")
    col2.metric('Rend. Période', f'{total_return:.1f}%')
    col3.metric('Volatilité Annuelle', f'{volatility:.1f}%', f"{volatility - VOL_MARCHE:+.1f}% vs Marché", delta_color="inverse")
    col4.metric('Hit Ratio', f'{acc_stock:.1f}%')
    col5.metric('Proba Hausse Demain', f'{proba_up:.1%}')

    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if proba_up >= 0.52: st.markdown('<div class="signal-buy">🟢 SIGNAL LONG : Asymétrie Favorable</div>', unsafe_allow_html=True)
        elif proba_up <= 0.48: st.markdown('<div class="signal-sell">🔴 SIGNAL SHORT / CASH : Pression Baissière</div>', unsafe_allow_html=True)
        else: st.markdown('<div class="signal-wait">🟠 SIGNAL NEUTRE : Bruit de marché élevé</div>', unsafe_allow_html=True)

    st.divider()
    
    c_graph, c_text = st.columns([3, 1])
    with c_graph:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Prix', line=dict(color='white', width=1.5)))
        fig_price.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'].rolling(20).mean(), name='SMA 20', line=dict(color='#00d4aa', width=1, dash='dash')))
        fig_price.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'].rolling(50).mean(), name='SMA 50', line=dict(color='#ffd166', width=1, dash='dash')))
        st.plotly_chart(style_fig(fig_price, titre="Action des prix et Momentum (SMA)", titre_y="Prix ($)"), use_container_width=True)
    with c_text:
        add_explication("Le croisement du prix avec ses moyennes mobiles (20 et 50 jours) est l'un des indicateurs de tendance les plus scrutés par les algorithmes.")

    # Rendements Journaliers
    fig_ret = go.Figure()
    colors_ret = ['green' if r > 0 else 'red' for r in stock_data['return']]
    fig_ret.add_trace(go.Bar(x=stock_data['date'], y=stock_data['return'] * 100, marker_color=colors_ret, name='Rendement'))
    st.plotly_chart(style_fig(fig_ret, hauteur=300, titre="Rendements journaliers de l'action", titre_y='Rendement (%)'), use_container_width=True)

    # Volatilité glissante
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['return'].rolling(30).std() * np.sqrt(252) * 100, fill='tozeroy', fillcolor='rgba(255,165,0,0.2)', line=dict(color='#ffa500', width=1.5)))
    st.plotly_chart(style_fig(fig_vol, hauteur=300, titre="Régime de Volatilité (Fenêtre de 30 jours)", titre_y='Volatilité (%)'), use_container_width=True)

    # Historique prédictions vs réalité
    fig_pred = go.Figure()
    correct = stock_preds[stock_preds['xgb_pred_direction'] == stock_preds['target_direction']]
    incorrect = stock_preds[stock_preds['xgb_pred_direction'] != stock_preds['target_direction']]
    fig_pred.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Prix', line=dict(color='white', width=1)))
    fig_pred.add_trace(go.Scatter(x=correct['date'], y=stock_data[stock_data['date'].isin(correct['date'])]['close'], mode='markers', marker=dict(color='green', size=5, symbol='circle'), name='Juste'))
    fig_pred.add_trace(go.Scatter(x=incorrect['date'], y=stock_data[stock_data['date'].isin(incorrect['date'])]['close'], mode='markers', marker=dict(color='red', size=5, symbol='x'), name='Faux'))
    st.plotly_chart(style_fig(fig_pred, titre="Distribution des Signaux (Correct vs Incorrect)"), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 — COMPARAISON DE STOCKS
# ─────────────────────────────────────────────
elif page == '⚖️ Comparaison de stocks':
    st.title('⚖️ Comparaison et Corrélation')
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1: stock1 = st.selectbox('Sous-jacent 1', ALL_TICKERS, index=ALL_TICKERS.index('AAPL'))
    with col2: stock2 = st.selectbox('Sous-jacent 2', ALL_TICKERS, index=ALL_TICKERS.index('MSFT'))
    with col3: stock3 = st.selectbox('Sous-jacent 3 (Optionnel)', ['Aucun'] + ALL_TICKERS)

    selected_stocks = [stock1, stock2] if stock3 == 'Aucun' else [stock1, stock2, stock3]
    colors_comp = ['#00d4aa', '#4e9af1', '#ffd166']
    
    fig_norm = go.Figure()
    for i, ticker in enumerate(selected_stocks):
        s_data = features_df[features_df['ticker'] == ticker].sort_values('date')
        fig_norm.add_trace(go.Scatter(x=s_data['date'], y=s_data['close'] / s_data['close'].iloc[0] * 100, name=ticker, line=dict(color=colors_comp[i], width=2)))
    fig_norm.add_hline(y=100, line_dash='dash', line_color='white', line_width=1, opacity=0.5)
    st.plotly_chart(style_fig(fig_norm, titre="Performance Relative (Normalisée base 100)", titre_y='Base 100'), use_container_width=True)

    st.subheader('📋 Métriques de performance comparées')
    comp_data = []
    for ticker in selected_stocks:
        s_data  = features_df[features_df['ticker'] == ticker].sort_values('date')
        s_preds = predictions_df[predictions_df['ticker'] == ticker]
        comp_data.append({
            'Actif': ticker, 'Secteur': SECTOR_MAP.get(ticker, ''),
            'Perf Absolue': f"{(s_data['close'].iloc[-1] / s_data['close'].iloc[0] - 1) * 100:.1f}%",
            'Volatilité': f"{s_data['return'].std() * np.sqrt(252) * 100:.1f}%",
            'Ratio Sharpe': f"{(s_data['return'].mean() / s_data['return'].std()) * np.sqrt(252):.2f}",
            'Max Drawdown': f"{((s_data['close'] / s_data['close'].cummax()) - 1).min() * 100:.1f}%",
            'Hit Ratio IA': f"{(s_preds['xgb_pred_direction'] == s_preds['target_direction']).mean() * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.divider()
    col_graph1, col_graph2 = st.columns(2)
    with col_graph1:
        returns_dict = {t: features_df[features_df['ticker'] == t].set_index('date')['return'] for t in selected_stocks}
        fig_corr = px.imshow(pd.DataFrame(returns_dict).dropna().corr(), text_auto='.2f', color_continuous_scale='RdYlGn')
        st.plotly_chart(style_fig(fig_corr, hauteur=350, titre="Matrice de Corrélation Croisée", type_x='cat'), use_container_width=True)
    with col_graph2:
        fig_vol_comp = go.Figure()
        for i, ticker in enumerate(selected_stocks):
            s_data = features_df[features_df['ticker'] == ticker].sort_values('date')
            fig_vol_comp.add_trace(go.Scatter(x=s_data['date'], y=s_data['return'].rolling(30).std() * np.sqrt(252) * 100, name=ticker, line=dict(color=colors_comp[i])))
        st.plotly_chart(style_fig(fig_vol_comp, hauteur=350, titre="Volatilité comparée (Fenêtre 30j)", titre_y='Volatilité (%)'), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — VUE PAR SECTEUR
# ─────────────────────────────────────────────
elif page == '🏭 Vue par secteur':
    st.title('🏭 Analyse Fondamentale : Les Secteurs')
    st.divider()

    selected_sector = st.selectbox('Industrie ciblée', ALL_SECTORS)
    sector_tickers  = sorted([t for t in ALL_TICKERS if SECTOR_MAP.get(t) == selected_sector])

    st.divider()
    sector_data  = features_df[features_df['sector'] == selected_sector]
    sector_preds = predictions_df[predictions_df['sector'] == selected_sector]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Échantillon', f'{len(sector_tickers)} actions')
    col2.metric('Rendement Moyen', f"{sector_data['return'].mean() * 252 * 100:.1f}%")
    col3.metric('Volatilité Sectorielle', f"{sector_data['return'].std() * np.sqrt(252) * 100:.1f}%")
    col4.metric('Hit Ratio Moyen', f"{(sector_preds['xgb_pred_direction'] == sector_preds['target_direction']).mean() * 100:.1f}%")

    fig_sect = go.Figure()
    for ticker in sector_tickers:
        s_data = features_df[features_df['ticker'] == ticker].sort_values('date')
        fig_sect.add_trace(go.Scatter(x=s_data['date'], y=s_data['close'] / s_data['close'].iloc[0] * 100, name=ticker, opacity=0.8))
    fig_sect.add_hline(y=100, line_dash='dash', line_color='white', line_width=1, opacity=0.5)
    st.plotly_chart(style_fig(fig_sect, hauteur=450, titre=f"Dispersion des rendements (Base 100) — {selected_sector}", titre_y='Base 100'), use_container_width=True)

    st.subheader(f'🏆 Classement des actions — {selected_sector}')
    sort_by = st.selectbox('Trier par', ['Rendement Total', 'Volatilité', 'Sharpe', 'Hit Ratio'])

    ranking_data = []
    for ticker in sector_tickers:
        s_data  = features_df[features_df['ticker'] == ticker].sort_values('date')
        s_preds = predictions_df[predictions_df['ticker'] == ticker]
        ranking_data.append({
            'Ticker': ticker,
            'Rendement Total': (s_data['close'].iloc[-1] / s_data['close'].iloc[0] - 1) * 100,
            'Volatilité': s_data['return'].std() * np.sqrt(252) * 100,
            'Sharpe': (s_data['return'].mean() / s_data['return'].std()) * np.sqrt(252),
            'Hit Ratio': (s_preds['xgb_pred_direction'] == s_preds['target_direction']).mean() * 100,
            'Dernière Proba': s_preds['xgb_pred_proba'].iloc[-1] * 100 if len(s_preds) > 0 else 50
        })

    st.dataframe(pd.DataFrame(ranking_data).sort_values(sort_by, ascending=False).style.format({'Rendement Total': '{:.1f}%', 'Volatilité': '{:.1f}%', 'Sharpe': '{:.2f}', 'Hit Ratio': '{:.1f}%', 'Dernière Proba': '{:.1f}%'}).background_gradient(subset=['Rendement Total', 'Hit Ratio'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

    st.divider()
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        returns_sector = {t: features_df[features_df['ticker'] == t].set_index('date')['return'] for t in sector_tickers}
        fig_corr_sect = px.imshow(pd.DataFrame(returns_sector).dropna().corr(), text_auto='.2f', color_continuous_scale='RdYlGn')
        st.plotly_chart(style_fig(fig_corr_sect, hauteur=450, titre="Matrice de Corrélation Sectorielle", type_x='cat'), use_container_width=True)
        
    with col_graph2:
        fig_vol_sect = go.Figure()
        for ticker in sector_tickers:
            s_data = features_df[features_df['ticker'] == ticker].sort_values('date')
            fig_vol_sect.add_trace(go.Scatter(x=s_data['date'], y=s_data['return'].rolling(30).std() * np.sqrt(252) * 100, name=ticker, opacity=0.7))
        st.plotly_chart(style_fig(fig_vol_sect, hauteur=450, titre="Volatilité Sectorielle Comparée (30j)", titre_y='Volatilité (%)'), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 5 — BACKTESTING
# ─────────────────────────────────────────────
elif page == '💰 Backtesting':
    st.title('💰 Backtesting et Exécution')
    st.caption('Évaluation rigoureuse des signaux via la simulation d\'un portefeuille')
    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1: sector_filter_bt = st.selectbox('Secteur', ['Tous'] + ALL_SECTORS, key='bt_sector')
    with col2:
        tickers_bt = ALL_TICKERS if sector_filter_bt == 'Tous' else sorted([t for t in ALL_TICKERS if SECTOR_MAP.get(t) == sector_filter_bt])
        selected_ticker_bt = st.selectbox('Périmètre', ['Tous les stocks'] + tickers_bt, key='bt_ticker')

    TRANSACTION_COST = st.slider('Frictions & Frais de courtage (%)', 0.0, 0.5, 0.1, 0.05) / 100
    st.divider()

    def run_backtest(preds_df, cost):
        preds_df = preds_df.copy().sort_values('date')
        # CORRECTION : Remplacer les valeurs nulles générées par le décalage (shift)
        strat_ret = (preds_df['xgb_pred_direction'].shift(1) * preds_df['target_return'] - preds_df['xgb_pred_direction'].diff().abs().fillna(0) * cost).fillna(0)
        return strat_ret, preds_df['target_return'].fillna(0)

    bt_preds = predictions_df if selected_ticker_bt == 'Tous les stocks' else predictions_df[predictions_df['ticker'] == selected_ticker_bt]
    
    if selected_ticker_bt == 'Tous les stocks':
        strat_returns = pd.concat([run_backtest(bt_preds[bt_preds['ticker'] == t], TRANSACTION_COST)[0] for t in bt_preds['ticker'].unique()]).groupby(level=0).mean()
        bah_returns = pd.concat([run_backtest(bt_preds[bt_preds['ticker'] == t], TRANSACTION_COST)[1] for t in bt_preds['ticker'].unique()]).groupby(level=0).mean()
    else:
        strat_returns, bah_returns = run_backtest(bt_preds, TRANSACTION_COST)

    def compute_metrics(r):
        r = r.dropna()
        cum = (1 + r).prod() - 1
        return {'Cumulé': cum, 'Annualisé': (1 + cum) ** (252 / len(r)) - 1, 'Vol': r.std() * np.sqrt(252), 'Sharpe': (r.mean() - 0.05/252) / r.std() * np.sqrt(252), 'Max DD': (((1+r).cumprod() - (1+r).cumprod().cummax()) / (1+r).cumprod().cummax()).min(), 'Trades': (r != 0).sum()}

    m_strat, m_bah = compute_metrics(strat_returns), compute_metrics(bah_returns)

    st.subheader('📊 Bilan des performances')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**🟢 Stratégie Quantitative (XGBoost)**')
        m1, m2, m3 = st.columns(3)
        m1.metric('Rend. Cumulé', f'{m_strat["Cumulé"]:.1%}')
        m2.metric('Ratio Sharpe', f'{m_strat["Sharpe"]:.3f}')
        m3.metric('Max Drawdown', f'{m_strat["Max DD"]:.1%}')
    with col2:
        st.markdown('**🟡 Benchmark Naïf (Buy & Hold)**')
        m1, m2, m3 = st.columns(3)
        m1.metric('Rend. Cumulé', f'{m_bah["Cumulé"]:.1%}')
        m2.metric('Ratio Sharpe', f'{m_bah["Sharpe"]:.3f}')
        m3.metric('Max Drawdown', f'{m_bah["Max DD"]:.1%}')

    c_graph, c_text = st.columns([3, 1])
    with c_graph:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=strat_returns.index, y=(1 + strat_returns).cumprod(), name='Exécution Stratégie', line=dict(color='#00d4aa', width=2)))
        fig_bt.add_trace(go.Scatter(x=bah_returns.index, y=(1 + bah_returns).cumprod(), name='Benchmark (Buy & Hold)', line=dict(color='#ffd166', width=2, dash='dash')))
        fig_bt.add_hline(y=1, line_dash='dot', line_color='white', line_width=1, opacity=0.5)
        st.plotly_chart(style_fig(fig_bt, titre=f"Courbe de Richesse — {selected_ticker_bt}", titre_y="Capital (Base 1)"), use_container_width=True)
    with c_text:
        add_explication("<b>Philosophie de la stratégie :</b><br>Notre approche 'Long/Cash' n'a pas pour seul but de battre le marché en rentabilité pure. <br><br>Le véritable atout quantitatif est de lisser le risque : en passant en 'Cash' (liquidité) lors des signaux baissiers, la courbe verte évite structurellement les effondrements soudains (Max Drawdown) que subit le Buy & Hold (courbe jaune).")

    # DRAWDOWN 
    cum_s = (1 + strat_returns).cumprod()
    dd_strat = (cum_s - cum_s.cummax()) / cum_s.cummax()
    cum_b = (1 + bah_returns).cumprod()
    dd_bah = (cum_b - cum_b.cummax()) / cum_b.cummax()

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd_strat.index, y=dd_strat.values * 100, fill='tozeroy', fillcolor='rgba(0,212,170,0.2)', line=dict(color='#00d4aa', width=1), name='Stratégie'))
    fig_dd.add_trace(go.Scatter(x=dd_bah.index, y=dd_bah.values * 100, fill='tozeroy', fillcolor='rgba(255,209,102,0.2)', line=dict(color='#ffd166', width=1, dash='dash'), name='Buy & Hold'))
    st.plotly_chart(style_fig(fig_dd, hauteur=300, titre="Analyse du Drawdown (Mesure de la douleur financière)", titre_y='Drawdown (%)'), use_container_width=True)
    
    st.subheader('💶 Simulation Investissement Financier')
    initial_investment = st.number_input('Montant investi (€)', 100, 1000000, 1000, 100)
    col1, col2, col3 = st.columns(3)
    col1.metric('Investissement initial', f'{initial_investment:,.0f}€')
    col2.metric('Capital final Stratégie', f"{initial_investment * (1 + m_strat['Cumulé']):,.0f}€", f"{initial_investment * m_strat['Cumulé']:+,.0f}€")
    col3.metric('Capital final Marché', f"{initial_investment * (1 + m_bah['Cumulé']):,.0f}€", f"{initial_investment * m_bah['Cumulé']:+,.0f}€")


# ─────────────────────────────────────────────
# PAGE 6 — FOCUS ACTION (Star)
# ─────────────────────────────────────────────
elif page == '⭐ Focus Action':
    st.title('⭐ Étude de Cas Spécifique')
    star_ticker = st.selectbox('Sélection du sous-jacent de référence', ALL_TICKERS, index=ALL_TICKERS.index('NVDA'))
    st.divider()

    s_data  = features_df[features_df['ticker'] == star_ticker].sort_values('date')
    s_preds = predictions_df[predictions_df['ticker'] == star_ticker].sort_values('date')

    col1, col2, col3, col4 = st.columns(4)
    total_ret_star  = (s_data['close'].iloc[-1] / s_data['close'].iloc[0] - 1) * 100
    acc_star = (s_preds['xgb_pred_direction'] == s_preds['target_direction']).mean() * 100
    
    col1.metric('Dernier Cours', f"{s_data['close'].iloc[-1]:.2f}$", f"{s_data['return'].iloc[-1]*100:+.2f}%")
    col2.metric('Appréciation', f'+{total_ret_star:.0f}%')
    col3.metric('Hit Ratio', f'{acc_star:.1f}%')
    col4.metric('Volatilité', f"{s_data['return'].std() * np.sqrt(252) * 100:.1f}%")

    st.divider()
    last_proba = s_preds['xgb_pred_proba'].iloc[-1]
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if last_proba >= 0.52: st.markdown(f'<div class="signal-buy">🟢 ORIENTATION : EXPOSITION LONGUE REQUISE</div>', unsafe_allow_html=True)
        elif last_proba <= 0.48: st.markdown(f'<div class="signal-sell">🔴 ORIENTATION : LIQUIDATION DE POSITION</div>', unsafe_allow_html=True)
        else: st.markdown(f'<div class="signal-wait">🟠 ORIENTATION : EXPOSITION NEUTRE (ATTENTE)</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center; color:#a0a0a0;">Score probabiliste : {last_proba:.1%}</p>', unsafe_allow_html=True)

    st.divider()
    
    c_graph, c_text = st.columns([3, 1])
    with c_graph:
        fig_star = go.Figure()
        fig_star.add_trace(go.Scatter(x=s_data['date'], y=s_data['close'], name=star_ticker, line=dict(color='#76b900', width=2), fill='tozeroy', fillcolor='rgba(118,185,0,0.1)'))
        for date, label in [('2020-03-23', 'Choc Pandémique'), ('2022-01-01', 'Resserrement FED'), ('2023-01-01', 'Boom Technologique')]:
            fig_star.add_vline(x=date, line_dash='dash', line_color='rgba(255,255,255,0.4)', line_width=1)
            fig_star.add_annotation(x=date, y=s_data['close'].max()*0.9, text=label, showarrow=False, font=dict(color='white', size=10), bgcolor='rgba(0,0,0,0.5)')
        st.plotly_chart(style_fig(fig_star, hauteur=400, titre=f"Histoire boursière : {star_ticker}", titre_y='Prix ($)'), use_container_width=True)
    with c_text:
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 50px;">
            <div style="color:#a0a0a0; font-size:0.9rem;">Multiplicateur</div>
            <div class="big-metric">x{s_data['close'].iloc[-1] / s_data['close'].iloc[0]:.1f}</div>
            <div style="color:#a0a0a0; font-size:0.8rem;">Sur la période affichée</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    fig_comp_star = go.Figure()
    sp500_norm  = macro_df['SP500'] / macro_df['SP500'].iloc[0] * 100
    nasdaq_norm = macro_df['NASDAQ'] / macro_df['NASDAQ'].iloc[0] * 100 if 'NASDAQ' in macro_df.columns else None
    star_norm   = s_data.set_index('date')['close'] / s_data['close'].iloc[0] * 100

    fig_comp_star.add_trace(go.Scatter(x=star_norm.index, y=star_norm.values, name=star_ticker, line=dict(color='#76b900', width=2.5)))
    fig_comp_star.add_trace(go.Scatter(x=sp500_norm.index, y=sp500_norm.values, name='S&P500', line=dict(color='#4e9af1', width=1.5, dash='dash')))
    if nasdaq_norm is not None:
        fig_comp_star.add_trace(go.Scatter(x=nasdaq_norm.index, y=nasdaq_norm.values, name='Nasdaq', line=dict(color='#ffd166', width=1.5, dash='dot')))
    fig_comp_star.add_hline(y=100, line_dash='dot', line_color='white', line_width=1, opacity=0.3)
    st.plotly_chart(style_fig(fig_comp_star, hauteur=400, titre=f"📊 {star_ticker} vs Marché (S&P500 / Nasdaq)", titre_y='Base 100'), use_container_width=True)

    fig_pred_star = go.Figure()
    correct_star   = s_preds[s_preds['xgb_pred_direction'] == s_preds['target_direction']]
    incorrect_star = s_preds[s_preds['xgb_pred_direction'] != s_preds['target_direction']]
    s_test_data = s_data[s_data['date'] >= s_preds['date'].min()]

    fig_pred_star.add_trace(go.Scatter(x=s_test_data['date'], y=s_test_data['close'], name='Prix', line=dict(color='#76b900', width=1.5)))
    fig_pred_star.add_trace(go.Scatter(x=correct_star['date'], y=s_test_data[s_test_data['date'].isin(correct_star['date'])]['close'], mode='markers', marker=dict(color='green', size=5), name='Signal Vrai'))
    fig_pred_star.add_trace(go.Scatter(x=incorrect_star['date'], y=s_test_data[s_test_data['date'].isin(incorrect_star['date'])]['close'], mode='markers', marker=dict(color='red', size=5, symbol='x'), name='Signal Faux'))
    st.plotly_chart(style_fig(fig_pred_star, titre=f"🎯 Historique des signaux sur {star_ticker}"), use_container_width=True)

    # CORRECTION DES LIGNES VIDES POUR LE BACKTESTING NVDA
    star_signal   = s_preds['xgb_pred_direction']
    pos_change    = star_signal.diff().abs().fillna(0)
    star_strat    = (star_signal.shift(1) * s_preds['target_return'] - pos_change * 0.001).fillna(0)
    star_bah      = s_preds['target_return'].fillna(0)

    cum_star_strat = (1 + star_strat).cumprod()
    cum_star_bah   = (1 + star_bah).cumprod()

    fig_bt_star = go.Figure()
    fig_bt_star.add_trace(go.Scatter(x=s_preds['date'], y=cum_star_strat, name='Algo Trading', line=dict(color='#00d4aa', width=2)))
    fig_bt_star.add_trace(go.Scatter(x=s_preds['date'], y=cum_star_bah, name='Investissement Passif', line=dict(color='#76b900', width=2, dash='dash')))
    st.plotly_chart(style_fig(fig_bt_star, hauteur=350, titre=f"💰 Backtesting Exclusif : Stratégie vs Buy & Hold", titre_y='Rendement (Base 1)'), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 7 — EXPLAINABILITY SHAP
# ─────────────────────────────────────────────
elif page == '🔍 Forces de Marché (SHAP)':
    st.title('🔍 Décomposition du Signal Quantitatif (SHAP)')
    st.caption('Transparence algorithmique : Identifier les vecteurs qui dictent la probabilité.')
    st.divider()

    col_graph, col_text = st.columns([2, 1])
    with col_graph:
        top_n = st.slider('Périmètre d\'analyse', 5, len(shap_df), 15)
        shap_top = shap_df.head(top_n)
        fig_shap = go.Figure(go.Bar(x=shap_top['importance'][::-1], y=shap_top['feature'][::-1], orientation='h', marker=dict(color=shap_top['importance'][::-1], colorscale='Teal')))
        st.plotly_chart(style_fig(fig_shap, hauteur=max(400, top_n * 25), titre="Importance structurelle des variables (Top N)", titre_y='Indicateur Financier'), use_container_width=True)
    with col_text:
        add_explication("<b>Comment lire les valeurs SHAP ?</b><br><br>Plus la barre est étendue, plus la variable a un pouvoir discriminant dans la création du signal d'achat ou de vente.<br><br>Généralement, les <i>'Moyennes Mobiles (SMA)'</i> dominent car elles encadrent mathématiquement la notion de 'Tendance'.")

    st.divider()
    
    def get_category(feature):
        if '_level' in feature: return 'Macro (Niveaux)'
        elif '_change' in feature: return 'Macro (Variations)'
        elif feature in ['day_of_week', 'month', 'quarter']: return 'Saisonnalité'
        elif feature in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position']: return 'Oscillateurs'
        elif 'SMA' in feature or 'EMA' in feature or 'price_to' in feature: return 'Suivi de Tendance'
        elif 'return_lag' in feature or 'volume' in feature.lower() or 'volatility' in feature: return 'Dynamique Prix/Volume'
        else: return 'Autres'

    shap_df['category'] = shap_df['feature'].apply(get_category)
    cat_importance = shap_df.groupby('category')['importance'].sum().sort_values(ascending=False)

    fig_cat = px.pie(values=cat_importance.values, names=cat_importance.index, hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(style_fig(fig_cat, hauteur=400, titre="Poids par catégorie de données", type_x='cat'), use_container_width=True)

    st.divider()

    st.subheader('🔎 Explication locale d\'un signal quotidien')
    col1, col2 = st.columns([1, 3])
    with col1: sector_filter_shap = st.selectbox('Secteur (Analyse locale)', ['Tous'] + ALL_SECTORS, key='shap_sector')
    with col2: selected_ticker_shap = st.selectbox('Sous-jacent', ALL_TICKERS if sector_filter_shap == 'Tous' else sorted([t for t in ALL_TICKERS if SECTOR_MAP.get(t) == sector_filter_shap]), key='shap_ticker')

    @st.cache_resource
    def get_model_and_explainer():
        from xgboost import XGBClassifier
        import shap as shap_lib
        with open('data/feature_cols.txt', 'r') as f: feature_cols = [line.strip() for line in f.readlines()]
        train_df = features_df[features_df['date'] <= '2022-12-31']
        val_df   = features_df[(features_df['date'] > '2022-12-31') & (features_df['date'] <= '2023-12-31')]
        X_train, y_train = train_df[feature_cols], train_df['target_direction']
        X_val, y_val     = val_df[feature_cols], val_df['target_direction']

        model = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42, verbosity=0, early_stopping_rounds=50, eval_metric='logloss')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        explainer = shap_lib.TreeExplainer(model)
        return model, explainer, feature_cols

    with st.spinner('Chargement du moteur SHAP...'):
        xgb_model, explainer, FEATURE_COLS = get_model_and_explainer()

    stock_shap = features_df[features_df['ticker'] == selected_ticker_shap].sort_values('date')
    test_shap  = stock_shap[stock_shap['date'] > '2023-12-31']
    X_shap     = test_shap[FEATURE_COLS]

    dates_available = test_shap['date'].dt.strftime('%Y-%m-%d').tolist()
    selected_date   = st.selectbox('Sélection de la date (Période Test)', dates_available, index=len(dates_available)-1)

    date_idx    = dates_available.index(selected_date)
    X_day       = X_shap.iloc[[date_idx]]
    shap_values = explainer.shap_values(X_day)[0]
    proba_day   = xgb_model.predict_proba(X_day)[0][1]
    real_ret    = test_shap.iloc[date_idx]['target_return']
    real_dir    = test_shap.iloc[date_idx]['target_direction']

    col1, col2, col3 = st.columns(3)
    col1.metric('Probabilité calculée', f'{proba_day:.1%}')
    col2.metric('Mouvement réel du lendemain', f'{real_ret:.2%}')
    col3.metric('Pertinence du signal', '✅ Confirmé' if (proba_day > 0.5) == (real_dir == 1) else '❌ Erroné')

    shap_day_df = pd.DataFrame({'feature' : FEATURE_COLS, 'shap': shap_values, 'value': X_day.values[0]}).sort_values('shap', key=abs, ascending=False).head(15)
    shap_sorted = shap_day_df.sort_values('shap')
    colors_shap = ['#00d4aa' if x > 0 else '#ff4444' for x in shap_sorted['shap']]

    fig_waterfall = go.Figure(go.Bar(x=shap_sorted['shap'], y=shap_sorted['feature'], orientation='h', marker_color=colors_shap, text=[f'{v:.4f}' for v in shap_sorted['shap']], textposition='outside'))
    fig_waterfall.add_vline(x=0, line_color='white', line_width=1)
    st.plotly_chart(style_fig(fig_waterfall, hauteur=500, titre=f"Construction du signal — {selected_ticker_shap} au {selected_date}", titre_y="Forces acheteuses (Vert) / Vendeuses (Rouge)"), use_container_width=True)

    st.subheader('💬 Rapport Analytique Automatisé')
    top_pos = shap_day_df[shap_day_df['shap'] > 0].head(3)
    top_neg = shap_day_df[shap_day_df['shap'] < 0].head(3)
    direction  = 'Haussier' if proba_day > 0.5 else 'Baissier'
    
    st.markdown(f"**Diagnostic Algo :** Le modèle a émis un signal **{direction}** avec une probabilité de **{proba_day:.1%}**.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**✅ Catalyseurs Positifs :**')
        for _, row in top_pos.iterrows(): st.markdown(f"- `{row['feature']}` (Val: {row['value']:.3f}) → Impact: +{row['shap']:.4f}")
    with col2:
        st.markdown('**❌ Freins Baissiers :**')
        for _, row in top_neg.iterrows(): st.markdown(f"- `{row['feature']}` (Val: {row['value']:.3f}) → Impact: {row['shap']:.4f}")