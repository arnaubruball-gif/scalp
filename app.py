import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="JDetector- Institutional Edge", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #000000; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3d4463; }
    .bank-card { background-color: #0e1117; padding: 10px; border-left: 5px solid #ffd700; margin-bottom: 5px; font-size: 0.85rem; }
    .metric-box { background-color: #0e1117; padding: 15px; border-radius: 8px; border: 1px solid #30363d; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MOTOR DE CÁLCULO ---
def calcular_hurst(ts):
    if len(ts) < 30: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=600)
def analyze_asset(ticker):
    try:
        df = yf.download(ticker, period='15d', interval='1h', progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['Ret'] = df['Close'].pct_change()
        df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
        df['RMF'] = df['Close'] * df['Vol_Proxy']
        df['RVOL'] = df['Vol_Proxy'] / df['Vol_Proxy'].rolling(20).mean()
        
        r2_series = []
        for i in range(len(df)):
            if i < 30: r2_series.append(0); continue
            subset = df.iloc[i-30:i].dropna()
            r2 = sm.OLS(subset['Ret'], sm.add_constant(subset['RMF'])).fit().rsquared
            r2_series.append(r2)
        
        df['R2_Dynamic'] = r2_series
        diff = df['Ret'].rolling(40).sum() - df['RMF'].pct_change().rolling(40).sum()
        z_series = (diff - diff.rolling(40).mean()) / (diff.rolling(40).std() + 1e-10)
        z_val = z_series.iloc[-1]
        
        hurst = calcular_hurst(df['Close'].tail(50).values.flatten())
        
        return {
            'df': df, 'price': float(df['Close'].iloc[-1]), 'z': z_val, 
            'z_series': z_series, 'r2': df['R2_Dynamic'].iloc[-1], 
            'hurst': hurst, 'vol': df['Ret'].tail(30).std(), 'rvol': df['RVOL'].iloc[-1]
        }
    except: return None

# --- 3. LISTA DE ACTIVOS ---
ASSETS = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDJPY=X', 'USDCHF=X', 'GC=F', 'BTC-USD', '^GSPC']

# --- 4. PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ADN", "🎯 Ejecución Pro", "🎲 Montecarlo", "🌊 Vol-Monitor Pro", "🏦 Banks Detector", "🏛️ COT Insight"
])

with tab1:
    if st.button('📡 ESCANEO ADN'):
        results = []
        for t in ASSETS:
            d = analyze_asset(t)
            if d:
                status = "🚨 VENTA" if d['z'] > 1.6 else "🟢 COMPRA" if d['z'] < -1.6 else "⚪ NEUTRAL"
                results.append([t.replace('=X',''), d['price'], round(d['r2'],3), round(d['z'],2), round(d['hurst'],2), status])
        st.dataframe(pd.DataFrame(results, columns=['Activo', 'Precio', 'R2', 'Z-Diff', 'Hurst', 'Veredicto']), use_container_width=True)

with tab2:
    st.subheader("🎯 Auditoría de Señales Recientes")
    target_e = st.selectbox("Analizar historial de:", ASSETS, key="exec_s")
    de = analyze_asset(target_e)
    if de:
        df_h = de['df'].tail(5).copy()
        df_h['Z-Diff'] = de['z_series'].tail(5)
        df_h['ADN_Signal'] = df_h['Z-Diff'].apply(lambda x: "🟢 COMPRA" if x < -1.6 else ("🚨 VENTA" if x > 1.6 else "⚪ Neutral"))
        audit_df = df_h[['Close', 'Z-Diff', 'ADN_Signal']].copy()
        audit_df.index = audit_df.index.strftime('%Y-%m-%d')
        st.table(audit_df.style.format({'Close': '{:.5f}', 'Z-Diff': '{:.2f}'}))
        fig_z = px.line(de['z_series'].tail(15), title="Evolución ADN (15d)")
        fig_z.add_hline(y=1.6, line_dash="dash", line_color="red")
        fig_z.add_hline(y=-1.6, line_dash="dash", line_color="green")
        st.plotly_chart(fig_z.update_layout(template="plotly_dark", height=250), use_container_width=True)

with tab3:
    st.subheader("🎲 Montecarlo (Sesgo Institucional)")
    target_m = st.selectbox("Analizar Probabilidades:", ASSETS, key="mc_s")
    dm = analyze_asset(target_m)
    if dm:
        sims, dias = 1000, 30
        rets = np.random.normal(dm['df']['Ret'].mean(), dm['vol'], (sims, dias))
        caminos = dm['price'] * (1 + rets).cumprod(axis=1)
        
        z_score = dm['z']
        precios_finales = caminos[:, -1]
        
        if z_score <= 0:
            exitos = (precios_finales > dm['price']).sum()
            tesis, color = "ALCISTA 🟢", "#00ffcc"
        else:
            exitos = (precios_finales < dm['price']).sum()
            tesis, color = "BAJISTA 🚨", "#ff4b4b"
        
        prob = (exitos / sims) * 100
        st.markdown(f'<div style="background-color:{color}; padding:15px; border-radius:10px; color:black; text-align:center;"><b>TESIS ADN: {tesis} | PROBABILIDAD ÉXITO: {prob:.1f}%</b></div>', unsafe_allow_html=True)
        
        fig_m = go.Figure()
        for i in range(15): fig_m.add_trace(go.Scatter(y=caminos[i], line=dict(width=1), opacity=0.2, showlegend=False))
        fig_m.add_trace(go.Scatter(y=np.percentile(caminos, 50, axis=0), line=dict(color=color, width=4), name="Mediana"))
        st.plotly_chart(fig_m.update_layout(template="plotly_dark", height=350), use_container_width=True)

with tab4:
    st.subheader("🌊 Vol-Monitor Pro: Veredicto de Fuerza")
    target_v = st.selectbox("Activo Detalle Fuerza:", ASSETS, key="v_s")
    dv = analyze_asset(target_v)
    if dv:
        df_v = dv['df']
        # Kaufman ER
        n_er = 10
        change = abs(df_v['Close'] - df_v['Close'].shift(n_er))
        volat = abs(df_v['Close'] - df_v['Close'].shift(1)).rolling(n_er).sum()
        er = (change / (volat + 1e-10)).iloc[-1]
        # ADX Básico
        plus_dm = df_v['High'].diff()
        minus_dm = df_v['Low'].diff()
        tr = pd.concat([df_v['High']-df_v['Low'], abs(df_v['High']-df_v['Close'].shift(1)), abs(df_v['Low']-df_v['Close'].shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        dx = 100 * abs((plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()) / (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean() + 1e-10))
        adx = dx.rolling(14).mean().iloc[-1]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-box"><b>RVOL</b><br><h2>{dv["rvol"]:.2f}x</h2></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-box"><b>ER (Eficiencia)</b><br><h2>{er:.2f}</h2></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-box"><b>ADX (Fuerza)</b><br><h2>{adx:.1f}</h2></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-box"><b>ROC</b><br><h2>{df_v["Ret"].iloc[-1]*100:+.2f}%</h2></div>', unsafe_allow_html=True)
        
        puntos = sum([dv['rvol'] > 1.5, er > 0.55, adx > 25, abs(df_v['Ret'].iloc[-1]*100) > 0.5])
        v_color = "#00ffcc" if puntos >= 3 else "#ffd700" if puntos == 2 else "#ff4b4b"
        st.markdown(f'<div style="background-color:{v_color}; padding:20px; border-radius:10px; color:black; text-align:center; font-weight:bold; margin-top:15px;">VEREDICTO: {"ALTA CALIDAD" if puntos >= 3 else "CALIDAD MEDIA" if puntos == 2 else "RUIDO ESTRUCTURAL"} ({puntos}/4)</div>', unsafe_allow_html=True)

with tab5:
    st.subheader("🏦 Banks Detector")
    target_b = st.selectbox("Shadow RMF:", ASSETS, key="b_s")
    db = analyze_asset(target_b)
    if db:
        df_b = db['df'].copy()
        df_b['Anom'] = df_b['RMF'].abs() / df_b['RMF'].abs().rolling(20).mean()
        clrs = ['#ffd700' if x > 2.5 else '#3d4463' for x in df_b['Anom']]
        st.plotly_chart(go.Figure(data=[go.Bar(x=df_b.index, y=df_b['RMF'].abs(), marker_color=clrs)]).update_layout(template="plotly_dark", height=400), use_container_width=True)

with tab6:
    st.subheader("🏛️ COT Insight: Institutional Sentiment (10 Weeks)")
    
    # 1. Base de datos histórica (Simulando las últimas 10 semanas)
    # En un entorno real, estos datos vendrían de un CSV o API de la CFTC
    cot_trend_db = {
        'USD (Dólar Index)': [22000, 24500, 21000, 28000, 31000, 29500, 33000, 35000, 32000, 36000],
        'EUR (Euro)': [140000, 135000, 120000, 115000, 110000, 105000, 98000, 90000, 85000, 82000],
        'GBP (Libra)': [-35000, -38000, -42000, -45000, -40000, -35000, -30000, -25000, -22000, -18000],
        'JPY (Yen)': [-110000, -115000, -120000, -125000, -130000, -135000, -140000, -145000, -142000, -148000],
        'BTC (Bitcoin)': [4200, 4800, 5100, 5500, 6200, 7000, 7800, 8200, 8500, 9100],
        'GC=F (Oro)': [180000, 185000, 192000, 200000, 215000, 210000, 225000, 230000, 235000, 242000]
    }

    selected_asset = st.selectbox("Seleccionar Activo Institucional:", list(cot_trend_db.keys()))
    
    # Extraer datos del activo seleccionado
    hist_data = cot_trend_db[selected_asset]
    net_actual = hist_data[-1]
    net_prev = hist_data[-2]
    cambio = net_actual - net_prev

    # 2. Layout de Métricas Superiores
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        st.metric("Net Positioning Actual", f"{net_actual:+,}", delta=f"{cambio:+,} vs prev")
    
    with c2:
        # Lógica de Bias Institucional
        if net_actual > 50000: bias = "EXTREME BULLISH 🚀"
        elif net_actual > 0: bias = "BULLISH 🟢"
        elif net_actual < -50000: bias = "EXTREME BEARISH 📉"
        else: bias = "BEARISH 🚨"
        st.write(f"**Bias Institucional:** \n### {bias}")

    with c3:
        # Gráfico de Indicador de Sentimiento (Gauge)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = net_actual,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fuerza Neta (Contratos)"},
            gauge = {
                'axis': {'range': [min(hist_data)-10000, max(hist_data)+10000]},
                'bar': {'color': "#00ffcc" if net_actual > 0 else "#ff4b4b"},
                'steps': [
                    {'range': [min(hist_data)-10000, 0], 'color': "#3d0000"},
                    {'range': [0, max(hist_data)+10000], 'color': "#002b2b"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge.update_layout(height=250, margin=dict(t=50, b=0), template="plotly_dark"), use_container_width=True)

    # 3. Gráfico de Tendencia de 10 Semanas
    st.markdown("### 📈 Evolución del Posicionamiento (Últimas 10 Semanas)")
    
    fig_trend = go.Figure()
    
    # Añadimos la línea de tendencia
    fig_trend.add_trace(go.Scatter(
        x=[f"W-{i}" for i in range(10, 0, -1)], 
        y=hist_data,
        mode='lines+markers',
        name='Net Position',
        line=dict(color='#00ffcc' if net_actual > 0 else '#ff4b4b', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 204, 0.1)' if net_actual > 0 else 'rgba(255, 75, 75, 0.1)'
    ))

    # Línea de equilibrio (cero)
    fig_trend.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig_trend.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Semanas Atrás",
        yaxis_title="Contratos Netos",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

    # 4. Interpretación Smart Money
    st.markdown("""
    <div style="background-color:#1c1c1c; padding:15px; border-radius:10px; border:1px solid #ffd700;">
        <b>💡 Análisis de Flujo:</b> Si el gráfico muestra una pendiente ascendente mientras el precio baja, estamos ante una 
        <b>Divergencia Institucional</b>. El Smart Money está acumulando contratos antes del giro del mercado.
    </div>
    """, unsafe_allow_html=True)
