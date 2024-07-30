import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
np.float_ = np.float64
import matplotlib.pyplot as plt
import plotly
from prophet import Prophet
from io import BytesIO
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Data'] = pd.to_datetime(df['Data'])
    return df

def aggregate_data(df, period):
    df_agg = df.groupby(['Data', 'Articolo'])['Vendite'].sum().reset_index()
    if period == 'Settimanale':
        df_agg['Data'] = df_agg['Data'].dt.to_period('W').dt.start_time
    elif period == 'Mensile':
        df_agg['Data'] = df_agg['Data'].dt.to_period('M').dt.start_time
    df_agg = df_agg.groupby(['Data', 'Articolo'])['Vendite'].sum().reset_index()
    return df_agg

def prepare_data_for_prophet(df):
    df_prophet = df.rename(columns={'Data': 'ds', 'Vendite': 'y'})
    return df_prophet

def train_and_forecast(df, periods):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D', include_history=True)
    forecast = model.predict(future)
    return forecast

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mape, rmse

st.title("App di Previsione Vendite")

# Carica il modello CSV
st.header("1. Carica il modello CSV")
model_csv = pd.DataFrame({'Data': ['YYYY-MM-DD'], 'Articolo': ['Nome Articolo'], 'Vendite': [0]})
csv_model = model_csv.to_csv(index=False)
st.download_button(
    label="Scarica modello CSV",
    data=csv_model,
    file_name="modello_vendite.csv",
    mime="text/csv"
)

# Carica i dati
uploaded_file = st.file_uploader("Carica il file CSV con i dati delle vendite", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    # Format date
    df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d')
    st.write("Dati caricati con successo!")

    # Scegli il range di previsione
    st.header("2. Scegli il range di previsione")
    period = st.selectbox("Seleziona il periodo di aggregazione", ["Giornaliero", "Settimanale", "Mensile"])

    # Aggrega i dati
    df_agg = aggregate_data(df, period)

    # Menu a tendina per la visualizzazione del grafico
    st.header("3. Visualizzazione dei dati")
    visualization_option = st.selectbox(
        "Seleziona l'articolo da visualizzare",
        ["Dati aggregati"] + list(df_agg['Articolo'].unique())
    )

    if visualization_option == "Dati aggregati":
        df_plot = df_agg.groupby('Data')['Vendite'].sum().reset_index()
        title = "Vendite Aggregate"
    else:
        df_plot = df_agg[df_agg['Articolo'] == visualization_option]
        title = f"Vendite per {visualization_option}"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_plot['Data'], df_plot['Vendite'])
    ax.set_title(title)
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendite")
    st.pyplot(fig)

    # Scegli il numero di periodi per la previsione
    st.header("4. Previsione")
    max_periods = 52 if period == 'Settimanale' else 12 if period == 'Mensile' else 365
    forecast_periods = st.slider("Seleziona il numero di periodi per la previsione", 1, max_periods, 1)

    # Selezione della metrica per l'intervallo di confidenza
    confidence_metric = st.selectbox(
        "Seleziona la metrica per l'intervallo di confidenza",
        ["MAPE", "RMSE"], help="""
        - MAPE (Mean Absolute Percentage Error): Misura la dimensione dell'errore in termini percentuali. Un valore più basso indica una previsione migliore.
        - RMSE (Root Mean Square Error): Rappresenta la deviazione standard dei residui (errori di previsione). Un valore più basso indica una previsione migliore."""
    )

    if st.button("Genera previsione"):
        # Preparazione dei dati e previsione per ogni articolo
        forecasts = {}
        for article in df_agg['Articolo'].unique():
            df_art = df_agg[df_agg['Articolo'] == article]
            df_prophet = prepare_data_for_prophet(df_art)
            forecast = train_and_forecast(df_prophet, forecast_periods)
            forecasts[article] = forecast

        # Crea una tabella con le previsioni per tutti gli articoli
        st.subheader("Previsioni per tutti gli articoli")
        forecast_table = pd.DataFrame(columns=['Articolo'] + [f'Previsione Periodo {i+1}' for i in range(forecast_periods)])
        for article, forecast in forecasts.items():
            row = [article] + list(forecast['yhat'].tail(forecast_periods))
            forecast_table.loc[len(forecast_table)] = row
        st.dataframe(forecast_table)

        # Genera CSV di output
        output = BytesIO()
        forecast_table.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        st.download_button(
            label="Scarica previsione CSV",
            data=output,
            file_name="previsione_vendite.csv",
            mime="text/csv"
        )

        # Grafico con dati storici e previsione aggregata
        st.subheader("Grafico dati storici e previsione aggregata")
        df_agg_total = df_agg.groupby('Data')['Vendite'].sum().reset_index()
        df_prophet_total = prepare_data_for_prophet(df_agg_total)
        forecast_total = train_and_forecast(df_prophet_total, forecast_periods)

        # Calcolo delle metriche di affidabilità
        historical_data = forecast_total[forecast_total['ds'] <= df_agg_total['Data'].max()]
        r2 = r2_score(df_agg_total['Vendite'], historical_data['yhat'])
        mape = mean_absolute_percentage_error(df_agg_total['Vendite'], historical_data['yhat'])
        rmse = np.sqrt(mean_squared_error(df_agg_total['Vendite'], historical_data['yhat']))

        # Visualizzazione delle metriche
        st.subheader("Statistiche di affidabilità della previsione")
        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared", f"{r2:.3f}")
        col2.metric("MAPE", f"{mape:.3f}")
        col3.metric("RMSE", f"{rmse:.3f}")

        # Spiegazione delle metriche
        st.write("""
        - R-squared: Indica la proporzione della varianza nella variabile dipendente che è prevedibile dalla variabile indipendente. Varia da 0 a 1, dove 1 indica una previsione perfetta.
        - MAPE (Mean Absolute Percentage Error): Misura la dimensione dell'errore in termini percentuali. Un valore più basso indica una previsione migliore.
        - RMSE (Root Mean Square Error): Rappresenta la deviazione standard dei residui (errori di previsione). Un valore più basso indica una previsione migliore.
        """)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Trova l'ultima data storica
        last_historical_date = df_agg_total['Data'].max()
        
        # Crea una maschera per i dati storici e le previsioni
        historical_mask = forecast_total['ds'] <= last_historical_date
        forecast_mask = forecast_total['ds'] > last_historical_date

        # Plotta i dati storici in blu
        ax.plot(forecast_total.loc[historical_mask, 'ds'], 
                forecast_total.loc[historical_mask, 'yhat'], 
                color='blue', label='Dati storici')

        # Plotta le previsioni in rosso
        ax.plot(forecast_total.loc[forecast_mask, 'ds'], 
                forecast_total.loc[forecast_mask, 'yhat'], 
                color='red', label='Previsione')

        # Calcola l'intervallo di confidenza basato sulla metrica selezionata
        if confidence_metric == "MAPE":
            forecast_total['yhat_lower'] = forecast_total['yhat'] * (1 - mape)
            forecast_total['yhat_upper'] = forecast_total['yhat'] * (1 + mape)
            confidence_label = f"Intervallo di confidenza (±{mape:.1%})"
        else:  # RMSE
            forecast_total['yhat_lower'] = forecast_total['yhat'] - rmse
            forecast_total['yhat_upper'] = forecast_total['yhat'] + rmse
            confidence_label = f"Intervallo di confidenza (±{rmse:.0f})"

        # Intervallo di confidenza solo per la parte di previsione
        ax.fill_between(forecast_total.loc[forecast_mask, 'ds'], 
                        forecast_total.loc[forecast_mask, 'yhat_lower'], 
                        forecast_total.loc[forecast_mask, 'yhat_upper'], 
                        color='red', alpha=0.2, label=confidence_label)
        
        ax.set_title(f"Dati storici e previsione aggregata con intervallo di confidenza basato su {confidence_metric}")
        ax.set_xlabel("Data")
        ax.set_ylabel("Vendite")
        ax.legend()
        
        # Formattazione dell'asse x per una migliore leggibilità
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)