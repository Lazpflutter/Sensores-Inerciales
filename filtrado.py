import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
import matplotlib.pyplot as plt
import os
import re

# -----------------------------
# ⚙️ FUNCIONES DE FILTRO
# -----------------------------
def hampel_filter(series, window_size=10, n_sigmas=3):
    s = series.copy()
    n = len(s)
    new_series = s.copy()
    k = 1.4826  # factor MAD normalizado

    for i in range(n):
        start = max(i - window_size, 0)
        end = min(i + window_size, n)
        window = s[start:end]
        median = np.median(window)
        mad = k * np.median(np.abs(window - median))
        if mad == 0:
            continue
        diff = np.abs(s[i] - median)
        if diff > n_sigmas * mad:
            new_series[i] = median
    return new_series

def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y

# -----------------------------
# ⚙️ CONFIGURACIÓN
# -----------------------------
fs = 100
fc_butter = 15.0
orden_butter = 4
ventana_hampel = int(0.2 * fs)
n_sigmas_hampel = 3
ventana_sg = 11  # debe ser impar
orden_sg = 3

carpeta_entrada = "3000datos"
carpeta_salida = "Filtrados"
os.makedirs(carpeta_salida, exist_ok=True)

# -----------------------------
# 🔄 RECORRER TODAS LAS CARPETAS DE SUJETOS
# -----------------------------
for sujeto in sorted(os.listdir(carpeta_entrada)):
    ruta_sujeto = os.path.join(carpeta_entrada, sujeto)
    if not os.path.isdir(ruta_sujeto):
        continue

    for archivo in sorted(os.listdir(ruta_sujeto)):
        if not archivo.endswith(".csv"):
            continue
        
        ruta_archivo = os.path.join(ruta_sujeto, archivo)
        print(f"Procesando: {ruta_archivo}")

        # Leer CSV y limitar a 3000 filas
        df = pd.read_csv(ruta_archivo).head(3000)

        # Extraer clase del nombre de archivo
        match = re.search(r'_(Caminar|Correr|Quieto)_', archivo)
        clase = match.group(1) if match else "Desconocida"

        # Columnas a filtrar
        cols_acel = ['Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)']
        cols_gyro = ['Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)']
        cols_ang  = ['Angle X(°)', 'Angle Y(°)', 'Angle Z(°)']

        # Crear columna de tiempo en milisegundos (0 a 30000)
        tiempo_ms = np.linspace(0, 30000, len(df))
        df_filtrado = pd.DataFrame({'Time (ms)': tiempo_ms})

        # Aplicar pipeline a cada columna
        for col in cols_acel + cols_gyro + cols_ang:
            serie = pd.to_numeric(df[col], errors='coerce')
            serie = serie.fillna(method='ffill').fillna(method='bfill')

            temp = hampel_filter(serie, window_size=ventana_hampel, n_sigmas=n_sigmas_hampel)
            temp = butterworth_filter(temp, cutoff=fc_butter, fs=fs, order=orden_butter)
            temp = savgol_filter(temp, window_length=ventana_sg, polyorder=orden_sg)
            df_filtrado[col] = temp

        # Carpeta de salida por sujeto y clase
        carpeta_salida_sujeto = os.path.join(carpeta_salida, sujeto, clase)
        os.makedirs(carpeta_salida_sujeto, exist_ok=True)

        nombre_salida = os.path.join(carpeta_salida_sujeto, archivo.replace(".csv","_pipeline.csv"))
        df_filtrado.to_csv(nombre_salida, index=False)
        print(f"✅ Guardado: {nombre_salida}")

        # Graficar ejemplo solo para el primer archivo del primer sujeto
        if sujeto == "S01" and clase == "Caminar" and archivo.endswith("_1.csv"):
            col_ejemplo = 'Acceleration X(g)'
            plt.figure(figsize=(10,4))
            plt.plot(tiempo_ms/1000, df[col_ejemplo], label='Original', alpha=0.6)
            plt.plot(tiempo_ms/1000, df_filtrado[col_ejemplo], label='Filtrado final', linewidth=2)
            plt.title(f'{archivo} - {col_ejemplo}')
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Aceleración (g)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
