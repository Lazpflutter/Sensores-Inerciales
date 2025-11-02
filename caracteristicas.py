import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy

# -----------------------------
# ⚙️ CONFIGURACIÓN
# -----------------------------
carpeta_entrada = "Filtrados"
carpeta_salida = "Caracteristicas"
os.makedirs(carpeta_salida, exist_ok=True)

fs = 100               # Frecuencia de muestreo [Hz]
ventana_seg = 2.56     # Duración de ventana [s]
n_muestras = int(fs * ventana_seg)  # 512 muestras
solapamiento = 0.5
paso = int(n_muestras * (1 - solapamiento))  # 256 muestras

# -----------------------------
# 🔍 FUNCIÓN PARA EXTRAER FEATURES DE UNA SEÑAL
# -----------------------------
def extraer_caracteristicas(signal, fs):
    signal = np.array(signal)

    # --- Dominio del tiempo ---
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    sma = np.sum(np.abs(signal)) / len(signal)

    # --- Dominio de la frecuencia ---
    f, Pxx = welch(signal, fs=fs)
    Pxx_norm = Pxx / np.sum(Pxx)
    freq_dom = f[np.argmax(Pxx)]
    spec_entropy = entropy(Pxx_norm + 1e-12)  # evitar log(0)

    return {
        'mean': mean,
        'std': std,
        'var': var,
        'rms': rms,
        'sma': sma,
        'freq_dom': freq_dom,
        'spec_entropy': spec_entropy
    }

# -----------------------------
# 🔄 PROCESAR TODOS LOS ARCHIVOS
# -----------------------------
def procesar_archivo(ruta_archivo, salida_csv):
    df = pd.read_csv(ruta_archivo)

    # Columnas relevantes (ajusta si tus CSV usan otros nombres)
    columnas = [
        'Acceleration X(g)', 'Acceleration Y(g)', 'Acceleration Z(g)',
        'Angular velocity X(°/s)', 'Angular velocity Y(°/s)', 'Angular velocity Z(°/s)'
    ]

    # Crear DataFrame para características
    features_list = []

    for inicio in range(0, len(df) - n_muestras + 1, paso):
        fin = inicio + n_muestras
        ventana = df.iloc[inicio:fin]

        caracteristicas_ventana = {}

        # Calcular características por canal
        for col in columnas:
            carac = extraer_caracteristicas(ventana[col], fs)
            for k, v in carac.items():
                caracteristicas_ventana[f"{col}_{k}"] = v

        # (Opcional) incluir etiqueta de clase si está en la ruta
        if "_Caminar_" in ruta_archivo:
            caracteristicas_ventana["Clase"] = "Caminar"
        elif "_Correr_" in ruta_archivo:
            caracteristicas_ventana["Clase"] = "Correr"
        elif "_Quieto_" in ruta_archivo:
            caracteristicas_ventana["Clase"] = "Quieto"
        else:
            caracteristicas_ventana["Clase"] = "Desconocida"

        features_list.append(caracteristicas_ventana)

    df_features = pd.DataFrame(features_list)
    df_features.to_csv(salida_csv, index=False)
    print(f"✅ Características guardadas en: {salida_csv}")

# -----------------------------
# 🚀 RECORRER TODA LA CARPETA
# -----------------------------
for sujeto in sorted(os.listdir(carpeta_entrada)):
    ruta_sujeto = os.path.join(carpeta_entrada, sujeto)
    if not os.path.isdir(ruta_sujeto):
        continue

    for clase in sorted(os.listdir(ruta_sujeto)):
        ruta_clase = os.path.join(ruta_sujeto, clase)
        if not os.path.isdir(ruta_clase):
            continue

        for archivo in sorted(os.listdir(ruta_clase)):
            if not archivo.endswith("_pipeline.csv"):
                continue

            ruta_archivo = os.path.join(ruta_clase, archivo)

            carpeta_salida_sujeto = os.path.join(carpeta_salida, sujeto)
            os.makedirs(carpeta_salida_sujeto, exist_ok=True)
            nombre_salida = archivo.replace("_pipeline.csv", "_features.csv")
            salida_csv = os.path.join(carpeta_salida_sujeto, nombre_salida)

            procesar_archivo(ruta_archivo, salida_csv)
