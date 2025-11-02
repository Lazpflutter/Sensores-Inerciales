import os
import pandas as pd

# -------------------------------
# ⚙️ CONFIGURACIÓN
# -------------------------------
carpeta_entrada = "Caracteristicas_Datos"
archivo_salida = "Matriz_Caracteristicas.csv"

# Carpetas destino
carpeta_SVM = "Metricas_SVM"
carpeta_KNN = "Metricas_KNN"

# Crear las carpetas si no existen
os.makedirs(carpeta_SVM, exist_ok=True)
os.makedirs(carpeta_KNN, exist_ok=True)

# -------------------------------
# 📂 UNIFICAR LOS ARCHIVOS
# -------------------------------
todos_dfs = []

for root, dirs, files in os.walk(carpeta_entrada):
    for file in files:
        if file.endswith(".csv"):
            ruta_archivo = os.path.join(root, file)
            df = pd.read_csv(ruta_archivo)
            
            # Detectar clase según el nombre del archivo
            if "_Quieto_" in file:
                clase = "Quieto"
            elif "_Caminar_" in file:
                clase = "Caminar"
            elif "_Correr_" in file:
                clase = "Correr"
            else:
                clase = "Desconocida"
            
            sujeto = os.path.basename(root)
            
            df["Clase"] = clase
            df["Sujeto"] = sujeto
            
            todos_dfs.append(df)

# Concatenar y ordenar
df_final = pd.concat(todos_dfs, ignore_index=True)
orden_clases = {"Quieto": 0, "Caminar": 1, "Correr": 2, "Desconocida": 3}
df_final["Clase_orden"] = df_final["Clase"].map(orden_clases)
df_final = df_final.sort_values(by=["Clase_orden", "Sujeto"]).drop(columns=["Clase_orden"])

# -------------------------------
# 💾 GUARDAR EN MÚLTIPLES CARPETAS
# -------------------------------
# Guardar en la carpeta principal (opcional)
df_final.to_csv(archivo_salida, index=False)

# Guardar copias en Metricas_SVM y Metricas_KNN
ruta_svm = os.path.join(carpeta_SVM, archivo_salida)
ruta_knn = os.path.join(carpeta_KNN, archivo_salida)

df_final.to_csv(ruta_svm, index=False)
df_final.to_csv(ruta_knn, index=False)

print(f"✅ Archivo combinado guardado en:")
print(f" - {os.path.abspath(archivo_salida)}")
print(f" - {os.path.abspath(ruta_svm)}")
print(f" - {os.path.abspath(ruta_knn)}")
print(f"\n📊 Filas totales: {len(df_final)} | Columnas: {len(df_final.columns)}")
