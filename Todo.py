import os
import pandas as pd

# Carpeta raíz donde están tus archivos de características
carpeta_entrada = "Caracteristicas"
archivo_salida = "Todas_caracteristicas.csv"

# Lista para guardar todos los DataFrames
todos_dfs = []

# Recorrer carpetas y subcarpetas
for root, dirs, files in os.walk(carpeta_entrada):
    for file in files:
        if file.endswith(".csv"):
            ruta_archivo = os.path.join(root, file)
            df = pd.read_csv(ruta_archivo)
            
            # Extraer clase y sujeto de la ruta o archivo
            if "_Quieto_" in file:
                clase = "Quieto"
            elif "_Caminar_" in file:
                clase = "Caminar"
            elif "_Correr_" in file:
                clase = "Correr"
            else:
                clase = "Desconocida"
            
            sujeto = os.path.basename(root)
            
            # Añadir columnas de referencia
            df["Clase"] = clase
            df["Sujeto"] = sujeto
            
            todos_dfs.append(df)

# Concatenar todos los DataFrames
df_final = pd.concat(todos_dfs, ignore_index=True)

# Ordenar primero por Clase y luego por Sujeto
orden_clases = {"Quieto": 0, "Caminar": 1, "Correr": 2, "Desconocida": 3}
df_final["Clase_orden"] = df_final["Clase"].map(orden_clases)
df_final = df_final.sort_values(by=["Clase_orden", "Sujeto"]).drop(columns=["Clase_orden"])

# Guardar en un solo CSV
df_final.to_csv(archivo_salida, index=False)
print(f"✅ Todos los archivos combinados y ordenados en: {archivo_salida}")
print(f"Filas totales: {len(df_final)}")
