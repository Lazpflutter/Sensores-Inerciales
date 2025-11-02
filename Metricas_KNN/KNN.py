import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -------------------------------
# ⚙️ CONFIGURACIÓN
# -------------------------------
archivo = "Todas_caracteristicas.csv"
carpeta_salida = "Metricas_KNN"
k_folds = 5
k_vecinos = 5  # Número de vecinos K para el modelo

# Crear carpeta si no existe
os.makedirs(carpeta_salida, exist_ok=True)

caracteristicas = [
    "Acceleration X(g)_var",
    "Acceleration X(g)_std",
    "Acceleration Z(g)_var",
    "Acceleration Z(g)_std",
    "Angular velocity Y(°/s)_rms",
    "Acceleration X(g)_rms",
    "Acceleration Z(g)_sma",
    "Acceleration Z(g)_mean",
    "Acceleration Y(g)_sma",
    "Acceleration Z(g)_rms",
    "Acceleration X(g)_sma",
    "Angular velocity Y(°/s)_var",
    "Angular velocity Y(°/s)_std"
]

# -------------------------------
# 📂 CARGA DE DATOS
# -------------------------------
df = pd.read_csv(archivo)
X = df[caracteristicas]
y = df["Clase"].values
labels = np.unique(y)

# -------------------------------
# 🔄 VALIDACIÓN CRUZADA
# -------------------------------
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

metricas_fold = []
metricas_clase = []
cm_total = np.zeros((len(labels), len(labels)), dtype=int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"🔹 Entrenando Fold {fold}/{k_folds}...")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo KNN
    knn = KNeighborsClassifier(n_neighbors=k_vecinos)
    knn.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = knn.predict(X_test_scaled)

    # -------------------------------
    # 📊 MÉTRICAS GLOBALES
    # -------------------------------
    acc = accuracy_score(y_test, y_pred)
    ppv = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metricas_fold.append({
        "Fold": fold,
        "ACC": acc,
        "PPV": ppv,
        "REC": rec,
        "F1": f1
    })

    # -------------------------------
    # 📈 MÉTRICAS POR CLASE
    # -------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_total += cm
    precisiones = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    recalls = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    f1s = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    for i, clase in enumerate(labels):
        metricas_clase.append({
            "Fold": fold,
            "Clase": clase,
            "Precision": precisiones[i],
            "Recall": recalls[i],
            "F1": f1s[i]
        })

    # -------------------------------
    # 🖼️ MATRIZ DE CONFUSIÓN POR FOLD
    # -------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - Fold {fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, f"matriz_confusion_fold_{fold}.png"), dpi=300)
    plt.close()

# -------------------------------
# 📊 PROMEDIO GLOBAL
# -------------------------------
df_metricas = pd.DataFrame(metricas_fold)
df_promedio = df_metricas.mean(numeric_only=True).to_dict()
df_promedio["Fold"] = "Promedio"
df_metricas = pd.concat([df_metricas, pd.DataFrame([df_promedio])], ignore_index=True)

# Guardar métricas globales
df_metricas.to_csv(os.path.join(carpeta_salida, "Metricas_globales_KNN.csv"), index=False)
print("✅ Métricas globales guardadas en 'Metricas_globales_KNN.csv'")

# -------------------------------
# 📊 MÉTRICAS POR CLASE
# -------------------------------
df_metricas_clase = pd.DataFrame(metricas_clase)
df_prom_clase = df_metricas_clase.groupby("Clase").mean(numeric_only=True).reset_index()
df_prom_clase["Fold"] = "Promedio"
df_metricas_clase = pd.concat([df_metricas_clase, df_prom_clase], ignore_index=True)

# Guardar métricas por clase
df_metricas_clase.to_csv(os.path.join(carpeta_salida, "Metricas_por_clase_KNN.csv"), index=False)
print("✅ Métricas por clase guardadas en 'Metricas_por_clase_KNN.csv'")

# -------------------------------
# 🖼️ MATRIZ DE CONFUSIÓN PROMEDIO
# -------------------------------
cm_promedio = cm_total / k_folds
plt.figure(figsize=(6, 5))
sns.heatmap(cm_promedio, annot=True, fmt=".1f", cmap="Greens",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de Confusión Promedio ({k_folds}-fold CV)")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "matriz_confusion_promedio_KNN.png"), dpi=300)
plt.close()

print(f"\n✅ Todas las métricas y figuras fueron guardadas en la carpeta '{carpeta_salida}'")
