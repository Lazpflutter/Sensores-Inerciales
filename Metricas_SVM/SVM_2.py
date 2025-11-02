import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# ⚙️ CONFIGURACIÓN
# -------------------------------
archivo = "Todas_caracteristicas.csv"
salida_metricas = "Metricas_SVM_CV_completo.csv"
k_folds = 5  # número de folds para la validación cruzada

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
metricas_por_clase = []
cm_total = np.zeros((len(labels), len(labels)), dtype=int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n📘 Fold {fold}/{k_folds}")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento SVM
    svm = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = svm.predict(X_test_scaled)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_total += cm

    # -------------------------------
    # 🔢 Métricas globales
    # -------------------------------
    acc = accuracy_score(y_test, y_pred)
    ppv = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metricas_fold.append({
        "Fold": fold,
        "ACC": acc,
        "PPV": ppv,
        "NPV": np.nan,
        "REC": rec,
        "SPEC": np.nan,
        "F1": f1,
        "Precision": ppv
    })

    # -------------------------------
    # 📈 Métricas por clase
    # -------------------------------
    precision_cls = precision_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    recall_cls = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    f1_cls = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)

    for i, label in enumerate(labels):
        metricas_por_clase.append({
            "Fold": fold,
            "Clase": label,
            "Precision": precision_cls[i],
            "Recall": recall_cls[i],
            "F1": f1_cls[i]
        })

    # -------------------------------
    # 🖼️ Guardar matriz de confusión individual
    # -------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - Fold {fold}")
    plt.tight_layout()
    plt.savefig(f"matriz_confusion_fold_{fold}.png", dpi=300)
    plt.close()
    print(f"✅ Matriz de confusión del fold {fold} guardada.")

# -------------------------------
# 📊 PROMEDIO GLOBAL
# -------------------------------
df_metricas = pd.DataFrame(metricas_fold)
df_promedio = df_metricas.mean(numeric_only=True).to_dict()
df_promedio["Fold"] = "Promedio"
df_metricas = pd.concat([df_metricas, pd.DataFrame([df_promedio])], ignore_index=True)

# Promedio por clase
df_metricas_clase = pd.DataFrame(metricas_por_clase)
df_promedio_clase = df_metricas_clase.groupby("Clase").mean(numeric_only=True).reset_index()
df_promedio_clase["Fold"] = "Promedio"

# -------------------------------
# 💾 GUARDAR RESULTADOS
# -------------------------------
df_final = pd.concat([df_metricas_clase, df_promedio_clase], ignore_index=True)
df_final.to_csv(salida_metricas, index=False)
print(f"\n✅ Métricas por clase y globales guardadas en '{salida_metricas}'")

# -------------------------------
# 🖼️ MATRIZ DE CONFUSIÓN PROMEDIO
# -------------------------------
cm_promedio = cm_total / k_folds
plt.figure(figsize=(6, 5))
sns.heatmap(cm_promedio, annot=True, fmt=".1f", cmap="Purples",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de Confusión Promedio ({k_folds}-fold CV)")
plt.tight_layout()
plt.savefig("matriz_confusion_promedio.png", dpi=300)
plt.show()

print("✅ Matriz de confusión promedio guardada como 'matriz_confusion_promedio.png'")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# ⚙️ CONFIGURACIÓN
# -------------------------------
carpeta_salida = "Metricas_SVM"
os.makedirs(carpeta_salida, exist_ok=True)  # Crear carpeta si no existe

archivo = "Todas_caracteristicas.csv"
salida_metricas = os.path.join(carpeta_salida, "Metricas_SVM_CV_completo.csv")
k_folds = 5  # número de folds para la validación cruzada

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
metricas_por_clase = []
cm_total = np.zeros((len(labels), len(labels)), dtype=int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n📘 Fold {fold}/{k_folds}")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento SVM
    svm = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = svm.predict(X_test_scaled)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_total += cm

    # -------------------------------
    # 🔢 Métricas globales
    # -------------------------------
    acc = accuracy_score(y_test, y_pred)
    ppv = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metricas_fold.append({
        "Fold": fold,
        "ACC": acc,
        "PPV": ppv,
        "NPV": np.nan,
        "REC": rec,
        "SPEC": np.nan,
        "F1": f1,
        "Precision": ppv
    })

    # -------------------------------
    # 📈 Métricas por clase
    # -------------------------------
    precision_cls = precision_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    recall_cls = recall_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    f1_cls = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)

    for i, label in enumerate(labels):
        metricas_por_clase.append({
            "Fold": fold,
            "Clase": label,
            "Precision": precision_cls[i],
            "Recall": recall_cls[i],
            "F1": f1_cls[i]
        })

    # -------------------------------
    # 🖼️ Guardar matriz de confusión individual
    # -------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - Fold {fold}")
    plt.tight_layout()
    ruta_fig = os.path.join(carpeta_salida, f"matriz_confusion_fold_{fold}.png")
    plt.savefig(ruta_fig, dpi=300)
    plt.close()
    print(f"✅ Matriz de confusión del fold {fold} guardada en {ruta_fig}")

# -------------------------------
# 📊 PROMEDIO GLOBAL
# -------------------------------
df_metricas = pd.DataFrame(metricas_fold)
df_promedio = df_metricas.mean(numeric_only=True).to_dict()
df_promedio["Fold"] = "Promedio"
df_metricas = pd.concat([df_metricas, pd.DataFrame([df_promedio])], ignore_index=True)

# Promedio por clase
df_metricas_clase = pd.DataFrame(metricas_por_clase)
df_promedio_clase = df_metricas_clase.groupby("Clase").mean(numeric_only=True).reset_index()
df_promedio_clase["Fold"] = "Promedio"

# -------------------------------
# 💾 GUARDAR RESULTADOS
# -------------------------------
df_final = pd.concat([df_metricas_clase, df_promedio_clase], ignore_index=True)
df_final.to_csv(salida_metricas, index=False)
print(f"\n✅ Métricas por clase y globales guardadas en '{salida_metricas}'")

# -------------------------------
# 🖼️ MATRIZ DE CONFUSIÓN PROMEDIO
# -------------------------------
cm_promedio = cm_total / k_folds
plt.figure(figsize=(6, 5))
sns.heatmap(cm_promedio, annot=True, fmt=".1f", cmap="Purples",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de Confusión Promedio ({k_folds}-fold CV)")
plt.tight_layout()
ruta_prom = os.path.join(carpeta_salida, "matriz_confusion_promedio.png")
plt.savefig(ruta_prom, dpi=300)
plt.show()

print(f"✅ Matriz de confusión promedio guardada en '{ruta_prom}'")
