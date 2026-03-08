from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calcular_metricas(y_true, y_pred, nombres_clases):
    """
    Calcula accuracy, macro F1, F1 por clase y matriz de confusión.

    Args:
        y_true:         lista de etiquetas reales (int)
        y_pred:         lista de predicciones (int)
        nombres_clases: lista de strings con el nombre de cada clase

    Returns:
        dict con accuracy, f1_macro, f1_por_clase, confusion_matrix
    """
    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_clases = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)
    reporte   = classification_report(
        y_true, y_pred, target_names=nombres_clases, zero_division=0
    )

    print(reporte)
    return {
        "accuracy":         acc,
        "f1_macro":         f1_macro,
        "f1_por_clase":     dict(zip(nombres_clases, f1_clases)),
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, nombres_clases):
    """Visualiza la matriz de confusión."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(nombres_clases)))
    ax.set_yticks(range(len(nombres_clases)))
    ax.set_xticklabels(nombres_clases, rotation=45, ha="right")
    ax.set_yticklabels(nombres_clases)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()
