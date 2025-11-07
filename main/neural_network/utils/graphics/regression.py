import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def dispersion_graph(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_true, y_pred,
        s=55,
        edgecolors="black",
        linewidths=0.6,
        color="#E26BAF"
    )

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--", linewidth=2, color="#C96A9F")

    plt.xlabel("Valores Reales", fontsize=12)
    plt.ylabel("Valores Predichos", fontsize=12)
    plt.title("Dispersión Real vs Predicción", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def learning_curve(loss_during_train):
    epochs = np.arange(len(loss_during_train))
    n = len(epochs)
    colors = np.linspace(0.4, 1.0, n)
    dynamic_color = [(1.0, 0.3, 0.7, c) for c in colors]

    plt.figure(figsize=(8, 5))
    for i in range(n - 1):
        plt.plot(epochs[i:i+2], loss_during_train[i:i+2], linewidth=2, color=dynamic_color[i])
    plt.title("Curva de Aprendizaje (Escala Logarítmica)", fontsize=14, fontweight='bold')
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel("Pérdida (Loss)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def residual_errors_graph(y_true, y_pred):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="#E26BAF", edgecolor="black", alpha=0.75)
    plt.xlabel("Error de Predicción", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Histograma de Errores Residuales", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_3d_points_and_plane_interactive(X, Y, coefficients):
    plt.ion()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x1 = X[:, 0, 0]
    x2 = X[:, 1, 0]

    ax.scatter(x1, x2, Y.flatten(), color="#E26BAF", s=60, edgecolors="black", linewidth=0.6)

    x1_range = np.linspace(x1.min(), x1.max(), 20)
    x2_range = np.linspace(x2.min(), x2.max(), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_plane = coefficients[0] * x1_grid + coefficients[1] * x2_grid + coefficients[2]

    ax.plot_surface(x1_grid, x2_grid, z_plane, alpha=0.45, color="#F4A3D5")

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Plano Ajustado en 3D', fontsize=14, fontweight='bold')

    plt.show(block=True)


def noisy_plane_points(n_points, n_dimensions, noise_level=0.3, seed=42, include_bias=True):
    rng = np.random.default_rng(seed)

    X = rng.uniform(0.0, 10.0, size=(n_points, n_dimensions)).astype(np.float32)
    w = rng.uniform(-1.0, 1.0, size=(n_dimensions,)).astype(np.float32)
    b = np.float32(rng.uniform(-2.0, 2.0) if include_bias else 0.0)

    Y_true = X @ w + b
    noise_std = float(noise_level) * (np.std(Y_true) + 1e-8)
    Y_noisy = (Y_true + rng.normal(0.0, noise_std, size=n_points)).astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_noisy, test_size=0.2, random_state=seed, shuffle=True
    )

    X_train = X_train.reshape(-1, n_dimensions, 1)
    X_test  = X_test.reshape(-1, n_dimensions, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test  = Y_test.reshape(-1, 1)

    return X_train, X_test, Y_train, Y_test, w, b