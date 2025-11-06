import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

def draw_nn(nn):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.set_aspect("equal")
    ax.axis("off")

    pink_nodes = LinearSegmentedColormap.from_list("pink_nodes", ["#ffb3d9", "#ff66b3", "#d9368f"])
    pink_edges = LinearSegmentedColormap.from_list("pink_edges", ["#ff99cc", "#ff4da6", "#cc0077"])

    layer_sizes = []
    for i in range(0, len(nn), 2):
        layer_sizes.append(nn[i].input_size)
        if i == len(nn) - 1 or i == len(nn) - 2:
            layer_sizes.append(nn[i].output_size)

    vmax = max(layer_sizes)
    v_spacing = 1 / float(vmax if vmax > 0 else 1)
    h_spacing = 1 / float(len(layer_sizes))

    weights = []
    for i in range(0, len(nn), 2):
        W = getattr(nn[i], "weights", None)
        if W is not None:
            weights.append(np.asarray(W))

    def norm_w(W):
        if W is None:
            return None
        m = np.percentile(np.abs(W), 95) or 1.0
        return np.clip(np.abs(W) / m, 0, 1)

    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2
        node_scale = np.interp(layer_size, [1, vmax], [0.8, 1.2])
        radius = (v_spacing / 3) * node_scale
        node_color = pink_nodes(i / max(1, len(layer_sizes) - 1))
        for j in range(layer_size):
            circle = plt.Circle(
                (i * h_spacing, layer_top - j * v_spacing),
                radius,
                facecolor=node_color,
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
                path_effects=[pe.withStroke(linewidth=6, foreground=(0, 0, 0, 0.25))]
            )
            ax.add_artist(circle)

    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2
        layer_top_b = v_spacing * (layer_size_b - 1) / 2
        W = norm_w(weights[i] if i < len(weights) else None)
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                strength = 0.25
                if W is not None:
                    strength = float(W[k, j])
                color = pink_edges(strength)
                lw = 0.8 + 2.2 * strength
                alpha = 0.12 + 0.35 * strength
                line = plt.Line2D(
                    [i * h_spacing + (v_spacing/3)*0.5, (i + 1) * h_spacing - (v_spacing/3)*0.5],
                    [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing],
                    c=color, lw=lw, alpha=alpha, zorder=2
                )
                ax.add_artist(line)

    out_i = len(layer_sizes) - 1
    for k in range(layer_sizes[out_i]):
        ax.text(
            out_i * h_spacing + 0.06,
            -v_spacing * (k - (layer_sizes[out_i] - 1) / 2),
            f"y{k + 1}",
            ha="left", va="center", fontsize=11, color="#ffb3d9"
        )

    in_i = 0
    for k in range(layer_sizes[in_i]):
        ax.text(
            in_i * h_spacing - 0.06,
            -v_spacing * (k - (layer_sizes[in_i] - 1) / 2),
            f"x{k + 1}",
            ha="right", va="center", fontsize=11, color="#ffb3d9"
        )

    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.6, 0.6)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    plt.show()