import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation

def draw_nn(
    nn=None,
    layer_sizes=None,
    weights=None,
    neuron_labels=True,
    show_axis=False,
    title="Neural Network",
    node_color="#4CAF50",
    edge_color="#4CAF50",
    bg_color="#0f1117",
    animate=False,
    anim_frames=80,
    anim_interval=30,
    save_path=None
):
    if layer_sizes is None and nn is not None:
        layers = getattr(nn, "layers", nn)
        sizes = []
        for i, L in enumerate(layers):
            if hasattr(L, "input_size") and hasattr(L, "output_size"):
                if i == 0:
                    sizes.append(int(L.input_size))
                sizes.append(int(L.output_size))
        layer_sizes = sizes

    if layer_sizes is None:
        raise ValueError()

    L = len(layer_sizes)
    Nmax = max(layer_sizes)

    fig_w = max(8.0, 1.3 * L + 4)
    fig_h = max(6.0, 0.6 * Nmax + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis("equal")
    ax.axis("off" if not show_axis else "on")

    x_pad = 0.12
    y_pad = 0.12
    xs = np.linspace(x_pad, 1 - x_pad, L)

    def y_positions(n):
        if n == 1:
            return np.array([0.5])
        top = 0.5 + 0.5 * (1 - y_pad) * (n - 1) / max(1, (Nmax - 1))
        bottom = 0.5 - 0.5 * (1 - y_pad) * (n - 1) / max(1, (Nmax - 1))
        return np.linspace(top, bottom, n)

    r = 0.025 * (8 / fig_w) * (max(6, 0.6 * Nmax + 3) / fig_h)

    nodes = []
    for i, n in enumerate(layer_sizes):
        ys = y_positions(n)
        nodes.append([(xs[i], y) for y in ys])

    def weight_style_matrix(W):
        if W is None:
            return None
        W = np.asarray(W)
        if W.size == 0:
            return None
        m = np.percentile(np.abs(W), 95) or 1.0
        return np.clip(np.abs(W) / m, 0, 1)

    edge_lines = []
    for i in range(L - 1):
        src = nodes[i]
        dst = nodes[i + 1]
        style = weight_style_matrix(None if weights is None or i >= len(weights) else weights[i])
        for j, (x1, y1) in enumerate(src):
            for k, (x2, y2) in enumerate(dst):
                lw = 1.3
                alpha = 0.18
                if style is not None:
                    alpha = 0.08 + 0.35 * float(style[k, j])
                    lw = 0.8 + 2.0 * float(style[k, j])
                line = plt.Line2D([x1 + r * 0.6, x2 - r * 0.6], [y1, y2],
                                  color=edge_color, lw=lw, alpha=alpha, zorder=2)
                ax.add_line(line)
                edge_lines.append(line)

    for layer in nodes:
        for (x, y) in layer:
            circ = plt.Circle((x, y), r,
                              facecolor=node_color, edgecolor="white",
                              linewidth=1.0, zorder=5,
                              path_effects=[pe.withStroke(linewidth=6, foreground=(0, 0, 0, 0.25))])
            ax.add_patch(circ)

    ax.set_title(title, color="white", fontsize=16, pad=14)

    if neuron_labels:
        for idx, (_, y) in enumerate(nodes[0]):
            ax.text(xs[0] - 0.04, y, f"x{idx+1}", ha="right", va="center", color="#dfe7ef", fontsize=10)
        for idx, (_, y) in enumerate(nodes[-1]):
            ax.text(xs[-1] + 0.04, y, f"y{idx+1}", ha="left", va="center", color="#dfe7ef", fontsize=10)

    for i in range(L):
        name = "Input" if i == 0 else ("Output" if i == L - 1 else f"Hidden {i}")
        ax.text(xs[i], 1 - y_pad/2, name, ha="center", va="center", color="#c7d1db", fontsize=11)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    anim = None
    if animate:
        edge_initial_alpha = [ln.get_alpha() for ln in edge_lines]
        edge_initial_lw = [ln.get_linewidth() for ln in edge_lines]

        def ease(t):
            return 0.5 - 0.5 * math.cos(2 * math.pi * t)

        def update(frame):
            t = (frame % anim_frames) / anim_frames
            pulse = 0.35 * ease(t)
            for ln, a0, lw0 in zip(edge_lines, edge_initial_alpha, edge_initial_lw):
                ln.set_alpha(min(1.0, a0 + pulse))
                ln.set_linewidth(lw0 + 0.8 * pulse)
            return edge_lines

        anim = FuncAnimation(fig, update, frames=anim_frames, interval=anim_interval, blit=False)

    if save_path:
        if animate and (save_path.endswith(".mp4") or save_path.endswith(".gif")):
            anim.save(save_path)
        else:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax, anim
