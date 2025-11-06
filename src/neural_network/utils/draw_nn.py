import matplotlib.pyplot as plt

def draw_nn(nn):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    layer_sizes = []

    for i in range(0, len(nn), 2):
        layer_sizes.append(nn[i].input_size)
        if i == len(nn) - 1 or i == len(nn) - 2:
            layer_sizes.append(nn[i].output_size)
    v_spacing = 1 / float(max(layer_sizes))
    h_spacing = 1 / float(len(layer_sizes))

    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing, layer_top - j * v_spacing), v_spacing / 3,
                                color='green', ec='k', zorder=4)
            ax.add_artist(circle)

    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2
        layer_top_b = v_spacing * (layer_size_b - 1) / 2
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing],
                                  c='green')
                ax.add_artist(line)

    output_layer_index = len(layer_sizes) - 1
    for k in range(layer_sizes[output_layer_index]):
        ax.text(output_layer_index * h_spacing + 0.15, -v_spacing * (k - (layer_sizes[output_layer_index] - 1) / 2),
                f'output y{k + 1}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

    output_layer_index = 0
    for k in range(layer_sizes[output_layer_index]):
        ax.text(output_layer_index * h_spacing - 0.15, -v_spacing * (k - (layer_sizes[output_layer_index] - 1) / 2),
                f'input x{k + 1}', horizontalalignment='center', verticalalignment='center', fontsize=12,
                color='black')

    ax.set_xlim(-0.9, 2)
    ax.set_ylim(-0.8, 0.5)
    plt.subplots_adjust(left=0.01, right=0.95 * 2, top=0.95, bottom=0.05 * 2)
    plt.show()
