import matplotlib.pyplot as plt

def draw_nn(nn):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    layer_sizes = []
