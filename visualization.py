from typing import List
import numpy as np
import matplotlib.pyplot as plt


def show_grayscale(images: List[np.ndarray]) -> plt:
    n_images = len(images)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    _, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
    
    if grid_size == 1:
        axes = [[axes]]
    
    axes = np.array(axes).flatten()

    for ax, img, i in zip(axes, images, range(n_images)):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Image {i+1}")
        ax.axis('off')
    
    for ax in axes[n_images:]:
        ax.axis('off')

    plt.tight_layout()
    return plt
