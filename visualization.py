from typing import List
import numpy as np
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go


def show_grayscale_plt(images: List[np.ndarray]) -> plt:
    n_images = len(images)
    grid_size = int(np.ceil(np.sqrt(n_images)))

    _, axes = plt.subplots(grid_size, grid_size,
                           figsize=(5 * grid_size, 5 * grid_size))

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


def show_grayscale(images: List[np.ndarray]) -> go.Figure:
    n_images = len(images)
    height, width = images[0].shape[0], images[0].shape[1]
    grid_size = int(np.ceil(np.sqrt(n_images)))

    fig = sp.make_subplots(
        rows=grid_size,
        cols=grid_size,
        subplot_titles=[f"Image {i+1}" for i in range(n_images)],
    )

    for i, img in enumerate(images):
        row, col = divmod(i, grid_size)
        fig.add_trace(
            go.Heatmap(z=img, showscale=False, colorscale="Greys"),
            row=row + 1,
            col=col + 1,
        )

    fig.update_layout(
        height=500 * grid_size,
        width=500 * grid_size,
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

    return fig

###################################################################################


def full_test() -> None:
    np.random.seed(100)

    RES = 500
    tensors = [np.random.rand(RES, RES) for _ in range(5)]

    show_grayscale(tensors).show()

#########################################


def main() -> None:
    full_test()


if __name__ == "__main__":
    main()
