from typing import List, Tuple
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


def plot_bboxes(frame: np.ndarray,
                bboxes: List[Tuple[int, int, int, int]]) -> go.Figure:
    """
    Plot grayscale image with bounding boxes using Plotly.

    Args:
        frame: 2D numpy array of shape (height, width)
        bboxes: List of bounding boxes from detect_shapes()

    Returns:
        Plotly Figure object
    """
    height, width = frame.shape

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=frame,
        colorscale='gray',
        showscale=False,
        dx=1,
        dy=1,
        x0=0,
        y0=0
    ))

    # Add bounding boxes
    for x, y, w, h in bboxes:
        fig.add_shape(
            type="rect",
            x0=x,
            y0=y,
            x1=x + w,
            y1=y + h,
            line=dict(color="red", width=2),
            xref="x",
            yref="y"
        )

    # Configure layout
    fig.update_layout(
        width=800,
        height=600,
        xaxis=dict(
            showgrid=False,
            range=[0, width],
            scaleanchor="y",
            constrain="domain"
        ),
        yaxis=dict(
            showgrid=False,
            range=[height, 0],
            scaleanchor="x",
            autorange=False
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

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
