from typing import List, Optional, Tuple
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
                bboxes: List[Tuple[int, int, int, int]],
                fig: Optional[go.Figure] = None) -> go.Figure:
    """
    Plot grayscale image with bounding boxes using Plotly.

    Args:
        frame: 2D numpy array of shape (height, width)
        bboxes: List of bounding boxes from detect_shapes()
        fig: Optional existing figure to modify

    Returns:
        Plotly Figure object
    """
    height, width = frame.shape

    if fig is None:
        fig = go.Figure()
        add_axes_config = True
    else:
        fig.data = []
        fig.layout.shapes = []
        add_axes_config = False

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=frame,
        colorscale='gray',
        showscale=False,
        x0=0,
        y0=0,
        dx=1,
        dy=1
    ))

    # Add bounding boxes
    for x, y, w, h in bboxes:
        fig.add_shape(
            type="rect",
            x0=x,
            y0=y,
            x1=x + w,
            y1=y + h,
            line=dict(color="red", width=2)
        )

    if add_axes_config:
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


def plot_bboxes_video(video: np.ndarray,
                      bboxes_list: List[List[Tuple[int, int, int, int]]],
                      frame_duration: int = 100) -> go.Figure:
    """
    Create interactive video visualization with bounding boxes and a slider.

    Args:
        video: 3D numpy array (num_frames, height, width)
        bboxes_list: List of bbox lists (one per frame)
        frame_duration: Milliseconds between frames in animation

    Returns:
        Plotly Figure object with slider
    """
    assert video.ndim == 3, "Video must be 3D (frames, height, width)"
    num_frames, height, width = video.shape
    assert len(
        bboxes_list) == num_frames, "bboxes_list length must match video frames"

    # Create initial figure with first frame
    fig = plot_bboxes(video[0], bboxes_list[0])

    # Create animation frames
    frames = []
    for i in range(num_frames):
        frame = go.Frame(
            data=[go.Heatmap(z=video[i])],
            layout=go.Layout(
                shapes=[_create_shape(x, y, w, h)
                        for (x, y, w, h) in bboxes_list[i]]
            ),
            name=f"frame_{i}"
        )
        frames.append(frame)

    # Create slider
    sliders = [{
        'active': 0,
        'currentvalue': {'prefix': 'Frame: '},
        'steps': [
            {
                'args': [[f.name], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                'label': str(i),
                'method': 'animate'
            } for i, f in enumerate(frames)
        ]
    }]

    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': frame_duration}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=sliders
    )

    fig.frames = frames
    return fig


def _create_shape(x: int, y: int, w: int, h: int) -> dict:
    """Helper to create rectangle shape dictionary"""
    return {
        'type': 'rect',
        'x0': x,
        'y0': y,
        'x1': x + w,
        'y1': y + h,
        'line': {'color': 'red', 'width': 2}
    }


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
