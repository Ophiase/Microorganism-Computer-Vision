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


###################################################################################


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

#########################################


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


def add_optical_flow(fig: go.Figure,
                     optical_flow_frame: np.ndarray,
                     stride: int = 10,
                     scale: float = 5.0) -> go.Figure:
    """
    Add optical flow vectors to plot
    Args:
        stride: Show every nth vector
        scale: Arrow size multiplier
    """
    h, w = optical_flow_frame.shape[:2]

    # Create grid
    xx, yy = np.meshgrid(np.arange(0, w, stride),
                         np.arange(0, h, stride))

    # Get vectors
    u = optical_flow_frame[::stride, ::stride, 0]
    v = optical_flow_frame[::stride, ::stride, 1]

    # Create line segments
    arrows = []
    for x, y, dx, dy in zip(xx.flatten(), yy.flatten(), u.flatten(), v.flatten()):
        if abs(dx) + abs(dy) > 0.1:  # Filter static vectors
            arrows.append(
                go.Scatter(
                    x=[x, x + dx*scale],
                    y=[y, y + dy*scale],
                    mode='lines',
                    line=dict(color='cyan', width=1),
                    hoverinfo='none',
                    showlegend=False
                )
            )

    fig.add_traces(arrows)
    return fig

###################################################################################


def plot_tracked_bboxes(fig: go.Figure,
                        frame: np.ndarray,
                        tracked_bboxes: List[Tuple[int, int, int, int, int]],
                        optical_flow: Optional[np.ndarray] = None,
                        flow_stride: int = 10) -> go.Figure:
    """
    Plot frame with tracked bounding boxes and IDs
    Returns modified Plotly figure
    """
    fig = plot_bboxes(frame, [bbox[1:] for bbox in tracked_bboxes], fig)

    for bbox_id, x, y, w, h in tracked_bboxes:
        fig.add_annotation(
            x=x + w/2,
            y=y + h/2,
            text=str(bbox_id),
            showarrow=False,
            font=dict(color="red", size=14),
            bordercolor="white",
            borderwidth=2
        )

    if optical_flow is not None:
        fig = add_optical_flow(fig, optical_flow, stride=flow_stride)

    return fig


def plot_tracked_video(video: np.ndarray,
                       tracked_boxes: List[List[Tuple[int, int, int, int, int]]],
                       optical_flow_video: np.ndarray = None,
                       frame_duration: int = 100,
                       show_flow: bool = False) -> go.Figure:
    """
    Create interactive video visualization with tracked bounding boxes
    """
    fig = plot_bboxes_video(
        video, [[bbox[1:] for bbox in frame_boxes] for frame_boxes in tracked_boxes])
    
    # Add ID annotations to each frame
    for i, frame_boxes in enumerate(tracked_boxes):
        annotations = [
            dict(
                x=x + w/2,
                y=y + h/2,
                text=str(bbox_id),
                showarrow=False,
                font=dict(color="red", size=14)
            )
            for bbox_id, x, y, w, h in frame_boxes
        ]

        fig.frames[i].layout.annotations = annotations

        if show_flow:
            flow_frame = optical_flow_video[i]
            fig.frames[i].data += tuple(
                add_optical_flow(go.Figure(), flow_frame).data
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
