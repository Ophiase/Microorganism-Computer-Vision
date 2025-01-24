from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from logic.structure.bounding_box import BoundingBox


def plot_bboxes(frame: np.ndarray,
                bboxes: List[BoundingBox],
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
    for bbox in bboxes:
        fig.add_shape(
            type="rect",
            x0=bbox.x,
            y0=bbox.y,
            x1=bbox.x + bbox.w,
            y1=bbox.y + bbox.h,
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
                      bboxes_list: List[List[BoundingBox]],
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
                shapes=[_create_shape(*bbox.getBbox())
                        for bbox in bboxes_list[i]]
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
                        tracked_bboxes: List[BoundingBox],
                        optical_flow: Optional[np.ndarray] = None,
                        flow_stride: int = 10) -> go.Figure:
    """
    Plot frame with tracked bounding boxes and IDs
    Returns modified Plotly figure
    """
    fig = plot_bboxes(frame, tracked_bboxes, fig)

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
                       tracked_boxes: List[List[BoundingBox]],
                       optical_flow_video: np.ndarray = None,
                       show_flow: bool = False) -> go.Figure:
    """
    Create interactive video visualization with tracked bounding boxes
    """
    fig = plot_bboxes_video(
        video, tracked_boxes)

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
            for bbox_id, x, y, w, h, visible in frame_boxes
        ]

        fig.frames[i].layout.annotations = annotations

        if show_flow:
            flow_frame = optical_flow_video[i]
            fig.frames[i].data += tuple(
                add_optical_flow(go.Figure(), flow_frame).data
            )

    return fig

