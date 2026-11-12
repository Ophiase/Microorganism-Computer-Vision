import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_VALUE = 255.0
ARROW_STEP = 5


def normalize_to_bytes(tensor: np.ndarray) -> np.ndarray:
    """Stretch a float tensor onto 0..255, keeping its shape."""
    destination = np.zeros(tensor.shape, dtype=np.float32)
    cv2.normalize(tensor, destination, 0, MAX_VALUE, cv2.NORM_MINMAX)
    return destination


def visualize_video_and_optical_flow(
    video: np.ndarray, optical_flow: np.ndarray, frame_index: int = 0
) -> go.Figure:
    """One frame beside its flow, the two components in the red and green channels."""
    if not 0 <= frame_index < len(video):
        raise ValueError(f"frame {frame_index} is out of the decoded range")

    frame = video[frame_index]
    flow = optical_flow[frame_index]
    flow_rgb = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
    flow_rgb[..., 0] = normalize_to_bytes(flow[..., 0])
    flow_rgb[..., 1] = normalize_to_bytes(flow[..., 1])

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Original frame", "Optical flow (red and green)"),
    )
    frame_bytes = normalize_to_bytes(frame)
    if frame.ndim == 2:
        figure.add_trace(go.Heatmap(z=frame_bytes, showscale=False), row=1, col=1)
    else:
        figure.add_trace(go.Image(z=frame_bytes), row=1, col=1)
    figure.add_trace(go.Image(z=flow_rgb), row=1, col=2)

    figure.update_layout(
        title=f"Frame {frame_index} and its flow",
        xaxis={"showticklabels": False},
        yaxis={"showticklabels": False},
    )
    return figure


def visualize_flow_arrows(
    video: np.ndarray,
    optical_flow: np.ndarray,
    frame_index: int = 0,
    step: int = ARROW_STEP,
) -> go.Figure:
    """A frame with one arrow per `step` pixels, the arrows normalized together."""
    if not 0 <= frame_index < len(optical_flow):
        raise ValueError(f"frame {frame_index} is out of the flow range")

    flow = optical_flow[frame_index]
    ys, xs = np.mgrid[0 : flow.shape[0] : step, 0 : flow.shape[1] : step]
    flow_x = flow[::step, ::step, 0]
    flow_y = flow[::step, ::step, 1]
    scale = np.sqrt(flow_x**2 + flow_y**2).max() + 1e-6

    figure = go.Figure(
        go.Heatmap(z=video[frame_index], colorscale="gray", showscale=False)
    )
    figure.add_trace(
        go.Scatter(
            x=xs.flatten(),
            y=ys.flatten(),
            mode="markers",
            marker={"color": "red", "size": 4},
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=(xs + flow_x / scale).flatten(),
            y=(ys + flow_y / scale).flatten(),
            mode="lines",
            line={"color": "blue", "width": 2},
            showlegend=False,
        )
    )
    figure.update_layout(
        title=f"Flow arrows on frame {frame_index}",
        xaxis={"scaleanchor": "y", "showgrid": False},
        yaxis={"showgrid": False},
    )
    return figure
