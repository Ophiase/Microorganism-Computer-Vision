import numpy as np
import plotly.graph_objects as go

from logic.structure.bounding_box import BoundingBox

BOX_COLOUR = "red"
BOX_WIDTH = 2
MIN_FLOW = 0.1


def box_shape(box: BoundingBox) -> dict[str, object]:
    """The rectangle a Plotly layout draws for one box."""
    return {
        "type": "rect",
        "x0": box.x,
        "y0": box.y,
        "x1": box.x + box.width,
        "y1": box.y + box.height,
        "line": {"color": BOX_COLOUR, "width": BOX_WIDTH},
    }


def plot_bboxes(
    frame: np.ndarray,
    boxes: list[BoundingBox],
    figure: go.Figure | None = None,
) -> go.Figure:
    """A grayscale frame with its boxes drawn over it."""
    height, width = frame.shape

    if figure is None:
        figure = go.Figure()
        first_call = True
    else:
        figure.data = []
        figure.layout.shapes = []
        first_call = False

    figure.add_trace(
        go.Heatmap(z=frame, colorscale="gray", showscale=False, x0=0, y0=0, dx=1, dy=1)
    )
    for box in boxes:
        figure.add_shape(**box_shape(box))

    if first_call:
        figure.update_layout(
            width=800,
            height=600,
            xaxis={
                "showgrid": False,
                "range": [0, width],
                "scaleanchor": "y",
                "constrain": "domain",
            },
            yaxis={
                "showgrid": False,
                "range": [height, 0],
                "scaleanchor": "x",
                "autorange": False,
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
    return figure


def add_optical_flow(
    figure: go.Figure,
    flow: np.ndarray,
    stride: int = 10,
    scale: float = 5.0,
) -> go.Figure:
    """Draw one arrow every `stride` pixels, its length scaled by `scale`."""
    height, width = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
    flow_x = flow[::stride, ::stride, 0]
    flow_y = flow[::stride, ::stride, 1]

    arrows = [
        go.Scatter(
            x=[x, x + dx * scale],
            y=[y, y + dy * scale],
            mode="lines",
            line={"color": "cyan", "width": 1},
            hoverinfo="none",
            showlegend=False,
        )
        for x, y, dx, dy in zip(
            xs.flatten(), ys.flatten(), flow_x.flatten(), flow_y.flatten(), strict=True
        )
        if abs(dx) + abs(dy) > MIN_FLOW
    ]
    figure.add_traces(arrows)
    return figure
