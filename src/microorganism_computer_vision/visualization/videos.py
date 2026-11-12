import numpy as np
import plotly.graph_objects as go
from motility.tracking.bounding_box import BoundingBox

from microorganism_computer_vision.visualization.frames import (
    add_optical_flow,
    box_shape,
    plot_bboxes,
)

FRAME_MS = 100
ID_COLOUR = "red"


def plot_boxes_video(
    video: np.ndarray,
    boxes_per_frame: list[list[BoundingBox]],
    frame_duration: int = FRAME_MS,
) -> go.Figure:
    """An animated figure of a video, its boxes and a slider to scrub it."""
    frame_count = len(video)
    if len(boxes_per_frame) != frame_count:
        raise ValueError("one list of boxes per frame is expected")

    figure = plot_bboxes(video[0], boxes_per_frame[0])
    frames = [
        go.Frame(
            data=[go.Heatmap(z=video[index])],
            layout=go.Layout(shapes=[box_shape(box) for box in boxes_per_frame[index]]),
            name=f"frame_{index}",
        )
        for index in range(frame_count)
    ]

    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                        "label": str(index),
                        "method": "animate",
                    }
                    for index, frame in enumerate(frames)
                ],
            }
        ],
    )
    figure.frames = frames
    return figure


def plot_tracked_video(
    video: np.ndarray,
    tracked_boxes: list[list[BoundingBox]],
    optical_flow_video: np.ndarray | None = None,
    show_flow: bool = False,
) -> go.Figure:
    """The video with the track index on each box, and its flow when asked."""
    figure = plot_boxes_video(video, tracked_boxes)

    for index, frame_boxes in enumerate(tracked_boxes):
        figure.frames[index].layout.annotations = [
            {
                "x": box.x + box.width / 2,
                "y": box.y + box.height / 2,
                "text": str(box.index),
                "showarrow": False,
                "font": {"color": ID_COLOUR, "size": 14},
            }
            for box in frame_boxes
        ]
        if show_flow and optical_flow_video is not None:
            figure.frames[index].data += tuple(
                add_optical_flow(go.Figure(), optical_flow_video[index]).data
            )

    return figure
