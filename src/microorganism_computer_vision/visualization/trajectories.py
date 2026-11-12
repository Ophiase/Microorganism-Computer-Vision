import numpy as np
import plotly.graph_objects as go
from motility.tracking.trajectories import Trajectory

SPEED_BINS = 50
OPACITY = 0.2


def plot_trajectories(
    trajectories: tuple[Trajectory, ...], resolution: tuple[int, int] | None = None
) -> go.Figure:
    figure = go.Figure()
    for index, trajectory in enumerate(trajectories):
        if len(trajectory.seen) < 2:
            continue
        xs, ys = zip(*trajectory.positions, strict=True)
        figure.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=f"Track {index}", opacity=0.7
            )
        )

    figure.update_layout(
        title="Microorganism trajectories",
        xaxis_title="X position",
        yaxis_title="Y position",
    )
    if resolution is not None:
        figure.update_xaxes(range=(0, resolution[0]))
        figure.update_yaxes(range=(0, resolution[1]))
    return figure


def _speeds(trajectory: Trajectory) -> np.ndarray:
    displacements = np.array(trajectory.displacements)
    deltas = np.array(trajectory.time_deltas, dtype=float)
    return np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2) / deltas


def plot_speed_distribution(trajectories: tuple[Trajectory, ...]) -> go.Figure:
    speeds = [
        speed
        for trajectory in trajectories
        if len(trajectory.displacements)
        for speed in _speeds(trajectory).tolist()
    ]
    figure = go.Figure(go.Histogram(x=speeds, nbinsx=SPEED_BINS))
    figure.update_layout(
        title="Speed distribution",
        xaxis_title="Speed (pixels per frame)",
        yaxis_title="Count",
    )
    return figure


def plot_speed_distribution_per_trajectory(
    trajectories: tuple[Trajectory, ...],
) -> go.Figure:
    """One faint curve per trajectory: the shape of the speeds, not their count."""
    figure = go.Figure()
    for trajectory in trajectories:
        if len(trajectory.positions) < 2:
            continue
        figure.add_trace(
            go.Scatter(
                x=np.arange(len(_speeds(trajectory))),
                y=_speeds(trajectory),
                mode="lines",
                line={"width": 1, "color": "blue"},
                opacity=OPACITY,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.update_layout(
        title="Collective speed pattern",
        xaxis_title="Time step index",
        yaxis_title="Speed (pixels per frame)",
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return figure


def plot_angular_distribution(trajectories: tuple[Trajectory, ...]) -> go.Figure:
    """One faint curve per trajectory: the heading in degrees over the frames."""
    figure = go.Figure()
    for trajectory in trajectories:
        if len(trajectory.positions) < 2:
            continue
        displacements = np.array(trajectory.displacements)
        headings = (
            np.degrees(np.arctan2(displacements[:, 1], displacements[:, 0])) % 360
        )
        figure.add_trace(
            go.Scatter(
                x=np.arange(len(headings)),
                y=headings,
                mode="lines",
                line={"width": 1, "color": "red"},
                opacity=OPACITY,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.update_layout(
        title="Collective heading pattern",
        xaxis_title="Time step index",
        yaxis_title="Heading (degrees)",
        yaxis={"range": [0, 360], "dtick": 45},
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return figure
