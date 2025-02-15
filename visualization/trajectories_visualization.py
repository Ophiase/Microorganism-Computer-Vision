from typing import List, Tuple
import numpy as np
import plotly.graph_objects as go
from logic.trajectories import Trajectory


def plot_trajectories(trajectories: List[Trajectory], resolution: Tuple[int] = None) -> go.Figure:
    fig = go.Figure()
    for i, traj in enumerate(trajectories):
        if len(traj.valid_entries) < 2:
            continue
        x, y = zip(*traj.positions())
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name=f'Track {i}',
            opacity=0.7
        ))

    fig.update_layout(
        title='Microorganism Trajectories',
        xaxis_title='X Position',
        yaxis_title='Y Position'
    )

    if resolution is not None:
        fig.update_xaxes(range=(0, resolution[0]))
        fig.update_yaxes(range=(0, resolution[1]))

    return fig


def plot_speed_distribution(trajectories: List[Trajectory]) -> go.Figure:
    speeds = []
    for traj in trajectories:
        if len(traj.displacements()) == 0:
            continue
        dx, dy = zip(*traj.displacements())
        dt = traj.time_deltas()
        speed = np.sqrt(np.array(dx)**2 + np.array(dy)**2) / np.array(dt)
        speeds.extend(speed.tolist())

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=speeds, nbinsx=50))
    fig.update_layout(title='Speed Distribution',
                      xaxis_title='Speed (pixels/frame)',
                      yaxis_title='Count')
    return fig


def plot_speed_distribution_per_trajectory(trajectories: List[Trajectory]) -> go.Figure:
    """
    Displays speed distributions across trajectories using transparent overlapping curves.
    Creates a density-like visualization of speed patterns.
    """
    fig = go.Figure()
    for i, traj in enumerate(trajectories):
        if len(traj.positions()) < 2:
            continue

        positions = np.array(traj.positions())
        dx = positions[1:, 0] - positions[:-1, 0]
        dy = positions[1:, 1] - positions[:-1, 1]
        dt = traj.time_deltas()[:len(dx)]

        speed = np.sqrt(dx**2 + dy**2) / dt
        indices = np.arange(len(speed))

        fig.add_trace(go.Scatter(
            x=indices,
            y=speed,
            mode='lines',
            line=dict(width=1, color='blue'),
            opacity=0.2,
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title='Collective Speed Distribution Pattern',
        xaxis_title='Time Step Index',
        yaxis_title='Speed (pixels/frame)',
        plot_bgcolor='white',
        hovermode='x unified'
    )
    return fig


def plot_angular_distribution(trajectories: List[Trajectory]) -> go.Figure:
    """
    Visualizes angular distribution patterns across all trajectories using
    semi-transparent overlapping curves. Emphasizes common turning patterns.
    """
    fig = go.Figure()
    for i, traj in enumerate(trajectories):
        if len(traj.positions()) < 2:
            continue

        positions = np.array(traj.positions())
        dx = positions[1:, 0] - positions[:-1, 0]
        dy = positions[1:, 1] - positions[:-1, 1]

        # Use absolute angles for clearer distribution pattern
        angles = np.degrees(np.arctan2(dy, dx)) % 360
        indices = np.arange(len(angles))

        fig.add_trace(go.Scatter(
            x=indices,
            y=angles,
            mode='lines',
            line=dict(width=1, color='red'),
            opacity=0.2,
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title='Collective Angular Distribution Pattern',
        xaxis_title='Time Step Index',
        yaxis_title='Angle (degrees)',
        yaxis=dict(range=[0, 360], dtick=45),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    return fig
