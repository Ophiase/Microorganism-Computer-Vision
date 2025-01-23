from typing import List
import numpy as np
import plotly.graph_objects as go
from logic.trajectories import Trajectory

###################################################################################


def plot_trajectories(trajectories: List[Trajectory]) -> go.Figure:
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
    fig.update_layout(title='Microorganism Trajectories',
                      xaxis_title='X Position',
                      yaxis_title='Y Position')
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
