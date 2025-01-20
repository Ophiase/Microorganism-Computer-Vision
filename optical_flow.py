from plotly.subplots import make_subplots
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

###################################################################################


VERBOSE = False

###################################################################################


def compute_optical_flow(video: np.ndarray, verbose: bool = VERBOSE) -> np.ndarray:
    optical_flow = []
    prev_frame = video[0]
    for i in range(1, len(video)):
        next_frame = video[i]
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optical_flow.append(flow)
        prev_frame = next_frame
        if verbose and (i % 30 == 0):
            print(
                f"Computed optical flow for frame {i}, flow shape: {flow.shape}")

    # Duplicate the first flow to match the video length
    optical_flow.insert(0, optical_flow[0])
    return np.array(optical_flow)

###################################################################################


def preview_optical_flow(video: np.ndarray, optical_flow: np.ndarray, verbose: bool = False):
    for i in range(len(optical_flow)):
        frame = video[i]
        flow = optical_flow[i]
        hsv = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Optical Flow', rgb_flow)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        if verbose:
            print(
                f"Previewing optical flow for frame {i}, rgb_flow shape: {rgb_flow.shape}")
    cv2.destroyAllWindows()


#########################################


def visualize_video_and_optical_flow_side_by_side(video: np.ndarray, optical_flow: np.ndarray, frame_idx: int = 0) -> go.Figure:
    """
    Visualizes a video frame and its optical flow side by side.

    Parameters:
        video (np.ndarray): Video data of shape (frames, height, width). # value between 0 and 1
        optical_flow (np.ndarray): Optical flow data of shape (frames, height, width, 2). # value between 0 and 1
        frame_idx (int): Index of the frame to visualize.

    Returns:
        go.Figure: Plotly figure showing video and optical flow side by side.
    """
    if frame_idx >= len(video):
        raise ValueError(
            f"Frame index {frame_idx} out of range. Maximum allowed index: {len(video)-1}.")

    frame = video[frame_idx]
    flow = optical_flow[frame_idx]

    # Convert optical flow to RG (using XY components)
    flow_rg = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    flow_rg[..., 0] = cv2.normalize(
        flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)  # R channel
    flow_rg[..., 1] = cv2.normalize(
        flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)  # G channel

    # Remap frame values to 0-255
    frame_255 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    # Plot side-by-side visualization
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Original Video Frame", "Optical Flow (RG Channels)"))

    is_grayscale = len(frame.shape) == 2
    if is_grayscale:
        fig.add_trace(go.Heatmap(
            z=frame_255, name="Original Video Frame",
            showscale=False
            ), row=1, col=1)
    else:
        fig.add_trace(
            go.Image(z=frame_255, name="Original Video Frame"), row=1, col=1)

    fig.add_trace(
        go.Image(z=flow_rg, name="Optical Flow (RG Channels)"), row=1, col=2)

    fig.update_layout(
        title=f"Video Frame and Optical Flow for Frame {frame_idx}",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        grid=dict(rows=1, columns=2, pattern='independent')
    )

    return fig

#########################################


def visualize_optical_flow_with_slider(video: np.ndarray, optical_flow: np.ndarray) -> go.Figure:
    frames = []
    for i in range(len(optical_flow)):
        frame = video[i]
        flow = optical_flow[i]
        y, x = np.mgrid[0:flow.shape[0], 0:flow.shape[1]]

        fig = go.Frame(
            data=[
                go.Image(z=frame),
                go.Scatter(
                    x=x.flatten(),
                    y=y.flatten(),
                    mode='markers',
                    marker=dict(color='red', size=2),
                ),
                go.Scatter(
                    x=x.flatten() + flow[..., 0].flatten(),
                    y=y.flatten() + flow[..., 1].flatten(),
                    mode='lines',
                    line=dict(color='blue', width=1),
                ),
            ],
            name=f"Frame {i}",
        )
        frames.append(fig)

    frame_0 = video[0]
    fig = go.Figure(
        data=[
            go.Image(z=frame_0),
        ],
        layout=go.Layout(
            title="Optical Flow Visualization",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}}]
                    },
                ]
            }],
        ),
        frames=frames,
    )
    return fig


def visualize_optical_flow_with_arrows(video: np.ndarray, optical_flow: np.ndarray, frame_idx: int = 0, step: int = 5) -> go.Figure:
    """
    Visualizes a video frame with optical flow arrows on a subgrid.

    Parameters:
        video (np.ndarray): Video data of shape (frames, height, width).
        optical_flow (np.ndarray): Optical flow data of shape (frames, height, width, 2).
        frame_idx (int): Index of the frame to visualize.
        step (int): Step size for subgrid selection (larger values = fewer arrows).

    Returns:
        go.Figure: Plotly figure with the visualized optical flow arrows.
    """
    if frame_idx >= len(optical_flow):
        raise ValueError(
            f"Frame index {frame_idx} out of range. Maximum allowed index: {len(optical_flow)-1}.")

    frame = video[frame_idx]
    flow = optical_flow[frame_idx]

    # Create a subgrid
    y, x = np.mgrid[0:flow.shape[0]:step, 0:flow.shape[1]:step]
    flow_x = flow[::step, ::step, 0]
    flow_y = flow[::step, ::step, 1]

    # Normalize flow for better visualization (scaling arrows)
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    scale = magnitude.max() + 1e-6  # Avoid division by zero
    flow_x /= scale
    flow_y /= scale

    # Plot the frame
    fig = px.imshow(frame, binary_string=True)

    # Add arrows (quiver-like effect)
    fig.add_trace(go.Scatter(
        x=x.flatten(),
        y=y.flatten(),
        mode='markers',
        marker=dict(color='red', size=4),  # Larger markers for visibility
    ))
    fig.add_trace(go.Scatter(
        x=(x + flow_x).flatten(),
        y=(y + flow_y).flatten(),
        mode='lines',
        line=dict(color='blue', width=2),
    ))

    fig.update_layout(
        title=f"Optical Flow Visualization with Arrows for Frame {frame_idx}",
        xaxis=dict(scaleanchor="y", showgrid=False),  # Ensure equal scaling
        yaxis=dict(showgrid=False)
    )
    return fig
