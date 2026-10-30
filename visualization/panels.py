import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots

IMAGE_SIZE = 500


def show_grayscale(images: list[np.ndarray]) -> go.Figure:
    """Lay grayscale arrays out in a square grid, for a look at what a step did."""
    if not images:
        raise ValueError("no image to show")

    columns = int(np.ceil(np.sqrt(len(images))))
    figure = subplots.make_subplots(
        rows=columns,
        cols=columns,
        subplot_titles=[f"Image {index + 1}" for index in range(len(images))],
    )

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        figure.add_trace(
            go.Heatmap(z=image, showscale=False, colorscale="Greys"),
            row=row + 1,
            col=column + 1,
        )

    figure.update_layout(
        height=IMAGE_SIZE * columns, width=IMAGE_SIZE * columns, showlegend=False
    )
    figure.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    return figure
