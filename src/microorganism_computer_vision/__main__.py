import argparse
from collections.abc import Sequence

from microorganism_computer_vision.pipeline.extract import process as extract
from microorganism_computer_vision.pipeline.object_detection import (
    process_with_tracking,
    show_tracks,
)
from microorganism_computer_vision.pipeline.paths import (
    DATA_FOLDER,
    DEFAULT_VIDEO,
    PREPROCESSED_FOLDER,
    TRACKING_FOLDER,
)
from microorganism_computer_vision.pipeline.render import process as render
from microorganism_computer_vision.pipeline.statistical_tests import process as analyze
from microorganism_computer_vision.pipeline.transform import process as transform
from microorganism_computer_vision.tracking.synthetic.dataset import (
    generate_synthetic_data,
)

TASKS = ("extract", "transform", "detection", "render", "analysis", "synthetic")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microorganism Computer Vision pipeline"
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="video file in data/")
    parser.add_argument(
        "--task", choices=TASKS, required=True, help="stage of the pipeline to run"
    )
    parser.add_argument(
        "--keep",
        choices=("all", "moving", "fastest"),
        default="all",
        help="which trajectories the analysis keeps",
    )
    parser.add_argument(
        "--debug", action="store_true", help="open the figures and print the shapes"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    video_path = DATA_FOLDER / arguments.video
    tensor_path = PREPROCESSED_FOLDER / f"{arguments.video}.npy"
    tracking_path = TRACKING_FOLDER / f"{arguments.video}.npy"

    match arguments.task:
        case "extract":
            extract()
        case "transform":
            transform(video_path=video_path, debug=arguments.debug)
        case "detection":
            if arguments.debug:
                show_tracks(tensor_path)
            else:
                process_with_tracking(tensor_path)
        case "synthetic":
            generate_synthetic_data(verbose=arguments.debug)
        case "render":
            render()
        case "analysis":
            analyze(
                video=arguments.video,
                tracking_file=tracking_path,
                keep=arguments.keep,
                debug=arguments.debug,
            )


if __name__ == "__main__":
    main()
