import argparse
import os
from common import DATA_FOLDER, DEFAULT_VIDEO, PREPROCESSED_FOLDER, TRACKING_FOLDER
from script.extract import process as extract_process
from script.transform import process as transform_process
from script.object_detection import process_with_tracking as detection_process, test_kalman_filter
from script.render import process as render_process
from script.statistical_tests import process as analysis_process
from script.synthetic_generation import generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(
        description="Microorganism Computer Vision Pipeline")
    parser.add_argument(
        "--video", type=str, help="Path to the video file to process",
        default=DEFAULT_VIDEO
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Disable verbose/debug output", default=False)
    parser.add_argument(
        "--task", type=str,
        choices=["extract", "transform", "detection", "render", "analysis", "synthetic"], required=True,
        help="Task to perform")

    args = parser.parse_args()

    video_path = os.path.join(DATA_FOLDER, args.video)
    npy_path = os.path.join(PREPROCESSED_FOLDER, args.video + ".npy")
    tracking_path = os.path.join(TRACKING_FOLDER, args.video + ".npy")

    match args.task:
        case "extract":
            extract_process()
        case "transform":
            transform_process(
                video_path=video_path,
                debug=args.verbose
            )
        case "detection":
            if args.verbose:
                test_kalman_filter(file_path=npy_path)
            else:
                detection_process(file_path=npy_path)
        case "synthetic":
            generate_synthetic_data(
                output_folder=TRACKING_FOLDER,
                verbose=args.verbose
            )
        case "render":
            render_process()
        case "analysis":
            analysis_process(
                video=args.video,
                tracking_file=tracking_path,
                debug=args.verbose
            )


if __name__ == "__main__":
    main()
