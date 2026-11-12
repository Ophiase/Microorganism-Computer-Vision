import argparse
from collections.abc import Sequence

from microorganism_computer_vision.pipeline.extract import process as extract


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microorganism Computer Vision")
    parser.add_argument("--task", choices=("extract",), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    parse_arguments(argv)
    extract()


if __name__ == "__main__":
    main()
