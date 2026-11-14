from illustrate import internal, models, representation
from illustrate.interactions import decor, neighbours


def main() -> None:
    models.draw()
    representation.draw()
    neighbours.draw()
    decor.draw()
    internal.draw()


if __name__ == "__main__":
    main()
