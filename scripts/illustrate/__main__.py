from illustrate import hypotheses
from illustrate.interactions import decor, neighbours
from illustrate.model import internal, models, representation


def main() -> None:
    models.draw()
    representation.draw()
    neighbours.draw()
    decor.draw()
    internal.draw()
    hypotheses.draw()


if __name__ == "__main__":
    main()
