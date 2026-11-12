import json

from motility.synthetic import Population

from microorganism_computer_vision.__main__ import main


def test_a_synthetic_video_goes_through_probe_run_and_bench(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    population = Population(
        stationary=1, diffusive=1, directed=3, confined=1, circular=1
    )
    scene = ["--set", "frames=30", "--set", "width=96", "--set", "height=80"]
    scene += [f"--set=population.{name}={count}" for name, count in population]

    assert main(["synth", "tiny", *scene]) == 0
    assert main(["probe", "output/synthetic/tiny/frames"]) == 0
    suggested = "output/runs/tiny-frames/probe/suggested.toml"
    assert main(["run", "output/synthetic/tiny/frames", "--config", suggested]) == 0
    assert main(["bench", "--scene", "output/synthetic/tiny/scene.toml"]) == 0

    run = tmp_path / "output/runs/tiny-frames/001"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["frames"] == 30
    assert (run / "tracks.csv").exists()
    assert (run / "report.md").exists()
