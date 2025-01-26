import numpy as np
import os
from typing import List, Tuple, Dict
from logic.structure.bounding_box import BoundingBox
from logic.structure.rectangle import Rectangle
from common import TRACKING_FOLDER


class SyntheticGenerator:
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.current_id = 0

    def _create_bbox(self, x: float, y: float, size: float = 20.0) -> BoundingBox:
        return BoundingBox(
            Rectangle(x - size/2, y - size/2, size, size),
            index=self.current_id,
            visible=True
        )

    def brownian_motion(self, num_frames: int, speed: float) -> List[BoundingBox]:
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        for _ in range(num_frames):
            trajectory.append(self._create_bbox(x, y))
            dx, dy = np.random.normal(0, speed, 2)
            x = np.clip(x + dx, 0, self.width)
            y = np.clip(y + dy, 0, self.height)
        self.current_id += 1
        return trajectory

    def directed_motion(self, num_frames: int, speed: float, angle: float) -> List[BoundingBox]:
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        vx = speed * np.cos(np.radians(angle))
        vy = speed * np.sin(np.radians(angle))
        for _ in range(num_frames):
            trajectory.append(self._create_bbox(x, y))
            x = np.clip(x + vx + np.random.normal(0, 0.1), 0, self.width)
            y = np.clip(y + vy + np.random.normal(0, 0.1), 0, self.height)
        self.current_id += 1
        return trajectory

    def confined_motion(self, num_frames: int, center: Tuple[float, float],
                        radius: float) -> List[BoundingBox]:
        x, y = center
        trajectory = []
        for _ in range(num_frames):
            trajectory.append(self._create_bbox(x, y))
            dx, dy = np.random.uniform(-radius, radius, 2)
            x = np.clip(center[0] + dx, 0, self.width)
            y = np.clip(center[1] + dy, 0, self.height)
        self.current_id += 1
        return trajectory

    def ctrw_motion(self, num_frames: int, base_speed: float) -> List[BoundingBox]:
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        wait_time = 0
        for _ in range(num_frames):
            trajectory.append(self._create_bbox(x, y))
            if wait_time > 0:
                wait_time -= 1
                continue
            dx, dy = np.random.normal(0, base_speed, 2)
            x = np.clip(x + dx, 0, self.width)
            y = np.clip(y + dy, 0, self.height)
            wait_time = np.random.poisson(5)
        self.current_id += 1
        return trajectory

    def circular_motion(self, num_frames: int, radius: float,
                        angular_speed: float) -> List[BoundingBox]:
        center = np.random.uniform(200, 824, 2)
        theta = np.random.uniform(0, 2*np.pi)
        trajectory = []
        for _ in range(num_frames):
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            trajectory.append(self._create_bbox(x, y))
            theta += np.radians(angular_speed)
            radius += np.random.normal(0, 0.1)  # Add slight radius variation
        self.current_id += 1
        return trajectory


def generate_synthetic_data(output_folder: str = TRACKING_FOLDER,
                            trajectories_per_type: int = 25,
                            num_frames: int = 200,
                            verbose: bool = False):
    generator = SyntheticGenerator()
    motion_types = [
        {
            'name': 'brownian',
            'method': generator.brownian_motion,
            'params': {'speed': 2.0}
        },
        {
            'name': 'directed',
            'method': generator.directed_motion,
            'params': {'speed': 1.5, 'angle': lambda: np.random.uniform(0, 360)}
        },
        {
            'name': 'confined',
            'method': generator.confined_motion,
            'params': {'center': (512, 512), 'radius': 50}
        },
        {
            'name': 'ctrw',
            'method': generator.ctrw_motion,
            'params': {'base_speed': 3.0}
        },
        {
            'name': 'circular',
            'method': generator.circular_motion,
            'params': {'radius': 40, 'angular_speed': 5}
        }
    ]

    for motion in motion_types:
        all_trajectories = []
        print(
            f"Generating {trajectories_per_type} {motion['name']} trajectories...")

        for _ in range(trajectories_per_type):
            resolved_params = {}
            for k, v in motion['params'].items():
                resolved_params[k] = v() if callable(v) else v
            trajectory = motion['method'](num_frames, **resolved_params)
            all_trajectories.append(trajectory)

        synthetic_data = [[] for _ in range(num_frames)]
        for traj in all_trajectories:
            for frame_idx, bbox in enumerate(traj):
                if frame_idx < num_frames:
                    synthetic_data[frame_idx].append(bbox)

        output_path = os.path.join(
            output_folder, f"synthetic_{motion['name']}.npy")
        np.save(output_path, np.array(synthetic_data, dtype=object))
        print(f"Saved {motion['name']} data to {output_path}")

    return True
