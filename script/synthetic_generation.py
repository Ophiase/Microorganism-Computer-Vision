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

    def _create_bbox(self, x: float, y: float, size: float = 20.0, visible: bool = True) -> BoundingBox:
        return BoundingBox(
            Rectangle(x - size/2, y - size/2, size, size),
            index=self.current_id,
            visible=visible
        )

    def brownian_motion(self, num_frames: int, speed: float, noise_level: float=5.0) -> List[BoundingBox]:
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        visible = True
        for _ in range(num_frames):
            if visible:
                # Generate noise components separately
                dx = np.random.normal(0, speed) + np.random.normal(0, noise_level)
                dy = np.random.normal(0, speed) + np.random.normal(0, noise_level)
                
                new_x = x + dx
                new_y = y + dy
                
                if 0 <= new_x <= self.width and 0 <= new_y <= self.height:
                    x = new_x  # Update x coordinate
                    y = new_y  # Update y coordinate separately
                    trajectory.append(self._create_bbox(x, y))
                else:
                    trajectory.append(self._create_bbox(new_x, new_y, visible=False))
                    visible = False
            else:
                trajectory.append(self._create_bbox(0, 0, visible=False))
        self.current_id += 1
        return trajectory

    def directed_motion(self, num_frames: int, speed: float, angle: float) -> List[BoundingBox]:
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        visible = True
        theta = np.radians(angle)
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        
        for _ in range(num_frames):
            if visible:
                noise = np.random.normal(0, 0.3, 2)  # Increased noise
                new_x = x + vx + noise[0]
                new_y = y + vy + noise[1]
                
                if 0 <= new_x <= self.width and 0 <= new_y <= self.height:
                    x, y = new_x, new_y
                    trajectory.append(self._create_bbox(x, y))
                else:
                    trajectory.append(self._create_bbox(new_x, new_y, visible=False))
                    visible = False
            else:
                trajectory.append(self._create_bbox(0, 0, visible=False))
        self.current_id += 1
        return trajectory

    def confined_motion(self, num_frames: int, center: Tuple[float, float], 
                       radius: float) -> List[BoundingBox]:
        x, y = center
        trajectory = []
        visible = True
        for _ in range(num_frames):
            if visible:
                dx = np.random.uniform(-radius, radius) + np.random.normal(0, 0.5)  # Added noise
                dy = np.random.uniform(-radius, radius) + np.random.normal(0, 0.5)
                new_x = center[0] + dx
                new_y = center[1] + dy
                
                if 0 <= new_x <= self.width and 0 <= new_y <= self.height:
                    x, y = new_x, new_y
                    trajectory.append(self._create_bbox(x, y))
                else:
                    trajectory.append(self._create_bbox(new_x, new_y, visible=False))
                    visible = False
            else:
                trajectory.append(self._create_bbox(0, 0, visible=False))
        self.current_id += 1
        return trajectory

    def ctrw_motion(self, num_frames: int, base_speed: float) -> List[BoundingBox]:
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        wait_time = 0
        visible = True
        for _ in range(num_frames):
            if visible:
                if wait_time > 0:
                    trajectory.append(self._create_bbox(x, y))
                    wait_time -= 1
                else:
                    dx = np.random.normal(0, base_speed) + np.random.normal(0, 0.3)  # Added noise
                    dy = np.random.normal(0, base_speed) + np.random.normal(0, 0.3)
                    new_x = x + dx
                    new_y = y + dy
                    
                    if 0 <= new_x <= self.width and 0 <= new_y <= self.height:
                        x, y = new_x, new_y
                        trajectory.append(self._create_bbox(x, y))
                    else:
                        trajectory.append(self._create_bbox(new_x, new_y, visible=False))
                        visible = False
                    wait_time = np.random.poisson(5)
            else:
                trajectory.append(self._create_bbox(0, 0, visible=False))
        self.current_id += 1
        return trajectory

    def sinusoidal_motion(self, num_frames: int, speed: float, 
                         amplitude: float, frequency: float, 
                         noise_level: float=5.0
                         ) -> List[BoundingBox]:
        """New motion type: Sinusoidal swimming pattern"""
        x, y = np.random.uniform(100, 924, 2)
        trajectory = []
        visible = True
        angle = np.random.uniform(0, 2*np.pi)  # Random initial direction
        phase = 0
        
        for _ in range(num_frames):
            if visible:
                # Base movement with noise
                dx = speed * np.cos(angle) + np.random.normal(0, noise_level)
                dy = speed * np.sin(angle) + np.random.normal(0, noise_level)
                
                # Sinusoidal component perpendicular to movement
                perp_angle = angle + np.pi/2
                dx += amplitude * np.sin(phase) * np.cos(perp_angle)
                dy += amplitude * np.sin(phase) * np.sin(perp_angle)
                
                new_x = x + dx
                new_y = y + dy
                phase += frequency * 2*np.pi
                
                if 0 <= new_x <= self.width and 0 <= new_y <= self.height:
                    x, y = new_x, new_y
                    trajectory.append(self._create_bbox(x, y))
                else:
                    trajectory.append(self._create_bbox(new_x, new_y, visible=False))
                    visible = False
            else:
                trajectory.append(self._create_bbox(0, 0, visible=False))
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
            'name': 'sinusoidal',
            'method': generator.sinusoidal_motion,
            'params': {'speed': 2.2, 'amplitude': 10.0, 'frequency': 0.01}
        }
    ]

    for motion in motion_types:
        all_trajectories = []
        print(f"Generating {trajectories_per_type} {motion['name']} trajectories...")

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

        output_path = os.path.join(output_folder, f"synthetic_{motion['name']}.npy")
        np.save(output_path, np.array(synthetic_data, dtype=object))
        print(f"Saved {motion['name']} data to {output_path}")

    return True