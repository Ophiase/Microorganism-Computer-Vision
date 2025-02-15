import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from logic.structure.rectangle import Rectangle
from .structure.bounding_box import BoundingBox


def _state_to_bbox(state: np.ndarray, old: Rectangle) -> Rectangle:
    x, y, vx, vy = state
    w, h = old.w, old.h
    return Rectangle(int(x - w/2), int(y - h/2), w, h)


class BacterialTracker:
    def __init__(self,
                 optical_flow_video: np.ndarray,
                 max_missed_frames: int = 15,
                 process_noise: float = 1e-5,
                 measurement_noise: float = 1e-5,
                 max_assignment_distance: float = 30.0,
                 max_velocity: float = 50.0):
        self.optical_flow = optical_flow_video
        self.next_id = 0
        self.tracks: Dict[int, dict] = {}
        self.max_missed = max_missed_frames
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.max_assignment_distance = max_assignment_distance
        self.max_velocity = max_velocity

    def _initialize_kalman(self, rectangle: Rectangle) -> dict:
        cx, cy = rectangle.centroid()
        return {
            'id': self.next_id,
            'bbox': rectangle,
            'age': 0,
            'missed': 0,
            'state': np.array([cx, cy, 0.0, 0.0]),
            'covariance': np.eye(4) * 0.1,
            'history': []
        }

    def _predict(self, track: dict) -> dict:
        F = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        track['state'] = F @ track['state']
        track['state'][2:] = np.clip(
            track['state'][2:], -self.max_velocity, self.max_velocity)
        track['covariance'] = F @ track['covariance'] @ F.T + self.Q
        track['bbox'] = _state_to_bbox(track['state'], track['bbox'])
        return track

    def _update(self, track: dict, measurement: Tuple[float, float]) -> dict:
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        y = measurement - H @ track['state']
        S = H @ track['covariance'] @ H.T + self.R
        K = track['covariance'] @ H.T @ np.linalg.inv(S)
        track['state'] += K @ y
        track['state'][2:] = np.clip(
            track['state'][2:], -self.max_velocity, self.max_velocity)
        track['covariance'] = (np.eye(4) - K @ H) @ track['covariance']
        track['bbox'] = _state_to_bbox(track['state'], track['bbox'])
        return track

    def _match_bboxes(self, predicted: List[dict], detections: List[BoundingBox]) -> List[Tuple[int, int]]:
        cost_matrix = np.full((len(predicted), len(detections)), 1e9)
        for i, track in enumerate(predicted):
            pred_pos = track['state'][:2]
            for j, bbox in enumerate(detections):
                det_pos = bbox.centroid()
                dist = np.hypot(*(pred_pos - det_pos))
                if dist <= self.max_assignment_distance:
                    cost_matrix[i, j] = dist
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= self.max_assignment_distance]

    def update_tracks(self, frame_bboxes: List[BoundingBox], frame_idx: int) -> List[BoundingBox]:
        for track in self.tracks.values():
            self._predict(track)

        matches = self._match_bboxes(list(self.tracks.values()), frame_bboxes)
        used_dets = set()
        for i, j in matches:
            track = list(self.tracks.values())[i]
            self._update(track, frame_bboxes[j].centroid())
            track['missed'] = 0
            track['age'] += 1
            track['history'].append((frame_idx, track['bbox']))
            used_dets.add(j)

        for j in (set(range(len(frame_bboxes))) - used_dets):
            self.tracks[self.next_id] = self._initialize_kalman(
                frame_bboxes[j].rectangle)
            self.next_id += 1

        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if (t.update({'missed': t['missed'] + 1}) or t['missed'] <= self.max_missed)
        }
        return [BoundingBox(t['bbox'], t['id'], t['missed'] > 0) for t in self.tracks.values()]
