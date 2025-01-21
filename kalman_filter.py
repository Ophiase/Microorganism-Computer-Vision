import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional

from bounding_box import BoundingBox


class BacterialTracker:
    def __init__(self, 
                optical_flow_video: np.ndarray,
                max_missed_frames: int = 10,  
                process_noise: float = 1e-3,  # Adjusted for pixel coordinates
                measurement_noise: float = 5e-1,
                flow_percentile: float = 75):
        """
        Track bacteria across frames using Kalman Filter with optical flow integration

        Args:
            optical_flow_video: (num_frames, height, width, 2) array of optical flow vectors
            max_missed_frames: Remove tracks after this many undetected frames
        """
        self.optical_flow = optical_flow_video
        self.next_id = 0
        self.tracks: Dict[int, dict] = {}
        self.max_missed = max_missed_frames
        self.Q = np.eye(4) * process_noise  # Process noise
        self.R = np.eye(2) * measurement_noise  # Measurement noise
        self.flow_percentile = flow_percentile

    def _get_representative_flow(self, frame_idx: int, x: int, y: int, w: int, h: int) -> Tuple[float, float]:
        """Get representative flow using fastest vectors in ROI"""
        # Extract flow vectors in bounding box
        flow_x = self.optical_flow[frame_idx, y:y+h, x:x+w, 0].flatten()
        flow_y = self.optical_flow[frame_idx, y:y+h, x:x+w, 1].flatten()
        
        # Calculate speeds
        speeds = np.sqrt(flow_x**2 + flow_y**2)
        
        # Threshold for fastest vectors
        threshold = np.percentile(speeds, self.flow_percentile)
        mask = speeds >= threshold
        
        if np.any(mask):
            return np.median(flow_x[mask]), np.median(flow_y[mask])
        return np.median(flow_x), np.median(flow_y)

    def _initialize_kalman(self, bbox: BoundingBox, frame_idx: int) -> dict:
        x, y, w, h = bbox.getBbox()
        
        # Get representative flow using improved method
        of_x, of_y = self._get_representative_flow(frame_idx, x, y, w, h)
        
        return {
            'id': self.next_id,
            'bbox': bbox.getBbox(),
            'age': 0,
            'missed': 0,
            'state': np.array([x + w/2, y + h/2, of_x, of_y]),
            'covariance': np.eye(4) * 0.1,  # Initial uncertainty
            'history': []
        }

    def _predict(self, track: dict):
        """Kalman prediction step"""
        F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        track['state'] = F @ track['state']
        track['covariance'] = F @ track['covariance'] @ F.T + self.Q
        return track

    def _update(self, track: dict, measurement: Tuple[float, float]):
        """Kalman update step"""
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        y = measurement - H @ track['state']
        S = H @ track['covariance'] @ H.T + self.R
        K = track['covariance'] @ H.T @ np.linalg.inv(S)

        track['state'] = track['state'] + K @ y
        track['covariance'] = (np.eye(4) - K @ H) @ track['covariance']
        return track

    def _match_bboxes(self,
                      predicted: List[dict],
                      detections: List[BoundingBox],
                      frame_idx: int) -> List[Tuple[int, int]]:
        """Hungarian algorithm matching between predictions and detections"""
        cost_matrix = np.zeros((len(predicted), len(detections)))

        for i, track in enumerate(predicted):
            pred_cx, pred_cy = track['state'][:2]
            for j, bbox in enumerate(detections):
                det_cx = bbox.x + bbox.w/2
                det_cy = bbox.y + bbox.h/2
                cost_matrix[i, j] = np.sqrt(
                    (pred_cx - det_cx)**2 + (pred_cy - det_cy)**2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def update_tracks(self,
                      frame_bboxes: List[BoundingBox],
                      frame_idx: int) -> List[BoundingBox]:
        """
        Update tracks with new detections
        Returns list of (id, x, y, w, h) tuples
        """
        # Predict existing tracks
        for track in self.tracks.values():
            self._predict(track)

        # Match detections to predictions
        matches = self._match_bboxes(
            list(self.tracks.values()), frame_bboxes, frame_idx)

        # Update matched tracks
        used_detections = set()
        for i, j in matches:
            track = list(self.tracks.values())[i]
            x, y, w, h = frame_bboxes[j].getBbox()
            self._update(track, (x + w/2, y + h/2))
            track['bbox'] = (x, y, w, h)
            track['missed'] = 0
            track['age'] += 1
            track['history'].append((frame_idx, track['bbox']))
            used_detections.add(j)

        # Create new tracks for unmatched detections
        for j in range(len(frame_bboxes)):
            if j not in used_detections:
                new_track = self._initialize_kalman(frame_bboxes[j], frame_idx)
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Remove stale tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['missed'] > self.max_missed:
                to_remove.append(track_id)
            else:
                track['missed'] += 1

        for track_id in to_remove:
            del self.tracks[track_id]

        return [
            BoundingBox(t['id'], *t['bbox'], t['missed'] > 0)
            for t in self.tracks.values()
        ]
