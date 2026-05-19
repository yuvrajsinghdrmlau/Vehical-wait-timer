"""
Vehicle Wait Time Monitoring System
Author: Yuvraj Singh

Description:
Reads input video, detects & tracks vehicles,
monitors ROI, calculates wait time (MM:SS),
and writes annotated output video.
"""

import cv2
import logging
from ultralytics import YOLO
from collections import defaultdict


# ------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# -------------------------------------------------------
# ROI Manager
# -------------------------------------------------------
class ROIMonitor:
    def __init__(self, roi_coordinates):
        self.x1, self.y1, self.x2, self.y2 = roi_coordinates

    def contains(self, bbox):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        return self.x1 < cx < self.x2 and self.y1 < cy < self.y2

    def draw(self, frame):
        cv2.rectangle(frame,
                      (self.x1, self.y1),
                      (self.x2, self.y2),
                      (255, 0, 0), 2)


# -------------------------------------------------------
# Wait Time Tracker
# -------------------------------------------------------
class WaitTimeTracker:
    def __init__(self, fps, tolerance_frames=5):
        self.fps = fps
        self.entry_frames = {}
        self.missed_counter = defaultdict(int)
        self.tolerance = tolerance_frames

    def update(self, track_id, frame_number, inside_roi):
        if inside_roi:
            if track_id not in self.entry_frames:
                self.entry_frames[track_id] = frame_number
                logging.debug(f"Vehicle {track_id} entered ROI")

            self.missed_counter[track_id] = 0

            elapsed = (frame_number - self.entry_frames[track_id]) / self.fps
            return elapsed

        else:
            if track_id in self.entry_frames:
                self.missed_counter[track_id] += 1

                if self.missed_counter[track_id] > self.tolerance:
                    logging.debug(f"Vehicle {track_id} exited ROI")
                    del self.entry_frames[track_id]
                    del self.missed_counter[track_id]

        return None

    def active_count(self):
        return len(self.entry_frames)


# -------------------------------------------------------
# Main System
# -------------------------------------------------------
class VehicleWaitTimeSystem:
    def __init__(self, input_path, output_path, roi_coords):
        self.input_path = input_path
        self.output_path = output_path
        self.roi_monitor = ROIMonitor(roi_coords)
        self.model = YOLO("yolov8n.pt")

    def run(self):

        cap = cv2.VideoCapture(self.input_path)

        if not cap.isOpened():
            logging.error("Unable to open input video.")
            return

        ret, frame = cap.read()
        if not ret:
            logging.error("Unable to read first frame.")
            return

        height, width, _ = frame.shape
        fps = 30  # stable FPS

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        tracker = WaitTimeTracker(fps)
        frame_index = 0
        max_wait_global = 0

        logging.info("Processing started.")

        while ret:
            frame_index += 1

            results = self.model.track(frame,
                                       persist=True,
                                       tracker="bytetrack.yaml")

            for r in results:
                for box in r.boxes:

                    if box.id is None:
                        continue

                    # Only car class (COCO class 2)
                    if int(box.cls) != 2:
                        continue

                    track_id = int(box.id)
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    cv2.putText(frame, f"ID {track_id}",
                                (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)

                    inside = self.roi_monitor.contains(bbox)

                    wait_time = tracker.update(
                        track_id,
                        frame_index,
                        inside
                    )

                    if wait_time is not None:

                        # Update global max
                        if wait_time > max_wait_global:
                            max_wait_global = wait_time

                        minutes = int(wait_time // 60)
                        seconds = int(wait_time % 60)

                        time_text = f"{minutes:02}:{seconds:02}"

                        cv2.putText(frame, time_text,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)

            # Draw ROI
            self.roi_monitor.draw(frame)

            # Display Metrics
            cv2.putText(frame,
                        f"Vehicles in ROI: {tracker.active_count()}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            max_minutes = int(max_wait_global // 60)
            max_seconds = int(max_wait_global % 60)

            cv2.putText(frame,
                        f"Max Wait (Overall): {max_minutes:02}:{max_seconds:02}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            out.write(frame)
            ret, frame = cap.read()

        cap.release()
        out.release()

        logging.info(f"Processing completed. Total frames: {frame_index}")


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == "__main__":

    INPUT_VIDEO = "highway.mp4"
    OUTPUT_VIDEO = "output.avi"

    ROI_COORDS = (300, 200, 900, 700)

    system = VehicleWaitTimeSystem(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        roi_coords=ROI_COORDS
    )

    system.run()
