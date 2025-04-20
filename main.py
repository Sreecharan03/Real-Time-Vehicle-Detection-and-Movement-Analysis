import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Dict, List, Set, Iterable
import tempfile
import ast
import os
from pathlib import Path
import gdown

# Color palette for annotations
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])


def download_from_drive(url: str) -> str:
    """
    Downloads a file from Google Drive and returns the local path
    """
    if not url:
        return None

    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "input_video.mp4")

        # Convert sharing URL to direct download URL if needed
        if "drive.google.com/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, output_path, quiet=False)
        return output_path
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {str(e)}")
        return None

def parse_zone_coordinates(zone_text):
    try:
        coords = ast.literal_eval(zone_text)
        return np.array(coords)
    except:
        st.error("Invalid zone format. Please check the format and try again.")
        return None

class DetectionsManager:
    def _init_(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections]
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)

        return detections_all[detections_all.class_id != -1]

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position]
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(polygon=polygon, triggering_anchors=triggering_anchors)
        for polygon in polygons
    ]

class VideoProcessor:
    def _init_(
        self,
        model_path: str,
        zones_in: List[np.ndarray],
        zones_out: List[np.ndarray],
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        trace_length: int = 50
    ):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.zones_in = initiate_polygon_zones(zones_in, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(zones_out, [sv.Position.CENTER])
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.BLACK)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS,
            position=sv.Position.CENTER,
            trace_length=trace_length,
            thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidence = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        detections = sv.Detections(xyxy=boxes, confidence=confidence, class_id=class_ids)
        # For demonstration, set all class_ids to 0
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []
        # Iterate through matching zone_in/zone_out pairs
        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            detections_in_zones.append(detections[zone_in.trigger(detections=detections)])
            detections_out_zones.append(detections[zone_out.trigger(detections=detections)])

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()

        # Draw zones
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        # Draw detections
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        # Draw counts
        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id]
                    )

        return annotated_frame

def process_video(
    model_path,
    video_path,
    output_path,
    zones_in,
    zones_out,
    conf_threshold,
    iou_threshold,
    trace_length
):
    processor = VideoProcessor(
        model_path=model_path,
        zones_in=zones_in,
        zones_out=zones_out,
        confidence_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        trace_length=trace_length
    )

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = processor.process_frame(frame)
        out.write(processed_frame)

        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count} of {total_frames}")

    cap.release()
    out.release()
    status_text.text("Processing complete!")
    return output_path

def generate_image_from_video(video_path: str) -> str:
    """
    Generates an image from the input video and returns the path to the generated image.
    """
    if not video_path:
        return None

    try:
        # Example palette usage
        colors = sv.ColorPalette(
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        )

        video_info = sv.VideoInfo.from_video_path(video_path)
        print(video_info)

        # extract video frame
        generator = sv.get_video_frames_generator(video_path)
        iterator = iter(generator)
        frame = next(iterator)

        # save first frame
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "generated_image.png")
        cv2.imwrite(image_path, frame)
        return image_path
    except Exception as e:
        st.error(f"Error generating image from video: {str(e)}")
        return None

def main():
    st.title("Traffic Flow Analysis")

    # Model path
    model_path = st.text_input(
        "Enter path to YOLO model file (best.pt)",
        "best.pt"
    )

    # Video input options
    input_option = st.radio(
        "Choose video input method",
        ["Upload Video", "Google Drive Link"]
    )

    video_path = None

    # --- KEY PART: Use session state to avoid repeated downloads ---
    if "last_drive_url" not in st.session_state:
        st.session_state.last_drive_url = ""
    if "stored_video_path" not in st.session_state:
        st.session_state.stored_video_path = None

    if input_option == "Upload Video":
        video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        if video_file:
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            # Reset anything saved for Drive link
            st.session_state.last_drive_url = ""
            st.session_state.stored_video_path = video_path
    else:
        drive_url = st.text_input(
            "Enter Google Drive video link (make sure the file is shared with 'Anyone with the link')"
        )
        if drive_url:
            # Only download if the Drive URL changed or we have no stored path
            if drive_url != st.session_state.last_drive_url or not st.session_state.stored_video_path:
                video_path = download_from_drive(drive_url)
                st.session_state.last_drive_url = drive_url
                st.session_state.stored_video_path = video_path
            else:
                video_path = st.session_state.stored_video_path

    # Number of zones
    num_zones = st.number_input("Enter number of zones", min_value=1, max_value=4, value=1)

    # Zone input fields
    st.subheader("Zone Coordinates")
    st.write("Enter coordinates as list of lists: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")

    zones_out = []
    zones_in = []

    col1, col2 = st.columns(2)

    with col1:
        st.write("In Zones")
        for i in range(num_zones):
            zone_text = st.text_area(
                f"Zone In {i + 1}",
                "[[0, 0], [100, 0], [100, 100], [0, 100]]"
            )
            if zone_text:
                zone = parse_zone_coordinates(zone_text)
                if zone is not None:
                    zones_out.append(zone)

    with col2:
        st.write("Out Zones")
        for i in range(num_zones):
            zone_text = st.text_area(
                f"Zone Out {i + 1}",
                "[[0, 0], [100, 0], [100, 100], [0, 100]]"
            )
            if zone_text:
                zone = parse_zone_coordinates(zone_text)
                if zone is not None:
                    zones_in.append(zone)

    # Parameters
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.7)
    trace_length = st.slider("Trace Length", 10, 100, 50)

    # Process if we have enough data
    if (
        st.session_state.stored_video_path
        and len(zones_in) == num_zones
        and len(zones_out) == num_zones
    ):
        # Make sure we pick up the 'final' video_path from session
        video_path = st.session_state.stored_video_path

        if st.button("Generate Image from Video"):
            image_path = generate_image_from_video(video_path)
            if image_path:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                st.download_button(
                    label="Download generated image",
                    data=image_bytes,
                    file_name="generated_image.png",
                    mime="image/png"
                )

        if st.button("Process Video"):
            try:
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "output.mp4")

                output_path = process_video(
                    model_path=model_path,
                    video_path=video_path,
                    output_path=output_path,
                    zones_in=zones_in,
                    zones_out=zones_out,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    trace_length=trace_length
                )
                # Provide download link
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label="Download processed video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")


if _name_ == "_main_":
    main()
