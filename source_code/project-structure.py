import os

PROJECT_NAME = "TAAM_Workspace"

# Define the folder structure
folders = [
    "ui",
    "gui",
    "backend",
    "workers",
    "utils"
]

# Define the file contents
files = {
    "main.py": """import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import TAAMMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TAAMMainWindow()
    window.show()
    sys.exit(app.exec())
""",

    "ui/__init__.py": "",
    
    "ui/video_selector.py": """# ==========================================
# PASTE YOUR EXISTING video_selector.py HERE
# ==========================================

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt

# This is just a dummy placeholder so the app runs before you paste your code.
class VideoSelectorWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        lbl = QLabel("VideoSelectorWidget Placeholder\\n(Paste your real code in ui/video_selector.py)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: white; font-size: 16px;")
        layout.addWidget(lbl)
        self.setStyleSheet("background-color: black;")
        self.annotations = {0: []} # Dummy annotation

    def set_current_frame(self, val, frame):
        pass
""",

    "gui/__init__.py": "",

    "gui/main_window.py": """import os
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QListWidget, QGroupBox, 
    QTextEdit, QLabel, QProgressBar, QTabWidget, QComboBox, QMessageBox, QSlider
)
from PyQt6.QtCore import Qt

# Import UI and Worker
from ui.video_selector import VideoSelectorWidget
from workers.training_worker import TrainingWorker

class TAAMMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAAM - Track Any Aquatic Model")
        self.resize(1200, 850)
        
        self.project_dir = os.path.join(os.getcwd(), "TAAM_Data")
        os.makedirs(self.project_dir, exist_ok=True)
        
        self.current_video_path = None
        self.current_video_cap = None
        self.total_frames = 0
        self.project_annotations = {} 
        self.worker = None
        
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        header = QLabel("🦍 TAAM: Universal Animal Tracking Pipeline")
        header.setStyleSheet("font-size: 22px; font-weight: bold; background-color: #2b2b2b; color: #4db8ff; padding: 15px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.build_tab_1_annotate()
        self.build_tab_2_train()
        self.build_tab_3_track()

        log_group = QGroupBox("System Activity")
        log_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFixedHeight(100)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00;")
        
        log_layout.addWidget(self.progress_bar)
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

    def build_tab_1_annotate(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        left_panel = QVBoxLayout()
        self.btn_load_video = QPushButton("📁 Load Video")
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_video.setStyleSheet("padding: 10px; background-color: #0d6efd; color: white;")
        
        self.list_videos = QListWidget()
        self.list_videos.currentRowChanged.connect(self.select_video_from_list)
        
        left_panel.addWidget(self.btn_load_video)
        left_panel.addWidget(QLabel("Project Videos:"))
        left_panel.addWidget(self.list_videos)
        
        right_panel = QVBoxLayout()
        instruction = QLabel("Step 1: Draw bounding boxes on 3-5 different frames.")
        instruction.setStyleSheet("color: #ffc107; font-weight:bold;")
        
        self.video_selector = VideoSelectorWidget()
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        self.lbl_frame = QLabel("Frame: 0 / 0")
        
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(self.lbl_frame)
        timeline_layout.addWidget(self.slider)

        self.btn_proceed = QPushButton("Save Annotations & Proceed to Step 2 ➡️")
        self.btn_proceed.clicked.connect(self.save_and_proceed)
        self.btn_proceed.setStyleSheet("padding: 10px; background-color: #28a745; color: white; font-weight: bold;")
        
        right_panel.addWidget(instruction)
        right_panel.addWidget(self.video_selector, 1)
        right_panel.addLayout(timeline_layout)
        right_panel.addWidget(self.btn_proceed)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 3)
        self.tabs.addTab(tab, "Step 1: Annotate")

    def build_tab_2_train(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.btn_start_training = QPushButton("🚀 START AUTOMATED AI TRAINING")
        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_start_training.setMinimumHeight(60)
        self.btn_start_training.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #dc3545; color: white;")
        
        layout.addWidget(QLabel("TAAM will use SAM 3 to auto-generate a dataset from your annotations, then train YOLO."))
        layout.addWidget(self.btn_start_training)
        layout.addStretch()
        self.tabs.addTab(tab, "Step 2: Train AI")

    def build_tab_3_track(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.combo_models = QComboBox()
        self.combo_models.addItem("Select Custom Trained Model...")
        self.btn_track = QPushButton("🎯 START FAST TRACKING")
        self.btn_track.setMinimumHeight(50)
        
        layout.addWidget(QLabel("Select your trained model to perform high-speed inference on the entire video."))
        layout.addWidget(self.combo_models)
        layout.addWidget(self.btn_track)
        layout.addStretch()
        self.tabs.addTab(tab, "Step 3: Fast Track")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video (*.mp4 *.avi *.MP4 *.MOV)")
        if path:
            self.list_videos.addItem(path)
            self.list_videos.setCurrentRow(self.list_videos.count() - 1)

    def select_video_from_list(self, row):
        if row < 0: return
        self.current_video_path = self.list_videos.item(row).text()
        
        if self.current_video_cap: self.current_video_cap.release()
        self.current_video_cap = cv2.VideoCapture(self.current_video_path)
        self.total_frames = int(self.current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.setValue(0)
        self.slider_moved(0)

    def slider_moved(self, val):
        self.lbl_frame.setText(f"Frame: {val} / {self.total_frames}")
        if self.current_video_cap:
            self.current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            ret, frame = self.current_video_cap.read()
            if ret:
                self.video_selector.set_current_frame(val, frame)

    def save_and_proceed(self):
        if not self.video_selector.annotations:
            QMessageBox.warning(self, "Wait!", "Please annotate at least one frame.")
            return
        self.project_annotations = self.video_selector.annotations
        self.log(f"Saved {len(self.project_annotations)} frames of annotations.")
        self.tabs.setCurrentIndex(1)

    def start_training(self):
        if not self.current_video_path or not self.project_annotations:
            QMessageBox.warning(self, "Error", "Go back to Step 1 and annotate a video first.")
            return
            
        self.btn_start_training.setEnabled(False)
        self.log("Initializing TAAM Training Pipeline...")
        
        self.worker = TrainingWorker(self.current_video_path, self.project_annotations, self.project_dir)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.log_signal.connect(self.log)
        self.worker.error_signal.connect(lambda e: self.log(f"ERROR: {e}"))
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.start()

    def training_finished(self, weights_path):
        self.btn_start_training.setEnabled(True)
        self.combo_models.addItem(os.path.basename(weights_path))
        self.combo_models.setCurrentIndex(self.combo_models.count() - 1)
        QMessageBox.information(self, "Success", "AI Model Trained Successfully! Proceed to Step 3.")
        self.tabs.setCurrentIndex(2)

    def log(self, msg):
        self.log_console.append(f"> {msg}")
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())
""",

    "backend/__init__.py": "",

    "backend/dataset_generator.py": """import os
import cv2
import numpy as np
import torch

GLOBAL_SAM_MODEL = None
GLOBAL_SAM_PREDICTOR = None

class DatasetGenerator:
    def __init__(self, output_dir, log_signal):
        self.output_dir = output_dir
        self.log_signal = log_signal
        
        self.dataset_dir = os.path.join(self.output_dir, "YOLO_Dataset")
        self.images_dir = os.path.join(self.dataset_dir, "images", "train")
        self.labels_dir = os.path.join(self.dataset_dir, "labels", "train")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def load_sam(self):
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        if GLOBAL_SAM_MODEL is not None: return GLOBAL_SAM_PREDICTOR
        
        try:
            import sam3
            from sam3.model_builder import build_sam3_video_model
            with torch.inference_mode():
                GLOBAL_SAM_MODEL = build_sam3_video_model()
                GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
                GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
            return GLOBAL_SAM_PREDICTOR
        except ImportError:
            self.log_signal.emit("WARNING: SAM 3 not installed. Running in Mock Mode for UI testing.")
            return None

    def generate_yolo_dataset(self, video_path, annotations, max_frames=300):
        predictor = self.load_sam()
        
        # MOCK MODE if SAM 3 isn't installed yet
        if predictor is None:
            self.log_signal.emit("MOCK DATASET GEN: Creating fake dataset to test YOLO pipeline...")
            yaml_path = os.path.join(self.dataset_dir, "dataset.yaml")
            with open(yaml_path, 'w') as f:
                f.write(f"path: {self.dataset_dir}\\n")
                f.write(f"train: images/train\\n")
                f.write(f"val: images/train\\n")
                f.write(f"names:\\n  0: custom_animal\\n")
            return yaml_path

        # REAL MODE
        cap = cv2.VideoCapture(video_path)
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        state = predictor.init_state(video_path=video_path)
        
        for f_idx, boxes in annotations.items():
            for obj_id, (rect, cls_id) in enumerate(boxes):
                cx = rect.x() + rect.width() / 2
                cy = rect.y() + rect.height() / 2
                pt = torch.tensor([[cx, cy]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points(state, f_idx, obj_id, pt, lbl)

        self.log_signal.emit(f"Propagating SAM 3 for {max_frames} frames...")
        
        frames_processed = 0
        for f_idx, oids, _, masks, _ in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=max_frames
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            img_name = f"frame_{f_idx:05d}.jpg"
            cv2.imwrite(os.path.join(self.images_dir, img_name), frame)
            
            label_path = os.path.join(self.labels_dir, f"frame_{f_idx:05d}.txt")
            with open(label_path, 'w') as f_txt:
                for j, oid in enumerate(oids):
                    m = (masks[j] > 0.0).cpu().numpy().squeeze()
                    rows, cols = np.where(m)
                    if len(rows) > 0:
                        x_min, x_max = np.min(cols), np.max(cols)
                        y_min, y_max = np.min(rows), np.max(rows)
                        x_center = ((x_min + x_max) / 2) / img_width
                        y_center = ((y_min + y_max) / 2) / img_height
                        w = (x_max - x_min) / img_width
                        h = (y_max - y_min) / img_height
                        f_txt.write(f"0 {x_center} {y_center} {w} {h}\\n")
            
            frames_processed += 1
            if frames_processed >= max_frames: break

        predictor.clear_all_points_in_video(state)
        cap.release()
        
        yaml_path = os.path.join(self.dataset_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {self.dataset_dir}\\n")
            f.write(f"train: images/train\\n")
            f.write(f"val: images/train\\n")
            f.write(f"names:\\n  0: custom_animal\\n")
            
        return yaml_path
""",

    "workers/__init__.py": "",

    "workers/training_worker.py": """import os
from PyQt6.QtCore import QThread, pyqtSignal
from backend.dataset_generator import DatasetGenerator

class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, annotations, project_dir):
        super().__init__()
        self.video_path = video_path
        self.annotations = annotations
        self.project_dir = project_dir

    def run(self):
        try:
            self.log_signal.emit("Initializing SAM 3 Model...")
            self.progress_signal.emit(10)

            generator = DatasetGenerator(self.project_dir, self.log_signal)
            dataset_yaml = generator.generate_yolo_dataset(self.video_path, self.annotations, max_frames=300)
            
            self.progress_signal.emit(50)
            self.log_signal.emit(f"Dataset generated at: {dataset_yaml}")
            self.log_signal.emit("Starting YOLO Training...")

            try:
                from ultralytics import YOLO
                model = YOLO("yolov8n.pt") 
                
                results = model.train(
                    data=dataset_yaml,
                    epochs=10, # Kept low for testing, increase to 50 later
                    imgsz=640,
                    project=os.path.join(self.project_dir, "Models"),
                    name="TAAM_Tracker",
                    exist_ok=True,
                    device=0 
                )
                weights_path = os.path.join(self.project_dir, "Models", "TAAM_Tracker", "weights", "best.pt")
                self.progress_signal.emit(100)
                self.log_signal.emit(f"✅ Training Complete! Weights saved to {weights_path}")
                self.finished_signal.emit(weights_path)
            except Exception as yolo_err:
                self.log_signal.emit(f"YOLO Training Failed: {yolo_err}")
                self.error_signal.emit(str(yolo_err))

        except Exception as e:
            self.error_signal.emit(str(e))
""",

    "requirements.txt": """PyQt6==6.5.2
opencv-python==4.8.0.76
numpy==1.24.3
torch
ultralytics
# sam3 (Add your local sam3 installation command here)
"""
}

def build_project():
    if not os.path.exists(PROJECT_NAME):
        os.makedirs(PROJECT_NAME)
    os.chdir(PROJECT_NAME)

    # 1. Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # 2. Create files
    for filepath, content in files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"\\n✅ SUCCESS! Project structure created at: {os.path.abspath('.')}")
    print("\\n🚀 NEXT STEPS:")
    print("1. Open the folder: ", PROJECT_NAME)
    print("2. Navigate to 'ui/video_selector.py'")
    print("3. Delete the dummy placeholder code in that file, and paste your REAL video_selector.py code into it.")
    print("4. Run 'python main.py' to launch your new app!")

if __name__ == "__main__":
    build_project()