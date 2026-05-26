import os
import traceback
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# IMPORTANT: Make sure the import matches your actual file name
from arena_processor import ArenaWorker 

class AdvancedTrackingPopup(QDialog):
    def __init__(self, workspace, video_paths, model_names, rois, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAAM | Advanced Tracking Configuration")
        self.resize(1100, 950)
        self.workspace, self.video_paths, self.model_names, self.rois = workspace, video_paths, model_names, rois
        self.setStyleSheet("QDialog { background-color: #2b2b2b; color: white; } QGroupBox { font-weight: bold; color: #00a2ed; }")
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); form = QVBoxLayout(content)

        # 1. Inputs
        g1 = QGroupBox("1. Batch Selection"); l1 = QVBoxLayout()
        self.vid_list = QListWidget()
        for v in self.video_paths: self.vid_list.addItem(os.path.basename(v))
        self.vid_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection); self.vid_list.selectAll()
        self.model_combo = QComboBox(); self.model_combo.addItems(self.model_names)
        l1.addWidget(QLabel("Videos:")); l1.addWidget(self.vid_list); l1.addWidget(QLabel("Model:")); l1.addWidget(self.model_combo)
        g1.setLayout(l1); form.addWidget(g1)

        # 2. Parameters
        g2 = QGroupBox("2. Inference & Tracking Tuning"); l2 = QFormLayout()
        self.combo_task = QComboBox(); self.combo_task.addItems(["Detection", "Segmentation"])
        self.combo_method = QComboBox(); self.combo_method.addItems(["Custom Force-N", "Norfair", "BoTSORT"])
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setValue(0.25)
        self.spin_max_n = QSpinBox(); self.spin_max_n.setRange(1, 999); self.spin_max_n.setValue(1)
        self.spin_jump = QDoubleSpinBox(); self.spin_jump.setRange(1, 9999); self.spin_jump.setValue(50.0)
        self.spin_jump.setToolTip("Smoothing: Prevents 'zipping' by ignoring movements larger than this pixel value.")
        self.chk_auto_stitch = QCheckBox("Enable Ethogrid Auto-Stitching (Hungarian Ghost Match)"); self.chk_auto_stitch.setChecked(True)
        
        l2.addRow("Task Type:", self.combo_task); l2.addRow("Algorithm:", self.combo_method); l2.addRow("Confidence:", self.spin_conf)
        l2.addRow("Animals per Arena:", self.spin_max_n); l2.addRow("Max Smooth Jump (px):", self.spin_jump)
        l2.addRow("", self.chk_auto_stitch)
        g2.setLayout(l2); form.addWidget(g2)

        # 3. Export
        g3 = QGroupBox("3. Scientific Analytics & Video Exports"); l3 = QGridLayout()
        self.chk_track_vid = QCheckBox("Save Tracked Video (IDs)"); self.chk_track_vid.setChecked(True)
        self.chk_inf_vid = QCheckBox("Save Raw AI Inference Video"); self.chk_inf_vid.setChecked(True)
        self.chk_csv = QCheckBox("Export CSV Data"); self.chk_csv.setChecked(True)
        self.chk_xlsx = QCheckBox("Export Multi-Sheet Excel"); self.chk_xlsx.setChecked(True)
        self.chk_traj = QCheckBox("Render Trajectories"); self.chk_traj.setChecked(True)
        self.chk_heat = QCheckBox("Render Heatmaps"); self.chk_heat.setChecked(True)
        l3.addWidget(self.chk_track_vid, 0, 0); l3.addWidget(self.chk_inf_vid, 0, 1)
        l3.addWidget(self.chk_csv, 1, 0); l3.addWidget(self.chk_xlsx, 1, 1)
        l3.addWidget(self.chk_traj, 2, 0); l3.addWidget(self.chk_heat, 2, 1)
        g3.setLayout(l3); form.addWidget(g3)

        # 4. Logs & Progress 
        g4 = QGroupBox("4. Execution Monitor"); l4 = QVBoxLayout()
        lbl_batch = QLabel("Overall Batch Progress:")
        self.prog_batch = QProgressBar(); self.prog_batch.setValue(0)
        
        self.lbl_file = QLabel("Current File Progress:")
        self.prog_file = QProgressBar(); self.prog_file.setValue(0)
        
        stats_layout = QHBoxLayout()
        self.lbl_time = QLabel("⏱️ Time: 00:00:00 | ETR: --:--:--")
        self.lbl_speed = QLabel("⚡ Speed: 0.0 FPS")
        stats_layout.addWidget(self.lbl_time); stats_layout.addWidget(self.lbl_speed)
        
        self.txt_logs = QTextEdit(); self.txt_logs.setReadOnly(True)
        self.txt_logs.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        
        l4.addWidget(lbl_batch); l4.addWidget(self.prog_batch)
        l4.addWidget(self.lbl_file); l4.addWidget(self.prog_file)
        l4.addLayout(stats_layout); l4.addWidget(self.txt_logs)
        g4.setLayout(l4); form.addWidget(g4)

        scroll.setWidget(content); layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_go = QPushButton("🚀 LAUNCH BATCH ARENA PIPELINE")
        self.btn_go.setFixedHeight(50); self.btn_go.setStyleSheet("background: #28a745; font-weight: bold;")
        self.btn_go.clicked.connect(self.start_processing)
        
        self.btn_stop = QPushButton("🛑 STOP PIPELINE")
        self.btn_stop.setFixedHeight(50); self.btn_stop.setStyleSheet("background: #dc3545; font-weight: bold;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        
        btn_layout.addWidget(self.btn_go); btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

    def get_config(self):
        v_idx = [i for i in range(self.vid_list.count()) if self.vid_list.item(i).isSelected()]
        return {
            "videos": [self.video_paths[i] for i in v_idx],
            "model_name": self.model_combo.currentText(),
            "task_type": self.combo_task.currentText(),
            "method": self.combo_method.currentText(),
            "conf": self.spin_conf.value(),
            "max_n": self.spin_max_n.value(),
            "auto_stitch": self.chk_auto_stitch.isChecked(),
            "jump_thresh": self.spin_jump.value(),
            "rois": self.rois,
            "save_video": self.chk_track_vid.isChecked(),
            "save_inference_vid": self.chk_inf_vid.isChecked(),
            "save_csv": self.chk_csv.isChecked(),
            "save_xlsx": self.chk_xlsx.isChecked(),
            "save_traj": self.chk_traj.isChecked(),
            "save_heat": self.chk_heat.isChecked()
        }

    def start_processing(self):
        try:
            config = self.get_config()
            if not config["videos"]: return
            
            self.btn_go.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.txt_logs.clear()
            self.append_log("Initializing background engine...")
            
            self.worker = ArenaWorker(config, self.workspace)
            self.worker.overall_progress.connect(self.update_batch)
            self.worker.file_progress.connect(self.update_file)
            self.worker.log_signal.connect(self.append_log)
            self.worker.time_updated.connect(self.update_time)
            self.worker.speed_updated.connect(self.update_speed)
            self.worker.finished_signal.connect(self.on_finished)
            self.worker.start()
        except Exception as e:
            traceback.print_exc()
            self.append_log(f"❌ Failed to start worker: {str(e)}")
            self.btn_go.setEnabled(True)

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    @pyqtSlot(int, int, str)
    def update_batch(self, current, total, filename):
        try:
            self.prog_batch.setMaximum(total)
            self.prog_batch.setValue(current)
            self.prog_batch.setFormat(f"Video {current}/{total} - {filename}")
        except: pass

    @pyqtSlot(int, int, int)
    def update_file(self, percent, current_frame, total_frames):
        try:
            self.prog_file.setValue(percent)
            self.lbl_file.setText(f"Current File Progress: Frame {current_frame}/{total_frames}")
        except: pass

    @pyqtSlot(str, str)
    def update_time(self, elapsed, etr):
        try:
            self.lbl_time.setText(f"⏱️ Time: {elapsed} | ETR: {etr}")
        except: pass

    @pyqtSlot(float)
    def update_speed(self, fps):
        try:
            self.lbl_speed.setText(f"⚡ Speed: {fps:.1f} FPS")
        except: pass

    @pyqtSlot(str)
    def append_log(self, msg):
        try:
            # Using append automatically handles scrolling safely without manual scrollbar updates
            self.txt_logs.append(msg)
        except: pass

    @pyqtSlot(str)
    def on_finished(self, msg):
        try:
            self.btn_go.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.append_log(f"\n{msg}")
        except: pass