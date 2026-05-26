import os
import traceback
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class AdvancedTrackingPopup(QDialog):
    def __init__(self, workspace, video_paths, model_names, rois, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAAM | EthoGrid Tracking Engine")
        self.resize(1150, 950)
        self.workspace, self.video_paths, self.model_names, self.rois = workspace, video_paths, model_names, rois
        self.setStyleSheet("QDialog { background-color: #2b2b2b; color: white; } QGroupBox { font-weight: bold; color: #00a2ed; }")
        self.setup_ui()
        self.load_settings()

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
        self.combo_method = QComboBox(); self.combo_method.addItems(["Custom Force-N", "Confidence Filter", "Norfair", "BoTSORT"])
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setValue(0.25)
        self.spin_max_n = QSpinBox(); self.spin_max_n.setRange(1, 999); self.spin_max_n.setValue(1)
        self.spin_iou = QDoubleSpinBox(); self.spin_iou.setRange(0.01, 1.0); self.spin_iou.setValue(0.45)
        self.spin_time_gap = QDoubleSpinBox(); self.spin_time_gap.setRange(0.1, 999.0); self.spin_time_gap.setValue(2.0)
        self.spin_sample_rate = QSpinBox(); self.spin_sample_rate.setRange(1, 100); self.spin_sample_rate.setValue(1)
        self.chk_auto_stitch = QCheckBox("Enable Auto-Stitching (Hungarian Ghost Match)"); self.chk_auto_stitch.setChecked(True)
        
        l2.addRow("Task Type:", self.combo_task); l2.addRow("Tracking Algorithm:", self.combo_method)
        l2.addRow("Confidence Threshold:", self.spin_conf); l2.addRow("Max Animals per Tank/Arena:", self.spin_max_n)
        l2.addRow("NMS IOU Threshold:", self.spin_iou); l2.addRow("Trajectory Time Gap (sec):", self.spin_time_gap)
        l2.addRow("Frame Sample Rate:", self.spin_sample_rate); l2.addRow("", self.chk_auto_stitch)
        g2.setLayout(l2); form.addWidget(g2)

        # 3. Visualization Tuning
        g_viz = QGroupBox("3. Visual Output Adjustments"); l_viz = QFormLayout()
        self.spin_bbox_thick = QSpinBox(); self.spin_bbox_thick.setRange(1, 15); self.spin_bbox_thick.setValue(3)
        self.spin_lbl_scale = QDoubleSpinBox(); self.spin_lbl_scale.setRange(0.1, 5.0); self.spin_lbl_scale.setSingleStep(0.1); self.spin_lbl_scale.setValue(0.6)
        self.spin_lbl_thick = QSpinBox(); self.spin_lbl_thick.setRange(1, 10); self.spin_lbl_thick.setValue(2)
        self.spin_dot_size = QSpinBox(); self.spin_dot_size.setRange(1, 30); self.spin_dot_size.setValue(5)
        
        l_viz.addRow("Bounding Box Thickness:", self.spin_bbox_thick)
        l_viz.addRow("Label Text Size (Scale):", self.spin_lbl_scale)
        l_viz.addRow("Label Text Thickness:", self.spin_lbl_thick)
        l_viz.addRow("Centroid Dot Size:", self.spin_dot_size)
        g_viz.setLayout(l_viz); form.addWidget(g_viz)

        # 4. Export
        g3 = QGroupBox("4. Scientific Analytics & Video Exports"); l3 = QGridLayout()
        self.chk_inf_vid = QCheckBox("Save Inference Video"); self.chk_inf_vid.setChecked(True)
        self.chk_track_vid = QCheckBox("Save Tracked Video"); self.chk_track_vid.setChecked(True)
        self.chk_csv = QCheckBox("Export Standard CSV"); self.chk_csv.setChecked(True)
        self.chk_centroid = QCheckBox("Export Centroid CSV"); self.chk_centroid.setChecked(True)
        self.chk_xlsx_track = QCheckBox("Export Excel (By Track)"); self.chk_xlsx_track.setChecked(True)
        self.chk_xlsx_tank = QCheckBox("Export Excel (By Tank)"); self.chk_xlsx_tank.setChecked(True)
        self.chk_traj = QCheckBox("Save Trajectory Image"); self.chk_traj.setChecked(True)
        self.chk_heat = QCheckBox("Save Heatmap Image"); self.chk_heat.setChecked(True)
        
        l3.addWidget(self.chk_inf_vid, 0, 0); l3.addWidget(self.chk_track_vid, 0, 1)
        l3.addWidget(self.chk_csv, 1, 0); l3.addWidget(self.chk_centroid, 1, 1)
        l3.addWidget(self.chk_xlsx_track, 2, 0); l3.addWidget(self.chk_xlsx_tank, 2, 1)
        l3.addWidget(self.chk_traj, 3, 0); l3.addWidget(self.chk_heat, 3, 1)
        g3.setLayout(l3); form.addWidget(g3)

        scroll.setWidget(content); layout.addWidget(scroll)

        self.btn_go = QPushButton("🚀 LAUNCH BATCH PIPELINE")
        self.btn_go.setFixedHeight(60); self.btn_go.setStyleSheet("background: #28a745; font-weight: bold; font-size: 14px;")
        self.btn_go.clicked.connect(self.on_launch_clicked) 
        layout.addWidget(self.btn_go)

    def on_launch_clicked(self):
        self.save_settings()
        self.accept()

    def load_settings(self):
        settings = QSettings("TAAM", "AdvancedTracking")
        if settings.contains("task_type"):
            self.combo_task.setCurrentText(settings.value("task_type", "Detection"))
            self.combo_method.setCurrentText(settings.value("method", "Custom Force-N"))
            self.spin_conf.setValue(float(settings.value("conf", 0.25)))
            self.spin_max_n.setValue(int(settings.value("max_n", 1)))
            self.spin_iou.setValue(float(settings.value("iou", 0.45)))
            self.spin_time_gap.setValue(float(settings.value("time_gap", 2.0)))
            self.spin_sample_rate.setValue(int(settings.value("sample_rate", 1)))
            self.chk_auto_stitch.setChecked(settings.value("auto_stitch", True, type=bool))

            self.spin_bbox_thick.setValue(int(settings.value("bbox_thick", 3)))
            self.spin_lbl_scale.setValue(float(settings.value("lbl_scale", 0.6)))
            self.spin_lbl_thick.setValue(int(settings.value("lbl_thick", 2)))
            self.spin_dot_size.setValue(int(settings.value("dot_size", 5)))

            self.chk_inf_vid.setChecked(settings.value("inf_vid", True, type=bool))
            self.chk_track_vid.setChecked(settings.value("track_vid", True, type=bool))
            self.chk_csv.setChecked(settings.value("csv", True, type=bool))
            self.chk_centroid.setChecked(settings.value("centroid", True, type=bool))
            self.chk_xlsx_track.setChecked(settings.value("xlsx_track", True, type=bool))
            self.chk_xlsx_tank.setChecked(settings.value("xlsx_tank", True, type=bool))
            self.chk_traj.setChecked(settings.value("traj", True, type=bool))
            self.chk_heat.setChecked(settings.value("heat", True, type=bool))

    def save_settings(self):
        settings = QSettings("TAAM", "AdvancedTracking")
        settings.setValue("task_type", self.combo_task.currentText())
        settings.setValue("method", self.combo_method.currentText())
        settings.setValue("conf", self.spin_conf.value())
        settings.setValue("max_n", self.spin_max_n.value())
        settings.setValue("iou", self.spin_iou.value())
        settings.setValue("time_gap", self.spin_time_gap.value())
        settings.setValue("sample_rate", self.spin_sample_rate.value())
        settings.setValue("auto_stitch", self.chk_auto_stitch.isChecked())

        settings.setValue("bbox_thick", self.spin_bbox_thick.value())
        settings.setValue("lbl_scale", self.spin_lbl_scale.value())
        settings.setValue("lbl_thick", self.spin_lbl_thick.value())
        settings.setValue("dot_size", self.spin_dot_size.value())

        settings.setValue("inf_vid", self.chk_inf_vid.isChecked())
        settings.setValue("track_vid", self.chk_track_vid.isChecked())
        settings.setValue("csv", self.chk_csv.isChecked())
        settings.setValue("centroid", self.chk_centroid.isChecked())
        settings.setValue("xlsx_track", self.chk_xlsx_track.isChecked())
        settings.setValue("xlsx_tank", self.chk_xlsx_tank.isChecked())
        settings.setValue("traj", self.chk_traj.isChecked())
        settings.setValue("heat", self.chk_heat.isChecked())

    def get_config(self):
        v_idx = [i for i in range(self.vid_list.count()) if self.vid_list.item(i).isSelected()]
        return {
            "videos": [self.video_paths[i] for i in v_idx],
            "model_name": self.model_combo.currentText(),
            "task_type": self.combo_task.currentText(),
            "method": self.combo_method.currentText(),
            "conf": self.spin_conf.value(),
            "max_n": self.spin_max_n.value(),
            "iou_threshold": self.spin_iou.value(),
            "time_gap_seconds": self.spin_time_gap.value(),
            "frame_sample_rate": self.spin_sample_rate.value(),
            "auto_stitch": self.chk_auto_stitch.isChecked(),
            
            "bbox_thick": self.spin_bbox_thick.value(),
            "lbl_scale": self.spin_lbl_scale.value(),
            "lbl_thick": self.spin_lbl_thick.value(),
            "dot_size": self.spin_dot_size.value(),

            "rois": self.rois,
            "save_inference_vid": self.chk_inf_vid.isChecked(),
            "save_video": self.chk_track_vid.isChecked(),
            "save_csv": self.chk_csv.isChecked(),
            "save_centroid_csv": self.chk_centroid.isChecked(),
            "save_excel_track": self.chk_xlsx_track.isChecked(),
            "save_excel_tank": self.chk_xlsx_tank.isChecked(),
            "save_traj": self.chk_traj.isChecked(),
            "save_heat": self.chk_heat.isChecked()
        }