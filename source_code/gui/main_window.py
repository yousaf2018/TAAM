import os, sys, cv2, glob, shutil, json
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRectF
from ui.video_selector import VideoSelectorWidget
from ui.roi_designer import ROIDesigner
from workers.pipeline_worker import TAAMWorker
from workers.splitter_worker import SplitterWorker
from workers.arena_processor import ArenaWorker
from gui.tracking_popup import AdvancedTrackingPopup
from utils.stream_logger import StreamRedirector
from ui.analysis_module import AnalysisModule

class TAAMMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAAM | Track Aquatic Animal Model")
        self.resize(1650, 950)
        
        # Paths & State
        self.workspace = os.path.abspath(os.path.join(os.getcwd(), "TAAM_Workspace"))
        os.makedirs(self.workspace, exist_ok=True)
        self.project_annotations = {} 
        self.video_data = {} 
        self.current_video = None
        self.worker = None 
        self.sw = None 
        
        # Apply Professional High-Contrast Styling
        self.setStyleSheet(self.get_style())
        
        # Terminal Redirection for Blue Panel
        self.stdout_redirect = StreamRedirector(sys.stdout)
        self.stderr_redirect = StreamRedirector(sys.stderr)
        sys.stdout = self.stdout_redirect; sys.stderr = self.stderr_redirect
        
        self.setup_ui()
        
        # Connect Streams
        self.stdout_redirect.text_written.connect(self.log_sam.append)
        self.stderr_redirect.text_written.connect(self.log_sam.append)
        
        self.refresh_model_list()

    def get_style(self):
        return """
            QMainWindow { background-color: #050505; }
            QWidget { background-color: #050505; color: #efefef; font-family: 'Segoe UI'; }
            QFrame#Sidebar { background-color: #0d0d0d; border-right: 1px solid #333; }
            
            /* Action Buttons */
            QPushButton { background-color: #0078d4; border: none; padding: 10px; font-weight: bold; border-radius: 4px; color: white; }
            QPushButton:hover { background-color: #008af0; }
            
            /* RED STOP BUTTON - PROFESSIONAL HIGHLIGHT */
            QPushButton#CancelBtn { 
                background-color: #d9534f; 
                color: white; 
                font-weight: bold; 
                border: 1px solid #b52b27; 
            }
            QPushButton#CancelBtn:hover { background-color: #c9302c; }
            QPushButton#CancelBtn:disabled { background-color: #222; color: #555; border: none; }
            
            QLineEdit, QSpinBox, QComboBox, QDoubleSpinBox { background: #1a1a1a; border: 1px solid #333; padding: 5px; color: #39FF14; }
            QTextEdit { background-color: #000; font-family: 'Consolas'; border: 1px solid #333; font-size: 11px; }
            
            #LogApp { color: #39FF14; } #LogSAM { color: #4db8ff; }
            QLabel#StatVal { color: #ffc107; font-weight: bold; font-size: 18px; }
            QLabel#PathBox { color: #39FF14; font-family: 'Consolas'; font-size: 10px; background: #111; padding: 5px; border-radius: 4px; }
            
            QProgressBar { height: 12px; border-radius: 6px; background: #111; text-align: center; }
            QProgressBar::chunk { background-color: #0078d4; }
            
            QTabWidget::pane { border: 1px solid #222; top: -1px; background: #0a0a0a; }
            QTabBar::tab { background: #111; padding: 15px 30px; border: 1px solid #222; margin-right: 2px; }
            QTabBar::tab:selected { background: #0078d4; color: white; }
            ROIDesigner { background-color: #1a1a1a; border: 1px solid #0078d4; }
        """

    def _btn(self, text, func, name=""):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(func)
        if name: btn.setObjectName(name)
        return btn

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QHBoxLayout(central); layout.setContentsMargins(0,0,0,0)
        
        # --- SIDEBAR (CONTROL CENTER) ---
        sidebar = QFrame(); sidebar.setObjectName("Sidebar"); sidebar.setFixedWidth(360)
        side_lay = QVBoxLayout(sidebar)
        side_lay.addWidget(QLabel("🐬 TAAM CONTROL CENTER"))
        side_lay.addWidget(self._btn("📂 Select Workspace", self.set_workspace))
        self.lbl_ws = QLabel(self.workspace); self.lbl_ws.setObjectName("PathBox"); self.lbl_ws.setWordWrap(True)
        side_lay.addWidget(self.lbl_ws)
        
        side_lay.addWidget(QLabel("SOURCE VIDEO QUEUE"))
        self.list_vids = QListWidget(); self.list_vids.setCursor(Qt.CursorShape.PointingHandCursor)
        self.list_vids.currentRowChanged.connect(self.load_preview); side_lay.addWidget(self.list_vids)
        btn_box = QHBoxLayout(); btn_box.addWidget(self._btn("📥 Add Video", self.add_videos)); btn_box.addWidget(self._btn("🗑️ Remove", self.remove_video))
        side_lay.addLayout(btn_box)
        
        model_group = QGroupBox("Model Manager")
        m_lay = QVBoxLayout(); self.list_models = QListWidget(); self.list_models.setCursor(Qt.CursorShape.PointingHandCursor)
        m_lay.addWidget(self.list_models); m_lay.addWidget(self._btn("📂 Import External .pt", self.import_custom_model))
        m_lay.addWidget(self._btn("❌ Delete Selected", self.delete_model))
        model_group.setLayout(m_lay); side_lay.addWidget(model_group)
        
        stat_group = QGroupBox("Stats")
        sl = QFormLayout(); self.lbl_f = QLabel("0"); self.lbl_f.setObjectName("StatVal"); self.lbl_b = QLabel("0"); self.lbl_b.setObjectName("StatVal")
        sl.addRow("Annotated Vids:", self.lbl_f); sl.addRow("Total Bboxes:", self.lbl_b)
        stat_group.setLayout(sl); side_lay.addWidget(stat_group); side_lay.addStretch(); layout.addWidget(sidebar)
        # Add this to the Sidebar (side_lay)
        ann_mgmt_group = QGroupBox("Annotation Management")
        am_lay = QVBoxLayout()
        am_lay.addWidget(self._btn("💾 Save All Annotations", self.save_project_annotations))
        am_lay.addWidget(self._btn("📂 Load Annotations File", self.load_project_annotations))
        ann_mgmt_group.setLayout(am_lay)
        side_lay.addWidget(ann_mgmt_group)

        # Create the button in the Sidebar side_lay
        self.btn_behavior_analysis = self._btn("📉 Behavioral Endpoints", self.open_behavior_analysis)
        side_lay.addWidget(self.btn_behavior_analysis)

        # --- MAIN TABS ---
        content = QVBoxLayout(); content.setContentsMargins(15,15,15,15)
        self.tabs = QTabWidget(); self.tabs.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # TAB 1: SPLIT
        t0 = QWidget(); l0 = QVBoxLayout(t0)
        l0.addWidget(QLabel("1. Pre-Processor: Splits all videos currently in the 'Control Center' sidebar queue.", wordWrap=True))
        slay = QFormLayout()
        self.spin_split_sec = QSpinBox(); self.spin_split_sec.setRange(1, 999999); self.spin_split_sec.setValue(60)
        self.lbl_split_out = QLabel(os.path.join(self.workspace, "Splitted_Videos")); self.lbl_split_out.setObjectName("PathBox")
        slay.addRow("Split Seconds:", self.spin_split_sec); slay.addRow("Target Path:", self.lbl_split_out)
        l0.addLayout(slay); l0.addWidget(self._btn("📁 Change Split Folder", self.set_split_out))
        br_split = QHBoxLayout()
        self.btn_run_split = self._btn("✂️ START BATCH SPLITTING", self.run_video_splitter); self.btn_run_split.setFixedHeight(70)
        self.btn_stop_split = self._btn("🛑 STOP", self.stop_splitter, "CancelBtn")
        self.btn_stop_split.setFixedHeight(70); self.btn_stop_split.setEnabled(False)
        br_split.addWidget(self.btn_run_split, 2); br_split.addWidget(self.btn_stop_split, 1)
        l0.addLayout(br_split); l0.addStretch()
        self.tabs.addTab(t0, "1. SPLIT")
        
        # TAB 2: ANNOTATE
        t1 = QWidget(); l1 = QVBoxLayout(t1)
        c_lay = QHBoxLayout(); c_lay.addWidget(QLabel("Define Classes (comma sep):"))
        self.edit_classes = QLineEdit("zebrafish, medaka"); self.edit_classes.textChanged.connect(self.update_class_dropdown)
        c_lay.addWidget(self.edit_classes); l1.addLayout(c_lay)
        tool_lay = QHBoxLayout(); tool_lay.addWidget(QLabel("Active Label:")); self.combo_cls = QComboBox()
        self.combo_cls.currentIndexChanged.connect(self.change_active_annotation_class)
        tool_lay.addWidget(self.combo_cls)
        tool_lay.addWidget(self._btn("Clear Frame", lambda: self.video_ui.clear_current_frame())); tool_lay.addStretch()
        l1.addLayout(tool_lay)
        self.video_ui = VideoSelectorWidget(); self.video_ui.selection_changed.connect(self.on_ann_change)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.valueChanged.connect(self.seek)
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        l1.addWidget(self.lbl_frame_info); l1.addWidget(self.video_ui, 1); l1.addWidget(self.slider)
        self.tabs.addTab(t1, "2. ANNOTATE"); self.update_class_dropdown()

        # TAB 3: TRAIN
        t2 = QWidget(); l2 = QVBoxLayout(t2)
        self.edit_train_name = QLineEdit("Exp_1"); l2.addWidget(QLabel("Experiment/Model Name:")); l2.addWidget(self.edit_train_name)
        cfg_box = QGroupBox("Optimization & YOLO Training Parameters"); cfg_lay = QFormLayout()
        self.combo_task = QComboBox(); self.combo_task.addItems(["Detection", "Segmentation"]); self.combo_task.currentTextChanged.connect(self.update_yolo_dropdown)
        self.combo_yolo_ver = QComboBox(); self.edit_weights = QLineEdit(""); self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 999999); self.spin_epochs.setValue(25)
        self.spin_batch_yolo = QSpinBox(); self.spin_batch_yolo.setRange(1, 999999); self.spin_batch_yolo.setValue(16); self.spin_imgsz = QSpinBox(); self.spin_imgsz.setRange(1, 999999); self.spin_imgsz.setValue(640) 
        self.spin_max = QSpinBox(); self.spin_max.setRange(1, 999999); self.spin_max.setValue(500); self.spin_sam_batch = QSpinBox(); self.spin_sam_batch.setValue(16); self.spin_sam_chunk = QSpinBox(); self.spin_sam_chunk.setValue(5)
        cfg_lay.addRow("Task:", self.combo_task); cfg_lay.addRow("Model:", self.combo_yolo_ver); cfg_lay.addRow("SAM3 Batch:", self.spin_sam_batch); cfg_lay.addRow("SAM3 Chunk:", self.spin_sam_chunk); cfg_lay.addRow("Base Weights:", self.edit_weights); cfg_lay.addRow("", self._btn("📁 Browse .pt", self.browse_custom_pt)); cfg_lay.addRow("Epochs:", self.spin_epochs); cfg_lay.addRow("Batch:", self.spin_batch_yolo); cfg_lay.addRow("Img Size:", self.spin_imgsz); cfg_lay.addRow("Max Frames:", self.spin_max)
        cfg_box.setLayout(cfg_lay); l2.addWidget(cfg_box); self.update_yolo_dropdown()
        # Add this under your existing YOLO configuration layout
        self.combo_bbox_mode = QComboBox()
        self.combo_bbox_mode.addItems(["Dynamic (From Mask)", "Fixed (Centered)"])
        self.combo_bbox_mode.setCursor(Qt.CursorShape.PointingHandCursor)

        self.spin_fixed_bbox = QSpinBox()
        self.spin_fixed_bbox.setRange(5, 5000)
        self.spin_fixed_bbox.setValue(60)
        self.spin_fixed_bbox.setSuffix(" px")
        self.spin_fixed_bbox.setCursor(Qt.CursorShape.PointingHandCursor)

        # Add to your form layout (cfg_lay)
        cfg_lay.addRow("Detection BBox Mode:", self.combo_bbox_mode)
        cfg_lay.addRow("Fixed Box Size:", self.spin_fixed_bbox)
        br = QHBoxLayout()
        # LAUNCH BUTTON
        self.btn_run_full = self._btn("🚀 LAUNCH FULL AUTO PIPELINE", lambda: self.start_pipeline("FULL")); self.btn_run_full.setFixedHeight(70)
        # STOP BUTTON - SIZE MATCHED TO LAUNCH BUTTON
        self.btn_stop = self._btn("🛑 STOP TASK", self.stop_pipeline, "CancelBtn")
        self.btn_stop.setFixedHeight(70); self.btn_stop.setEnabled(False)
        br.addWidget(self.btn_run_full, 2); br.addWidget(self.btn_stop, 1)

        row_btns = QHBoxLayout()
        row_btns.addWidget(self._btn("🔍 STAGE 1: SAM3 ONLY", lambda: self.start_pipeline("SAM3_ONLY")))
        row_btns.addWidget(self._btn("🔥 STAGE 2: YOLO ONLY", lambda: self.start_pipeline("YOLO_ONLY")))
        
        l2.addLayout(br); l2.addLayout(row_btns); l2.addStretch()
        self.tabs.addTab(t2, "3. TRAIN")

        # TAB 4: ADVANCED ARENA (ROI)
        t3 = QWidget(); l3 = QHBoxLayout(t3); scroll = QScrollArea(); scroll.setFixedWidth(350); scroll.setWidgetResizable(True); s_widget = QWidget(); s_lay = QVBoxLayout(s_widget); roi_grp = QGroupBox("Arena Designer"); rl = QVBoxLayout(); self.combo_roi_type = QComboBox(); self.combo_roi_type.addItems(["rect", "circle", "grid"]); self.combo_roi_type.currentTextChanged.connect(self.update_roi_type); rl.addWidget(QLabel("Shape Type:")); rl.addWidget(self.combo_roi_type); 
        trans_grp = QGroupBox("Adjust Selected"); tl = QVBoxLayout()
        tl.addWidget(QLabel("Rotation (°)")); self.sld_rot = QSlider(Qt.Orientation.Horizontal); self.sld_rot.setRange(-180, 180); self.sld_rot.valueChanged.connect(self.transform_roi); tl.addWidget(self.sld_rot)
        tl.addWidget(QLabel("Width")); self.sld_w = QSlider(Qt.Orientation.Horizontal); self.sld_w.setRange(10, 3000); self.sld_w.valueChanged.connect(self.transform_roi); tl.addWidget(self.sld_w)
        tl.addWidget(QLabel("Height")); self.sld_h = QSlider(Qt.Orientation.Horizontal); self.sld_h.setRange(10, 3000); self.sld_h.valueChanged.connect(self.transform_roi); tl.addWidget(self.sld_h)
        trans_grp.setLayout(tl); rl.addWidget(trans_grp); rl.addWidget(self._btn("📂 Load ROI", self.load_roi)); rl.addWidget(self._btn("💾 Save ROI", self.save_roi)); rl.addWidget(self._btn("🗑️ Delete Selected ROI", self.delete_roi))
        roi_grp.setLayout(rl); s_lay.addWidget(roi_grp); s_lay.addStretch(); scroll.setWidget(s_widget); l3.addWidget(scroll)
        v_panel = QVBoxLayout(); self.roi_designer = ROIDesigner(); self.roi_designer.roi_selected.connect(self.sync_sliders)
        v_panel.addWidget(self.roi_designer, 1)
        btn_adv_row = QHBoxLayout(); self.btn_adv_run = self._btn("🎯 START ARENA TRACKING", self.start_tracking); self.btn_adv_run.setFixedHeight(60); self.btn_adv_run.setStyleSheet("background: #28a745")
        self.btn_adv_stop = self._btn("🛑 STOP TRACKING", self.stop_pipeline, "CancelBtn"); self.btn_adv_stop.setFixedHeight(60); self.btn_adv_stop.setEnabled(False)
        btn_adv_row.addWidget(self.btn_adv_run, 2); btn_adv_row.addWidget(self.btn_adv_stop, 1)
        v_panel.addLayout(btn_adv_row); l3.addLayout(v_panel, 1); self.tabs.addTab(t3, "4. ADVANCED")

        content.addWidget(self.tabs, 3); self.prog_bar = QProgressBar(); content.addWidget(self.prog_bar)
        log_lay = QHBoxLayout(); g1 = QGroupBox("Project Events"); v1 = QVBoxLayout(); self.log_app = QTextEdit(); self.log_app.setObjectName("LogApp"); self.log_app.setReadOnly(True); v1.addWidget(self.log_app); g1.setLayout(v1)
        g2 = QGroupBox("AI Engine Monitor"); v2 = QVBoxLayout(); self.log_sam = QTextEdit(); self.log_sam.setObjectName("LogSAM"); self.log_sam.setReadOnly(True); v2.addWidget(self.log_sam); g2.setLayout(v2)
        log_lay.addWidget(g1); log_lay.addWidget(g2); content.addLayout(log_lay, 1); layout.addLayout(content)

    # ==========================================
    # LOGIC (METHODS)
    # ==========================================
    def on_ann_change(self):
        if self.current_video:
            self.project_annotations[self.current_video] = self.video_ui.annotations
            active_vids = len([v for v in self.project_annotations.values() if any(len(b)>0 for b in v.values())])
            total_bboxes = sum(sum(len(b) for b in v.values()) for v in self.project_annotations.values())
            self.lbl_f.setText(str(active_vids)); self.lbl_b.setText(str(total_bboxes))

    def update_yolo_dropdown(self, t=""):
        self.combo_yolo_ver.clear()
        vers = ["yolov8n", "yolov9n", "yolov10n", "yolov11n", "yolov12n", "yolo26"]
        suffix = "-seg.pt" if self.combo_task.currentText() == "Segmentation" else ".pt"
        self.combo_yolo_ver.addItems([v + suffix for v in vers])

    def update_roi_type(self, t): self.roi_designer.current_type = t
    def delete_roi(self): 
        if self.roi_designer.selected_idx != -1: del self.roi_designer.shapes[self.roi_designer.selected_idx]; self.roi_designer.selected_idx = -1; self.roi_designer.update()
    def save_roi(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save ROI", os.path.join(self.workspace, "roi.json"), "JSON (*.json)")
        if p: self.roi_designer.save_to_json(p)
    def load_roi(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load ROI", os.path.join(self.workspace, "roi.json"), "JSON (*.json)")
        if p: self.roi_designer.load_from_json(p)
    def sync_sliders(self, idx):
        if idx == -1: return
        s = self.roi_designer.shapes[idx]
        self.sld_rot.blockSignals(True); self.sld_rot.setValue(int(s.get('angle', 0))); self.sld_rot.blockSignals(False)
        self.sld_w.blockSignals(True); self.sld_w.setValue(int(s['points'].width())); self.sld_w.blockSignals(False)
        self.sld_h.blockSignals(True); self.sld_h.setValue(int(s['points'].height())); self.sld_h.blockSignals(False)
    def transform_roi(self):
        idx = self.roi_designer.selected_idx
        if idx != -1:
            s = self.roi_designer.shapes[idx]; c = s['points'].center()
            s['angle'] = self.sld_rot.value(); s['points'] = QRectF(c.x()-self.sld_w.value()/2, c.y()-self.sld_h.value()/2, self.sld_w.value(), self.sld_h.value()); self.roi_designer.update()

    def set_workspace(self):
        d = QFileDialog.getExistingDirectory(self, "Workspace")
        if d: self.workspace = os.path.abspath(d); self.lbl_ws.setText(self.workspace); self.refresh_model_list(); self.lbl_split_out.setText(os.path.join(self.workspace, "Splitted_Videos"))

    def set_split_out(self):
        d = QFileDialog.getExistingDirectory(self, "Folder"); self.lbl_split_out.setText(d) if d else None
    def browse_custom_pt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Weights", "", "Weights (*.pt)"); self.edit_weights.setText(p) if p else None
    def refresh_model_list(self):
        self.list_models.clear(); self.list_models.addItems([p.split(os.sep)[-3] for p in glob.glob(os.path.join(self.workspace, "Models", "*/weights/best.pt"))])

    def update_class_dropdown(self):
        classes = [c.strip() for c in self.edit_classes.text().split(',')]
        self.combo_cls.clear(); self.combo_cls.addItems(classes)
        if classes: self.video_ui.set_current_class(0)
    def change_active_annotation_class(self, idx):
        if idx >= 0: self.video_ui.set_current_class(idx)

    def add_videos(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add", "", "Videos (*.mp4 *.avi *.MP4 *.MOV)")
        for p in ps:
            if p not in self.video_data: cap = cv2.VideoCapture(p); self.video_data[p] = int(cap.get(7)); cap.release(); self.list_vids.addItem(p)
        if ps: self.list_vids.setCurrentRow(self.list_vids.count()-1)

    def run_video_splitter(self):
        vids = [self.list_vids.item(i).text() for i in range(self.list_vids.count())]
        if not vids: return
        self.log_app.clear(); self.btn_run_split.setEnabled(False); self.btn_stop_split.setEnabled(True)
        self.sw = SplitterWorker(vids, self.spin_split_sec.value(), self.lbl_split_out.text())
        self.sw.log_signal.connect(self.log_app.append); self.sw.finished_signal.connect(self.on_split_done); self.sw.start()

    def stop_splitter(self):
        if self.sw: self.sw.stop()
    def on_split_done(self, m):
        self.btn_run_split.setEnabled(True); self.btn_stop_split.setEnabled(False); QMessageBox.information(self, "Splitter", m)

    def remove_video(self):
        idx = self.list_vids.currentRow()
        if idx >= 0: p = self.list_vids.item(idx).text(); self.project_annotations.pop(p, None); self.list_vids.takeItem(idx)

    def load_preview(self, row):
        if row < 0: return
        self.current_video = self.list_vids.item(row).text()
        
        cap = cv2.VideoCapture(self.current_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setRange(0, total - 1)
        
        # Ensure the video has a dictionary entry in project_annotations
        if self.current_video not in self.project_annotations:
            self.project_annotations[self.current_video] = {}
        
        # Link the Widget's annotations directly to the main storage
        self.video_ui.annotations = self.project_annotations[self.current_video]
        
        ret, f = cap.read()
        if ret:
            self.video_ui.set_current_frame(0, f)
            self.roi_designer.set_frame(f)
        cap.release()
        
        self.on_ann_change() 
        self.lbl_frame_info.setText(f"Frame: 0 / {total}")
        self.video_ui.update()
    
    def seek(self, val):
        if not hasattr(self, 'current_video'): return
        cap = cv2.VideoCapture(self.current_video); cap.set(1, val); ret, f = cap.read(); self.video_ui.set_current_frame(val, f); self.roi_designer.set_frame(f); cap.release(); self.lbl_frame_info.setText(f"Frame: {val} / {self.video_data.get(self.current_video, 0)}")

    def stop_pipeline(self):
        if hasattr(self, 'worker') and self.worker: self.worker.stop(); self.btn_stop.setEnabled(False); self.btn_adv_stop.setEnabled(False)

    def start_pipeline(self, mode):
        vids = [self.list_vids.item(i).text() for i in range(self.list_vids.count())]
        if not vids: return
        if mode != "YOLO_ONLY":
            missing = [os.path.basename(v) for v in vids if not any(len(b)>0 for b in self.project_annotations.get(v, {}).values())]
            if missing: QMessageBox.critical(self, "Audit", f"Unannotated:\n" + "\n".join(missing)); return
        classes = [c.strip() for c in self.edit_classes.text().split(',')]
        config = {"task_type": self.combo_task.currentText(),"bbox_mode": self.combo_bbox_mode.currentText(),"fixed_bbox_size": self.spin_fixed_bbox.value(), "yolo_ver": self.combo_yolo_ver.currentText(), "custom_weights": self.edit_weights.text(), "epochs": self.spin_epochs.value(), "yolo_batch": self.spin_batch_yolo.value(), "imgsz": self.spin_imgsz.value(), "max_frames": self.spin_max.value(), "tr": 70, "va": 20, "chunk_duration": 5, "sam_batch": self.spin_sam_batch.value(), "class_names": classes}
        self.btn_run_full.setEnabled(False); self.btn_stop.setEnabled(True); self.log_app.clear()
        self.worker = TAAMWorker(vids, self.project_annotations, self.workspace, self.edit_train_name.text(), config, mode)
        self.worker.log_app_signal.connect(self.log_app.append); self.worker.progress_signal.connect(lambda v, m: (self.prog_bar.setValue(v), self.log_app.append(f"STATUS: {m}")))
        self.worker.finished_signal.connect(lambda m: (self.btn_run_full.setEnabled(True), self.btn_stop.setEnabled(False), self.refresh_model_list(), QMessageBox.information(self, "TAAM", m))); self.worker.start()

    def start_tracking(self):
        try:
            video_count = self.list_vids.count(); model_count = self.list_models.count()
            if video_count == 0 or model_count == 0 or not self.roi_designer.shapes:
                QMessageBox.warning(self, "TAAM", "Required: Videos + Models + ROIs"); return
            vids = [self.list_vids.item(i).text() for i in range(video_count)]
            models = [self.list_models.item(i).text() for i in range(model_count)]
            
            clean_rois = []
            for s in self.roi_designer.shapes:
                item = {'type': str(s['type']), 'x': float(s['points'].x()), 'y': float(s['points'].y()), 'w': float(s['points'].width()), 'h': float(s['points'].height()), 'angle': float(s.get('angle', 0))}
                if s['type'] == 'grid': item['grid'] = (int(s['grid'][0]), int(s['grid'][1]))
                clean_rois.append(item)

            popup = AdvancedTrackingPopup(self.workspace, vids, models, clean_rois, self)
            if popup.exec():
                cfg = popup.get_config(); self.log_app.clear(); self.btn_adv_stop.setEnabled(True)
                self.worker = ArenaWorker(cfg, self.workspace); self.worker.log_signal.connect(self.log_app.append); self.worker.progress_signal.connect(lambda v, m: (self.prog_bar.setValue(v), self.log_app.append(m))); self.worker.finished_signal.connect(self.on_adv_tracking_finished); self.worker.start()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def on_adv_tracking_finished(self, msg): self.btn_adv_stop.setEnabled(False); QMessageBox.information(self, "TAAM", msg)

    def import_custom_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import", "", "YOLO (*.pt)");
        if p: dest = os.path.join(self.workspace, "Models", os.path.basename(p).replace(".pt", ""), "weights"); os.makedirs(dest, exist_ok=True); shutil.copy(p, os.path.join(dest, "best.pt")); self.refresh_model_list()

    def delete_model(self):
        item = self.list_models.currentItem()
        if item and QMessageBox.question(self, 'Confirm', f"Delete {item.text()}?") == QMessageBox.StandardButton.Yes: shutil.rmtree(os.path.join(self.workspace, "Models", item.text()), ignore_errors=True); self.refresh_model_list()
    def save_project_annotations(self):
        """Saves all current annotations and class names to a JSON file."""
        if not self.project_annotations:
            QMessageBox.warning(self, "Save Error", "No annotations found to save.")
            return

        default_path = os.path.join(self.workspace, "annotations.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", default_path, "JSON Files (*.json)")
        
        if file_path:
            # Prepare data: Convert QRectF objects to lists for JSON compatibility
            serializable_ann = {}
            for v_path, frames in self.project_annotations.items():
                serializable_ann[v_path] = {}
                for f_idx, boxes in frames.items():
                    # box is (QRectF, class_id)
                    serializable_ann[v_path][f_idx] = [[b[0].x(), b[0].y(), b[0].width(), b[0].height(), b[1]] for b in boxes]

            data_to_save = {
                "class_names": self.edit_classes.text(),
                "annotations": serializable_ann
            }

            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            self.log_app.append(f"✅ SYSTEM: Annotations exported to {os.path.basename(file_path)}")

    def load_project_annotations(self):
        """Loads and converts JSON data strictly to prevent Core Dump."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", self.workspace, "JSON Files (*.json)")
        if not file_path: return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if "class_names" in data:
                self.edit_classes.setText(str(data["class_names"]))
                self.update_class_dropdown()

            loaded_paths = [self.list_vids.item(i).text() for i in range(self.list_vids.count())]
            
            count_imported = 0
            for v_path, frames in data.get("annotations", {}).items():
                if v_path in loaded_paths:
                    self.project_annotations[v_path] = {}
                    for f_idx_str, boxes in frames.items():
                        # CONVERT: JSON Key "0" (string) -> 0 (int)
                        f_idx = int(f_idx_str) 
                        converted_boxes = []
                        for b in boxes:
                            # CONVERT: JSON List [x, y, w, h] -> QRectF (float)
                            rect = QRectF(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                            cid = int(b[4])
                            converted_boxes.append((rect, cid))
                        self.project_annotations[v_path][f_idx] = converted_boxes
                    count_imported += 1

            if self.current_video in self.project_annotations:
                self.video_ui.annotations = self.project_annotations[self.current_video]
            
            self.video_ui.update()
            self.on_ann_change() 
            self.log_app.append(f"📂 IMPORT: Loaded {count_imported} videos.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"JSON Data Error: {str(e)}")

   # Add this inside TAAMMainWindow class
    def open_behavior_analysis(self):
            """Pops up the Advanced Analysis Module"""
            try:
                # Maintain reference to avoid garbage collection
                self.behavior_suite = AnalysisModule(self.workspace, self)
                self.behavior_suite.show()
            except Exception as e:
                QMessageBox.critical(self, "Crash Prevented", f"Could not open analysis module: {str(e)}")
if __name__ == "__main__":
    app = QApplication(sys.argv); w = TAAMMainWindow(); w.show(); sys.exit(app.exec())