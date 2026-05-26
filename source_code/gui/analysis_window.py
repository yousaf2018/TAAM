import os, sys, cv2, json, pandas as pd, numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from backend.analysis_engine import AnalysisEngine

class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)

    def __init__(self, groups, arenas, arena_configs, config, workspace):
        super().__init__()
        self.groups, self.arenas, self.arena_configs, self.config, self.workspace = groups, arenas, arena_configs, config, workspace

    def run(self):
        all_sums, all_kins, stems = [], [], []
        files = [p for paths in self.groups.values() for p in paths]
        try:
            for i, p in enumerate(files):
                g_name = next(g for g, paths in self.groups.items() if p in paths)
                stems.append(os.path.splitext(os.path.basename(p))[0])
                sums, kins = AnalysisEngine.calculate_behavior(p, self.config, self.arenas, self.arena_configs, self.log_signal)
                for s in sums: s['Group'] = g_name; s['Source'] = os.path.basename(p); all_sums.append(s)
                kins['Group'] = g_name; kins['Source'] = os.path.basename(p); all_kins.append(kins)
                self.progress_signal.emit(int(((i+1)/len(files))*100))
            
            if all_sums:
                name = "_".join(stems[:2]) + f"_n{len(stems)}_analysis.xlsx"
                with pd.ExcelWriter(os.path.join(self.workspace, name)) as writer:
                    pd.DataFrame(all_sums).to_excel(writer, sheet_name="Grand_Summary", index=False)
                    pd.concat(all_kins).to_excel(writer, sheet_name="Detailed_Kinetics", index=False)
                self.finished_signal.emit(f"✅ Exported: {name}")
        except Exception as e: self.log_signal.emit(f"❌ Error: {str(e)}")

class AnalysisModule(QDialog):
    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAAM | Scientific Analytics")
        self.resize(1600, 950); self.workspace = os.path.abspath(workspace)
        self.experimental_groups, self.arenas, self.arena_configs = {}, [], {}
        self.video_frame, self.video_path_stored = None, ""
        
        self.setStyleSheet("""
            QDialog { background-color: #050505; }
            QWidget { background-color: #050505; color: #efefef; font-family: 'Segoe UI'; }
            QFrame#Sidebar { background-color: #0d0d0d; border-right: 1px solid #333; }
            QGroupBox { border: 1px solid #333; margin-top: 10px; font-weight: bold; color: #0078d4; }
            QPushButton { background-color: #0078d4; border: none; padding: 10px; font-weight: bold; border-radius: 4px; color: white; cursor: pointinghand; }
            QPushButton:hover { background-color: #008af0; }
            QLineEdit, QDoubleSpinBox, QComboBox { background: #1a1a1a; border: 1px solid #333; padding: 5px; color: #39FF14; }
            QListWidget { background: #000; border: 1px solid #222; }
            QTextEdit#Console { background: #000; color: #39FF14; font-family: 'Consolas'; font-size: 11px; border: 1px solid #222; }
            QProgressBar { height: 12px; border-radius: 6px; background: #111; }
        """)
        self.init_ui()

    def _make_prec_spin(self, v):
        s = QDoubleSpinBox(); s.setDecimals(15); s.setRange(0, 1e18); s.setValue(v); return s

    def _make_btn(self, t, f):
        b = QPushButton(t); b.setCursor(Qt.CursorShape.PointingHandCursor); b.clicked.connect(f); return b

    def init_ui(self):
        self.main_layout = QHBoxLayout(self)
        sidebar = QVBoxLayout()
        
        g1 = QGroupBox("1. Scientific Parameters")
        pl = QFormLayout()
        self.sp_conv = self._make_prec_spin(18.74); self.sp_fps = self._make_prec_spin(24.99); self.sp_dur = self._make_prec_spin(3549.7)
        pl.addRow("Conversion:", self.sp_conv); pl.addRow("FPS:", self.sp_fps); pl.addRow("Duration:", self.sp_dur)
        g1.setLayout(pl); sidebar.addWidget(g1)

        g2 = QGroupBox("2. Persistence"); sl = QVBoxLayout()
        sl.addWidget(self._make_btn("💾 Save Session", self.save_settings)); sl.addWidget(self._make_btn("📂 Load Session", self.load_settings))
        g2.setLayout(sl); sidebar.addWidget(g2)

        g3 = QGroupBox("3. Grouping"); gl = QVBoxLayout()
        self.list_groups = QListWidget(); self.list_files = QListWidget(); self.list_groups.itemClicked.connect(self.update_file_list)
        btn_r = QHBoxLayout(); btn_r.addWidget(self._make_btn("+ Group", self.add_group)); btn_r.addWidget(self._make_btn("+ Excel", self.add_xlsx))
        gl.addWidget(self.list_groups); gl.addWidget(self.list_files); gl.addLayout(btn_r); gl.addWidget(self._make_btn("Remove Item", self.remove_item))
        g3.setLayout(gl); sidebar.addWidget(g3); sidebar.addStretch(); self.main_layout.addLayout(sidebar, 1)

        content = QVBoxLayout()
        self.view_label = QLabel("Reference Viewport"); self.view_label.setFixedSize(800, 450); self.view_label.setStyleSheet("background:black; border:2px solid #0078d4;")
        content.addWidget(self.view_label, 0, Qt.AlignmentFlag.AlignCenter)

        g5 = QGroupBox("4. Calibration (X and Y Dot Adjustment)")
        cl = QVBoxLayout(); cl_h = QHBoxLayout(); self.combo_arena = QComboBox(); self.combo_arena.setCursor(Qt.CursorShape.PointingHandCursor); self.combo_arena.currentIndexChanged.connect(self.update_view)
        cl_h.addWidget(QLabel("Tank:")); cl_h.addWidget(self.combo_arena, 1)
        self.sld_x = QSlider(Qt.Orientation.Horizontal); self.sld_y = QSlider(Qt.Orientation.Horizontal)
        for s in [self.sld_x, self.sld_y]: s.setRange(0, 100); s.setValue(50); s.valueChanged.connect(self.adjust_center)
        cl.addLayout(cl_h); cl.addWidget(QLabel("X %")); cl.addWidget(self.sld_x); cl.addWidget(QLabel("Y %")); cl.addWidget(self.sld_y); g5.setLayout(cl); content.addWidget(g5)

        self.prog_bar = QProgressBar(); content.addWidget(self.prog_bar)
        self.console = QTextEdit(); self.console.setObjectName("Console"); self.console.setReadOnly(True); self.console.setFixedHeight(120); content.addWidget(self.console)

        bl2 = QHBoxLayout(); bl2.addWidget(self._make_btn("🖼️ Load Video", self.load_video)); bl2.addWidget(self._make_btn("📐 Load ROI JSON", self.load_roi))
        content.addLayout(bl2); self.btn_go = self._make_btn("🚀 LAUNCH SCIENTIFIC EXCEL EXPORT", self.run_extraction); self.btn_go.setFixedHeight(70); self.btn_go.setStyleSheet("background:#28a745; font-size:18px;"); content.addWidget(self.btn_go)
        self.main_layout.addLayout(content, 3)

    def update_view(self):
        if self.video_frame is None: return
        canvas = self.video_frame.copy(); cur = self.combo_arena.currentIndex()
        for i, roi in enumerate(self.arenas):
            # Coordinates
            rx, ry, rw, rh = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
            config = self.arena_configs.get(i, {'cx_pct': 50.0, 'cy_pct': 50.0})
            is_active = (i == cur); color = (0,255,0) if not is_active else (255,255,0)
            
            # FIX: FULL BORDER DRAWING
            # Rectangle / Grid
            if roi.get('type') == 'rect' or roi.get('is_subcell'):
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, 2)
            # Circle
            elif roi.get('type') == 'circle':
                center = (int(rx + rw/2), int(ry + rh/2))
                cv2.circle(canvas, center, int(rw/2), color, 2)
            else: # Default Fallback
                cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color, 2)

            # Centroid Dot
            cx = int(rx + rw * (config['cx_pct']/100.0)); cy = int(ry + rh * (config['cy_pct']/100.0))
            cv2.circle(canvas, (cx, cy), 6, (0,0,255), -1)
            
        qimg = QImage(canvas.data, canvas.shape[1], canvas.shape[0], canvas.shape[1]*3, QImage.Format.Format_BGR888)
        self.view_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.view_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    # --- LOGIC ---
    def load_roi(self):
        p, _ = QFileDialog.getOpenFileName(self, "ROI", self.workspace, "JSON (*.json)")
        if p:
            with open(p, 'r') as f:
                data = json.load(f); self.arenas = []
                for d in data:
                    if d['type'] == 'grid':
                        rx, ry, rw, rh, r, c = d['x'], d['y'], d['w'], d['h'], d['grid'][0], d['grid'][1]; cw, ch = rw/c, rh/r
                        for row in range(r):
                            for col in range(c): self.arenas.append({'x':rx+col*cw, 'y':ry+row*ch, 'w':cw, 'h':ch, 'is_subcell':True})
                    else: self.arenas.append({'x':d['x'], 'y':d['y'], 'w':d['w'], 'h':d['h'], 'type':d.get('type','rect')})
            self.combo_arena.clear(); [self.combo_arena.addItem(f"Arena {i+1}") for i in range(len(self.arenas))]; self.update_view()

    def run_extraction(self):
        if not self.experimental_groups or not self.arenas: QMessageBox.warning(self, "Audit", "Add groups and load ROI first."); return
        cfg = {'fps': self.sp_fps.value(), 'conversion': self.sp_conv.value(), 'duration': self.sp_dur.value()}
        self.btn_go.setEnabled(False); self.console.clear()
        self.worker = AnalysisWorker(self.experimental_groups, self.arenas, self.arena_configs, cfg, self.workspace)
        self.worker.log_signal.connect(self.console.append); self.worker.progress_signal.connect(self.prog_bar.setValue)
        self.worker.finished_signal.connect(lambda m: (self.btn_go.setEnabled(True), QMessageBox.information(self, "TAAM", m))); self.worker.start()

    def save_settings(self):
        data = {"params": {"conv": self.sp_conv.value(), "fps": self.sp_fps.value(), "dur": self.sp_dur.value()}, "arenas": self.arena_configs, "groups": self.experimental_groups, "video": self.video_path_stored, "roi": self.arenas}
        with open(os.path.join(self.workspace, "endpoints.json"), 'w') as f: json.dump(data, f, indent=4); self.console.append("✅ endpoints.json saved.")
    def load_settings(self):
        p = os.path.join(self.workspace, "endpoints.json")
        if os.path.exists(p):
            with open(p, 'r') as f:
                d = json.load(f); pr = d.get('params', {})
                self.sp_conv.setValue(pr.get('conv', 18.74)); self.sp_fps.setValue(pr.get('fps', 24.99)); self.sp_dur.setValue(pr.get('dur', 3549.7))
                self.arena_configs = {int(k): v for k,v in d.get('arenas', {}).items()}; self.experimental_groups = d.get('groups', {}); self.arenas = d.get('roi', []); self.video_path_stored = d.get('video', "")
            self.list_groups.clear(); self.list_groups.addItems(self.experimental_groups.keys()); self.combo_arena.clear(); self.combo_arena.addItems([f"Arena {i+1}" for i in range(len(self.arenas))])
            if self.video_path_stored and os.path.exists(self.video_path_stored):
                cap = cv2.VideoCapture(self.video_path_stored); stat, f = cap.read(); cap.release(); self.video_frame = f if stat else None
            self.update_view(); self.console.append("📂 Session restored.")
    def add_group(self):
        t, ok = QInputDialog.getText(self, 'Group', 'Enter Name:');
        if ok and t: self.list_groups.addItem(t); self.experimental_groups[t] = []
    def add_xlsx(self):
        it = self.list_groups.currentItem()
        if it: fs, _ = QFileDialog.getOpenFileNames(self, "Excel", self.workspace, "Excel (*.xlsx)"); self.experimental_groups[it.text()].extend(fs); self.update_file_list(it)
    def update_file_list(self, it):
        self.list_files.clear(); [self.list_files.addItem(os.path.basename(p)) for p in self.experimental_groups[it.text()]]
    def remove_item(self):
        fi, gi = self.list_files.currentItem(), self.list_groups.currentItem()
        if fi: self.experimental_groups[gi.text()].remove(next(p for p in self.experimental_groups[gi.text()] if os.path.basename(p)==fi.text())); self.update_file_list(gi)
        elif gi: self.experimental_groups.pop(gi.text()); self.list_groups.takeItem(self.list_groups.row(gi)); self.list_files.clear()
    def load_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", self.workspace, "Videos (*.mp4 *.avi *.MP4 *.MOV)"); cap = cv2.VideoCapture(p); stat, f = cap.read()
        if stat: self.video_path_stored = p; self.video_frame = f; self.update_view()
        cap.release()
    def adjust_center(self):
        idx = self.combo_arena.currentIndex()
        if idx != -1: self.arena_configs[idx] = {'cx_pct': float(self.sld_x.value()), 'cy_pct': float(self.sld_y.value())}; self.update_view()