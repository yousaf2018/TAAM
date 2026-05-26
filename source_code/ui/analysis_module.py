import os, sys, cv2, json, math, pandas as pd, numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from backend.analysis_engine import AnalysisEngine

class AnalysisWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)

    def __init__(self, groups, arenas, arena_configs, config, metrics, workspace):
        super().__init__()
        self.groups, self.arenas, self.arena_configs = groups, arenas, arena_configs
        self.config, self.selected_metrics, self.workspace = config, metrics, workspace

    def run(self):
        all_sums, all_kins, stems = [], [], []
        files = [p for paths in self.groups.values() for p in paths]
        try:
            for i, p in enumerate(files):
                g_name = next(g for g, paths in self.groups.items() if p in paths)
                stems.append(os.path.splitext(os.path.basename(p))[0])
                
                sum_list, kinetics_df = AnalysisEngine.calculate_behavior(
                    p, self.config, self.arenas, self.arena_configs, self.log_signal
                )
                
                for s in sum_list:
                    row = {"Group": g_name, "Source": os.path.basename(p), "Arena_ID": s["Arena_ID"]}
                    for m in self.selected_metrics:
                        if m in s: row[m] = s[m]
                    all_sums.append(row)
                
                if not kinetics_df.empty:
                    kinetics_df.insert(0, 'Source', os.path.basename(p))
                    kinetics_df.insert(0, 'Group', g_name)
                    all_kins.append(kinetics_df)
                
                self.progress_signal.emit(int(((i+1)/len(files))*100))

            if all_sums:
                name = "_".join(stems[:2]) + f"_n{len(stems)}_analysis.xlsx"
                save_path = os.path.join(self.workspace, name)
                self.log_signal.emit(f"[TAAM IO] Generating Multi-sheet Workbook & Computing Group Math...")
                
                # Group-wise Calculations
                df_summary = pd.DataFrame(all_sums)
                numeric_cols = [c for c in df_summary.columns if c not in ["Group", "Source", "Arena_ID"]]
                
                df_mean = df_summary.groupby("Group")[numeric_cols].mean().reset_index()
                df_mean.insert(1, 'Aggregation_Type', 'AVERAGE')
                
                df_sum = df_summary.groupby("Group")[numeric_cols].sum().reset_index()
                df_sum.insert(1, 'Aggregation_Type', 'SUM')
                
                df_group_agg = pd.concat([df_mean, df_sum]).sort_values("Group")

                MAX_EXCEL_ROWS = 1000000 # Prevents Excel from crashing
                
                with pd.ExcelWriter(save_path) as writer:
                    df_group_agg.to_excel(writer, sheet_name="Group_Averages_Sums", index=False)
                    df_summary.to_excel(writer, sheet_name="Individual_Results", index=False)
                    
                    if all_kins:
                        master_kinetics = pd.concat(all_kins, ignore_index=True)
                        total_rows = len(master_kinetics)
                        
                        if total_rows > MAX_EXCEL_ROWS:
                            self.log_signal.emit(f"[TAAM IO] Large data ({total_rows} rows). Splitting sheets...")
                            num_chunks = math.ceil(total_rows / MAX_EXCEL_ROWS)
                            
                            for c_idx in range(num_chunks):
                                start_r = c_idx * MAX_EXCEL_ROWS
                                end_r = min((c_idx + 1) * MAX_EXCEL_ROWS, total_rows)
                                chunk = master_kinetics.iloc[start_r:end_r]
                                sheet_name = f"Frame_Kinetics_Pt{c_idx+1}"
                                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                                self.log_signal.emit(f"      Saved {sheet_name}")
                        else:
                            master_kinetics.to_excel(writer, sheet_name="Frame_Wise_Kinetics", index=False)

                self.finished_signal.emit(f"Success! TAAM Data exported safely to {name}")
        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")

class AnalysisModule(QDialog):
    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAAM | Advanced Scientific Analytics Suite")
        self.resize(1650, 950); self.workspace = os.path.abspath(workspace)
        self.experimental_groups, self.arenas, self.arena_configs = {}, [], {}
        self.video_frame, self.video_path_stored = None, ""
        self.metric_checkboxes = {}
        
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
            QProgressBar { height: 12px; border-radius: 6px; background: #111; text-align: center; }
        """)
        self.init_ui()

    def _make_prec_spin(self, v):
        s = QDoubleSpinBox(); s.setDecimals(3); s.setRange(0, 1e18); s.setValue(v); return s

    def _make_btn(self, t, f):
        b = QPushButton(t); b.setCursor(Qt.CursorShape.PointingHandCursor); b.clicked.connect(f); return b

    def init_ui(self):
        self.main_layout = QHBoxLayout(self)
        sidebar = QVBoxLayout()
        
        g1 = QGroupBox("1. Scientific Parameters")
        pl = QFormLayout()
        self.sp_conv = self._make_prec_spin(18.74); self.sp_fps = self._make_prec_spin(24.99); self.sp_dur = self._make_prec_spin(3549.7)
        self.sp_frz = self._make_prec_spin(0.1); self.sp_rpd = self._make_prec_spin(3.0)
        pl.addRow("Conversion:", self.sp_conv); pl.addRow("FPS:", self.sp_fps); pl.addRow("Duration (Legacy):", self.sp_dur)
        pl.addRow("Freezing (cm/s):", self.sp_frz); pl.addRow("Rapid (cm/s):", self.sp_rpd)
        g1.setLayout(pl); sidebar.addWidget(g1)

        g2 = QGroupBox("2. Settings Persistence")
        sl = QVBoxLayout(); sl.addWidget(self._make_btn("Save TAAM Session", self.save_settings)); sl.addWidget(self._make_btn("Load TAAM Session", self.load_settings))
        g2.setLayout(sl); sidebar.addWidget(g2)

        g3 = QGroupBox("3. Grouping Manager")
        gl = QVBoxLayout(); self.list_groups = QListWidget(); self.list_files = QListWidget(); self.list_groups.itemClicked.connect(self.update_file_list)
        btn_r = QHBoxLayout(); btn_r.addWidget(self._make_btn("+ Group", self.add_group)); btn_r.addWidget(self._make_btn("+ Excel", self.add_xlsx))
        gl.addWidget(QLabel("Groups:")); gl.addWidget(self.list_groups); gl.addWidget(QLabel("Excel Files:")); gl.addWidget(self.list_files); gl.addLayout(btn_r); gl.addWidget(self._make_btn("Remove Item", self.remove_item))
        g3.setLayout(gl); sidebar.addWidget(g3); sidebar.addStretch(); self.main_layout.addLayout(sidebar, 1)

        content = QVBoxLayout()
        g4 = QGroupBox("4. Behavioral Endpoints Selection")
        ml = QGridLayout()
        m_list = ["Average Speed (cm/s)", "Freezing Time Ratio (%)", "Swimming Time Ratio (%)", "Rapid movement time ratio (%)", "Time in Top Percentage (%)", "Time in Bottom Percentage (%)", "Average Thigmotaxis (cm)", "Fractal Dimension", "Entropy"]
        for i, m in enumerate(m_list):
            cb = QCheckBox(m); cb.setChecked(True); self.metric_checkboxes[m] = cb; ml.addWidget(cb, i//3, i%3); cb.setCursor(Qt.CursorShape.PointingHandCursor)
        g4.setLayout(ml); content.addWidget(g4)

        self.view_label = QLabel("TAAM Visualizer Window"); self.view_label.setFixedSize(800, 450); self.view_label.setStyleSheet("background:black; border:2px solid #0078d4;"); self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(self.view_label, 0, Qt.AlignmentFlag.AlignCenter)

        g5 = QGroupBox("5. Centroid Calibration")
        cl = QVBoxLayout(); top_h = QHBoxLayout(); self.combo_arena = QComboBox(); self.combo_arena.setCursor(Qt.CursorShape.PointingHandCursor); self.combo_arena.currentIndexChanged.connect(self.update_view)
        top_h.addWidget(QLabel("Select Tank:")); top_h.addWidget(self.combo_arena, 1); cl.addLayout(top_h)
        self.sld_x = QSlider(Qt.Orientation.Horizontal); self.sld_x.setRange(0, 100); self.sld_x.setValue(50)
        self.sld_y = QSlider(Qt.Orientation.Horizontal); self.sld_y.setRange(0, 100); self.sld_y.setValue(50)
        self.sld_x.valueChanged.connect(self.adjust_center); self.sld_y.valueChanged.connect(self.adjust_center)
        cl.addWidget(QLabel("Center X%")); cl.addWidget(self.sld_x); cl.addWidget(QLabel("Center Y%")); cl.addWidget(self.sld_y); g5.setLayout(cl); content.addWidget(g5)

        self.prog_bar = QProgressBar(); content.addWidget(self.prog_bar)
        self.console = QTextEdit(); self.console.setObjectName("Console"); self.console.setReadOnly(True); self.console.setFixedHeight(100); content.addWidget(self.console)

        bl2 = QHBoxLayout(); bl2.addWidget(self._make_btn("Load Video", self.load_video)); bl2.addWidget(self._make_btn("Load ROI JSON", self.load_roi))
        content.addLayout(bl2)

        self.btn_go = QPushButton("LAUNCH SCIENTIFIC EXCEL EXPORT"); self.btn_go.setFixedHeight(70); self.btn_go.setStyleSheet("background:#28a745; font-size: 18px;"); self.btn_go.clicked.connect(self.run_extraction)
        content.addWidget(self.btn_go); self.main_layout.addLayout(content, 3)

    def run_extraction(self):
        if not self.experimental_groups or not self.arenas: QMessageBox.warning(self, "Audit", "Add groups and load ROI first."); return
        self.btn_go.setEnabled(False); self.console.clear()
        cfg = {'fps': self.sp_fps.value(), 'conversion': self.sp_conv.value(), 'freeze_thresh': self.sp_frz.value(), 'rapid_thresh': self.sp_rpd.value()}
        metrics = [k for k,v in self.metric_checkboxes.items() if v.isChecked()]
        
        self.worker = AnalysisWorker(self.experimental_groups, self.arenas, self.arena_configs, cfg, metrics, self.workspace)
        self.worker.log_signal.connect(self.console.append); self.worker.progress_signal.connect(self.prog_bar.setValue)
        self.worker.finished_signal.connect(lambda m: (self.btn_go.setEnabled(True), QMessageBox.information(self, "TAAM Analysis", m))); self.worker.start()

    def save_settings(self):
        data = {"params": {"conv": self.sp_conv.value(), "fps": self.sp_fps.value(), "dur": self.sp_dur.value(), "frz": self.sp_frz.value(), "rpd": self.sp_rpd.value()}, "arenas": self.arena_configs, "groups": self.experimental_groups, "video": self.video_path_stored, "roi": self.arenas, "metrics": [k for k,v in self.metric_checkboxes.items() if v.isChecked()]}
        with open(os.path.join(self.workspace, "endpoints.json"), 'w') as f: json.dump(data, f, indent=4); self.console.append("TAAM endpoints.json saved.")

    def load_settings(self):
        p = os.path.join(self.workspace, "endpoints.json")
        if not os.path.exists(p): return
        try:
            with open(p, 'r') as f:
                d = json.load(f); pr = d.get('params', {})
                self.sp_conv.setValue(pr.get('conv', 18.74)); self.sp_fps.setValue(pr.get('fps', 24.99)); self.sp_dur.setValue(pr.get('dur', 3549.7))
                self.sp_frz.setValue(pr.get('frz', 0.1)); self.sp_rpd.setValue(pr.get('rpd', 3.0))
                self.arena_configs = {int(k): v for k,v in d.get('arenas', {}).items()}
                self.experimental_groups = d.get('groups', {}); self.arenas = d.get('roi', []); self.video_path_stored = d.get('video', "")
                m_list = d.get('metrics', [])
                for k,v in self.metric_checkboxes.items(): v.setChecked(k in m_list)
            self.list_groups.clear(); self.list_groups.addItems(self.experimental_groups.keys()); self.combo_arena.clear(); self.combo_arena.addItems([f"Arena {i+1}" for i in range(len(self.arenas))])
            if self.video_path_stored and os.path.exists(self.video_path_stored):
                cap = cv2.VideoCapture(self.video_path_stored); stat, f = cap.read(); cap.release(); self.video_frame = f if stat else None
            self.update_view(); self.console.append("TAAM Session restored.")
        except: pass

    def add_group(self):
        t, ok = QInputDialog.getText(self, 'Group', 'Enter Name:');
        if ok and t: self.list_groups.addItem(t); self.experimental_groups[t] = []
    def add_xlsx(self):
        it = self.list_groups.currentItem()
        if it: fs, _ = QFileDialog.getOpenFileNames(self, "Excel", self.workspace, "Excel (*.xlsx)"); self.experimental_groups[it.text()].extend(fs); self.update_file_list(it)
        else: QMessageBox.warning(self, "Audit", "Select group first.")
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
        if idx == -1: return

        roi = self.arenas[idx]
        px, py = float(self.sld_x.value()), float(self.sld_y.value())

        if roi.get('type', 'rect') == 'circle':
            rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
            cx0, cy0, r = rx+(rw/2), ry+(rh/2), min(rw, rh)/2

            tx, ty = rx+(rw*(px/100.0)), ry+(rh*(py/100.0))
            dx, dy = tx-cx0, ty-cy0
            dist = np.sqrt(dx*dx + dy*dy)

            if dist > r and dist > 0:
                tx, ty = cx0+(dx/dist)*r, cy0+(dy/dist)*r
                px = ((tx-rx)/rw)*100.0
                py = ((ty-ry)/rh)*100.0
                self.sld_x.blockSignals(True); self.sld_y.blockSignals(True)
                self.sld_x.setValue(int(px)); self.sld_y.setValue(int(py))
                self.sld_x.blockSignals(False); self.sld_y.blockSignals(False)

        self.arena_configs[idx] = {'cx_pct': px, 'cy_pct': py}
        self.update_view()

    def load_roi(self):
        p, _ = QFileDialog.getOpenFileName(self, "ROI", self.workspace, "JSON (*.json)")
        if p:
            with open(p, 'r') as f:
                data = json.load(f); self.arenas = []
                for d in data:
                    if d['type'] == 'grid':
                        rx, ry, rw, rh, r, c = d['x'], d['y'], d['w'], d['h'], d['grid'][0], d['grid'][1]; cw, ch = rw/c, rh/r
                        for row in range(r):
                            for col in range(c): self.arenas.append({'type':'grid', 'x':rx+col*cw, 'y':ry+row*ch, 'w':cw, 'h':ch})
                    else: self.arenas.append({'type':d['type'], 'x':d['x'], 'y':d['y'], 'w':d['w'], 'h':d['h']})
            self.combo_arena.clear(); [self.combo_arena.addItem(f"Arena {i+1}") for i in range(len(self.arenas))]; self.update_view()


    def update_view(self):
        if self.video_frame is None: return
        canvas = self.video_frame.copy(); cur = self.combo_arena.currentIndex()

        for i, roi in enumerate(self.arenas):
            rx, ry, rw, rh = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
            rtype = roi.get('type', 'rect'); color = (255,255,0) if i==cur else (0,255,0)

            if rtype == 'circle':
                cv2.circle(canvas, (rx+rw//2, ry+rh//2), min(rw,rh)//2, color, 4, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (rx,ry), (rx+rw, ry+rh), color, 4, cv2.LINE_AA)

            config = self.arena_configs.get(i, {'cx_pct':50.0, 'cy_pct':50.0}); px, py = config['cx_pct'], config['cy_pct']
            cx, cy = int(rx + rw*(px/100.0)), int(ry + rh*(py/100.0))

            cv2.circle(canvas, (cx,cy), 10, (0,0,255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx,cy), 12, (255,255,255), 2, cv2.LINE_AA)
            cv2.line(canvas, (cx-15,cy), (cx+15,cy), (255,255,255), 2)
            cv2.line(canvas, (cx,cy-15), (cx,cy+15), (255,255,255), 2)

            cv2.putText(canvas, f"ARENA {i+1}", (rx+10, ry+35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        qimg = QImage(canvas.data, canvas.shape[1], canvas.shape[0], canvas.shape[1]*3, QImage.Format.Format_BGR888)
        self.view_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.view_label.size(), Qt.AspectRatioMode.KeepAspectRatio))