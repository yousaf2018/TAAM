import os, time, random, traceback
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from PyQt6.QtCore import QThread, pyqtSignal

class Stopwatch:
    def __init__(self): self.start_time = 0
    def start(self): self.start_time = time.time()
    def get_elapsed_time(self, as_float=False):
        el = time.time() - self.start_time
        return el if as_float else time.strftime("%H:%M:%S", time.gmtime(el))
    def get_etr(self, current, total):
        if current <= 0 or total <= 0: return "--:--:--"
        etr = ((time.time() - self.start_time) / current) * (total - current)
        return time.strftime("%H:%M:%S", time.gmtime(etr))

class ArenaWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, config, workspace):
        super().__init__()
        self.config = config
        self.workspace = workspace
        self.is_running = True
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        
        self.b_thick = self.config.get("bbox_thick", 3)
        self.l_scale = self.config.get("lbl_scale", 0.6)
        self.l_thick = self.config.get("lbl_thick", 2)
        self.dot_sz = self.config.get("dot_size", 5)

        self.class_palette = [
            (57, 255, 20), (255, 0, 127), (0, 255, 255), (255, 102, 0),
            (204, 0, 255), (0, 102, 255), (255, 255, 255)
        ]

    def stop(self):
        self.is_running = False

    def _get_grid_cells(self, roi):
        cells = []
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        r, c = roi['grid']
        cw, ch = rw/c, rh/r
        for row in range(r):
            for col in range(c):
                cells.append({'type':'rect','x':rx+col*cw,'y':ry+row*ch,'w':cw,'h':ch})
        return cells

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if (boxAArea + boxBArea - interArea) == 0: return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def _merge_frame_duplicates_pre_tracking(self, detections_dict):
        merged_detections = defaultdict(list)
        total_merged = 0
        for frame_idx, dets in detections_dict.items():
            dets_by_tank = defaultdict(list)
            for det in dets:
                if det.get('tank_number') is not None:
                    dets_by_tank[int(det['tank_number'])].append(det)
            
            frame_output = []
            for tank_num, tank_dets in dets_by_tank.items():
                if len(tank_dets) == 1:
                    frame_output.append(tank_dets[0])
                    continue
                
                tank_dets.sort(key=lambda x: x.get('conf', 0.0), reverse=True)
                keep_list = []
                while tank_dets:
                    best_det = tank_dets.pop(0)
                    keep_list.append(best_det)
                    i = 0
                    while i < len(tank_dets):
                        other_det = tank_dets[i]
                        box_best = [best_det['x1'], best_det['y1'], best_det['x2'], best_det['y2']]
                        box_other = [other_det['x1'], other_det['y1'], other_det['x2'], other_det['y2']]
                        if self._calculate_iou(box_best, box_other) >= self.iou_threshold: 
                            tank_dets.pop(i)
                            total_merged += 1
                        else:
                            i += 1
                frame_output.extend(keep_list)
            merged_detections[frame_idx].extend(frame_output)
        self.log_signal.emit(f" > Pre-Tracking Cleanup: Removed {total_merged} overlapping NMS detections.")
        return merged_detections

    def _force_stitch_to_max(self, detections_dict):
        self.log_signal.emit(f" > Post-processing: Optimizing Track Stitching (Hungarian Algorithm)...")
        all_rows = []
        for frame_idx, dets in detections_dict.items():
            for d in dets:
                d['frame_idx'] = frame_idx; all_rows.append(d)
        if not all_rows: return detections_dict
        
        df = pd.DataFrame(all_rows)
        if 'track_id' not in df.columns: return detections_dict
        
        for tank_num in df['tank_number'].dropna().unique():
            tank_df = df[df['tank_number'] == tank_num]
            if tank_df.empty: continue
            
            track_meta = {}
            for tid in tank_df['track_id'].unique():
                t_data = tank_df[tank_df['track_id'] == tid]
                track_meta[tid] = {
                    'start_frame': t_data['frame_idx'].min(), 'end_frame': t_data['frame_idx'].max(),
                    'start_pos': (t_data.iloc[0]['cx'], t_data.iloc[0]['cy']),
                    'end_pos': (t_data.iloc[-1]['cx'], t_data.iloc[-1]['cy']),
                    'frames': set(t_data['frame_idx'].tolist())
                }
                
            track_durations = tank_df.groupby('track_id')['frame_idx'].count().sort_values(ascending=False)
            all_sorted_tids = track_durations.index.tolist()
            
            active_primaries = set(all_sorted_tids[:self.config['max_n']])
            unmatched_ghosts = set(all_sorted_tids[self.config['max_n']:])
            
            while True:
                merged_in_iteration = False
                list_primaries = list(active_primaries)
                list_ghosts = list(unmatched_ghosts)
                
                if not list_primaries or not list_ghosts: break
                    
                cost_matrix = np.full((len(list_primaries), len(list_ghosts)), 1e6)
                for i, p_id in enumerate(list_primaries):
                    for j, g_id in enumerate(list_ghosts):
                        t_p, t_g = track_meta[p_id], track_meta[g_id]
                        if not t_p['frames'].isdisjoint(t_g['frames']): continue 
                            
                        gap, dist = 0, 0
                        if t_g['start_frame'] > t_p['end_frame']:
                            gap = t_g['start_frame'] - t_p['end_frame']
                            dist = np.hypot(t_g['start_pos'][0] - t_p['end_pos'][0], t_g['start_pos'][1] - t_p['end_pos'][1])
                        elif t_p['start_frame'] > t_g['end_frame']:
                            gap = t_p['start_frame'] - t_g['end_frame']
                            dist = np.hypot(t_p['start_pos'][0] - t_g['end_pos'][0], t_p['start_pos'][1] - t_g['end_pos'][1])
                        cost_matrix[i, j] = dist + (gap * 2.0)
                        
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1e5:
                        p_id, g_id = list_primaries[r], list_ghosts[c]
                        mask = (df['tank_number'] == tank_num) & (df['track_id'] == g_id)
                        df.loc[mask, 'track_id'] = p_id
                        
                        track_meta[p_id]['frames'].update(track_meta[g_id]['frames'])
                        track_meta[p_id]['start_frame'] = min(track_meta[p_id]['start_frame'], track_meta[g_id]['start_frame'])
                        track_meta[p_id]['end_frame'] = max(track_meta[p_id]['end_frame'], track_meta[g_id]['end_frame'])
                        unmatched_ghosts.remove(g_id)
                        merged_in_iteration = True
                        
                if not merged_in_iteration:
                    if unmatched_ghosts and len(active_primaries) < self.config['max_n']:
                        longest_ghost = max(unmatched_ghosts, key=lambda x: len(track_meta[x]['frames']))
                        active_primaries.add(longest_ghost)
                        unmatched_ghosts.remove(longest_ghost)
                        continue
                    else: break
            
            if unmatched_ghosts:
                for g_id in unmatched_ghosts:
                    mask = (df['tank_number'] == tank_num) & (df['track_id'] == g_id)
                    df = df.drop(df[mask].index)
                    
            final_tank_indices = df[df['tank_number'] == tank_num].index
            id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(list(active_primaries)), 1)}
            df.loc[final_tank_indices, 'track_id'] = df.loc[final_tank_indices, 'track_id'].map(id_map)

        new_detections = defaultdict(list)
        for _, row in df.iterrows():
            d = row.to_dict()
            if 'frame_idx' in d: new_detections[int(d['frame_idx'])].append(d)
        return new_detections

    def run(self):
        try:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            import cv2
            import torch
            from ultralytics import YOLO
            
            cv2.setNumThreads(0)
            if torch.cuda.is_available(): torch.set_num_threads(1)

            start_batch_time = time.time()
            all_arenas = []
            for s in self.config['rois']:
                if s['type'] == 'grid': all_arenas.extend(self._get_grid_cells(s))
                else: all_arenas.append(s)
            self.log_signal.emit(f"📐 GEOMETRY: {len(all_arenas)} arenas mapped.")

            model_path = os.path.join(self.workspace, "Models", self.config['model_name'], "weights", "best.pt")
            model = YOLO(model_path)
            class_names = model.names
            
            device_target, use_half = 'cpu', False
            if torch.cuda.is_available():
                self.log_signal.emit(f"🔍 Hardware: {torch.cuda.get_device_name(0)}")
                dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
                try:
                    model.predict(dummy_img, device=0, half=True, verbose=False)
                    device_target, use_half = 0, True
                    self.log_signal.emit("✅ GPU FP16 Mode Active.")
                except Exception:
                    try:
                        model.predict(dummy_img, device=0, half=False, verbose=False)
                        device_target, use_half = 0, False
                        self.log_signal.emit("✅ GPU FP32 Mode Active.")
                    except:
                        self.log_signal.emit("❌ GPU Failed. CPU Fallback.")

            total_vids = len(self.config['videos'])
            
            for v_idx, v_path in enumerate(self.config['videos']):
                if not self.is_running: break
                
                base_n = os.path.splitext(os.path.basename(v_path))[0]
                out_dir = os.path.join(self.workspace, "Advanced_Results", base_n)
                os.makedirs(out_dir, exist_ok=True)
                
                cap = cv2.VideoCapture(v_path)
                width, height = int(cap.get(3)), int(cap.get(4))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret, bg_frame = cap.read()
                cap.release()

                if total_frames <= 0: continue

                # ==============================================================
                # PHASE 1: STREAMING INFERENCE
                # ==============================================================
                self.log_signal.emit(f"\n--- [PHASE 1] Inference Engine: {base_n} ---")
                
                inf_writer = None
                if self.config['save_inference_vid']:
                    inf_writer = cv2.VideoWriter(os.path.join(out_dir, f"{base_n}_inference.mp4"),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                
                raw_frame_data = defaultdict(list)
                is_seg = self.config['task_type'] == "Segmentation"
                
                results_gen = model.predict(source=v_path, conf=self.config['conf'], stream=True, 
                                           device=device_target, half=use_half, verbose=False)
                
                stopwatch = Stopwatch(); stopwatch.start()
                frame_count_fps, fps_time, current_fps = 0, 0, 0.0

                for frame_idx, res in enumerate(results_gen):
                    if not self.is_running: break
                    frame = res.orig_img.copy() if inf_writer else None
                    
                    if res.boxes is not None and len(res.boxes) > 0:
                        boxes_np = res.boxes.cpu()
                        masks_data = res.masks.data.cpu().numpy() if is_seg and res.masks else None

                        for j in range(len(boxes_np)):
                            b = boxes_np.xyxy[j].numpy()
                            cid = int(boxes_np.cls[j])
                            conf = float(boxes_np.conf[j])
                            color = self.class_palette[cid % len(self.class_palette)]
                            
                            cx, cy, poly_str, poly_pts = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0, "", None

                            if is_seg and masks_data is not None:
                                mask_res = cv2.resize(masks_data[j], (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                                M = cv2.moments(mask_res)
                                if M["m00"] != 0:
                                    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                                
                                contours, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    cnt = max(contours, key=cv2.contourArea)
                                    poly_pts = cnt
                                    poly_str = ";".join([f"{p[0][0]},{p[0][1]}" for p in cnt])

                            tank_num = None
                            for a_idx, arena in enumerate(all_arenas):
                                ax, ay, aw, ah = arena['x'], arena['y'], arena['w'], arena['h']
                                if arena['type'] == 'circle':
                                    acx, acy, r_sq = ax+(aw/2), ay+(ah/2), ((aw+ah)/4)**2
                                    if ((cx - acx)**2 + (cy - acy)**2) <= r_sq: tank_num = a_idx + 1; break
                                else:
                                    if (ax <= cx <= ax + aw and ay <= cy <= ay + ah): tank_num = a_idx + 1; break
                            
                            if tank_num is not None:
                                det_dict = {
                                    'frame_idx': frame_idx, 'tank_number': tank_num, 'class_name': class_names[cid],
                                    'conf': conf, 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3],
                                    'cx': cx, 'cy': cy, 'polygon': poly_str, 'poly_pts': poly_pts, 'cid': cid
                                }
                                raw_frame_data[frame_idx].append(det_dict)
                                
                                if inf_writer:
                                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, self.b_thick)
                                    if poly_pts is not None:
                                        cv2.drawContours(frame, [poly_pts], -1, color, max(1, self.b_thick - 1))
                                    cv2.circle(frame, (int(cx), int(cy)), self.dot_sz, (0, 0, 255), -1)

                    if inf_writer: inf_writer.write(frame)
                    
                    frame_count_fps += 1
                    curr_time = stopwatch.get_elapsed_time(as_float=True)
                    if curr_time > fps_time + 1.0:
                        current_fps = frame_count_fps / (curr_time - fps_time)
                        frame_count_fps, fps_time = 0, curr_time

                    if frame_idx % 5 == 0:
                        pct = int((frame_idx + 1) * 100 / total_frames)
                        self.progress_signal.emit(pct, f"Inference: {frame_idx}/{total_frames} | {current_fps:.1f} FPS | ETR: {stopwatch.get_etr(frame_idx, total_frames)}")
                
                if inf_writer: inf_writer.release()
                if not self.is_running: break

                # ==============================================================
                # PHASE 2: BATCH PROCESSING (TRACKING & NMS)
                # ==============================================================
                self.log_signal.emit(f"\n--- [PHASE 2] Tracking & Optimization ---")
                
                self.log_signal.emit("Running NMS Duplication Merging...")
                clean_detections = self._merge_frame_duplicates_pre_tracking(raw_frame_data)
                
                tracked_detections = defaultdict(list)
                num_tanks = len(all_arenas)
                
                if self.config['method'] == "Custom Force-N":
                    self.log_signal.emit("Applying Custom Force-N tracking...")
                    active_tracks = {t: {} for t in range(1, num_tanks + 1)}
                    for frame_idx in range(total_frames):
                        dets_this_frame = clean_detections.get(frame_idx, [])
                        for tank_num in range(1, num_tanks + 1):
                            tank_dets = [d for d in dets_this_frame if d.get('tank_number') == tank_num]
                            tank_dets.sort(key=lambda x: x.get('conf', 0.0), reverse=True)
                            tank_dets = tank_dets[:self.config['max_n']]
                            tank_active = active_tracks[tank_num]
                            
                            if len(tank_active) < self.config['max_n']:
                                unassigned_dets = []
                                for det in tank_dets:
                                    new_id = len(tank_active) + 1
                                    if new_id <= self.config['max_n']:
                                        det['track_id'] = new_id
                                        tank_active[new_id] = {'pos': (det['cx'], det['cy'])}
                                        tracked_detections[frame_idx].append(det)
                                    else: unassigned_dets.append(det)
                                tank_dets = unassigned_dets 
                            
                            if not tank_dets: continue 
                                
                            track_ids = list(tank_active.keys())
                            cost_matrix = np.full((len(track_ids), len(tank_dets)), 1e6) 
                            for i, tid in enumerate(track_ids):
                                for j, det in enumerate(tank_dets):
                                    cost_matrix[i, j] = np.hypot(tank_active[tid]['pos'][0] - det['cx'], tank_active[tid]['pos'][1] - det['cy'])

                            row_ind, col_ind = linear_sum_assignment(cost_matrix)
                            for r, c in zip(row_ind, col_ind):
                                tid = track_ids[r]; matched_det = tank_dets[c].copy()
                                matched_det['track_id'] = tid
                                tank_active[tid] = {'pos': (matched_det['cx'], matched_det['cy'])}
                                tracked_detections[frame_idx].append(matched_det)
                                
                elif self.config['method'] == "Confidence Filter":
                    self.log_signal.emit("Applying Confidence Filter...")
                    for frame_idx, dets in clean_detections.items():
                        tank_groups = defaultdict(list)
                        for det in dets: tank_groups[det.get('tank_number')].append(det)
                        for tank_num, tank_dets in tank_groups.items():
                            tank_dets.sort(key=lambda x: x.get('conf', 0.0), reverse=True)
                            valid_dets = tank_dets[:self.config['max_n']]
                            for i, d in enumerate(valid_dets):
                                d['track_id'] = i + 1
                                tracked_detections[frame_idx].append(d)

                if self.config['auto_stitch'] and self.config['method'] != "Custom Force-N":
                    tracked_detections = self._force_stitch_to_max(tracked_detections)

                # ==============================================================
                # PHASE 3: RENDER VIDEO & ANALYTICS EXPORT
                # ==============================================================
                if self.config['save_video']:
                    self.log_signal.emit(f"\n--- [PHASE 3] Rendering Tracked Video ---")
                    v_cap = cv2.VideoCapture(v_path)
                    trk_writer = cv2.VideoWriter(os.path.join(out_dir, f"{base_n}_tracked.mp4"),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                    
                    for f_idx in range(total_frames):
                        if not self.is_running: break
                        ret, frame = v_cap.read()
                        if not ret: break
                        
                        for d in tracked_detections.get(f_idx, []):
                            x1, y1, x2, y2 = map(int, [d['x1'], d['y1'], d['x2'], d['y2']])
                            cx, cy = int(d['cx']), int(d['cy'])
                            t_id = d.get('track_id', 0)
                            color = self.class_palette[int(d.get('cid', 0)) % len(self.class_palette)]
                            poly_pts = d.get('poly_pts', None)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.b_thick)
                            if is_seg and poly_pts is not None:
                                ov = frame.copy()
                                cv2.fillPoly(ov, [poly_pts], color)
                                cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)
                                cv2.drawContours(frame, [poly_pts], -1, color, max(1, self.b_thick - 1))
                            
                            cv2.circle(frame, (cx, cy), self.dot_sz, (0, 0, 255), -1)
                            
                            label = f"A{int(d['tank_number'])}|ID{t_id} {d['class_name']}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, self.l_scale, self.l_thick)
                            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                            cv2.putText(frame, label, (x1+5, y1-7), cv2.FONT_HERSHEY_DUPLEX, self.l_scale, (255,255,255), self.l_thick, cv2.LINE_AA)
                            
                        trk_writer.write(frame)
                        if f_idx % 10 == 0:
                            self.progress_signal.emit(int((f_idx + 1) * 100 / total_frames), f"Rendering: {f_idx}/{total_frames}")

                    v_cap.release(); trk_writer.release()

                if self.is_running:
                    self.log_signal.emit("Exporting Analytics Data...")
                    flat_data = [d for f, dets in tracked_detections.items() for d in dets]
                    self._generate_analytics(base_n, flat_data, bg_frame, out_dir, width, height)

            time_str = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_batch_time))
            self.finished_signal.emit(f"✅ Batch Completed in {time_str}")
        except Exception as e: 
            print("=== WORKER CRASH TRACEBACK ===")
            traceback.print_exc()
            self.finished_signal.emit(f"❌ Error: {str(e)}")

    def _generate_analytics(self, base_n, flat_data, bg, out, width, height):
        import cv2 # Local safety import
        
        if not flat_data: return
        df = pd.DataFrame(flat_data)
        
        export_df = df[['frame_idx', 'tank_number', 'track_id', 'class_name', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'conf', 'polygon']]
        export_df.columns = ["Frame", "Arena", "ID", "Class", "X", "Y", "x1", "y1", "x2", "y2", "Conf", "Polygon"]

        # 1. Standard CSV Export
        if self.config.get('save_csv'):
            export_df.to_csv(os.path.join(out, f"{base_n}_tracked_data.csv"), index=False)

        # 2. Centroid CSV Export
        if self.config.get('save_centroid_csv'):
            cent_df = export_df[['Frame', 'Arena', 'ID', 'Class', 'X', 'Y']]
            cent_df.to_csv(os.path.join(out, f"{base_n}_centroids.csv"), index=False)

        # -------------------------------
        # 3. Excel (By Tank) Export
        # -------------------------------
        if self.config.get('save_excel_tank'):
            try:
                MAX_EXCEL_ROWS = 1048576
                MAX_DATA_ROWS = MAX_EXCEL_ROWS - 1

                base_path = os.path.join(out, f"{base_n}_by_tank.xlsx")

                total_rows = len(export_df)

                if total_rows <= MAX_DATA_ROWS:
                    # DEFAULT BEHAVIOR (no splitting)
                    with pd.ExcelWriter(base_path, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name="Master_Data", index=False)

                        for a_id in sorted(export_df['Arena'].unique()):
                            arena_df = export_df[export_df['Arena'] == a_id]
                            if not arena_df.empty:
                                arena_df.to_excel(writer, sheet_name=f"Arena_{int(a_id)}", index=False)

                else:
                    # CHUNKED EXPORT
                    num_files = (total_rows // MAX_DATA_ROWS) + 1

                    for i in range(num_files):
                        start_row = i * MAX_DATA_ROWS
                        end_row = min((i + 1) * MAX_DATA_ROWS, total_rows)

                        chunk_df = export_df.iloc[start_row:end_row]

                        excel_file = os.path.join(
                            out,
                            f"{base_n}_by_tank_Part_{i+1}.xlsx"
                        )

                        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                            chunk_df.to_excel(writer, sheet_name="Master_Data", index=False)

                            for a_id in sorted(chunk_df['Arena'].unique()):
                                arena_df = chunk_df[chunk_df['Arena'] == a_id]
                                if not arena_df.empty:
                                    arena_df.to_excel(writer, sheet_name=f"Arena_{int(a_id)}", index=False)

            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed Excel By Tank: {str(e)}")

        # -------------------------------
        # 4. Excel (By Track) Export
        # -------------------------------
        if self.config.get('save_excel_track'):
            try:
                MAX_EXCEL_ROWS = 1048576
                MAX_DATA_ROWS = MAX_EXCEL_ROWS - 1

                base_path = os.path.join(out, f"{base_n}_by_track.xlsx")

                total_rows = len(export_df)

                if total_rows <= MAX_DATA_ROWS:
                    # DEFAULT BEHAVIOR
                    with pd.ExcelWriter(base_path, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name="Master_Data", index=False)

                        for t_id in sorted(export_df['ID'].unique()):
                            track_df = export_df[export_df['ID'] == t_id]
                            if not track_df.empty:
                                track_df.to_excel(writer, sheet_name=f"Track_{int(t_id)}", index=False)

                else:
                    # CHUNKED EXPORT
                    num_files = (total_rows // MAX_DATA_ROWS) + 1

                    for i in range(num_files):
                        start_row = i * MAX_DATA_ROWS
                        end_row = min((i + 1) * MAX_DATA_ROWS, total_rows)

                        chunk_df = export_df.iloc[start_row:end_row]

                        excel_file = os.path.join(
                            out,
                            f"{base_n}_by_track_Part_{i+1}.xlsx"
                        )

                        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                            chunk_df.to_excel(writer, sheet_name="Master_Data", index=False)

                            for t_id in sorted(chunk_df['ID'].unique()):
                                track_df = chunk_df[chunk_df['ID'] == t_id]
                                if not track_df.empty:
                                    track_df.to_excel(writer, sheet_name=f"Track_{int(t_id)}", index=False)

            except Exception as e:
                self.log_signal.emit(f"⚠️ Failed Excel By Track: {str(e)}")

        # 5. Render Trajectories
        if self.config.get('save_traj') and bg is not None:
            canvas = bg.copy()
            for a_id in sorted(export_df['Arena'].unique()):
                for o_id in sorted(export_df[export_df['Arena']==a_id]['ID'].unique()):
                    subset = export_df[(export_df['Arena']==a_id)&(export_df['ID']==o_id)].sort_values("Frame")
                    subset['diff'] = subset['Frame'].diff().fillna(1)
                    subset['group'] = (subset['diff'] > 1).cumsum()
                    
                    for _, group in subset.groupby('group'):
                        pts = group[['X','Y']].values.astype(np.int32)
                        random.seed(int(a_id*100 + o_id))
                        clr = (random.randint(60,255), random.randint(60,255), random.randint(60,255))
                        if len(pts)>1: cv2.polylines(canvas, [pts], False, clr, 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out, f"{base_n}_trajectories.png"), canvas)

        # 6. Render Heatmap
        if self.config.get('save_heat') and bg is not None:
            h, w = bg.shape[:2]; accum = np.zeros((h, w), dtype=np.float32)
            for _, r in export_df.iterrows(): cv2.circle(accum, (int(r['X']), int(r['Y'])), 10, 0.4, -1)
            accum = cv2.normalize(cv2.GaussianBlur(accum, (51, 51), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(accum, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out, f"{base_n}_heatmap.png"), cv2.addWeighted(bg, 0.6, heatmap, 0.4, 0))