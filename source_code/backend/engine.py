# import os, cv2, torch, numpy as np, shutil, glob, csv, gc, sys, random, pandas as pd
# from ultralytics import YOLO

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SAM3_PATH = os.path.join(BASE_DIR, "sam3")
# if SAM3_PATH not in sys.path: sys.path.append(SAM3_PATH)

# GLOBAL_SAM_MODEL = None
# GLOBAL_SAM_PREDICTOR = None

# class TAAMEngine:
#     def __init__(self, workspace, config, log_app):
#         self.workspace = os.path.abspath(workspace)
#         self.config = config
#         self.log_app = log_app
#         self.predictor = None
#         self.stop_flag = False 
#         self.colors = [(57, 255, 20), (50, 50, 255), (255, 100, 50), (0, 255, 255), (255, 0, 255)]

#     def log(self, msg):
#         if self.log_app: self.log_app.emit(msg)

#     def load_sam3(self):
#         global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
#         if GLOBAL_SAM_MODEL is not None:
#             self.predictor = GLOBAL_SAM_PREDICTOR
#             return True
#         try:
#             from sam3.model_builder import build_sam3_video_model
#             self.log("AI: Loading SAM 3 Architecture...")
#             with torch.inference_mode():
#                 GLOBAL_SAM_MODEL = build_sam3_video_model()
#                 GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
#                 GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
#             self.predictor = GLOBAL_SAM_PREDICTOR
#             return True
#         except Exception as e:
#             self.log(f"AI: ❌ SAM3 Fail: {e}"); return False

#     def run_full_pipeline(self, video_paths, annotations_map, model_name, progress_callback):
#         video_info = self.run_sam3_only(video_paths, annotations_map, model_name, progress_callback)
#         if video_info and not self.stop_flag:
#             return self.run_yolo_only(video_info, model_name, progress_callback)
#         return False

#     def run_sam3_only(self, video_paths, annotations_map, model_name, progress_callback):
#         try:
#             exp_root = os.path.join(self.workspace, "Experiments", model_name)
#             ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
#             pool_dir = os.path.join(ds_dir, "sampling_pool")
#             os.makedirs(exp_root, exist_ok=True); os.makedirs(ds_dir, exist_ok=True); os.makedirs(pool_dir, exist_ok=True)

#             video_info = {}
#             for v_idx, v_path in enumerate(video_paths):
#                 if self.stop_flag: break
#                 v_name = os.path.splitext(os.path.basename(v_path))[0]
#                 self.log(f"PROJECT: Tracking {v_name} ({v_idx+1}/{len(video_paths)})")
#                 v_exp_dir = os.path.join(exp_root, v_name); os.makedirs(v_exp_dir, exist_ok=True)
#                 csv_p = self._process_video_sam3(v_path, annotations_map.get(v_path, {}), v_exp_dir, pool_dir, progress_callback)
#                 if csv_p: video_info[v_path] = csv_p
#             return video_info
#         except Exception as e:
#             self.log(f"SAM3 ERROR: {e}"); return None

#     def run_yolo_only(self, video_info, model_name, progress_callback):
#         """Native YOLO training with strict image-file existence audit."""
#         try:
#             if not video_info:
#                 video_info = self._discover_existing_data(model_name)
            
#             if not video_info:
#                 self.log("❌ Error: No SAM3 CSV data discovered. Stage 2 aborted.")
#                 return False

#             ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
#             self.log("PROJECT: Starting Robust Dataset Audit & Generation...")
            
#             # THE FIX: Performs physical check of files before including them in the YAML
#             yaml_path = self._generate_yolo_dataset_robust(video_info, ds_dir, progress_callback)
            
#             if yaml_path and not self.stop_flag:
#                 self.predictor = None
#                 gc.collect(); torch.cuda.empty_cache()
                
#                 task = "segment" if self.config['task_type'] == "Segmentation" else "detect"
#                 model_base = self.config.get('custom_weights') or ("yolov8n-seg.pt" if task == "segment" else "yolov8n.pt")
                
#                 model = YOLO(model_base)
#                 model.train(data=yaml_path, epochs=self.config['epochs'], imgsz=self.config['imgsz'], batch=self.config['yolo_batch'],
#                             project=os.path.join(self.workspace, "Models"), name=model_name, workers=0, device=0, task=task, exist_ok=True)
#                 return True
#             return False
#         except Exception as e:
#             self.log(f"YOLO ERROR: {str(e)}"); return False
#     def _process_video_sam3(self, v_path, ann, out_dir, pool_dir, cb):
#         temp, proc = os.path.join(out_dir, "temp"), os.path.join(out_dir, "proc")
#         os.makedirs(temp, exist_ok=True); os.makedirs(proc, exist_ok=True)
#         name = os.path.splitext(os.path.basename(v_path))[0]
#         csv_p = os.path.join(out_dir, f"{name}_data.csv")
#         vid_p = os.path.join(out_dir, f"{name}_tracked.mp4")
        
#         with open(csv_p, 'w', newline='') as f: 
#             csv.writer(f).writerow(["Global_Frame_ID", "Object_ID", "Class_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2", "Polygon_Coords", "Image_Path"])

#         cap = cv2.VideoCapture(v_path); fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
#         chunk_paths = self._split_video(cap, fps, w, h, temp); cap.release()

#         prompts = {}
#         if 0 in ann:
#             for i, (rect, cid) in enumerate(ann[0]):
#                 prompts[i+1] = {'pt': (rect.center().x() / w, rect.center().y() / h), 'cls': cid}

#         total_c = len(chunk_paths)
#         for i, cp in enumerate(chunk_paths):
#             if self.stop_flag or not prompts: break
#             cb(10 + int((i/total_c)*40), f"SAM3 Batch {i+1}")
#             save_p = os.path.join(proc, f"p_{os.path.basename(cp)}")
#             prompts = self._track_chunk(cp, i, prompts, fps, csv_p, pool_dir, w, h, save_p)

#         self._stitch(proc, vid_p, fps, w, h)
#         shutil.rmtree(temp, ignore_errors=True); shutil.rmtree(proc, ignore_errors=True)
#         return csv_p

#     def _track_chunk(self, cp, c_idx, prompts, fps, csv_path, pool_dir, w_orig, h_orig, save_p):
#         state = self.predictor.init_state(video_path=cp, offload_video_to_cpu=True)
#         with torch.inference_mode():
#             for oid, data in prompts.items():
#                 pt_norm = data['pt'] 
#                 self.predictor.add_new_points(state, 0, oid, 
#                                               torch.tensor([[pt_norm[0], pt_norm[1]]], dtype=torch.float32, device="cuda"), 
#                                               torch.tensor([1], dtype=torch.int32, device="cuda"))
        
#         last_masks, next_p, vis_data = {}, {}, {}
#         cap_read = cv2.VideoCapture(cp); frame_count = int(cap_read.get(7))
        
#         with open(csv_path, 'a', newline='') as f_csv:
#             csv_w = csv.writer(f_csv)
#             gen = self.predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=frame_count, reverse=False, propagate_preflight=True)
#             for f_idx, oids, _, masks, _ in gen:
#                 ret, frame = cap_read.read()
#                 if not ret: break
                
#                 img_p = os.path.abspath(os.path.join(pool_dir, f"v_{c_idx}_{f_idx:04d}_{random.randint(100,999)}.jpg"))
#                 cv2.imwrite(img_p, frame)
                
#                 vis_data[f_idx] = {}
#                 for j, oid in enumerate(oids):
#                     m = (masks[j] > 0.0).cpu().numpy().squeeze()
#                     c, area = self._get_centroid_area(m)
#                     if c:
#                         poly = self._get_polygon_str(m)
#                         cls = prompts[oid]['cls']
#                         g_frame = (c_idx * frame_count) + f_idx
#                         csv_w.writerow([g_frame, oid, cls, c[0], c[1], area, 0, poly, img_p])
#                         vis_data[f_idx][oid] = {'pt': c, 'cls': cls}
#                         last_masks[oid] = m

#         self.predictor.clear_all_points_in_video(state); cap_read.release()
#         self._render(cp, save_p, fps, vis_data)
        
#         next_prompts = {}
#         for oid in prompts.keys():
#             if oid in last_masks:
#                 m = last_masks[oid]
#                 c, _ = self._get_centroid_area(m)
#                 if c:
#                     next_prompts[oid] = {'pt': (c[0]/w_orig, c[1]/h_orig), 'cls': prompts[oid]['cls']}
#         return next_prompts

#     def _render(self, inp, out, fps, data):
#         cap = cv2.VideoCapture(inp); w, h = int(cap.get(3)), int(cap.get(4))
#         writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         curr_f = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             if curr_f in data:
#                 for oid, info in data[curr_f].items():
#                     color = self.colors[info['cls'] % len(self.colors)]
#                     cv2.circle(frame, info['pt'], 5, color, -1)
#                     cv2.putText(frame, f"ID:{oid}", (info['pt'][0]+8, info['pt'][1]-8), 0, 0.6, (255,255,255), 2)
#             writer.write(frame); curr_f += 1
#         cap.release(); writer.release()

#     def _get_centroid_area(self, mask):
#         a = np.count_nonzero(mask); rows, cols = np.where(mask)
#         return ((int(np.mean(cols)), int(np.mean(rows))), a) if len(rows) > 0 else (None, 0)

#     def _get_polygon_str(self, mask):
#         m8 = (mask.astype(np.uint8)) * 255; cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not cnts: return ""
#         c = max(cnts, key=cv2.contourArea)
#         return ";".join([f"{p[0][0]},{p[0][1]}" for p in cv2.approxPolyDP(c, 0.005*cv2.arcLength(c, True), True)])

#     def _generate_yolo_dataset_robust(self, video_info_map, dataset_root, cb):
#         """Re-reads CSV files and ensures only existing image files are used for YOLO."""
#         all_valid_frame_refs = [] # (video_path, frame_id, csv_path)
        
#         for v_path, csv_p in video_info_map.items():
#             if not os.path.exists(csv_p): continue
#             df = pd.read_csv(csv_p)
#             df.columns = df.columns.str.strip()
            
#             # Check every frame index in the CSV
#             for fid in df["Global_Frame_ID"].unique():
#                 rows = df[df["Global_Frame_ID"] == fid]
#                 # Check if the image path recorded by SAM3 physically exists
#                 # Allows user to delete bad frames manually from the sampling_pool
#                 img_path = rows.iloc[0]["Image_Path"]
#                 if os.path.exists(img_path):
#                     all_valid_frame_refs.append((v_path, fid, csv_p))
        
#         if not all_valid_frame_refs:
#             self.log("⚠️ ERROR: No images found. Stage 1 must generate files first.")
#             return None

#         # Shuffle only the files that were confirmed to exist
#         random.shuffle(all_valid_frame_refs)
#         sampled = all_valid_frame_refs[:self.config['max_frames']]
        
#         # Split logic (70% Train, 20% Val, 10% Test)
#         tr_len = int(len(sampled) * 0.7)
#         va_len = int(len(sampled) * 0.2)
        
#         for s in ['train', 'val', 'test']:
#             os.makedirs(os.path.join(dataset_root, "images", s), exist_ok=True)
#             os.makedirs(os.path.join(dataset_root, "labels", s), exist_ok=True)

#         class_names = self.config.get('class_names', ['object'])

#         for i, (v_path, f_idx, csv_p) in enumerate(sampled):
#             if self.stop_flag: return None
#             split = "train" if i < tr_len else ("val" if i < (tr_len+va_len) else "test")
            
#             df = pd.read_csv(csv_p)
#             rows = df[df["Global_Frame_ID"] == f_idx]
#             img_src = rows.iloc[0]["Image_Path"]
            
#             # Final copy and label creation
#             shutil.copy(img_src, os.path.join(dataset_root, "images", split, os.path.basename(img_src)))
            
#             # Get dimensions from image itself to ensure normalization is perfect
#             tmp = cv2.imread(img_src)
#             fh, fw = tmp.shape[:2]
            
#             with open(os.path.join(dataset_root, "labels", split, os.path.basename(img_src).replace(".jpg", ".txt")), "w") as f_lbl:
#                 for _, r in rows.iterrows():
#                     poly = str(r["Polygon_Coords"]); cid = int(r["Class_ID"])
#                     if self.config['task_type'] == "Segmentation":
#                         f_lbl.write(f"{cid} " + " ".join([f"{float(p.split(',')[0])/fw:.6f} {float(p.split(',')[1])/fh:.6f}" for p in poly.split(';') if ',' in p]) + "\n")
#                     else:
#                         coords = [list(map(float, p.split(','))) for p in poly.split(';') if ',' in p]
#                         if coords:
#                             xs, ys = [c[0] for c in coords], [c[1] for c in coords]
#                             bw, bh = max(xs)-min(xs), max(ys)-min(ys)
#                             f_lbl.write(f"{cid} {(min(xs)+bw/2)/fw:.6f} {(min(ys)+bh/2)/fh:.6f} {bw/fw:.6f} {bh/fh:.6f}\n")
            
#             if i % 10 == 0: cb(50 + int((i/len(sampled))*40), f"YOLO: Processing {split} images...")

#         yaml_p = os.path.join(dataset_root, "data.yaml")
#         with open(yaml_p, "w") as f:
#             f.write(f"path: {dataset_root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
#             for idx, name in enumerate(class_names): f.write(f"  {idx}: {name}\n")
#         return yaml_p
#     def _discover_existing_data(self, model_name):
#         discovered = {}
#         search_path = os.path.join(self.workspace, "Experiments", model_name)
#         csv_files = glob.glob(os.path.join(search_path, "**", "*_data.csv"), recursive=True)
#         for cp in csv_files:
#             v_name = os.path.basename(cp).replace("_data.csv", "")
#             discovered[v_name] = cp
#         return discovered

#     def _split_video(self, cap, fps, w, h, out_dir):
#         per = int(fps * self.config['chunk_duration']); paths, idx = [], 0
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             out_f = os.path.abspath(os.path.join(out_dir, f"c_{idx:03d}.mp4"))
#             writer = cv2.VideoWriter(out_f, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#             cnt = 0
#             writer.write(frame)
#             while cnt < per - 1:
#                 ret, frame = cap.read()
#                 if not ret: break
#                 writer.write(frame); cnt += 1
#             writer.release(); paths.append(out_f); idx += 1
#         return paths

#     def _stitch(self, d, out, fps, w, h):
#         files = sorted(glob.glob(os.path.join(d, "*.mp4")))
#         writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for f in files:
#             cap = cv2.VideoCapture(f)
#             while True:
#                 ret, frame = cap.read()
#                 if not ret: break
#                 writer.write(frame)
#             cap.release()
#         writer.release()


import os, cv2, torch, numpy as np, shutil, glob, csv, gc, sys, random, pandas as pd
from ultralytics import YOLO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3_PATH = os.path.join(BASE_DIR, "sam3")
if SAM3_PATH not in sys.path: sys.path.append(SAM3_PATH)

GLOBAL_SAM_MODEL = None
GLOBAL_SAM_PREDICTOR = None

class TAAMEngine:
    def __init__(self, workspace, config, log_app):
        self.workspace = os.path.abspath(workspace)
        self.config = config
        self.log_app = log_app
        self.predictor = None
        self.stop_flag = False 
        self.colors = [(57, 255, 20), (50, 50, 255), (255, 100, 50), (0, 255, 255)]

    def log(self, msg):
        if self.log_app: self.log_app.emit(msg)
        else: print(msg)

    def load_sam3(self):
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        try:
            if GLOBAL_SAM_MODEL is not None:
                self.predictor = GLOBAL_SAM_PREDICTOR
                self.log("AI: SAM3 already loaded in memory.")
                return True

            from sam3.model_builder import build_sam3_video_model
            self.log("AI: Loading SAM3 Architecture...")
            with torch.inference_mode():
                GLOBAL_SAM_MODEL = build_sam3_video_model()
                GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
                GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
            self.predictor = GLOBAL_SAM_PREDICTOR
            self.log("AI: SAM3 Loaded Successfully.")
            return True
        except Exception as e:
            self.log(f"AI: ❌ Failed to load SAM3: {e}")
            self._free_sam3_memory()
            return False

    def _free_sam3_memory(self):
        """Safely delete SAM3 from GPU memory."""
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        try:
            if GLOBAL_SAM_MODEL is not None:
                self.log("AI: Releasing SAM3 from GPU memory...")
                del GLOBAL_SAM_MODEL
                GLOBAL_SAM_MODEL = None
            if GLOBAL_SAM_PREDICTOR is not None:
                del GLOBAL_SAM_PREDICTOR
                GLOBAL_SAM_PREDICTOR = None
            self.predictor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log("AI: SAM3 memory cleared successfully.")
        except Exception as e:
            self.log(f"AI: Error during SAM3 memory cleanup: {e}")

    def run_full_pipeline(self, video_paths, annotations_map, model_name, progress_callback):
        try:
            video_info = self.run_sam3_only(video_paths, annotations_map, model_name, progress_callback)
            # Free memory after SAM3 tracking
            self._free_sam3_memory()
            if video_info and not self.stop_flag:
                return self.run_yolo_only(video_info, model_name, progress_callback)
            return False
        except Exception as e:
            self.log(f"PIPELINE ERROR: {e}")
            self._free_sam3_memory()
            return False

    def run_sam3_only(self, video_paths, annotations_map, model_name, progress_callback):
        try:
            if not self.load_sam3():
                return None

            exp_root = os.path.join(self.workspace, "Experiments", model_name)
            ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
            pool_dir = os.path.join(ds_dir, "sampling_pool")
            os.makedirs(exp_root, exist_ok=True); os.makedirs(ds_dir, exist_ok=True); os.makedirs(pool_dir, exist_ok=True)

            video_info = {}
            for v_idx, v_path in enumerate(video_paths):
                if self.stop_flag: break

                v_filename_full = os.path.basename(v_path)
                v_name = os.path.splitext(v_filename_full)[0]
                self.log(f"PROJECT: Tracking {v_name} ({v_idx+1}/{len(video_paths)})")
                v_exp_dir = os.path.join(exp_root, v_name); os.makedirs(v_exp_dir, exist_ok=True)

                csv_p = self._process_video_sam3(
                    v_path,
                    annotations_map.get(v_path, {}),
                    v_exp_dir,
                    pool_dir,
                    progress_callback
                )

                if csv_p:
                    video_info[v_path] = csv_p

            return video_info
        except RuntimeError as re:
            # Likely CUDA OOM
            self.log(f"SAM3 MEMORY ERROR: {re}")
            self._free_sam3_memory()
            return None
        except Exception as e:
            self.log(f"SAM3 ERROR: {e}")
            self._free_sam3_memory()
            return None

    def run_yolo_only(self, video_info, model_name, progress_callback):
        try:
            ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
            os.makedirs(ds_dir, exist_ok=True)

            self.log("SYSTEM: Preparing YOLO dataset folders...")

            for sub in ["images", "labels", "train", "val", "test"]:
                target = os.path.join(ds_dir, sub)
                if os.path.exists(target):
                    shutil.rmtree(target)

            if not video_info:
                self.log("🔍 YOLO-Only Mode: Discovering existing SAM3 outputs...")
                video_info = self._discover_existing_data(model_name)

            clean_video_info = {}
            for v_path, csv_p in (video_info or {}).items():
                if os.path.exists(csv_p):
                    clean_video_info[v_path] = csv_p
                else:
                    self.log(f"SYSTEM: Skipping deleted SAM3 output for video: {os.path.basename(str(v_path))}")

            if not clean_video_info:
                self.log("❌ Error: No valid SAM3 tracking folders found.")
                return False

            yaml_path = self._generate_yolo_dataset(clean_video_info, ds_dir, progress_callback)

            if yaml_path and not self.stop_flag:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                task = "segment" if self.config['task_type'] == "Segmentation" else "detect"
                cust_w = self.config.get('custom_weights')
                model_base = cust_w if (cust_w and os.path.exists(cust_w)) else (self.config.get('yolo_ver') or "yolov8n.pt")

                self.log("🚀 Starting YOLO Training...")

                model = YOLO(model_base)
                model.train(
                    data=yaml_path,
                    epochs=self.config['epochs'],
                    imgsz=self.config['imgsz'],
                    batch=self.config['yolo_batch'],
                    project=os.path.join(self.workspace, "Models"),
                    name=model_name,
                    workers=0,
                    device=0,
                    task=task,
                    exist_ok=True
                )

                self.log("✅ YOLO Training Completed Successfully.")
                return True

            self.log("❌ ERROR: Dataset preparation failed.")
            return False

        except Exception as e:
            self.log(f"YOLO ERROR: {e}")
            return False

    def _generate_yolo_dataset(self, video_info_map, dataset_root, cb):
        
        all_p = []

        # 1. Collect all valid frames across all videos
        for v_path, csv_p in video_info_map.items():
            if not os.path.exists(csv_p):
                continue

            df = pd.read_csv(csv_p)
            df.columns = df.columns.str.strip()
            # Ensure images actually exist before adding to pool
            df = df[df["Image_Path"].apply(os.path.exists)]
            if df.empty:
                continue

            for fid in df["Global_Frame_ID"].unique():
                all_p.append((v_path, fid, csv_p))

        if not all_p:
            self.log("ERROR: No valid images found in pool for dataset generation.")
            return None

        # 2. Randomized Sampling based on UI "Max Frames"
        random.shuffle(all_p)
        max_f = self.config.get('max_frames', 500)
        sampled = all_p[:max_f]
        
        # 3. Scientific Splitting based on UI Percentages
        tr_ratio = self.config.get('tr', 70) / 100.0
        va_ratio = self.config.get('va', 20) / 100.0
        tr_lim = int(len(sampled) * tr_ratio)
        va_lim = tr_lim + int(len(sampled) * va_ratio)

        for s in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_root, "images", s), exist_ok=True)
            os.makedirs(os.path.join(dataset_root, "labels", s), exist_ok=True)

        class_names = self.config.get('class_names', ['object'])
        bbox_mode = self.config.get('bbox_mode', 'Dynamic (From Mask)')
        fixed_size = self.config.get('fixed_bbox_size', 60)

        # 4. Generate Files
        for i, (v_path, f_idx, csv_p) in enumerate(sampled):
            if self.stop_flag:
                return None

            # Determine split folder
            if i < tr_lim: split = "train"
            elif i < va_lim: split = "val"
            else: split = "test"

            df = pd.read_csv(csv_p)
            df.columns = df.columns.str.strip()
            rows = df[df["Global_Frame_ID"] == f_idx]
            if rows.empty:
                continue

            img_src = rows.iloc[0]["Image_Path"]
            tmp_img = cv2.imread(img_src)
            if tmp_img is None: continue
            fh, fw = tmp_img.shape[:2]

            vid_name = os.path.splitext(os.path.basename(v_path))[0]
            base_n = f"{vid_name}_frame_{int(f_idx):06d}_{random.randint(100,999)}.jpg"
            shutil.copy(img_src, os.path.join(dataset_root, "images", split, base_n))

            with open(os.path.join(dataset_root, "labels", split, base_n.replace(".jpg", ".txt")), "w") as f_lbl:
                for _, r in rows.iterrows():
                    poly = str(r.get("Polygon_Coords", ""))
                    cls_id = int(r.get("Class_ID", 0))

                    # Logic A: Segmentation
                    if self.config['task_type'] == "Segmentation":
                        f_lbl.write(
                            f"{cls_id} " +
                            " ".join([f"{float(p.split(',')[0])/fw:.6f} {float(p.split(',')[1])/fh:.6f}"
                                    for p in poly.split(';') if ',' in p]) + "\n"
                        )
                    # Logic B: Fixed Bounding Box
                    elif bbox_mode == "Fixed (Centered)":
                        cx, cy = float(r["Centroid_X"]), float(r["Centroid_Y"])
                        n_cx = cx / fw
                        n_cy = cy / fh
                        n_w = fixed_size / fw
                        n_h = fixed_size / fh
                        f_lbl.write(f"{cls_id} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f}\n")
                    
                    # Logic C: Dynamic Bounding Box (From Polygon)
                    else:
                        coords = [list(map(float, p.split(','))) for p in poly.split(';') if ',' in p]
                        if coords:
                            xs, ys = [c[0] for c in coords], [c[1] for c in coords]
                            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                            bw, bh = max_x - min_x, max_y - min_y
                            # Normalized YOLO Format: cls cx cy w h
                            f_lbl.write(
                                f"{cls_id} {(min_x+bw/2)/fw:.6f} {(min_y+bh/2)/fh:.6f} {bw/fw:.6f} {bh/fh:.6f}\n"
                            )

            cb(50 + int((i/len(sampled))*40), f"YOLO: {split} ({i+1}/{len(sampled)})")

        # 5. Write Data.YAML using Absolute Path
        yaml_p = os.path.join(dataset_root, "data.yaml")
        with open(yaml_p, "w") as f:
            f.write(f"path: {os.path.abspath(dataset_root)}\n")
            f.write(f"train: images/train\nval: images/val\ntest: images/test\n")
            f.write(f"names:\n")
            for idx, name in enumerate(class_names):
                f.write(f"  {idx}: {name}\n")

        return yaml_p
    def _discover_existing_data(self, model_name):
        discovered = {}
        search_path = os.path.join(self.workspace, "Experiments", model_name)
        csv_files = glob.glob(os.path.join(search_path, "**", "*_data.csv"), recursive=True)
        for cp in csv_files:
            v_name = os.path.basename(cp).replace("_data.csv", "")
            discovered[v_name] = cp
        return discovered

    # --- SAM3 Video Processing ---
    def _process_video_sam3(self, v_path, ann, out_dir, pool_dir, cb):
        temp, proc = os.path.join(out_dir, "temp"), os.path.join(out_dir, "proc")
        os.makedirs(temp, exist_ok=True); os.makedirs(proc, exist_ok=True)
        name = os.path.splitext(os.path.basename(v_path))[0]
        csv_p = os.path.join(out_dir, f"{name}_data.csv")
        vid_p = os.path.join(out_dir, f"{name}_tracked.mp4")
        
        with open(csv_p, 'w', newline='') as f: 
            csv.writer(f).writerow(["Global_Frame_ID", "Object_ID", "Class_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2", "Polygon_Coords", "Image_Path"])

        cap = cv2.VideoCapture(v_path); fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
        chunk_paths = self._split_video(cap, fps, w, h, temp); cap.release()

        # Initialize prompts from your UI annotations
        prompts = {}
        if 0 in ann:
            for i, (rect, cid) in enumerate(ann[0]):
                # Keeping your exact normalized logic: (x / w)
                prompts[i+1] = {'pt': (rect.center().x() / w, rect.center().y() / h), 'cls': cid}

        total_c = len(chunk_paths)
        for i, cp in enumerate(chunk_paths):
            if self.stop_flag or not prompts: break
            cb(10 + int((i/total_c)*40), f"SAM3 Chunk {i+1}/{total_c}")
            save_p = os.path.join(proc, f"p_{os.path.basename(cp)}")
            
            # This method now processes objects 4-at-a-time but returns prompts for the NEXT chunk
            prompts = self._track_chunk(cp, i, prompts, fps, csv_p, pool_dir, w, h, save_p)

        self._stitch(proc, vid_p, fps, w, h)
        shutil.rmtree(temp, ignore_errors=True); shutil.rmtree(proc, ignore_errors=True)
        return csv_p

    def _track_chunk(self, cp, c_idx, prompts, fps, csv_path, pool_dir, w_orig, h_orig, save_p):
        """
        Tracks objects in batches. The batch size is dynamically pulled from the UI config.
        """
        # --- DYNAMIC LINK TO UI ---
        # Pulls the "SAM3 Batch" value from your UI settings. Default to 1 if not found.
        obj_batch_size = int(self.config.get('sam_batch', 1))
        if obj_batch_size < 1: obj_batch_size = 1 
        
        prompt_items = list(prompts.items())
        all_frames_data = {} 
        last_masks_for_stitch = {}
        
        cap_temp = cv2.VideoCapture(cp)
        frame_count = int(cap_temp.get(7))
        cap_temp.release()

        # 1. Process groups of objects based on UI Batch Size
        for b_start in range(0, len(prompt_items), obj_batch_size):
            batch = prompt_items[b_start : b_start + obj_batch_size]
            self.log(f"AI: Tracking Object Batch {b_start//obj_batch_size + 1} (Size: {len(batch)})")

            # Initialize state for this specific batch group
            state = self.predictor.init_state(video_path=cp, offload_video_to_cpu=True)
            
            try:
                with torch.inference_mode():
                    # Inject points for objects in this specific batch group
                    for oid, data in batch:
                        pt_norm = data['pt']
                        self.predictor.add_new_points(
                            state, 0, oid,
                            torch.tensor([[pt_norm[0], pt_norm[1]]], dtype=torch.float32, device="cuda"),
                            torch.tensor([1], dtype=torch.int32, device="cuda")
                        )

                    # Propagate tracking for this batch
                    gen = self.predictor.propagate_in_video(
                        state, 
                        start_frame_idx=0, 
                        max_frame_num_to_track=frame_count, 
                        reverse=False, 
                        propagate_preflight=True
                    )

                    # Collect results into the shared chunk dictionary
                    for f_idx, oids, _, masks, _ in gen:
                        if f_idx not in all_frames_data:
                            all_frames_data[f_idx] = {}
                        
                        for j, oid in enumerate(oids):
                            mask_np = (masks[j] > 0.0).cpu().numpy().squeeze()
                            centroid, area = self._get_centroid_area(mask_np)
                            if centroid:
                                poly_str = self._get_polygon_str(mask_np)
                                all_frames_data[f_idx][oid] = {
                                    'pt': centroid, 
                                    'cls': prompts[oid]['cls'], 
                                    'area': area, 
                                    'poly': poly_str
                                }
                                # Save mask of the last frame to calculate the "Stitch" point for next chunk
                                if f_idx == frame_count - 1:
                                    last_masks_for_stitch[oid] = mask_np

                # Clear points and state for this batch to free GPU VRAM
                self.predictor.clear_all_points_in_video(state)
            
            except Exception as e:
                self.log(f"AI BATCH ERROR: {e}")
            
            finally:
                # Forcefully delete state and empty cache between batches
                del state
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 2. Write all merged results (from all batches) to CSV and save frames
        video_base = os.path.splitext(os.path.basename(csv_path))[0].replace("_data", "")
        cap_read = cv2.VideoCapture(cp)
        
        with open(csv_path, 'a', newline='') as f_csv:
            csv_w = csv.writer(f_csv)
            for f_idx in range(frame_count):
                ret, frame = cap_read.read()
                if not ret: break

                if f_idx in all_frames_data and all_frames_data[f_idx]:
                    global_frame_id = (c_idx * frame_count) + f_idx
                    img_name = f"{video_base}_frame_{int(global_frame_id):06d}.jpg"
                    img_p = os.path.abspath(os.path.join(pool_dir, img_name))
                    cv2.imwrite(img_p, frame)

                    for oid, info in all_frames_data[f_idx].items():
                        csv_w.writerow([
                            global_frame_id, oid, info['cls'],
                            info['pt'][0], info['pt'][1], info['area'], 0, info['poly'], img_p
                        ])
        cap_read.release()

        # 3. Render the preview using the merged data
        self._render(cp, save_p, fps, all_frames_data)

        # 4. Prepare normalized points for the NEXT chunk (STITCH LOGIC)
        next_prompts = {}
        for oid in prompts.keys():
            if oid in last_masks_for_stitch:
                m = last_masks_for_stitch[oid]
                c, _ = self._get_centroid_area(m)
                if c:
                    next_prompts[oid] = {
                        'pt': (c[0]/w_orig, c[1]/h_orig), 
                        'cls': prompts[oid]['cls']
                    }
        return next_prompts
    def _render(self, inp, out, fps, data):
        cap = cv2.VideoCapture(inp); w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        curr_f = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if curr_f in data:
                for oid, info in data[curr_f].items():
                    color = self.colors[info['cls'] % len(self.colors)]
                    cv2.circle(frame, info['pt'], 5, color, -1)
                    cv2.putText(frame, f"ID:{oid}", (info['pt'][0]+8, info['pt'][1]-8), 0, 0.6, (255,255,255), 2)
            writer.write(frame); curr_f += 1
        cap.release(); writer.release()

    def _get_centroid_area(self, mask):
        a = np.count_nonzero(mask); rows, cols = np.where(mask)
        return ((int(np.mean(cols)), int(np.mean(rows))), a) if len(rows) > 0 else (None, 0)

    def _get_polygon_str(self, mask):
        m8 = (mask.astype(np.uint8)) * 255; cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return ""
        c = max(cnts, key=cv2.contourArea)
        return ";".join([f"{p[0][0]},{p[0][1]}" for p in cv2.approxPolyDP(c, 0.005*cv2.arcLength(c, True), True)])

    def _split_video(self, cap, fps, w, h, out_dir):
        per = int(fps * self.config['chunk_duration']); paths, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            out_f = os.path.abspath(os.path.join(out_dir, f"c_{idx:03d}.mp4"))
            writer = cv2.VideoWriter(out_f, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            cnt = 0
            while cnt < per:
                if frame is None: break
                writer.write(frame); cnt += 1
                ret, frame = cap.read()
                if not ret: break
            writer.release(); paths.append(out_f); idx += 1
        return paths

    def _stitch(self, src_dir, out_path, fps, w, h):
        files = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in files:
            cap = cv2.VideoCapture(f)
            while True:
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
            cap.release()
        writer.release()

    # --- Video splitting, processing, and rendering methods remain exactly the same as provided ---
    # _process_video_sam3, _track_chunk, _render, _get_centroid_area, _get_polygon_str, _split_video, _stitch