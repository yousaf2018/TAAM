import os, cv2, torch, numpy as np, shutil, glob, csv, gc, sys, random, pandas as pd
import math
from scipy.stats import linregress

class AnalysisEngine:
    @staticmethod
    def calculate_behavior(xlsx_path, config, roi_list, arena_settings, logger):
        px_per_cm = float(config.get('conversion', 18.74))
        fps = float(config.get('fps', 24.99))
        f_thr = float(config.get('freeze_thresh', 0.1))
        r_thr = float(config.get('rapid_thresh', 3.0))
        
        results_summary, raw_kinetics = [], []
        xl = pd.ExcelFile(xlsx_path)
        sheet_names = xl.sheet_names

        for i, roi in enumerate(roi_list):
            arena_id = i + 1
            sheet_found = next((s for s in sheet_names if str(arena_id) in s), None)
            if not sheet_found: continue
            
            logger.emit(f"      [TAAM Math] Processing Arena {arena_id}: {sheet_found}")
            tdf = xl.parse(sheet_found).copy()
            if tdf.empty: continue

            # Circle ROI boundary filtering
            if roi.get('type', 'rect') == 'circle':
                cx, cy = roi['x'] + roi['w']/2, roi['y'] + roi['h']/2
                radius = min(roi['w'], roi['h'])/2
                dist = np.sqrt((tdf['X']-cx)**2 + (tdf['Y']-cy)**2)
                tdf = tdf[dist <= radius].copy()
                if tdf.empty: continue

            id_cols = ['Animal_ID', 'Track_ID', 'ID', 'Track', 'id', 'track']
            id_col = next((c for c in id_cols if c in tdf.columns), None)

            # 1. KINEMATICS (Calculated for EACH individual animal)
            if id_col:
                tdf = tdf.sort_values([id_col, 'Frame'])
                tdf['dx'] = tdf.groupby(id_col)['X'].diff()
                tdf['dy'] = tdf.groupby(id_col)['Y'].diff()
            else:
                tdf = tdf.sort_values('Frame')
                tdf['dx'] = tdf['X'].diff()
                tdf['dy'] = tdf['Y'].diff()

            tdf['dr'] = np.sqrt(tdf['dx']**2 + tdf['dy']**2)
            tdf['speed'] = (tdf['dr'] / px_per_cm) * fps
            tdf['speed'] = np.where(tdf['speed'] >= 50.0, np.nan, tdf['speed'])

            # 2. BEHAVIORAL STATES
            spd_clean = tdf['speed'].fillna(50.0).values
            tdf['freezing'] = (spd_clean < f_thr).astype(int)
            tdf['swimming'] = ((spd_clean >= f_thr) & (spd_clean < r_thr)).astype(int)
            tdf['rapid'] = (spd_clean >= r_thr).astype(int)

            # 3. SPATIAL POSITIONING
            set_val = arena_settings.get(i, arena_settings.get(str(i), {'cx_pct':50.0, 'cy_pct':50.0}))
            x_dot = float(roi['x']) + (float(roi['w']) * (float(set_val.get('cx_pct', 50.0))/100.0))
            y_dot = float(roi['y']) + (float(roi['h']) * (float(set_val.get('cy_pct', 50.0))/100.0))

            if id_col:
                tdf['prev_x'] = tdf.groupby(id_col)['X'].shift(1)
                tdf['prev_y'] = tdf.groupby(id_col)['Y'].shift(1)
                is_first = tdf.groupby(id_col).cumcount() == 0
                tdf.loc[is_first, ['freezing', 'swimming', 'rapid']] = 0
            else:
                tdf['prev_x'], tdf['prev_y'] = tdf['X'].shift(1), tdf['Y'].shift(1)
                tdf.iloc[0, [tdf.columns.get_loc(c) for c in ['freezing','swimming','rapid']]] = 0

            tdf['is_top'] = (tdf['prev_x'] < x_dot).astype(int)
            tdf['thigmo'] = np.sqrt((tdf['prev_x']-x_dot)**2 + (tdf['prev_y']-y_dot)**2) / px_per_cm
            if id_col: tdf.loc[is_first, 'is_top'] = 0
            else: tdf.iloc[0, tdf.columns.get_loc('is_top')] = 0

            # 4. COMPLEXITY (Entropy & Box-Counting Fractal Dimension)
            fd_list, ent_list = [], []
            
            if id_col:
                for _, sub_df in tdf.groupby(id_col):
                    fd_list.append(AnalysisEngine._calc_box_fd(sub_df['X'].values, sub_df['Y'].values))
                    ent_list.append(AnalysisEngine._calc_ent(sub_df['dx'].values, sub_df['dy'].values, sub_df['dr'].values))
            else:
                fd_list.append(AnalysisEngine._calc_box_fd(tdf['X'].values, tdf['Y'].values))
                ent_list.append(AnalysisEngine._calc_ent(tdf['dx'].values, tdf['dy'].values, tdf['dr'].values))

            # 5. SUMMARY EXPORT (Strict 100% ratios)
            valid_frames = tdf['speed'].notna().sum()
            valid_frames = valid_frames if valid_frames > 0 else 1

            frz_pct = round((tdf['freezing'].sum() / valid_frames) * 100, 3)
            swm_pct = round((tdf['swimming'].sum() / valid_frames) * 100, 3)
            rap_pct = round(100.0 - (frz_pct + swm_pct), 3)
            rap_pct = max(0.0, rap_pct) # Prevent edge-case negatives

            top_pct = round((tdf['is_top'].sum() / valid_frames) * 100, 3)
            bot_pct = round(100.0 - top_pct, 3)
            bot_pct = max(0.0, bot_pct)

            results_summary.append({
                "Arena_ID": int(arena_id),
                "Average Speed (cm/s)": round(np.nanmean(tdf['speed'].values), 3),
                "Freezing Time Ratio (%)": frz_pct,
                "Swimming Time Ratio (%)": swm_pct,
                "Rapid movement time ratio (%)": rap_pct,
                "Time in Top Percentage (%)": top_pct,
                "Time in Bottom Percentage (%)": bot_pct,
                "Average Thigmotaxis (cm)": round(np.nanmean(tdf['thigmo'].values), 3),
                "Fractal Dimension": round(np.mean(fd_list), 3) if fd_list else 1.000,
                "Entropy": round(np.mean(ent_list), 3) if ent_list else 1.009
            })

            tdf['Arena_ID'] = arena_id
            raw_kinetics.append(tdf)

        return results_summary, pd.concat(raw_kinetics) if raw_kinetics else pd.DataFrame()

    @staticmethod
    def _calc_ent(dx, dy, dr):
        """Calculates Shannon Entropy based on turning angles (>90 degrees)"""
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dot = dx[2:]*dx[1:-1] + dy[2:]*dy[1:-1]
                denom = dr[1:-1] * dr[2:]
                cos_v = np.clip(np.where(denom > 0, dot/denom, np.nan), -1.0, 1.0)
                theta = np.arccos(cos_v)*(180.0/np.pi)
            
            v = theta[~np.isnan(theta)]
            if len(v) == 0: return 1.009
            
            p1 = np.sum(v >= 90.0) / len(v)
            p2 = 1.0 - p1
            return ((-p1*math.log2(p1) if p1>0 else 0) + (-p2*math.log2(p2) if p2>0 else 0))
        except:
            return 1.009

    @staticmethod
    def _calc_box_fd(x, y):
        """Minkowski-Bouligand Box-Counting Dimension. Mathematically bounded to [1.0, 2.0]"""
        try:
            valid = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[valid], y[valid]
            if len(x) < 10: return 1.000
                
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            
            # If the animal essentially didn't move
            if (x_max - x_min) < 1e-5 and (y_max - y_min) < 1e-5:
                return 1.000
                
            # Normalize coordinates to a 0.0 -> 1.0 box
            scale = max(x_max - x_min, y_max - y_min)
            x_norm = (x - x_min) / scale
            y_norm = (y - y_min) / scale
            
            # Grid sizes
            scales = np.array([2, 4, 8, 16, 32, 64, 128])
            Ns = []
            
            for s in scales:
                x_idx = np.floor(x_norm * s).astype(int)
                y_idx = np.floor(y_norm * s).astype(int)
                hashes = x_idx * (s + 1) + y_idx
                Ns.append(len(np.unique(hashes)))
                
            # Fit line to Log(Boxes) vs Log(Scale)
            log_s = np.log(scales)
            log_N = np.log(Ns)
            slope, _, _, _, _ = linregress(log_s, log_N)
            
            # Restrict strictly to theoretical physics boundaries (1.0 to 2.0)
            return float(np.clip(slope, 1.0, 2.0))
        except Exception:
            return 1.000