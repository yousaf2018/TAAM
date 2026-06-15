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

            # --- DYNAMIC PARAMETER EXTRACTION FROM THE WORKBOOK ---
            # Extract scale column S (19th column, index 18)
            try:
                S_values = pd.to_numeric(tdf.iloc[:, 18], errors='coerce').dropna().values
            except Exception:
                S_values = np.array([])

            # Extract AA9 (Column AA index 26, Row 9 index 7)
            try:
                aa9_val = float(tdf.iloc[7, 26])
                if np.isnan(aa9_val):
                    aa9_val = 0.0
            except Exception:
                aa9_val = 0.0

            # Extract Z16 (Column Z index 25, Row 16 index 14)
            try:
                z16_val = int(tdf.iloc[14, 25])
                if np.isnan(z16_val):
                    z16_val = 1
            except Exception:
                z16_val = 1

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

            # 3. SPATIAL POSITIONING (Calculated dynamically from ROI parameters)
            set_val = arena_settings.get(i, arena_settings.get(str(i), {'cx_pct':50.0, 'cy_pct':50.0}))
            x_dot = float(roi['x']) + (float(roi['w']) * (float(set_val.get('cx_pct', 50.0))/100.0))
            y_dot = float(roi['y']) + (float(roi['h']) * (float(set_val.get('cy_pct', 50.0))/100.0))

            if id_col:
                tdf['prev_x'] = tdf.groupby(id_col)['X'].shift(1)
                tdf['prev_y'] = tdf.groupby(id_col)['Y'].shift(1)
                is_first = tdf.groupby(id_col).cumcount() == 0
                tdf.loc[is_first, 'swimming'] = 1
                tdf.loc[is_first, ['freezing', 'rapid']] = 0
            else:
                tdf['prev_x'], tdf['prev_y'] = tdf['X'].shift(1), tdf['Y'].shift(1)
                tdf.loc[tdf.index[0], 'swimming'] = 1
                tdf.loc[tdf.index[0], ['freezing', 'rapid']] = 0

            tdf['is_top'] = (tdf['prev_x'] < x_dot).astype(int)
            tdf['thigmo'] = np.sqrt((tdf['prev_x']-x_dot)**2 + (tdf['prev_y']-y_dot)**2) / px_per_cm
            if id_col: tdf.loc[is_first, 'is_top'] = 0
            else: tdf.loc[tdf.index[0], 'is_top'] = 0

            # 4. COMPLEXITY (Slicing on consecutive 15,000 blocks)
            fd_list, ent_list = [], []
            slice_size = 15000

            if id_col:
                for _, sub_df in tdf.groupby(id_col):
                    fd_slices, ent_slices = [], []
                    for start_idx in range(0, len(sub_df), slice_size):
                        slice_df = sub_df.iloc[start_idx : start_idx + slice_size]
                        if len(slice_df) < 50: 
                            continue
                        
                        fd_slices.append(AnalysisEngine._calc_excel_fd(slice_df['dr'].values, S_values, aa9_val, z16_val))
                        ent_slices.append(AnalysisEngine._calc_ent(slice_df['dx'].values, slice_df['dy'].values, slice_df['dr'].values))
                    
                    fd_list.append(np.mean(fd_slices) if fd_slices else 1.000)
                    ent_list.append(np.mean(ent_slices) if ent_slices else 1.009)
            else:
                fd_slices, ent_slices = [], []
                for start_idx in range(0, len(tdf), slice_size):
                    slice_df = tdf.iloc[start_idx : start_idx + slice_size]
                    if len(slice_df) < 50: 
                        continue
                    
                    fd_slices.append(AnalysisEngine._calc_excel_fd(slice_df['dr'].values, S_values, aa9_val, z16_val))
                    ent_slices.append(AnalysisEngine._calc_ent(slice_df['dx'].values, slice_df['dy'].values, slice_df['dr'].values))
                
                fd_list.append(np.mean(fd_slices) if fd_slices else 1.000)
                ent_list.append(np.mean(ent_slices) if ent_slices else 1.009)

            # 5. SUMMARY EXPORT
            valid_frames = int(tdf['freezing'].sum() + tdf['swimming'].sum() + tdf['rapid'].sum())
            valid_frames = valid_frames if valid_frames > 0 else 1

            frz_pct = round((tdf['freezing'].sum() / valid_frames) * 100, 3)
            swm_pct = round((tdf['swimming'].sum() / valid_frames) * 100, 3)
            rap_pct = round(100.0 - (frz_pct + swm_pct), 3)
            rap_pct = max(0.0, rap_pct)

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
        """Calculates Shannon Entropy matching turning angle column R and cell AD6"""
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dot = dx[2:]*dx[1:-1] + dy[2:]*dy[1:-1]  # TYPO FIXED
                denom = dr[1:-1] * dr[2:]
                cos_v = np.clip(np.where(denom > 0, dot/denom, np.nan), -1.0, 1.0)
                theta = np.arccos(cos_v)*(180.0/np.pi)
            
            v = theta[~np.isnan(theta)]
            
            # COUNTA(R4:R15001) evaluates every cell in the range (exactly len - 2)
            total_angles = len(dx) - 2
            if total_angles <= 0: return 1.009
            
            p1 = np.sum(v >= 90.0) / total_angles
            p2 = np.sum(v < 90.0) / total_angles
            
            return ((-p1*math.log2(p1) if p1>0 else 0) + (-p2*math.log2(p2) if p2>0 else 0))
        except:
            return 1.009

    @staticmethod
    def _calc_excel_fd(dr_array, S_values, aa9_val, z16_val):
        """Calculates Fractal Dimension over the dynamic regression window match of Excel"""
        try:
            q = dr_array[~np.isnan(dr_array)]
            q = q[q > 0]
            if len(q) < 50: 
                return 1.000

            # Fallback in case S_values was not loaded from Excel column S
            if len(S_values) == 0:
                S_values = np.sort(np.unique(q))
                if len(S_values) < 26: return 1.000

            log_S = np.log10(S_values)
            log_V = []
            
            # Recreate cumulative probabilities P(r < s)
            for s in S_values:
                t = np.sum(q < s)
                u = np.sum(q > s)
                v = t / (t + u) if (t + u) > 0 else 0.0
                log_V.append(np.log10(v) if v > 0 else -np.inf)

            log_V = np.array(log_V)

            # Replicate Excel match logic to find AA10, AA11, AA12, and AA13
            # COUNTIF(log_S, <= aa9)
            count_le = np.sum(log_S <= aa9_val)
            aa10_val = log_S[count_le - 1] if count_le > 0 else log_S[0]

            # COUNTIF(log_S, >= aa9)
            count_ge = np.sum(log_S >= aa9_val)
            aa11_val = log_S[len(log_S) - count_ge] if count_ge > 0 else log_S[-1]

            # AA12 = MIN(AA10, AA11)
            aa12_val = min(aa10_val, aa11_val)

            # MATCH(AA12, log_S) (1-based index)
            # Safe float lookup using argmin to eliminate floating point mismatch errors
            match_idx = np.argmin(np.abs(log_S - aa12_val))
            aa13_val = int(match_idx) + 1

            # Determine the exact start position of the 11-point regression window
            start_idx = aa13_val + z16_val - 1
            
            # Boundary check
            if start_idx + 11 > len(log_S):
                start_idx = max(0, len(log_S) - 11)

            x_window = log_S[start_idx : start_idx + 11]
            y_window = log_V[start_idx : start_idx + 11]

            # Drop inf values if any exist in the edge scaling window
            valid_mask = ~np.isinf(y_window)
            if np.sum(valid_mask) < 2: return 1.000

            slope, _, _, _, _ = linregress(x_window[valid_mask], y_window[valid_mask])
            # CLIP REMOVED to align with Excel outputting values below 1.0 (e.g. 0.147)
            return float(slope)
        except:
            return 1.000