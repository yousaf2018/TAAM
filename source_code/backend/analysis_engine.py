import os, cv2, torch, numpy as np, pandas as pd
import math
import zipfile  # Needed for zip file validation checks
from scipy.stats import linregress

class AnalysisEngine:
    @staticmethod
    def calculate_behavior(xlsx_path, config, roi_list, arena_settings, logger):
        px_per_cm = float(config.get('conversion', 18.74))
        fps = float(config.get('fps', 24.99))
        f_thr = float(config.get('freeze_thresh', 0.1))
        r_thr = float(config.get('rapid_thresh', 3.0))
        
        # Body length conversion settings
        body_length_cm = float(config.get('body_length', 3.0))
        f_thr_bl = float(config.get('freeze_thresh_bl', 0.05))
        r_thr_bl = float(config.get('rapid_thresh_bl', 1.0))
        
        # Performance check: If True, returns full frame-by-frame data. If False, skips allocation.
        export_kinetics = config.get('export_kinetics', True)
        
        results_summary, raw_kinetics = [], []

        # --- PRE-READ INTEGRITY CHECK FOR CRIPPLED ZIP FILES ---
        if not os.path.exists(xlsx_path):
            logger.emit(f"      [TAAM Math ERROR] File path does not exist: {xlsx_path}")
            return [], pd.DataFrame()
            
        if os.path.getsize(xlsx_path) == 0:
            logger.emit(f"      [TAAM Math ERROR] File is empty (0 bytes): {os.path.basename(xlsx_path)}")
            return [], pd.DataFrame()

        # Check if the filename claims to be a zipped OpenXML format and has zip integrity
        if xlsx_path.lower().endswith(('.xlsx', '.xlsm', '.xltx', '.xltm', '.xlsx_processed')):
            if not zipfile.is_zipfile(xlsx_path):
                logger.emit(f"      [TAAM Math WARNING] File '{os.path.basename(xlsx_path)}' has a zipped Excel extension but is not a valid zip archive (corrupt, empty, or locked). Skipping.")
                return [], pd.DataFrame()

        try:
            xl = pd.ExcelFile(xlsx_path)
        except Exception as file_read_err:
            logger.emit(f"      [TAAM Math ERROR] Failed to load workbook '{os.path.basename(xlsx_path)}': {str(file_read_err)}")
            return [], pd.DataFrame()

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
            try:
                S_values = pd.to_numeric(tdf.iloc[:, 18], errors='coerce').dropna().values
            except Exception:
                S_values = np.array([])

            try:
                aa9_val = float(tdf.iloc[7, 26])
                if np.isnan(aa9_val):
                    aa9_val = 0.0
            except Exception:
                aa9_val = 0.0

            try:
                z16_val = int(tdf.iloc[14, 25])
                if np.isnan(z16_val):
                    z16_val = 5
            except Exception:
                z16_val = 5

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
            
            # Speed scaling relative to the specimen's body length
            tdf['speed_bl'] = tdf['speed'] / body_length_cm

            # 2. BEHAVIORAL STATES (Standard cm/s boundaries)
            spd_clean = tdf['speed'].fillna(50.0).values
            tdf['freezing'] = (spd_clean < f_thr).astype(int)
            tdf['swimming'] = ((spd_clean >= f_thr) & (spd_clean < r_thr)).astype(int)
            tdf['rapid'] = (spd_clean >= r_thr).astype(int)

            # 2b. BEHAVIORAL STATES (Relative body-length/s boundaries)
            spd_bl_clean = tdf['speed_bl'].fillna(50.0 / body_length_cm).values
            tdf['freezing_bl'] = (spd_bl_clean < f_thr_bl).astype(int)
            tdf['swimming_bl'] = ((spd_bl_clean >= f_thr_bl) & (spd_bl_clean < r_thr_bl)).astype(int)
            tdf['rapid_bl'] = (spd_bl_clean >= r_thr_bl).astype(int)

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
                
                # Apply initialization adjustments to BL indices as well
                tdf.loc[is_first, 'swimming_bl'] = 1
                tdf.loc[is_first, ['freezing_bl', 'rapid_bl']] = 0
            else:
                tdf['prev_x'], tdf['prev_y'] = tdf['X'].shift(1), tdf['Y'].shift(1)
                tdf.loc[tdf.index[0], 'swimming'] = 1
                tdf.loc[tdf.index[0], ['freezing', 'rapid']] = 0
                
                # Apply initialization adjustments to BL indices as well
                tdf.loc[tdf.index[0], 'swimming_bl'] = 1
                tdf.loc[tdf.index[0], ['freezing_bl', 'rapid_bl']] = 0

            tdf['is_top'] = (tdf['prev_x'] < x_dot).astype(int)
            tdf['thigmo'] = np.sqrt((tdf['prev_x']-x_dot)**2 + (tdf['prev_y']-y_dot)**2) / px_per_cm
            if id_col: tdf.loc[is_first, 'is_top'] = 0
            else: tdf.loc[tdf.index[0], 'is_top'] = 0

            # 4. COMPLEXITY (Optimized Dual Slicing)
            fd_list, ent_list = [], []
            slice_size = 15000

            if id_col:
                for _, sub_df in tdf.groupby(id_col):
                    dr_global = sub_df['dr'].values
                    
                    # A. Fractal Dimension on globally pre-calculated displacement array
                    fd_slices = []
                    for start_idx in range(0, len(dr_global), slice_size):
                        slice_dr = dr_global[start_idx : start_idx + slice_size]
                        if len(slice_dr) >= 50:
                            fd_slices.append(AnalysisEngine._calc_excel_fd(slice_dr, S_values, aa9_val, z16_val))
                    
                    # B. Entropy on sliced coordinates to match Excel's row-gap offsets exactly
                    ent_slices = []
                    for start_idx in range(0, len(sub_df), slice_size):
                        slice_df = sub_df.iloc[start_idx : start_idx + slice_size]
                        if len(slice_df) >= 50:
                            ent_slices.append(AnalysisEngine._calc_ent(
                                slice_df['dx'].values, 
                                slice_df['dy'].values, 
                                slice_df['dr'].values
                            ))
                    
                    fd_list.append(np.mean(fd_slices) if fd_slices else 1.000)
                    ent_list.append(np.mean(ent_slices) if ent_slices else 1.009)
            else:
                dr_global = tdf['dr'].values
                
                # A. Fractal Dimension on globally pre-calculated displacement array
                fd_slices = []
                for start_idx in range(0, len(dr_global), slice_size):
                    slice_dr = dr_global[start_idx : start_idx + slice_size]
                    if len(slice_dr) >= 50:
                        fd_slices.append(AnalysisEngine._calc_excel_fd(slice_dr, S_values, aa9_val, z16_val))
                
                # B. Entropy on sliced coordinates to match Excel's row-gap offsets exactly
                ent_slices = []
                for start_idx in range(0, len(tdf), slice_size):
                    slice_df = tdf.iloc[start_idx : start_idx + slice_size]
                    if len(slice_df) >= 50:
                        ent_slices.append(AnalysisEngine._calc_ent(
                            slice_df['dx'].values, 
                            slice_df['dy'].values, 
                            slice_df['dr'].values
                        ))
                
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

            # Calculate ratio endpoints using Body Length / second groupings
            valid_frames_bl = int(tdf['freezing_bl'].sum() + tdf['swimming_bl'].sum() + tdf['rapid_bl'].sum())
            valid_frames_bl = valid_frames_bl if valid_frames_bl > 0 else 1

            frz_bl_pct = round((tdf['freezing_bl'].sum() / valid_frames_bl) * 100, 3)
            swm_bl_pct = round((tdf['swimming_bl'].sum() / valid_frames_bl) * 100, 3)
            rap_bl_pct = round(100.0 - (frz_bl_pct + swm_bl_pct), 3)
            rap_bl_pct = max(0.0, rap_bl_pct)

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
                "Entropy": round(np.mean(ent_list), 3) if ent_list else 1.009,
                
                # New Body-Length Scaled Endpoints using body-length/s
                "Average Speed (body-length/s)": round(np.nanmean(tdf['speed_bl'].values), 3),
                "Freezing Time Ratio (body-length/s) (%)": frz_bl_pct,
                "Swimming Time Ratio (body-length/s) (%)": swm_bl_pct,
                "Rapid movement time ratio (body-length/s) (%)": rap_bl_pct
            })

            # Check if we should compile high-frequency frame coordinates
            if export_kinetics:
                tdf['Arena_ID'] = arena_id
                raw_kinetics.append(tdf)

        return results_summary, pd.concat(raw_kinetics) if raw_kinetics else pd.DataFrame()

    @staticmethod
    def _calc_ent(dx, dy, dr):
        """Calculates Shannon Entropy matching turning angle column R and cell AD6"""
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dot = dx[2:]*dx[1:-1] + dy[2:]*dy[1:-1]  # consecutive vectors
                denom = dr[1:-1] * dr[2:]
                cos_v = np.clip(np.where(denom > 0, dot/denom, np.nan), -1.0, 1.0)
                theta = np.arccos(cos_v)*(180.0/np.pi)
            
            # Excel's COUNTA(R4:R15001) is exactly len(dx) - 2 (total possible slots)
            M = len(dx) - 2
            if M <= 0: 
                return 1.009
            
            # Replicates Excel's cell-registry limits by using a double-precision float epsilon
            M1 = np.sum(theta >= (90.0 - 1e-9))
            M2 = np.sum(theta < (90.0 - 1e-9))
            
            p1 = M1 / M
            p2 = M2 / M
            
            return ((-p1*math.log2(p1) if p1>0 else 0) + (-p2*math.log2(p2) if p2>0 else 0))
        except:
            return 1.009

    @staticmethod
    def _calc_excel_fd(dr_array, S_values, aa9_val, z16_val):
        """Calculates Fractal Dimension over the strict centered regression window of Excel"""
        try:
            # Drop NaN values, but KEEP 0.0 values (just like Excel's Q column!)
            q = dr_array[~np.isnan(dr_array)]
            if len(q) < 50: 
                return 1.000

            # Fallback in case S_values was not loaded from Excel column S (recreates linear sequence r)
            if len(S_values) == 0:
                S_values = np.arange(0.1, 185.1, 0.1)
                if len(S_values) >= 10:
                    S_values[9] = 1.01  # Set S11 exactly to 1.01 matching your sheet

            log_S = np.log10(S_values)
            log_V = []
            
            # Recreate cumulative probabilities P(r < s)
            for s in S_values:
                t = np.sum(q < s)
                u = np.sum(q > s)
                v = t / (t + u) if (t + u) > 0 else 0.0
                log_V.append(np.log10(v) if v > 0 else -np.inf)

            log_V = np.array(log_V)

            # Match Excel AA9 fallback value (typically 0.0, representing scale 1.01 / 1.0)
            if aa9_val is None or aa9_val == 0.0:
                aa9_val = 0.0

            # Find the closest match index (0-based) in log_S for aa9_val
            match_idx = np.argmin(np.abs(log_S - aa9_val))

            # Excel uses an 11-point window centered around match_idx (using Column Z offsets -5 to 5)
            # Row 16 of the table corresponds to offset +5 (index match_idx + 5)
            # Row 26 of the table corresponds to offset -5 (index match_idx - 5)
            if z16_val is None or z16_val == 0:
                z16_val = 5

            start_idx = match_idx + z16_val - 1
            
            # Regression is done over the 11 points centered around start_idx (extending down from start_idx)
            x_window = log_S[start_idx - 10 : start_idx + 1]
            y_window = log_V[start_idx - 10 : start_idx + 1]

            # Drop inf values if any exist in the edge scaling window
            valid_mask = ~np.isinf(y_window)
            if np.sum(valid_mask) < 2: return 1.000

            slope, _, _, _, _ = linregress(x_window[valid_mask], y_window[valid_mask])
            return float(slope)
        except:
            return 1.000