import os
import cv2
import subprocess
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class SplitterWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, videos, split_sec, output_dir, daynight_config=None):
        super().__init__()
        self.videos = videos
        self.split_sec = split_sec
        self.output_dir = output_dir
        self.daynight_config = daynight_config
        self.is_running = True

    def stop(self):
        self.is_running = False

    def check_ffmpeg(self):
        """Checks if FFmpeg is available in the system PATH."""
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            return False

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            ffmpeg_available = self.check_ffmpeg()
            
            if ffmpeg_available:
                self.log_signal.emit("⚡ FFmpeg detected! Enabling Fast Re-encoding mode (frame-accurate splits).")
            else:
                self.log_signal.emit("⚠️ FFmpeg not found in system PATH. Falling back to Sequential OpenCV rendering.")

            for video_path in self.videos:
                if not self.is_running:
                    break

                base_name = os.path.splitext(os.path.basename(video_path))[0]
                self.log_signal.emit(f"Processing: {base_name}")

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if total_frames <= 0 or fps <= 0:
                    self.log_signal.emit(f"⚠️ Skipping {base_name}: Invalid properties (frames={total_frames}, fps={fps})")
                    cap.release()
                    continue

                if self.daynight_config and self.daynight_config.get('enabled', False):
                    self.run_daynight_split(cap, video_path, base_name, total_frames, fps, ffmpeg_available)
                else:
                    self.run_uniform_split(cap, video_path, base_name, total_frames, fps, ffmpeg_available)

                cap.release()

            if self.is_running:
                self.finished_signal.emit("Splitting pipeline finished successfully.")
            else:
                self.finished_signal.emit("Splitting process stopped by user.")

        except Exception as e:
            self.log_signal.emit(f"❌ Splitter Error: {str(e)}")
            self.finished_signal.emit(f"Failed: {str(e)}")

    def run_uniform_split(self, cap, video_path, base_name, total_frames, fps, use_ffmpeg):
        chunk_frames = int(self.split_sec * fps)
        num_chunks = int(np.ceil(total_frames / chunk_frames))
        self.log_signal.emit(f"Splitting uniformly into {num_chunks} chunks of {self.split_sec}s.")

        chunks = []
        for i in range(num_chunks):
            start_f = i * chunk_frames
            end_f = min(total_frames, (i + 1) * chunk_frames)
            chunks.append((start_f, end_f, "part"))

        if use_ffmpeg:
            self.export_ffmpeg(video_path, chunks, base_name, fps)
        else:
            self.export_sequential(cap, chunks, base_name, total_frames, fps)

    def run_daynight_split(self, cap, video_path, base_name, total_frames, fps, use_ffmpeg):
        threshold = self.daynight_config['threshold']
        day_chunk_sec = self.daynight_config['day_chunk']
        night_chunk_sec = self.daynight_config['night_chunk']

        self.log_signal.emit(f"Running Multi-Transition Split. Threshold: {threshold}")

        # Timeline Scanner
        step = max(500, total_frames // 1000)
        brightnesses = []
        frames_checked = []

        for f_idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(np.mean(gray))
            frames_checked.append(f_idx)

        # Detect transition boundaries
        transition_points = []
        for i in range(len(brightnesses) - 1):
            b1, b2 = brightnesses[i], brightnesses[i+1]
            if (b1 >= threshold and b2 < threshold) or (b1 < threshold and b2 >= threshold):
                start_zone = frames_checked[i]
                end_zone = frames_checked[i+1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_zone)
                
                zone_transition = -1
                for fine_f in range(start_zone, end_zone + 1):
                    ret, f_frame = cap.read()
                    if not ret:
                        break
                    f_gray = cv2.cvtColor(f_frame, cv2.COLOR_BGR2GRAY)
                    f_avg = np.mean(f_gray)
                    if (b1 >= threshold and f_avg < threshold) or (b1 < threshold and f_avg >= threshold):
                        zone_transition = fine_f
                        break
                
                if zone_transition != -1:
                    # Require transitions to be at least 5 minutes apart to avoid light flicker trigger
                    if not transition_points or (zone_transition - transition_points[-1]) > (fps * 300):
                        transition_points.append(zone_transition)

        self.log_signal.emit(f"🎯 Located {len(transition_points)} transition boundaries across timeline.")

        # Partition video timeline into segments
        boundaries = [0] + transition_points + [total_frames]
        day_counter = 1
        night_counter = 1
        chunks = []

        day_chunk_frames = int(day_chunk_sec * fps)
        night_chunk_frames = int(night_chunk_sec * fps)

        def generate_segment_chunks(start_f, end_f, chunk_frames, direction, label):
            seg_chunks = []
            if chunk_frames <= 0:
                return seg_chunks
            if direction == "forward":
                curr = start_f
                while curr < end_f:
                    nxt = min(end_f, curr + chunk_frames)
                    seg_chunks.append((curr, nxt, label))
                    curr = nxt
            else: # backward
                curr = end_f
                while curr > start_f:
                    prev = max(start_f, curr - chunk_frames)
                    seg_chunks.insert(0, (prev, curr, label))
                    curr = prev
            return seg_chunks

        # Apply specific direction settings on every segment dynamically
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i+1]
            if seg_start >= seg_end:
                continue

            # Evaluate segment's safe midpoint frame to prevent transition-ramp bias
            mid_frame = (seg_start + seg_end) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, test_frame = cap.read()
            b_val = threshold
            if ret:
                b_val = np.mean(cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY))

            is_day = b_val >= threshold

            if is_day:
                label = f"day{day_counter}"
                day_counter += 1
                chunk_frames = day_chunk_frames
            else:
                label = f"night{night_counter}"
                night_counter += 1
                chunk_frames = night_chunk_frames

            # STRICT OVERRIDE RULE:
            # day1 uses backward chunking; all other phases (night1, day2, night2, etc.) use forward chunking.
            if label == "day1":
                direction = "backward"
            else:
                direction = "forward"

            self.log_signal.emit(f"Segment {i+1}: Frames {seg_start} to {seg_end} chunked as [{label.upper()}] using strict {direction.upper()} alignment.")
            seg_chunks = generate_segment_chunks(seg_start, seg_end, chunk_frames, direction, label)
            chunks.extend(seg_chunks)

        if use_ffmpeg:
            self.export_ffmpeg(video_path, chunks, base_name, fps)
        else:
            self.export_sequential(cap, chunks, base_name, total_frames, fps)

    def export_ffmpeg(self, video_path, chunks, base_name, fps):
        """High-speed H.264 re-encoding slice operation with chronological directory sorting and strict FPS lock."""
        num_chunks = len(chunks)
        video_out_dir = os.path.join(self.output_dir, base_name)
        self.log_signal.emit(f"Encoding {num_chunks} files via FFmpeg pipeline at strictly {fps:.4f} FPS...")

        for idx, (start, end, label) in enumerate(chunks):
            if not self.is_running:
                break

            start_sec = start / fps
            duration = (end - start) / fps
            
            # Map dynamic segment folders (day1, night1, day2, night2...)
            dest_folder = os.path.join(video_out_dir, label)
            os.makedirs(dest_folder, exist_ok=True)
            out_path = os.path.join(dest_folder, f"{base_name}_part{idx:03d}.mp4")

            # -r forces FFmpeg's encoder to output at the exact original source frame rate
            cmd = [
                "ffmpeg", "-y",
                "-threads", "2",
                "-ss", f"{start_sec:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                "-r", f"{fps:.4f}",  # FORCE exact original frame rate
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                out_path
            ]

            if idx % 5 == 0 or num_chunks < 20:
                self.log_signal.emit(f"⚡ [FFmpeg Encode] Slice {idx+1}/{num_chunks} [{label.upper()}]: {start_sec:.1f}s to {start_sec+duration:.1f}s")

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.msleep(50)

    def export_sequential(self, cap, chunks, base_name, total_frames, fps):
        """Sequential single-pass writing with chronological directory sorting using original FPS."""
        num_chunks = len(chunks)
        if num_chunks == 0:
            return

        video_out_dir = os.path.join(self.output_dir, base_name)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        current_chunk_idx = 0
        c_start, c_end, c_label = chunks[current_chunk_idx]
        
        dest_folder = os.path.join(video_out_dir, c_label)
        os.makedirs(dest_folder, exist_ok=True)
        out_path = os.path.join(dest_folder, f"{base_name}_part{current_chunk_idx:03d}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        self.log_signal.emit(f"📦 Exporting chunk {current_chunk_idx + 1}/{num_chunks} [{c_label.upper()}]: Frames {c_start} to {c_end}")

        for f_idx in range(total_frames):
            if not self.is_running:
                break

            ret, frame = cap.read()
            if not ret:
                break

            while f_idx >= c_end:
                writer.release()
                current_chunk_idx += 1
                if current_chunk_idx >= num_chunks:
                    writer = None
                    break

                c_start, c_end, c_label = chunks[current_chunk_idx]
                
                dest_folder = os.path.join(video_out_dir, c_label)
                os.makedirs(dest_folder, exist_ok=True)
                out_path = os.path.join(dest_folder, f"{base_name}_part{current_chunk_idx:03d}.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                if current_chunk_idx % 5 == 0 or num_chunks < 20:
                    self.log_signal.emit(f"📦 Exporting chunk {current_chunk_idx + 1}/{num_chunks} [{c_label.upper()}]: Frames {c_start} to {c_end}")

            if writer is not None:
                writer.write(frame)

            if f_idx % 100 == 0:
                self.msleep(1)

        if writer is not None:
            writer.release()