from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QWheelEvent
import cv2

class VideoSelectorWidget(QWidget):
    selection_changed = pyqtSignal() 

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Image State
        self.original_pixmap = None
        self.annotations = {} # { frame_idx: [(QRectF, class_id), ...] }
        self.current_frame_idx = 0
        self.selected_index = -1
        self.current_class_id = 0
        self.copy_buffer = [] 
        
        # Zoom & Pan State
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0) # Offset in screen pixels
        self.last_mouse_pos = QPoint()
        
        # Interaction State
        self.mode = "IDLE" # IDLE, DRAWING, MOVING, PANNING
        self.start_pt = QPointF() # Image coordinates
        self.drag_offset = QPointF() # Image coordinates
        
        self.class_colors = [
            QColor(57, 255, 20), QColor(255, 50, 50), 
            QColor(50, 100, 255), QColor(255, 255, 0)
        ]

    def set_current_class(self, cid): self.current_class_id = int(cid)

    def set_current_frame(self, frame_idx, frame_bgr):
        self.current_frame_idx = int(frame_idx)
        if frame_bgr is not None:
            h, w, ch = frame_bgr.shape
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qt_img)
        self.selected_index = -1
        self.update()

    def _get_base_scale(self):
        """Calculates the scale needed to fit the image to the widget size."""
        if not self.original_pixmap: return 1.0
        ww, wh = self.width(), self.height()
        iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
        return min(ww / iw, wh / ih)

    def map_to_image(self, pos: QPoint):
        """Converts Widget pixels -> Image pixels."""
        if not self.original_pixmap: return QPointF(0, 0)
        base_scale = self._get_base_scale()
        # Adjusted for zoom and pan
        ix = (pos.x() - self.width()/2 - self.pan_offset.x()) / (base_scale * self.zoom_level) + self.original_pixmap.width()/2
        iy = (pos.y() - self.height()/2 - self.pan_offset.y()) / (base_scale * self.zoom_level) + self.original_pixmap.height()/2
        return QPointF(ix, iy)

    def map_to_screen(self, ix, iy):
        """Converts Image pixels -> Widget pixels."""
        base_scale = self._get_base_scale()
        sx = (ix - self.original_pixmap.width()/2) * (base_scale * self.zoom_level) + self.width()/2 + self.pan_offset.x()
        sy = (iy - self.original_pixmap.height()/2) * (base_scale * self.zoom_level) + self.height()/2 + self.pan_offset.y()
        return sx, sy

    def wheelEvent(self, event: QWheelEvent):
        """Handles zooming in/out at the mouse cursor."""
        if not self.original_pixmap: return
        
        # Calculate cursor pos in image space before zoom
        pos_before = self.map_to_image(event.position().toPoint())
        
        # Zoom logic
        zoom_step = 1.15
        if event.angleDelta().y() > 0:
            self.zoom_level *= zoom_step
        else:
            self.zoom_level /= zoom_step
        
        # Clamp zoom
        self.zoom_level = max(0.5, min(self.zoom_level, 20.0))
        
        # Calculate cursor pos in image space after zoom to adjust pan
        pos_after = self.map_to_image(event.position().toPoint())
        
        # Adjust pan so the point under the mouse stays fixed
        base_scale = self._get_base_scale()
        self.pan_offset += (pos_after - pos_before) * (base_scale * self.zoom_level)
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.original_pixmap: return

        # 1. Draw Image with Zoom and Pan
        base_scale = self._get_base_scale()
        nw = self.original_pixmap.width() * base_scale * self.zoom_level
        nh = self.original_pixmap.height() * base_scale * self.zoom_level
        
        img_rect = QRectF(self.width()/2 + self.pan_offset.x() - nw/2, 
                          self.height()/2 + self.pan_offset.y() - nh/2, nw, nh)
        
        painter.drawPixmap(img_rect.toRect(), self.original_pixmap)

        # 2. Draw Annotations
        fid = int(self.current_frame_idx)
        cur_ann = self.annotations.get(fid, [])
        
        for i, (rect_data, cid) in enumerate(cur_ann):
            # Ensure we have a QRectF
            if isinstance(rect_data, list):
                rect = QRectF(rect_data[0], rect_data[1], rect_data[2], rect_data[3])
            else: rect = rect_data
            
            # Map image coords to screen coords
            sx, sy = self.map_to_screen(rect.x(), rect.y())
            sw = rect.width() * base_scale * self.zoom_level
            sh = rect.height() * base_scale * self.zoom_level
            
            color = self.class_colors[int(cid) % len(self.class_colors)]
            pen = QPen(color, 2)
            if i == self.selected_index: 
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(QRectF(sx, sy, sw, sh))
            painter.drawText(int(sx), int(sy-5), f"ID:{cid}")

    def mousePressEvent(self, event):
        if not self.original_pixmap: return
        self.last_mouse_pos = event.position().toPoint()
        img_pt = self.map_to_image(self.last_mouse_pos)

        # Right Click: Start Panning
        if event.button() == Qt.MouseButton.RightButton:
            self.mode = "PANNING"
            return

        # Middle Click: Reset Zoom
        if event.button() == Qt.MouseButton.MiddleButton:
            self.zoom_level = 1.0
            self.pan_offset = QPointF(0, 0)
            self.update()
            return

        # Left Click: Check for Selection or Draw
        if event.button() == Qt.MouseButton.LeftButton:
            fid = int(self.current_frame_idx)
            cur_ann = self.annotations.get(fid, [])
            
            for i in range(len(cur_ann)-1, -1, -1):
                rect, cid = cur_ann[i]
                if rect.contains(img_pt):
                    self.selected_index = i
                    self.mode = "MOVING"
                    self.drag_offset = img_pt - rect.topLeft()
                    self.update()
                    return

            self.mode = "DRAWING"
            self.start_pt = img_pt
            self.selected_index = -1
            self.update()

    def mouseMoveEvent(self, event):
        if not self.original_pixmap: return
        curr_pos = event.position().toPoint()
        img_pt = self.map_to_image(curr_pos)

        if self.mode == "PANNING":
            delta = curr_pos - self.last_mouse_pos
            self.pan_offset += QPointF(delta.x(), delta.y())
            self.update()
        
        elif self.mode == "MOVING" and self.selected_index != -1:
            fid = int(self.current_frame_idx)
            rect, cid = self.annotations[fid][self.selected_index]
            rect.moveTo(img_pt - self.drag_offset)
            self.update()

        self.last_mouse_pos = curr_pos

    def mouseReleaseEvent(self, event):
        if self.mode == "DRAWING":
            img_pt = self.map_to_image(event.position().toPoint())
            new_rect = QRectF(self.start_pt, img_pt).normalized()
            
            if new_rect.width() > 5:
                fid = int(self.current_frame_idx)
                if fid not in self.annotations: self.annotations[fid] = []
                self.annotations[fid].append((new_rect, int(self.current_class_id)))
        
        self.mode = "IDLE"
        self.selection_changed.emit()
        self.update()

    def keyPressEvent(self, event):
        fid = int(self.current_frame_idx)
        if event.key() in [Qt.Key.Key_Delete, Qt.Key.Key_Backspace]:
            if fid in self.annotations and self.selected_index != -1:
                del self.annotations[fid][self.selected_index]
                self.selected_index = -1
                self.selection_changed.emit()
                self.update()
        
        # Copy-Paste Logic
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_C:
            if fid in self.annotations and self.selected_index != -1:
                self.copy_buffer = [self.annotations[fid][self.selected_index]]
        
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_V:
            if self.copy_buffer:
                if fid not in self.annotations: self.annotations[fid] = []
                for rect, cid in self.copy_buffer:
                    self.annotations[fid].append((QRectF(rect), cid))
                self.selection_changed.emit()
                self.update()

    def clear_current_frame(self):
        fid = int(self.current_frame_idx)
        if fid in self.annotations:
            del self.annotations[fid]
            self.selected_index = -1
            self.selection_changed.emit()
            self.update()

    def get_stats(self):
        annotated_frames = len(self.annotations)
        total_boxes = sum(len(v) for v in self.annotations.values())
        return annotated_frames, total_boxes, 0