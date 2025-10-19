import sys
import re
import time
import torch
import logging
import os
import random
import tempfile
import json
import math
import hashlib
import shutil
import numpy as np
from numpy.core._exceptions import _ArrayMemoryError
from PIL import Image
import traceback
import gc
import psutil
import GPUtil
from pathlib import Path
from diffusers import DiffusionPipeline
from PySide6.QtWidgets import (QApplication, QDialog, QGraphicsObject, QGraphicsOpacityEffect, QInputDialog, QListWidget, QListWidgetItem, QProxyStyle, QStackedWidget, QStyle, QTextBrowser, QToolTip, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QMainWindow, QMenuBar, QFileDialog, QSlider, QProgressBar, QGroupBox, 
                             QMessageBox, QSizePolicy, QComboBox, QTextEdit, QCheckBox, QGridLayout, QMenu, QGraphicsDropShadowEffect,
                             QSplitter, QFrame, QScrollBar, QSpacerItem, QToolButton, QGraphicsScene, QGraphicsView, QGraphicsItem, QScrollArea)
from PySide6.QtGui import (QAction, QConicalGradient, QFontMetrics, QLinearGradient, QMouseEvent, QPainterPath, QPixmap, QImage, QFont, QPalette, QColor, QPainter, QIcon, QPen, QBrush, QRadialGradient, QSyntaxHighlighter, QTextCharFormat, QTextCursor, QTextFormat, QDrag)
from PySide6.QtCore import (QAbstractAnimation, QObject, QParallelAnimationGroup, QPointF, QRunnable, QThreadPool, Qt, QThread, Signal, QTimer, QSize, QPoint, QRectF, QSettings, QPropertyAnimation, QEasingCurve, QEvent, QRect, QMimeData)
from logging.handlers import RotatingFileHandler
from spellchecker import SpellChecker
from datetime import datetime

# --- Dynamic Path Resolution for Assets ---
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (e.g., by PyInstaller)
    BASE_DIR = sys._MEIPASS
else:
    # If run as a normal .py script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
# --- End of Dynamic Path Resolution ---


def setup_logger():
    logger = logging.getLogger('ImageGenerator')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        'image_generator.log', maxBytes=5*1024*1024, backupCount=2)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def clear_gpu_cache():
    if torch.cuda.is_available():
        logger.info("Clearing GPU cache...")
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")
    else:
        logger.info("CUDA is not available. No GPU cache to clear.")

class MemoryManager:
    @staticmethod
    def clear_cache():
        gc.collect()
        clear_gpu_cache()

    @staticmethod
    def get_memory_usage():
        process = psutil.Process()
        ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # in MB
            return ram_usage, gpu_usage
        return ram_usage, None
            
class SpellCheckHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.spell = SpellChecker()

    def highlightBlock(self, text):
        format = QTextCharFormat()
        format.setUnderlineColor(QColor("red"))
        format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)

        for word_object in re.finditer(r'\w+', text):
            if not self.spell.known([word_object.group()]):
                self.setFormat(word_object.start(),
                               word_object.end() - word_object.start(),
                               format)

class SpellCheckTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighter = SpellCheckHighlighter(self.document())
        self.spell = SpellChecker()

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        text = cursor.selectedText()

        if not self.spell.known([text]):
            suggestions = self.spell.candidates(text)
            if suggestions:
                submenu = QMenu("Spelling Suggestions", menu)
                for suggestion in suggestions:
                    action = submenu.addAction(suggestion)
                    action.triggered.connect(lambda _, s=suggestion: self.correctWord(cursor, s))
                menu.insertMenu(menu.actions()[0], submenu)

        # Add custom "Save to Library" action
        menu.addSeparator()
        save_action = menu.addAction("Save to Library")
        save_action.triggered.connect(self.save_to_library)

        menu.exec_(event.globalPos())

    def correctWord(self, cursor, suggestion):
        cursor.beginEditBlock()
        cursor.removeSelectedText()
        cursor.insertText(suggestion)
        cursor.endEditBlock()

    def save_to_library(self):
        # Find the main window (ImageGeneratorUI instance)
        main_window = self.window()
        if hasattr(main_window, 'save_prompt_to_library'):
            main_window.save_prompt_to_library()
        else:
            print("Error: Cannot find save_prompt_to_library method")

class ImageGeneratorThread(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(str, float)
    error = Signal(str)
    intermediate_result = Signal(Image.Image)
    

    def __init__(self, model, prompt, num_inference_steps, width, height, cache_dir, negative_prompt="", seed=None, guidance_scale=7.5, clip_skip=None, live_render=False):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.cache_dir = cache_dir
        self.seed = seed
        self.guidance_scale = guidance_scale        
        self.clip_skip = clip_skip
        self.is_cancelled = False
        self.live_render = live_render

    def run(self):
        try:
            start_time = time.time()
            
            def callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
                if self.is_cancelled:
                    raise InterruptedError("Generation cancelled by user")
                progress = int((step + 1) / self.num_inference_steps * 100)
                self.progress.emit(0, progress)
                self.status.emit(f"Step {step + 1}/{self.num_inference_steps}: Generating image")
                
                if self.live_render and self.width == 512 and self.height == 512:
                    with torch.no_grad():
                        latents = 1 / 0.18215 * latents
                        image = self.model.vae.decode(latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = (image * 255).round().astype("uint8")
                        image = Image.fromarray(image[0])
                        self.intermediate_result.emit(image)

            self.status.emit("Generating image...")
            
            # Validate input dimensions
            if self.width % 8 != 0 or self.height % 8 != 0:
                raise ValueError("Width and height must be multiples of 8")

            # Set the generator for reproducibility if seed is provided
            generator = torch.Generator("cuda").manual_seed(self.seed) if self.seed is not None else None

            # Try CUDA first
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(device)
                
                output = self.model(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt if self.negative_prompt else None,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    callback=callback,
                    callback_steps=1,
                    width=self.width,
                    height=self.height,
                    generator=generator,
                    clip_skip=self.clip_skip if self.clip_skip != 1 else None
                )
                
                # The model returns a list of PIL Images
                if isinstance(output.images, list) and len(output.images) > 0:
                    image = output.images[0]
                else:
                    raise ValueError("Model did not return any images")
            
            except RuntimeError as e:
                if "CUDA" in str(e) or "cuDNN" in str(e):
                    logging.warning(f"CUDA error occurred: {str(e)}. Falling back to CPU.")
                    self.status.emit("CUDA error occurred. Falling back to CPU (this may be slower)...")
                    
                    # Fall back to CPU
                    self.model = self.model.to("cpu")
                    output = self.model(
                        prompt=self.prompt,
                        num_inference_steps=self.num_inference_steps,
                        callback=callback,
                        callback_steps=1,
                        width=self.width,
                        height=self.height,
                        clip_skip=self.clip_skip
                    )
                    
                    if isinstance(output.images, list) and len(output.images) > 0:
                        image = output.images[0]
                    else:
                        raise ValueError("Model did not return any images")
                else:
                    raise

            self.status.emit("Post-processing image")
            
            # Process the image
            try:
                # If the image is already a PIL Image, we don't need to convert it
                if not isinstance(image, Image.Image):
                    image = Image.fromarray((image * 255).astype('uint8'))
            except _ArrayMemoryError as e:
                raise MemoryError(f"Not enough memory to process the image. Try a smaller resolution. Error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Error processing the image: {str(e)}")
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            cache_path = f"{self.cache_dir}/image_{int(time.time())}.png"
            image.save(cache_path)
            
            self.finished.emit(cache_path, generation_time)
            logger.info(f"Image generated successfully. Size: {self.width}x{self.height}, Time: {generation_time:.2f}s")
        except InterruptedError as e:
            logger.info(str(e))
            self.status.emit("Generation cancelled")
        except MemoryError as e:
            error_msg = str(e)
            logger.error(f"Memory error during image generation: {error_msg}")
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error during image generation: {error_msg}")
            self.error.emit(error_msg)

    def cancel(self):
        self.is_cancelled = True

class GridGeneratorThread(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    finished = Signal(list, float)
    error = Signal(str)

    def __init__(self, model, prompt, num_inference_steps, width, height, num_images, cache_dir, negative_prompt="", seed=None, guidance_scale=7.5, clip_skip=None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height
        self.num_images = num_images
        self.cache_dir = cache_dir
        self.seed = seed
        self.guidance_scale = guidance_scale
        self.clip_skip = clip_skip
        self.is_cancelled = False

    def run(self):
        try:
            start_time = time.time()
            cache_paths = []
            
            for i in range(self.num_images):
                if self.is_cancelled:
                    raise InterruptedError("Generation cancelled by user")
                self.status.emit(f"Generating image {i+1}/{self.num_images}")
                
                def callback(step: int, timestep: int, latents: torch.FloatTensor) -> None:
                    if self.is_cancelled:
                        raise InterruptedError("Generation cancelled by user")
                    progress = int((step + 1) / self.num_inference_steps * 100)
                    self.progress.emit(i, progress)
                
                # Set the generator for reproducibility if seed is provided
                generator = torch.Generator("cuda").manual_seed(self.seed + i) if self.seed is not None else None
                
                output = self.model(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt if self.negative_prompt else None,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    callback=callback,
                    callback_steps=1,
                    width=self.width,
                    height=self.height,
                    generator=generator,
                    clip_skip=self.clip_skip if self.clip_skip != 1 else None
                )
                
                image = output.images[0]
                
                cache_path = f"{self.cache_dir}/grid_image_{i}_{int(time.time())}.png"
                image.save(cache_path)
                cache_paths.append(cache_path)
                
                self.status.emit(f"Post-processing image {i+1}/{self.num_images}")
            
            end_time = time.time()
            generation_time = end_time - start_time
            self.finished.emit(cache_paths, generation_time)
            logger.info(f"Grid generated successfully. {self.num_images} images, Size: {self.width}x{self.height}, Time: {generation_time:.2f}s")
        except InterruptedError as e:
            logger.info(str(e))
            self.status.emit("Generation cancelled")
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error during grid generation: {error_msg}")
            self.error.emit(error_msg)

    def cancel(self):
        self.is_cancelled = True

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(32)  # Standard Windows title bar height
        self.setAutoFillBackground(True)
        
        # Setup the layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Icon
        self.icon_label = QLabel(self)
        self.icon_label.setFixedSize(32, 32)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_path = os.path.join(ASSETS_DIR, "Prism XL.ico")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.icon_label.setPixmap(scaled_pixmap)
            self.icon_label.setStyleSheet("padding: 4px;")
        self.layout.addWidget(self.icon_label)
        
        # Title
        self.title_label = QLabel("Sapphire - PrismXL")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                padding-left: 5px;
                font-size: 12px;
            }
        """)
        self.layout.addWidget(self.title_label)
        
        # Spacer
        self.layout.addStretch()
        
        # Window controls
        self.controls_layout = QHBoxLayout()
        self.controls_layout.setSpacing(0)
        
        # Minimize button
        self.minimize_button = QPushButton("🗕")
        self.minimize_button.setFixedSize(45, 32)
        self.minimize_button.clicked.connect(self.minimize_window)
        
        # Maximize button
        self.maximize_button = QPushButton("🗖")
        self.maximize_button.setFixedSize(45, 32)
        self.maximize_button.clicked.connect(self.toggle_maximize)
        
        # Close button
        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(45, 32)
        self.close_button.clicked.connect(self.close_window)
        
        # Add buttons to controls layout
        self.controls_layout.addWidget(self.minimize_button)
        self.controls_layout.addWidget(self.maximize_button)
        self.controls_layout.addWidget(self.close_button)
        
        # Add controls to main layout
        self.layout.addLayout(self.controls_layout)
        
        # Window dragging
        self.pressing = False
        self.start = QPoint()
        self.was_maximized = False

        # Set initial style
        self.set_style(False)  # Default to light mode

    def set_style(self, is_night_mode):
        if is_night_mode:
            bg_color = "#1e1e1e"
            text_color = "#ffffff"
            button_hover_bg = "#404040"
            close_hover_bg = "#e81123"
        else:
            bg_color = "#2e2e2e"
            text_color = "#ffffff"
            button_hover_bg = "#404040"
            close_hover_bg = "#e81123"

        self.setStyleSheet(f"""
            CustomTitleBar {{
                background-color: {bg_color};
            }}
            QLabel {{
                color: {text_color};
            }}
            QPushButton {{
                background-color: transparent;
                border: none;
                color: {text_color};
                font-family: Segoe UI;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background-color: {button_hover_bg};
            }}
            QPushButton:pressed {{
                background-color: #333333;
            }}
            QPushButton#close_button:hover {{
                background-color: {close_hover_bg};
            }}
        """)
        
        # Set object names for specific styling
        self.close_button.setObjectName("close_button")

    def set_title(self, title):
        self.title_label.setText(title)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressing = True
            self.start = event.globalPos()
            # Store window state before dragging
            self.was_maximized = self.parent.isMaximized()

    def mouseMoveEvent(self, event):
        if self.pressing and not self.parent.isMaximized():
            self.parent.move(self.parent.pos() + (event.globalPos() - self.start))
            self.start = event.globalPos()
        elif self.pressing and self.parent.isMaximized():
            # Restore window and calculate new position
            self.restore_from_maximized(event.globalPos())

    def mouseReleaseEvent(self, event):
        self.pressing = False

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_maximize()

    def minimize_window(self):
        self.parent.showMinimized()

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_button.setText("🗖")
        else:
            self.parent.showMaximized()
            self.maximize_button.setText("🗗")

    def close_window(self):
        self.parent.close()

    def restore_from_maximized(self, global_pos):
        # Calculate the relative position where the user clicked on the title bar
        normal_geometry = self.parent.normalGeometry()
        ratio = (global_pos.x() - self.parent.frameGeometry().left()) / self.parent.frameGeometry().width()
        
        # Restore the window
        self.parent.showNormal()
        
        # Calculate the new position maintaining the mouse under the same relative point
        new_x = int(global_pos.x() - (normal_geometry.width() * ratio))
        new_y = global_pos.y() - (self.height() / 2)  # Vertical center of title bar
        
        self.parent.move(new_x, new_y)
        self.maximize_button.setText("🗖")
        
        # Update the start position for continued dragging
        self.start = global_pos

    def update_maximize_button(self):
        if self.parent.isMaximized():
            self.maximize_button.setText("🗗")
        else:
            self.maximize_button.setText("🗖")

class ZoomWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.zoom_pixmap = None
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.zoom_factor = 1.5

    def update_zoom(self, pixmap, cursor_pos, view_rect):
        if pixmap:
            scale_x = pixmap.width() / view_rect.width()
            scale_y = pixmap.height() / view_rect.height()

            image_x = cursor_pos.x() * scale_x
            image_y = cursor_pos.y() * scale_y
            
            zoom_width = self.width() / self.zoom_factor
            zoom_height = self.height() / self.zoom_factor
            
            aspect_ratio = pixmap.width() / pixmap.height()
            if aspect_ratio > 1:
                zoom_height = zoom_width / aspect_ratio
            else:
                zoom_width = zoom_height * aspect_ratio
            
            zoom_rect = QRectF(
                image_x - zoom_width / 2,
                image_y - zoom_height / 2,
                zoom_width,
                zoom_height
            )
            
            if zoom_rect.left() < 0:
                zoom_rect.moveLeft(0)
            if zoom_rect.top() < 0:
                zoom_rect.moveTop(0)
            if zoom_rect.right() > pixmap.width():
                zoom_rect.moveRight(pixmap.width())
            if zoom_rect.bottom() > pixmap.height():
                zoom_rect.moveBottom(pixmap.height())
            
            self.zoom_pixmap = pixmap.copy(zoom_rect.toRect()).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.update()

    def paintEvent(self, event):
        if self.zoom_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), 10, 10)
            
            painter.setClipPath(path)
            
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(self.rect())
            
            x = (self.width() - self.zoom_pixmap.width()) / 2
            y = (self.height() - self.zoom_pixmap.height()) / 2
            
            painter.drawPixmap(QPointF(x, y), self.zoom_pixmap)
            
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 100), 2))
            painter.drawPath(path)

class GridDisplayWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)

        self.main_image = QLabel(self)
        self.main_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_image.setMinimumSize(640, 640)
        self.main_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_image.setMouseTracking(True)

        self.thumbnail_widget = QWidget(self)
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_widget)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_layout.setSpacing(5)

        self.layout.addWidget(self.main_image, 4)
        self.layout.addWidget(self.thumbnail_widget, 1)

        self.thumbnails = []
        self.current_image = None
        
        self.zoom_widget = ZoomWidget()
        self.zoom_widget.hide()
        self.original_pixmap = None
        self.is_zooming = False
        self.last_mouse_pos = None

        self.main_image.installEventFilter(self)

    def update_zoom(self, pos):
        if self.is_zooming and self.original_pixmap:
            image_rect = self.main_image.rect()
            pixmap_rect = self.original_pixmap.rect()
            
            scale_x = image_rect.width() / pixmap_rect.width()
            scale_y = image_rect.height() / pixmap_rect.height()
            scale = min(scale_x, scale_y)
            
            scaled_width = pixmap_rect.width() * scale
            scaled_height = pixmap_rect.height() * scale
            
            x = (image_rect.width() - scaled_width) / 2
            y = (image_rect.height() - scaled_height) / 2
            view_rect = QRectF(x, y, scaled_width, scaled_height)
            
            adjusted_pos = QPointF(pos.x() - view_rect.x(), pos.y() - view_rect.y())
            
            original_x = adjusted_pos.x() / scale
            original_y = adjusted_pos.y() / scale
            
            self.zoom_widget.update_zoom(self.original_pixmap, QPointF(original_x, original_y), pixmap_rect)
            
            global_pos = self.main_image.mapToGlobal(pos)
            zoom_pos = global_pos + QPoint(20, 20)
            
            screen = QApplication.primaryScreen().geometry()
            if zoom_pos.x() + self.zoom_widget.width() > screen.right():
                zoom_pos.setX(global_pos.x() - self.zoom_widget.width() - 20)
            if zoom_pos.y() + self.zoom_widget.height() > screen.bottom():
                zoom_pos.setY(global_pos.y() - self.zoom_widget.height() - 20)
            
            self.zoom_widget.move(zoom_pos)

    def eventFilter(self, obj, event):
        if obj == self.main_image:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.is_zooming = True
                self.last_mouse_pos = event.position().toPoint()
                self.zoom_widget.show()
                self.update_zoom(self.last_mouse_pos)
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self.is_zooming = False
                self.zoom_widget.hide()
            elif event.type() == QEvent.Type.MouseMove:
                self.last_mouse_pos = event.position().toPoint()
                if self.is_zooming:
                    self.update_zoom(self.last_mouse_pos)
        return super().eventFilter(obj, event)

    def set_images(self, image_paths):
        for thumbnail in self.thumbnails:
            self.thumbnail_layout.removeWidget(thumbnail)
            thumbnail.deleteLater()
        self.thumbnails.clear()

        for i, path in enumerate(image_paths):
            thumbnail = ThumbnailLabel(self, path)
            thumbnail.clicked.connect(lambda p=path, idx=i: self.select_image(p, idx))
            self.thumbnail_layout.addWidget(thumbnail)
            self.thumbnails.append(thumbnail)

        if image_paths:
            self.select_image(image_paths[0], 0)

    def select_image(self, image_path, index):
        self.current_image = image_path
        self.original_pixmap = QPixmap(image_path)
        scaled_pixmap = self.original_pixmap.scaled(self.main_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.main_image.setPixmap(scaled_pixmap)

        for i, thumbnail in enumerate(self.thumbnails):
            thumbnail.set_selected(i == index)

    def get_selected_image(self):
        return self.current_image

    def reset_main_image(self):
        if self.current_image:
            self.select_image(self.current_image, self.thumbnails.index(next(t for t in self.thumbnails if t.image_path == self.current_image)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image:
            scaled_pixmap = self.original_pixmap.scaled(self.main_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.main_image.setPixmap(scaled_pixmap)

    def update_live_render(self, image):
        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.main_image.setPixmap(pixmap.scaled(self.main_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

class ThumbnailLabel(QLabel):
    clicked = Signal()

    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.image_path = image_path
        self.setFixedSize(100, 100)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid transparent;
                border-radius: 10px;
                padding: 2px;
            }
            QLabel:hover {
                border: 2px solid #4a9;
                background-color: rgba(74, 153, 153, 0.1);
            }
        """)
        self.load_image()
        self.setToolTip("Click to select this image")

    def load_image(self):
        pixmap = QPixmap(self.image_path).scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        rounded_pixmap = QPixmap(pixmap.size())
        rounded_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(pixmap))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pixmap.rect(), 10, 10)
        painter.end()
        self.setPixmap(rounded_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setGraphicsEffect(QGraphicsOpacityEffect(opacity=0.7))

    def leaveEvent(self, event):
        self.unsetCursor()
        self.setGraphicsEffect(None)

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #4a9;
                    border-radius: 10px;
                    padding: 2px;
                }
                QLabel:hover {
                    background-color: rgba(74, 153, 153, 0.1);
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid transparent;
                    border-radius: 10px;
                    padding: 2px;
                }
                QLabel:hover {
                    border: 2px solid #4a9;
                    background-color: rgba(74, 153, 153, 0.1);
                }
            """)

class MovableSection(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setAcceptDrops(True)
        self.drag_start_position = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.position()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self.drag_start_position is None:
            return
        if (event.position() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.title())
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, event):
        self.drag_start_position = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        parent = self.parent()
        index = parent.layout().indexOf(self)
        dropped_widget = event.source()
        dropped_index = parent.layout().indexOf(dropped_widget)
        if index != dropped_index:
            parent.layout().removeWidget(dropped_widget)
            parent.layout().insertWidget(index, dropped_widget)
            event.acceptProposedAction()

class CustomToolTip(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("""
            background-color: #FFFFD8;
            color: black;
            border: 1px solid #76797C;
            padding: 5px;
            border-radius: 3px;
            font-size: 9pt;
        """)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_tooltip(self, text, pos, duration=5000):
        self.setText(text)
        self.adjustSize()
        self.move(pos)
        self.show()
        self.timer.start(duration)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.setBrush(QColor("#FFFFD8"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 3, 3)
        
        # Draw border
        painter.setPen(QPen(QColor("#76797C"), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)
        
        super().paintEvent(event)

class ToolTipFilter(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tooltip = CustomToolTip()
        self.last_target = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(750)  # 750ms delay
        self.timer.timeout.connect(self._show_tooltip_now)
        self.pending_text = ""
        self.pending_pos = QPoint()

    def _show_tooltip_now(self):
        if self.pending_text:
            self.tooltip.show_tooltip(self.pending_text, self.pending_pos)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.ToolTip:
            helpEvent = event
            text = obj.toolTip()
            if text:
                self.pending_text = text
                self.pending_pos = obj.mapToGlobal(helpEvent.pos()) + QPoint(10, 10)
                self.timer.start()
            return True
        elif event.type() == QEvent.Type.Leave:
            self.timer.stop()
            self.tooltip.hide()
        return super().eventFilter(obj, event)
    
class PromptListItem(QWidget):
    def __init__(self, title, prompt, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #4a9; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.prompt_label = QLabel(prompt)
        self.prompt_label.setStyleSheet("color: #ffffff; background-color: #2e2e2e; padding: 5px; border-radius: 3px;")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.prompt_label)
    
class PromptLibraryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt Library")
        self.setMinimumSize(600, 400)
        self.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, "Prism XL.ico")))
        
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        self.parent = parent
        self.setup_ui()
        self.load_prompts()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Add search functionality
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search prompts...")
        self.search_input.textChanged.connect(self.filter_prompts)
        layout.addWidget(self.search_input)
        
        self.prompt_list = QListWidget()
        self.prompt_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #3e3e3e;
            }
            QListWidget::item:selected {
                background-color: #2e2e2e;
            }
        """)
        layout.addWidget(self.prompt_list)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        self.edit_button = QPushButton("Edit")
        self.cast_button = QPushButton("Cast")
        self.delete_button = QPushButton("Delete")

        button_style = """
            QPushButton {
                background-color: #4a9999;
                color: #ffffff;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5aA9A9;
            }
            QPushButton:pressed {
                background-color: #398989;
            }
        """

        for button in [self.edit_button, self.cast_button, self.delete_button]:
            button.setStyleSheet(button_style)
            button_layout.addWidget(button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.edit_button.clicked.connect(self.edit_prompt)
        self.cast_button.clicked.connect(self.cast_prompt)
        self.delete_button.clicked.connect(self.delete_prompt)

    def load_prompts(self):
        self.prompts = self.parent.load_prompt_library()
        self.update_prompt_list()

    def update_prompt_list(self):
        self.prompt_list.clear()
        for title, prompt in self.prompts.items():
            self.add_prompt_to_list(title, prompt)

    def add_prompt_to_list(self, title, prompt):
        item = QListWidgetItem(self.prompt_list)
        item_widget = PromptListItem(title, prompt)
        item.setSizeHint(item_widget.sizeHint())
        self.prompt_list.addItem(item)
        self.prompt_list.setItemWidget(item, item_widget)

    def filter_prompts(self):
        search_text = self.search_input.text().lower()
        for i in range(self.prompt_list.count()):
            item = self.prompt_list.item(i)
            widget = self.prompt_list.itemWidget(item)
            title = widget.title_label.text().lower()
            prompt = widget.prompt_label.text().lower()
            if search_text in title or search_text in prompt:
                item.setHidden(False)
            else:
                item.setHidden(True)

    def edit_prompt(self):
        current_item = self.prompt_list.currentItem()
        if current_item:
            item_widget = self.prompt_list.itemWidget(current_item)
            old_title = item_widget.title_label.text()
            old_prompt = item_widget.prompt_label.text()
            
            new_title, ok = QInputDialog.getText(self, "Edit Title", "Enter new title:", text=old_title)
            if ok and new_title:
                new_prompt, ok = QInputDialog.getMultiLineText(self, "Edit Prompt", "Enter new prompt:", old_prompt)
                if ok:
                    del self.prompts[old_title]
                    self.prompts[new_title] = new_prompt
                    self.parent.save_prompt_library(self.prompts)
                    self.update_prompt_list()

    def cast_prompt(self):
        current_item = self.prompt_list.currentItem()
        if current_item:
            item_widget = self.prompt_list.itemWidget(current_item)
            prompt = item_widget.prompt_label.text()
            self.parent.text_input.setPlainText(prompt)
            self.close()

    def delete_prompt(self):
        current_item = self.prompt_list.currentItem()
        if current_item:
            item_widget = self.prompt_list.itemWidget(current_item)
            title = item_widget.title_label.text()
            confirm = QMessageBox.question(self, "Confirm Deletion", "Are you sure you want to delete this prompt?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.Yes:
                del self.prompts[title]
                self.parent.save_prompt_library(self.prompts)
                self.update_prompt_list()

    def closeEvent(self, event):
        self.reject()
        event.accept()

class LiabilityDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Terms of Use & Liability Disclaimer")
        self.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, "Prism XL.ico")))
        self.setMinimumSize(600, 450)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #2a2a2a; color: #ffffff; }
            QTextBrowser { background-color: #1e1e1e; border: 1px solid #3e3e3e; }
            QPushButton { background-color: #4a9999; color: #ffffff; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #5aA9A9; }
            QPushButton#disagree_button { background-color: #8b0000; }
            QPushButton#disagree_button:hover { background-color: #a52a2a; }
        """)

        layout = QVBoxLayout(self)

        disclaimer_text = """
        <h2>Terms of Use & Liability Disclaimer</h2>
        <p>Welcome to PrismXL. Before you begin, please read the following terms carefully.</p>
        <p><strong>1. User Responsibility:</strong> You, the user, are solely responsible for the content you create using this application. You agree not to create, share, or disseminate any images that are illegal, harmful, hateful, harassing, defamatory, or that infringe upon the rights of others. This includes, but is not limited to, content that is violent, sexually explicit, promotes discrimination, or is otherwise malicious.</p>
        <p><strong>2. No Guarantees:</strong> The image generation is performed by an AI model. The outputs can be unpredictable. The developers provide no warranty for the outputs and are not responsible for any content that may be generated, including content that may be offensive, inaccurate, or unintended.</p>
        <p><strong>3. Limitation of Liability:</strong> By using this application, you agree to release the developers and any associated parties from any and all liability for any damages, losses, or legal issues arising from the content you create or your use of the software. You agree to indemnify and hold harmless the developers from any claims related to your use of this application.</p>
        <p><strong>By clicking "Agree", you acknowledge that you have read, understood, and agree to be bound by these terms. If you do not agree, you may not use this application.</strong></p>
        """

        text_browser = QTextBrowser()
        text_browser.setHtml(disclaimer_text)
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)

        button_layout = QHBoxLayout()
        agree_button = QPushButton("Agree")
        disagree_button = QPushButton("Disagree")
        disagree_button.setObjectName("disagree_button")

        agree_button.clicked.connect(self.accept)
        disagree_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(disagree_button)
        button_layout.addWidget(agree_button)
        layout.addLayout(button_layout)

class ImageGeneratorUI(QMainWindow):
    VAE_TILING_RESOLUTIONS = [
        (768, 2048), (1920, 1080), (2560, 1440), (3200, 1800),
        (3840, 2160), (4096, 2160), (2560, 1600), (3200, 2000),
        (3840, 2400), (4096, 2304)
    ]

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setGeometry(100, 100, 1280, 900)
        self.setFont(QFont('Arial', 9))
        self.setMinimumSize(1000, 700)
        
        # Setup QSettings
        self.settings = QSettings("Sapphire", "PrismXL")
        
        self.app_data_dir = os.path.join(os.path.expanduser('~'), '.sapphire_prismxl')
        self.user_image_dir = os.path.join(self.app_data_dir, 'images')
        os.makedirs(self.user_image_dir, exist_ok=True)

        self.generation_time = "0.00s"
        self.prompts = {}
    
        # Initialize sections dictionary before it's used
        self.sections = {}

        # Main central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
    
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Custom title bar
        self.title_bar = CustomTitleBar(self)
        self.main_layout.addWidget(self.title_bar)

        # Menu bar - Now added directly to layout after title bar
        self.menubar = QMenuBar()
        self.menubar.setStyleSheet("""
            QMenuBar {
                background-color: transparent;
                border: none;
            }
            QMenuBar::item:selected {
                background-color: #4a9999;
            }
            QMenu {
                background-color: #2a2a2a;
                border: 1px solid #3e3e3e;
            }
            QMenu::item:selected {
                background-color: #4a9999;
            }
        """)
        self.main_layout.addWidget(self.menubar)
    
        # Settings menu
        self.settings_menu = self.menubar.addMenu('Settings')
        self.night_mode_action = QAction('Night Mode', self)
        self.night_mode_action.setCheckable(True)
        self.night_mode_action.triggered.connect(self.toggle_night_mode)
        self.settings_menu.addAction(self.night_mode_action)
    
        # Add Live Render option
        self.live_render_action = QAction('Live Render', self)
        self.live_render_action.setCheckable(True)
        self.live_render_action.triggered.connect(self.toggle_live_render)
        self.settings_menu.addAction(self.live_render_action)

        self.settings_menu.addSeparator()
    
        # Add Close option
        close_action = QAction('Close', self)
        close_action.triggered.connect(self.close)
        self.settings_menu.addAction(close_action)
    
        # Add Window menu
        self.window_menu = self.menubar.addMenu('Window')

        # Content layout (holds left panel and right area)
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Left panel
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(398)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(20, 20, 20, 20)
        self.left_layout.setSpacing(20)

        # Add left panel widgets
        self.setup_left_panel()

        # Now that sections are created, set up window menu
        self.setup_window_menu()
    
        # Add left panel to content layout
        self.content_layout.addWidget(self.left_panel)

        # Right area (for image display)
        self.right_area = QWidget()
        right_layout = QVBoxLayout(self.right_area)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(20)

        # Grid display
        self.grid_display = GridDisplayWidget(self)
        right_layout.addWidget(self.grid_display)

        # Error display
        self.error_display = QTextEdit()
        self.error_display.setReadOnly(True)
        self.error_display.setMaximumHeight(100)
        self.error_display.hide()
        right_layout.addWidget(self.error_display)

        # Add right area to content layout
        self.content_layout.addWidget(self.right_area)

        # Initialize UI state and other components
        self.initialize_ui_state()
        self.load_model()
        self.load_settings()
        self.load_prompt_library()
        self.set_style(self.is_night_mode)
        self.cache_dir = tempfile.mkdtemp()
        self.memory_manager = MemoryManager()
        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(5000)  # Update every 5 seconds               
    
        # Modify menu style
        self.set_menu_style()      
    
        # Prompt section
        self.text_input.setToolTip("Enter your creative prompt here.\nBe descriptive and specific for best results.")
        self.negative_text_input.setToolTip("Enter concepts you want to avoid in the generated image.\nUse this to refine your results.")
        self.generate_button.setToolTip("Start generating the image based on your prompt.\nMake sure you've entered a prompt before clicking.")
        self.cancel_button.setToolTip("Cancel the current image generation process.\nUse this if you want to stop the ongoing generation.")
        self.save_button.setToolTip("Save the generated image.\nRight-click to save all images in a grid.")

        # Advanced Options section
        self.steps_slider.setToolTip("Adjust the number of inference steps.\nMore steps can lead to better quality but take longer. 25-40 is recommended")
        self.num_images_slider.setToolTip("Set the number of images to generate in a grid.\nGenerating multiple images allows for more variety.")
        self.cfg_slider.setToolTip("Adjust the Classifier Free Guidance scale.\nHigher values adhere more closely to the prompt, but may limit creativity. 3-7 works best.")
        self.clip_skip_slider.setToolTip("Set the number of CLIP layers to skip.\nCan affect style and composition. Experiment to find what works best.")
        self.resolution_combo.setToolTip("Select the resolution for the generated image.\nHigher resolutions may require more processing time and memory.")
        self.use_custom_seed.setToolTip("Enable to use a custom seed for reproducible results.\nUseful if you want to recreate a specific image.")
        self.seed_input.setToolTip("Enter a custom seed number for reproducible results.\nUse the same number to generate the same image in the future.")

        # Progress Information section
        self.progress_bar.setToolTip("Shows the progress of the current generation.\nWatch this to estimate how long the process will take.")
        self.status_label.setToolTip("Displays the current status of the application.\nCheck here for real-time updates on the generation process.")
        self.timer_label.setToolTip("Shows the time taken for the current generation.\nUse this to gauge the efficiency of your settings.")
        self.estimation_label.setToolTip("Displays the estimated time remaining for the current generation.\nThis is an approximation and may vary.")

        # System Resources section
        self.ram_label.setToolTip("Shows the current RAM usage.\nMonitor this to ensure your system has enough memory.")
        self.gpu_label.setToolTip("Displays the current GPU memory usage.\nHigh usage may indicate need for lower resolution or fewer steps.")

        # Menu items
        self.night_mode_action.setToolTip("Toggle between light and dark themes.\nChoose the theme that's easiest on your eyes.")
        self.live_render_action.setToolTip("Enable/disable live rendering during image generation.\nLive rendering shows intermediate results but may slow down generation.")

    def set_menu_style(self):
        # Set the palette for the menu bar
        palette = self.menuBar().palette()
        palette.setColor(QPalette.ColorRole.Highlight, QColor(74, 153, 153))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.menuBar().setPalette(palette)

        # Set stylesheet for menu items
        self.menuBar().setStyleSheet("""
            QMenuBar::item:selected { 
                background-color: #4a9999;
                color: white;
            }
            QMenu::item:selected { 
                background-color: #4a9999;
                color: white;
            }
        """)
        
        # Add Library to the top menu bar
        self.library_menu = self.menubar.addMenu('Library')
        self.open_library_action = QAction('Open Prompt Library', self)
        self.open_library_action.triggered.connect(self.open_prompt_library)
        self.library_menu.addAction(self.open_library_action)

        # Add save prompt functionality
        save_icon_path = os.path.join(ASSETS_DIR, "save.png")
        self.save_prompt_icon = QIcon(save_icon_path)
        self.save_prompt_action = QAction(self.save_prompt_icon, "Save to Library", self)
        self.save_prompt_action.triggered.connect(self.save_prompt_to_library)
        self.save_prompt_action.setToolTip("Save selected text to Prompt Library")

        # Create a context menu for the text input
        self.text_input_menu = QMenu(self)
        self.text_input_menu.addAction(self.save_prompt_action)        

    def show_text_input_menu(self, position):
        menu = self.text_input.createStandardContextMenu()
        
        # Add a separator before our custom action
        menu.addSeparator()
        
        # Add our custom "Save to Library" action
        save_action = menu.addAction(self.save_prompt_icon, "Save to Library")
        save_action.triggered.connect(self.save_prompt_to_library)
        
        # Show the menu
        menu.exec(self.text_input.mapToGlobal(position))        
                
    def setup_left_panel(self):
        # Create and set up the left panel FIRST
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(398)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(20, 20, 20, 20)
        self.left_layout.setSpacing(20)

        self.sections = {}

        # Prompt group
        self.sections['prompt'] = MovableSection("Prompt")
        self.sections['prompt'].setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        prompt_layout = QVBoxLayout(self.sections['prompt'])
        prompt_layout.setSpacing(10)
        
        self.text_input = SpellCheckTextEdit(self)
        self.text_input.setPlaceholderText("Enter a creative prompt...")
        self.text_input.setFixedHeight(130)
        self.text_input.setStyleSheet("font-size: 10px;")
        prompt_layout.addWidget(self.text_input)

        prompt_weight_info = QLabel(" ")
        prompt_weight_info.setStyleSheet("font-size: 8px; color: #888;")
        prompt_layout.addWidget(prompt_weight_info)

        negative_prompt_label = QLabel("Negative Prompt:")
        negative_prompt_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        prompt_layout.addWidget(negative_prompt_label)

        self.negative_text_input = SpellCheckTextEdit()
        self.negative_text_input.setPlaceholderText("Enter a negative prompt (optional)...")
        self.negative_text_input.setFixedHeight(80)
        self.negative_text_input.setStyleSheet("font-size: 10px")
        prompt_layout.addWidget(self.negative_text_input)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_style = "font-size: 10px;"
        
        self.generate_button = QPushButton("Generate")
        self.generate_button.setStyleSheet(button_style)
        self.generate_button.clicked.connect(self.generate_image)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(button_style)
        self.cancel_button.clicked.connect(self.cancel_generation)
        
        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet(button_style)
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.save_button.customContextMenuRequested.connect(self.save_button_right_click)

        for button in [self.generate_button, self.cancel_button, self.save_button]:
            button.setFixedHeight(26)
            button_layout.addWidget(button)

        prompt_layout.addLayout(button_layout)

        # Advanced Options group
        self.sections['advanced'] = MovableSection("Advanced Options")
        self.sections['advanced'].setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        advanced_layout = QGridLayout(self.sections['advanced'])
        advanced_layout.setVerticalSpacing(10)

        label_style = "font-size: 10px; font-weight: bold;"
        value_style = "font-size: 10px;"

        # Steps
        steps_label = QLabel("Steps:")
        steps_label.setStyleSheet(label_style)
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setMaximumWidth(205)
        self.steps_slider.setRange(20, 100)
        self.steps_slider.setValue(50)
        self.steps_value = QLabel(str(self.steps_slider.value()))
        self.steps_value.setStyleSheet(value_style)
        self.steps_slider.valueChanged.connect(lambda v: self.steps_value.setText(str(v)))
        advanced_layout.addWidget(steps_label, 0, 0)
        advanced_layout.addWidget(self.steps_slider, 0, 1)
        advanced_layout.addWidget(self.steps_value, 0, 2)

        # Number of Images
        num_images_label = QLabel("Number of Images:")
        num_images_label.setStyleSheet(label_style)
        self.num_images_slider = QSlider(Qt.Orientation.Horizontal)
        self.num_images_slider.setMaximumWidth(205)
        self.num_images_slider.setRange(1, 4)
        self.num_images_slider.setValue(1)
        self.num_images_value = QLabel(str(self.num_images_slider.value()))
        self.num_images_value.setStyleSheet(value_style)
        self.num_images_slider.valueChanged.connect(lambda v: self.num_images_value.setText(str(v)))
        advanced_layout.addWidget(num_images_label, 1, 0)
        advanced_layout.addWidget(self.num_images_slider, 1, 1)
        advanced_layout.addWidget(self.num_images_value, 1, 2)

        # CFG Scale
        cfg_label = QLabel("CFG Scale:")
        cfg_label.setStyleSheet(label_style)
        self.cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self.cfg_slider.setMaximumWidth(205)
        self.cfg_slider.setRange(10, 200)
        self.cfg_slider.setValue(75)
        self.cfg_value = QLabel(str(self.cfg_slider.value() / 10))
        self.cfg_value.setStyleSheet(value_style)
        self.cfg_slider.valueChanged.connect(lambda v: self.cfg_value.setText(str(v / 10)))
        advanced_layout.addWidget(cfg_label, 2, 0)
        advanced_layout.addWidget(self.cfg_slider, 2, 1)
        advanced_layout.addWidget(self.cfg_value, 2, 2)

        # CLIP Skip
        clip_skip_label = QLabel("CLIP Skip:")
        clip_skip_label.setStyleSheet(label_style)
        self.clip_skip_slider = QSlider(Qt.Orientation.Horizontal)
        self.clip_skip_slider.setMaximumWidth(205)
        self.clip_skip_slider.setRange(0, 4)
        self.clip_skip_slider.setValue(0)
        self.clip_skip_value = QLabel(str(self.clip_skip_slider.value()))
        self.clip_skip_value.setStyleSheet(value_style)
        self.clip_skip_slider.valueChanged.connect(lambda v: self.clip_skip_value.setText(str(v)))
        advanced_layout.addWidget(clip_skip_label, 3, 0)
        advanced_layout.addWidget(self.clip_skip_slider, 3, 1)
        advanced_layout.addWidget(self.clip_skip_value, 3, 2)

        # Apply slider style
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4a4a4a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a9;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """
        for slider in [self.steps_slider, self.num_images_slider, self.cfg_slider, self.clip_skip_slider]:
            slider.setStyleSheet(slider_style)

        # Resolution
        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet(label_style)
        self.resolution_combo = QComboBox()
        self.resolution_combo.setStyleSheet("font-size: 12px;")
        self.setup_resolution_selection()
        advanced_layout.addWidget(resolution_label, 4, 0)
        advanced_layout.addWidget(self.resolution_combo, 4, 1, 1, 2)

        # Seed
        self.use_custom_seed = QCheckBox("Custom Seed:")
        self.use_custom_seed.setStyleSheet("font-size: 10px; font-weight: bold;")
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Enter seed number")
        self.seed_input.setEnabled(False)
        self.seed_input.setStyleSheet("font-size: 9px;")
        self.use_custom_seed.stateChanged.connect(self.toggle_seed_input)
        
        advanced_layout.addWidget(self.use_custom_seed, 5, 0)
        advanced_layout.addWidget(self.seed_input, 5, 1, 1, 2)

        # Progress Information group
        self.sections['progress'] = MovableSection("Progress Information")
        self.sections['progress'].setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        progress_layout = QVBoxLayout(self.sections['progress'])
        progress_layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #999999;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a9;
                width: 10px;
                margin: 0.5px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 9px; font-weight: bold; color: #888;")
        progress_layout.addWidget(self.status_label)

        time_layout = QHBoxLayout()
        self.timer_label = QLabel("Generate Time: 0.00s")
        self.timer_label.setStyleSheet("font-size: 10px;")
        self.estimation_label = QLabel("Estimated Time: --")
        self.estimation_label.setStyleSheet("font-size: 10px;")
        time_layout.addWidget(self.timer_label)
        time_layout.addWidget(self.estimation_label)
        progress_layout.addLayout(time_layout)

        # System Resources group
        self.sections['memory'] = MovableSection("System Resources")
        self.sections['memory'].setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; }")
        memory_layout = QVBoxLayout(self.sections['memory'])
        memory_layout.setSpacing(10)

        resource_style = "font-size: 10px;"
        self.ram_label = QLabel("RAM Usage: N/A")
        self.ram_label.setStyleSheet(resource_style)
        self.gpu_label = QLabel("GPU Memory: N/A")
        self.gpu_label.setStyleSheet(resource_style)

        memory_layout.addWidget(self.ram_label)
        memory_layout.addWidget(self.gpu_label)

        # Add all sections to the left layout with appropriate stretch factors
        self.left_layout.addWidget(self.sections['prompt'], 40)
        self.left_layout.addWidget(self.sections['advanced'], 30)
        self.left_layout.addWidget(self.sections['progress'], 15)
        self.left_layout.addWidget(self.sections['memory'], 15)

        self.left_layout.addStretch(1)
        
    def wheelEvent(self, event):
        if self.seed_input.isEnabled():
            delta = event.angleDelta().y()
            current_seed = int(self.seed_input.text()) if self.seed_input.text() else 0
            new_seed = current_seed + (1 if delta > 0 else -1)
            self.seed_input.setText(str(new_seed))
        elif hasattr(self, 'current_seed'):
            self.use_custom_seed.setChecked(True)
            self.seed_input.setText(str(self.current_seed))

    def setup_window_menu(self):
        for section_name, section in self.sections.items():
            action = QAction(section.title(), self)
            action.setCheckable(True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, s=section_name: self.toggle_section(s, checked))
            self.window_menu.addAction(action)

    def toggle_section(self, section_name, visible):
        self.sections[section_name].setVisible(visible)
        self.save_settings()

    def toggle_seed_input(self, state):
        is_checked = (state == Qt.CheckState.Checked.value)
        self.seed_input.setEnabled(is_checked)

    def setup_resolution_selection(self):
        resolutions = [
            "512 x 512",
            "768 x 768",
            "896 x 896",
            "1024 x 512",
            "512 x 1024",
            "1024 x 1024",
            "1024 x 768",
            "768 x 1024",
            "1152 x 896",
            "896 x 1152",            
            "1280 x 720",
            "720 x 1280",
            "1216 x 832",
            "832 x 1216",
            "1344 x 768",
            "768 x 1344",
            "1536 x 640",
            "640 x 1536",            
            "2048 x 768",
            "768 x 2048",
            "1024 x 2048",
            "2048 x 2048",
            "1920 x 1080",
            "2560 x 1440",
            "3200 x 1800",
            "3840 x 2160",
            "4096 x 2160",
            "2560 x 1600",
            "3200 x 2000",
            "3840 x 2400",
            "4096 x 2304"
        ]
        self.resolution_combo.addItems(resolutions)
        
    def validate_resolution(self, width, height):
        if width % 8 != 0 or height % 8 != 0:
            QMessageBox.warning(self, "Invalid Resolution", "Width and height must be multiples of 8. Adjusting to nearest valid resolution.")
            width = round(width / 8) * 8
            height = round(height / 8) * 8
        return width, height

    def initialize_ui_state(self):
        self.cancel_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.timer_label.setText("Generate Time: 0.00s")
        self.estimation_label.setText("Estimated Time: --")

    def load_model(self):
        try:
            self.model = DiffusionPipeline.from_pretrained(
                "RunDiffusion/Juggernaut-XL-v9",
                trust_remote_code=True,
                use_safetensors=True,
                variant="fp16",
                torch_dtype=torch.float16
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            logger.info("Juggernaut-XL-v9 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Juggernaut-XL-v9 model: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load the model: {str(e)}")
            sys.exit(1)

    def load_settings(self):
        self.is_night_mode = self.settings.value("night_mode", False, type=bool)
        self.night_mode_action.setChecked(self.is_night_mode)
        
        self.steps_slider.setValue(self.settings.value("inference_steps", 50, type=int))
        self.num_images_slider.setValue(self.settings.value("num_images", 1, type=int))
        
        self.is_live_render_enabled = self.settings.value("live_render", False, type=bool)
        self.live_render_action.setChecked(self.is_live_render_enabled)

        # Load section visibility
        self.settings.beginGroup("SectionVisibility")
        for action in self.window_menu.actions():
            section_name = action.text()
            is_visible = self.settings.value(section_name, True, type=bool)
            action.setChecked(is_visible)
            if section_name in self.sections:
                self.sections[section_name].setVisible(is_visible)
        self.settings.endGroup()
        
        logger.info("Settings loaded successfully")


    def set_style(self, is_night_mode):
        if is_night_mode:
            style = """
                QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
                QLineEdit, QTextEdit { background-color: #2e2e2e; color: #ffffff; border: 1px solid #3e3e3e; }
                QComboBox { 
                    background-color: #2e2e2e; 
                    color: #ffffff; 
                    border: 1px solid #3e3e3e;
                    padding: 1px 18px 1px 3px;
                    min-width: 6em;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 15px;
                    border-left-width: 1px;
                    border-left-color: #3e3e3e;
                    border-left-style: solid;
                }
                QPushButton { background-color: #3e3e3e; color: #ffffff; border: none; padding: 8px 16px; border-radius: 4px; }
                QPushButton:hover { background-color: #4e4e4e; }
                QPushButton:pressed { background-color: #2e2e2e; }
                QPushButton:disabled { background-color: #2a2a2a; color: #5a5a5a; }
                QGroupBox { border: 1px solid #3e3e3e; margin-top: 0.5em; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
                QDialog { background-color: #1e1e1e; color: #ffffff; }
                QTextBrowser { background-color: #2e2e2e; color: #ffffff; border: 1px solid #3e3e3e; }
                QCheckBox::indicator:checked { background-color: #4a9999; }
                QCheckBox::indicator:unchecked { background-color: #2e2e2e; }
            """
        else:
            style = """
                QMainWindow, QWidget { background-color: #2a2a2a; color: #ffffff; }
                QLineEdit, QTextEdit { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; }
                QComboBox { 
                    background-color: #3a3a3a; 
                    color: #ffffff; 
                    border: 1px solid #4a4a4a;
                    padding: 1px 18px 1px 3px;
                    min-width: 6em;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 15px;
                    border-left-width: 1px;
                    border-left-color: #4a4a4a;
                    border-left-style: solid;
                }
                QPushButton { background-color: #4a9999; color: #ffffff; border: none; padding: 8px 16px; border-radius: 4px; }
                QPushButton:hover { background-color: #5aA9A9; }
                QPushButton:pressed { background-color: #398989; }
                QPushButton:disabled { background-color: #3a3a3a; color: #5a5a5a; }
                QGroupBox { border: 1px solid #4a4a4a; margin-top: 0.5em; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
                QDialog { background-color: #2a2a2a; color: #ffffff; }
                QTextBrowser { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; }
                QCheckBox::indicator:checked { background-color: #4a9999; }
                QCheckBox::indicator:unchecked { background-color: #3a3a3a; }
            """
        
        self.setStyleSheet(style)
        self.title_bar.set_style(is_night_mode)
        QApplication.instance().setStyleSheet(style)

        progress_bar_style = """
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a9;
                width: 10px;
                margin: 0.5px;
                border-radius: 2px;
            }
        """
        self.progress_bar.setStyleSheet(progress_bar_style)

        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #4a4a4a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a9;
                border: 1px solid #5c5c5c;
                width: 15px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """
        self.steps_slider.setStyleSheet(slider_style)
        self.num_images_slider.setStyleSheet(slider_style)
        self.cfg_slider.setStyleSheet(slider_style)
        self.clip_skip_slider.setStyleSheet(slider_style)

        self.update()

    def should_use_vae_tiling(self, width, height):
        return (width, height) in self.VAE_TILING_RESOLUTIONS

    def generate_image(self):
        prompt = self.text_input.toPlainText().strip()
        negative_prompt = self.negative_text_input.toPlainText().strip()

        if not prompt:
            self.show_error("Prompt cannot be empty.")
            return

        num_inference_steps = self.steps_slider.value()
        num_images = self.num_images_slider.value()
        guidance_scale = self.cfg_slider.value() / 10.0
        clip_skip = self.clip_skip_slider.value()

        resolution_str = self.resolution_combo.currentText()        
        resolution_parts = resolution_str.replace('(default)', '').strip().split('x')
        width, height = map(int, map(str.strip, resolution_parts))

        # Validate and adjust resolution if necessary
        width, height = self.validate_resolution(width, height)
        
        # Get seed value
        seed = None
        if self.use_custom_seed.isChecked():
            try:
                seed = int(self.seed_input.text())
            except ValueError:
                self.show_error("Invalid seed. Please enter a valid integer.")
                return

        # Enable or disable VAE tiling based on resolution
        if self.should_use_vae_tiling(width, height):
            self.model.enable_vae_tiling()
            logger.info(f"VAE tiling enabled for resolution {width}x{height}")
        else:
            self.model.disable_vae_tiling()
            logger.info(f"VAE tiling disabled for resolution {width}x{height}")

        self.progress_bar.setValue(0)
        self.timer_label.show()
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.error_display.hide()

        if num_images > 1:
            self.thread = GridGeneratorThread(self.model, prompt, num_inference_steps, width, height, num_images, self.cache_dir, negative_prompt, seed, guidance_scale, clip_skip)
            self.thread.finished.connect(self.display_grid)
        else:
            self.thread = ImageGeneratorThread(self.model, prompt, num_inference_steps, width, height, self.cache_dir, negative_prompt, seed, guidance_scale, clip_skip, live_render=self.is_live_render_enabled and width == 512 and height == 512)
            self.thread.finished.connect(self.display_final_image)
            if self.is_live_render_enabled and width == 512 and height == 512:
                self.thread.intermediate_result.connect(self.grid_display.update_live_render)

        self.thread.progress.connect(self.update_progress)
        self.thread.status.connect(self.update_status)
        self.thread.error.connect(self.show_error)
        self.thread.start()

        self.start_time = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(10)

        logger.info(f"Starting image generation. Prompt: '{prompt}', Negative Prompt: '{negative_prompt}', Steps: {num_inference_steps}, Num Images: {num_images}, Resolution: {width}x{height}, VAE Tiling: {'Enabled' if self.should_use_vae_tiling(width, height) else 'Disabled'}, CLIP Skip: {clip_skip}")

    def update_progress(self, current_image, value):
        total_progress = (current_image * 100 + value) / self.num_images_slider.value()
        self.progress_bar.setValue(int(total_progress))

    def update_status(self, status):
        self.status_label.setText(status)

    def update_timer(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f}s"
        else:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            time_str = f"{minutes}m {seconds:.2f}s"
        self.timer_label.setText(f"Generate Time: {time_str}")
        
        # Update estimation
        if self.progress_bar.value() > 0:
            estimated_total = elapsed_time / (self.progress_bar.value() / 100)
            remaining = estimated_total - elapsed_time
            if remaining < 60:
                est_str = f"{remaining:.2f}s"
            else:
                est_minutes = int(remaining // 60)
                est_seconds = remaining % 60
                est_str = f"{est_minutes}m {est_seconds:.2f}s"
            self.estimation_label.setText(f"Estimated Time: {est_str}")

    def display_final_image(self, image_path, generation_time):
        if not self.thread.is_cancelled:
            self.grid_display.set_images([image_path])
            self.auto_save_image(image_path)
        self.finish_generation(generation_time)

    def display_grid(self, image_paths, generation_time):
        if not self.thread.is_cancelled:
            self.grid_display.set_images(image_paths)
            for path in image_paths:
                self.auto_save_image(path)
        self.finish_generation(generation_time)

    def finish_generation(self, generation_time):
        self.progress_bar.setValue(100)        
        self.timer.stop()
        if generation_time >= 60:
            minutes = int(generation_time // 60)
            seconds = generation_time % 60
            time_str = f"{minutes}m {seconds:.2f}s"
        else:
            time_str = f"{generation_time:.2f}s"
        self.generation_time = time_str
        self.timer_label.setText(f"Generate Time: {time_str}")
        self.estimation_label.setText("Estimated Time: --")
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.save_button.setEnabled(True)
        if hasattr(self, 'thread') and isinstance(self.thread, GridGeneratorThread):
            self.status_label.setText("Generation cancelled" if self.thread.is_cancelled else "Generation complete")
        else:
            self.status_label.setText("Generation complete")
        logger.info(f"Image generation completed. Time taken: {time_str}")
        self.memory_manager.clear_cache()
        

    def cancel_generation(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling...")
            self.thread.wait()  # Wait for the thread to finish
            self.finish_generation(time.time() - self.start_time)

    def save_image(self):
        image_to_save = self.grid_display.get_selected_image()
        if image_to_save:
            prompt = self.text_input.toPlainText().strip()
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
            safe_prompt = safe_prompt[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"{safe_prompt}_{timestamp}.png"
            
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(self.user_image_dir, default_name), "Images (*.png *.jpg *.bmp)")
            if file_path:
                try:
                    pixmap = QPixmap(image_to_save)
                    pixmap.save(file_path)
                    
                    self.save_image_info(file_path)
                    
                    logger.info(f"Image saved successfully to {file_path}")
                    QMessageBox.information(self, "Success", f"Image saved successfully to {file_path}")
                    
                    self.grid_display.reset_main_image()
                    
                except Exception as e:
                    error_msg = f"Failed to save image: {str(e)}"
                    logger.error(error_msg)
                    self.show_error(error_msg)

    def save_button_right_click(self, position):
        if self.grid_display.thumbnails:
            self.save_all_images()

    def save_all_images(self):
        if not self.grid_display.thumbnails:
            return

        prompt = self.text_input.toPlainText().strip()
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
        safe_prompt = safe_prompt[:50]

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Images")
        if save_dir:
            for i, thumbnail in enumerate(self.grid_display.thumbnails):
                file_name = f"{safe_prompt}_{i+1}.png"
                file_path = os.path.join(save_dir, file_name)
                image_path = thumbnail.image_path
                try:
                    pixmap = QPixmap(image_path)
                    pixmap.save(file_path)
                    logger.info(f"Image saved successfully to {file_path}")
                except Exception as e:
                    error_msg = f"Failed to save image {i+1}: {str(e)}"
                    logger.error(error_msg)
                    self.show_error(error_msg)

            QMessageBox.information(self, "Success", f"All images saved successfully to {save_dir}")

    def show_error(self, message):
        self.progress_bar.setValue(0)
        self.timer_label.hide()
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.error_display.show()
        
        error_msg = f"Error: {message}\n\n"
        if "memory" in message.lower():
            error_msg += ("This error is likely due to insufficient GPU memory. "
                          "Try the following:\n"
                          "1. Close other applications to free up GPU memory\n"
                          "2. Reduce the number of inference steps\n"
                          "3. If the issue persists, try restarting your computer")
        elif "cuda" in message.lower():
            error_msg += ("This error is related to CUDA. Try the following:\n"
                          "1. Update your GPU drivers\n"
                          "2. Ensure your GPU supports CUDA\n"
                          "3. Reinstall PyTorch with CUDA support")
        elif "model" in message.lower():
            error_msg += ("This error is related to the AI model. Try the following:\n"
                          "1. Check your internet connection\n"
                          "2. Ensure you have the latest version of the diffusers library\n"
                          "3. Try reinstalling the model")
        else:
            error_msg += "An unexpected error occurred. Please check the log file for more details."

        self.error_display.setText(error_msg)
        self.status_label.setText("Error occurred")
        
        if not hasattr(self, 'clear_error_button'):
            self.clear_error_button = QPushButton("Clear Error")
            self.clear_error_button.clicked.connect(self.clear_error)
            self.left_layout.addWidget(self.clear_error_button)
        self.clear_error_button.show()
        
        logger.error(f"Error displayed to user: {message}")

    def clear_error(self):
        self.error_display.hide()
        self.clear_error_button.hide()
        self.status_label.setText("Ready")

    def close_application(self):
        self.save_settings()
        logger.info("Application closed via menu option")
        self.close()

    def toggle_night_mode(self, checked):
        self.is_night_mode = checked
        self.set_style(self.is_night_mode)
        self.save_settings()
        logger.info(f"Night mode toggled: {'On' if checked else 'Off'}")

    def toggle_live_render(self, checked):
        self.is_live_render_enabled = checked
        self.save_settings()
        logger.info(f"Live Render toggled: {'On' if checked else 'Off'}")

    def save_settings(self):
        self.settings.setValue("night_mode", self.is_night_mode)
        self.settings.setValue("inference_steps", self.steps_slider.value())
        self.settings.setValue("num_images", self.num_images_slider.value())
        self.settings.setValue("live_render", self.is_live_render_enabled)

        # Save section visibility
        self.settings.beginGroup("SectionVisibility")
        for section in self.sections.values():
            self.settings.setValue(section.title(), section.isVisible())
        self.settings.endGroup()
        
        logger.info("Settings saved successfully")

    def update_memory_usage(self):
        # Update RAM usage
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024 ** 3)  # Convert to GB
        ram_total = ram.total / (1024 ** 3)
        ram_percent = ram.percent
        self.ram_label.setText(f"RAM Usage: {ram_used:.2f}GB / {ram_total:.2f}GB ({ram_percent}%)")

        # Update GPU usage
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming we're using the first GPU
                gpu_used = gpu.memoryUsed
                gpu_total = gpu.memoryTotal
                gpu_percent = (gpu_used / gpu_total) * 100
                self.gpu_label.setText(f"GPU Memory: {gpu_used:.2f}MB / {gpu_total:.2f}MB ({gpu_percent:.2f}%)")
            else:
                self.gpu_label.setText("GPU Memory: Not Available")
        except Exception as e:
            self.gpu_label.setText(f"GPU Memory: Error ({str(e)})")

    def closeEvent(self, event):
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting cached file {file_path}: {str(e)}")
        os.rmdir(self.cache_dir)
        
        self.save_settings()
        logger.info("Application closed")
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.grid_display.resize(self.grid_display.size())

    def load_prompt_library(self):
        filename = os.path.join(self.app_data_dir, 'prompt_library.json')
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    self.prompts = json.load(f)
            except json.JSONDecodeError:
                self.prompts = {}
                logger.error(f"Error decoding prompt library file: {filename}")
        else:
            self.prompts = {}
        return self.prompts

    def save_prompt_library(self, prompts):
        self.prompts = prompts
        filename = os.path.join(self.app_data_dir, 'prompt_library.json')
        with open(filename, 'w') as f:
            json.dump(prompts, f)
            
    def save_prompt_to_library(self):
        prompt = self.text_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Empty Prompt", "Cannot save an empty prompt.")
            return

        title, ok = QInputDialog.getText(self, "Save Prompt", "Enter a title for this prompt:")
        if ok and title:
            self.prompts[title] = prompt
            self.save_prompt_library(self.prompts)
            QMessageBox.information(self, "Prompt Saved", f"Prompt '{title}' saved to your library.")

    def open_prompt_library(self):
        dialog = PromptLibraryDialog(parent=self)
        dialog.exec()
        
    def auto_save_image(self, temp_image_path):
        prompt = self.text_input.toPlainText().strip()
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
        safe_prompt = safe_prompt[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{safe_prompt}_{timestamp}.png"
        new_path = os.path.join(self.user_image_dir, new_filename)
        
        shutil.copy2(temp_image_path, new_path)
        self.save_image_info(new_path)
        
        logger.info(f"Image automatically saved to {new_path}")

        
    def save_image_info(self, image_path):
        info = {
            'prompt': self.text_input.toPlainText(),
            'negative_prompt': self.negative_text_input.toPlainText(),
            'steps': self.steps_slider.value(),
            'cfg_scale': self.cfg_slider.value() / 10.0,
            'clip_skip': self.clip_skip_slider.value(),
            'resolution': self.resolution_combo.currentText(),
            'seed': int(self.seed_input.text()) if self.use_custom_seed.isChecked() else None,
            'gen_time': self.timer_label.text().split(': ')[1],
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(image_path + '.json', 'w') as f:
            json.dump(info, f, indent=4)        

def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("Sapphire")
    app.setApplicationName("PrismXL")
    
    settings = QSettings()
    
    if not settings.value("terms_agreed", False, type=bool):
        dialog = LiabilityDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings.setValue("terms_agreed", True)
        else:
            logger.info("User did not agree to the terms. Application will exit.")
            sys.exit(0)

    logger.info("Application starting...")
    
    main_window = ImageGeneratorUI()
    logger.info("Main window created")
    main_window.show()
    logger.info("Main window shown")
    
    return app.exec()

if __name__ == '__main__':
    try:
        exit_code = main()
        logger.info(f"Application exiting with code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.critical(f"Unhandled exception: {error_type}: {error_msg}", exc_info=True)
        
        # Log system info
        import psutil
        logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
        logger.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_max_memory = torch.cuda.max_memory_allocated()
                if gpu_max_memory > 0:
                    gpu_usage_percent = (gpu_memory_allocated / gpu_max_memory) * 100
                    logger.info(f"GPU Memory Usage: {gpu_usage_percent:.2f}%")
                else:
                    logger.info("GPU Memory Usage: 0% (No memory allocated)")
            except Exception as gpu_error:
                logger.error(f"Error getting GPU info: {str(gpu_error)}")
        else:
            logger.info("CUDA is not available")

        # Show error to user
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setText(f"A critical error has occurred: {error_type}")
        error_box.setInformativeText(f"Error details: {error_msg}")
        error_box.setWindowTitle("Critical Error")
        error_box.setDetailedText(f"Please check the log file for more information.\n\nLog file location: {os.path.abspath('image_generator.log')}")
        error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        error_box.exec()
        
        sys.exit(1)