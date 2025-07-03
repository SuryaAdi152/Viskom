import numpy as np
import cv2
import time
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import subprocess
import logging
import pygame

# Setup logging
logging.basicConfig(filename='parking_assist.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize pygame mixer untuk audio
pygame.mixer.init()

# Constants in centimeters
Known_distance = 200.0
Known_width = 180.0
thres = 0.5
nms_threshold = 0.2
default_object_width = 50.0
default_focal_length = 1000.0
DETECTION_INTERVAL = 7  
CAMERA_WIDTH = 320      
CAMERA_HEIGHT = 240
PROCESSING_WIDTH = 320  
PROCESSING_HEIGHT = 240 
UI_WIDTH = 480          
UI_HEIGHT = 360         
ALERT_DISTANCE = 100.0
BEEP_COOLDOWN = 2.2
AUDIO_DIR = os.path.join('audio')

object_widths = {
    'car': 180.0,
    'truck': 250.0,
    'bus': 300.0,
    'person': 50.0,
    'bicycle': 60.0,
    'motorcycle': 80.0,
    'traffic light': 30.0,
    'stop sign': 75.0
}

# Fixed object audio files mapping
object_audio_files = {
    'car': 'car.mp3',
    'truck': 'truck.mp3',
    'bus': 'bus.mp3',
    'person': 'person.mp3',
    'bicycle': 'bicycle.mp3',
    'motorcycle': 'motorcycle.mp3',
    'traffic light': 'traffic light.mp3',
    'stop sign': 'stop sign.mp3'
}

# Relevant classes for parking
relevant_classes = ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle', 'traffic light', 'stop sign']

# Colors in BGR format
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Font
font = cv2.FONT_HERSHEY_PLAIN

def get_camera_name_from_index(index):
    """Get camera name using multiple methods"""
    camera_name = f"Camera {index}"
    
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            backend_name = cap.getBackendName()
            if backend_name and backend_name != "DSHOW":
                camera_name = backend_name
            cap.release()
    except:
        pass
    
    try:
        wmi_queries = [
            'wmic path Win32_PnPEntity where "Caption like \'%camera%\' or Caption like \'%webcam%\' or Caption like \'%video%\' or Caption like \'%droid%\' or Caption like \'%cam%\' or Service like \'usbvideo%\'" get Caption',
            'wmic path Win32_VideoController get Caption',
            'wmic path Win32_USBHub where "Caption like \'%camera%\' or Caption like \'%video%\' or Caption like \'%droid%\'" get Caption'
        ]
        
        all_devices = []
        for query in wmi_queries:
            try:
                result = subprocess.check_output(query, shell=True).decode('utf-8', errors='ignore').splitlines()
                devices = [line.strip() for line in result if line.strip() and not line.strip().startswith("Caption")]
                all_devices.extend(devices)
            except:
                continue
        
        unique_devices = list(set(all_devices))
        cleaned_devices = []
        
        for device in unique_devices:
            clean_name = device
            replacements = [
                ('USB Video Device', ''),
                ('USB2.0', 'USB'),
                ('USB 2.0', 'USB'),
                ('(', ''),
                (')', ''),
                ('  ', ' ')
            ]
            for old, new in replacements:
                clean_name = clean_name.replace(old, new)
            clean_name = clean_name.strip()
            
            if clean_name and len(clean_name) > 3:
                cleaned_devices.append(clean_name)
        
        if index < len(cleaned_devices):
            camera_name = cleaned_devices[index]
            
    except Exception as e:
        logging.debug(f"Error in WMI query: {e}")
    
    if "droid" in camera_name.lower():
        camera_name = "DroidCam"
    
    return camera_name

def load_audio_files():
    audio_files = {}
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith('.mp3'):
            name = filename.replace('.mp3', '').replace(' ', '_').lower()
            audio_files[name] = os.path.join(AUDIO_DIR, filename)
    return audio_files

object_audio_files = load_audio_files()
warning_file = os.path.join(AUDIO_DIR, 'awas_terlalu_dekat.mp3')

def detect_cameras(max_index=4):
    available_cameras = [("None", -1, "ref_car.png")]
    
    for i in range(max_index):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                name = get_camera_name_from_index(i)
                
                if name == f"Camera {i}" and backend == cv2.CAP_DSHOW:
                    try:
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        if width > 0 and height > 0:
                            name = f"Camera {i} ({int(width)}x{int(height)})"
                    except:
                        pass
                
                available_cameras.append((name, i, "ref_car.png"))
                cap.release()
                logging.info(f"Detected camera: {name} at index {i}")
                break
            cap.release()

    if len(available_cameras) == 1:
        logging.warning("No cameras detected. Please check your camera connections.")
    
    return available_cameras

def preprocess_frame(frame):
    """Preprocess frame for detection: resize and optional noise reduction."""
    processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
    return processed_frame

def play_alert_sound(detected_objects):
    try:
        # Play general warning first
        if os.path.exists(warning_file):
            pygame.mixer.music.load(warning_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        
        # Play sound for the highest priority object detected
        priority_objects = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic_light', 'stop_sign']
        for obj in priority_objects:
            if obj in detected_objects:
                obj_file = object_audio_files.get(obj)
                if obj_file and os.path.exists(obj_file):
                    pygame.mixer.music.load(obj_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    break
                
    except Exception as e:
        logging.error(f"Error playing alert sound: {e}")
        try:
            import winsound
            winsound.Beep(1000, 200)
        except:
            pass

class ParkingAssistUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Assist System")
        self.root.geometry("1200x900")
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = "#1e1e1e"
        fg_color = "#ffffff"
        accent_color = "#4CAF50"
        danger_color = "#f44336"
        button_bg = "#2196F3"
        
        self.root.configure(bg=bg_color)
        
        # Configure styles
        style.configure("Title.TLabel", font=("Arial", 24, "bold"), background=bg_color, foreground=fg_color)
        style.configure("Heading.TLabel", font=("Arial", 12, "bold"), background=bg_color, foreground=fg_color)
        style.configure("Info.TLabel", font=("Arial", 10), background=bg_color, foreground=fg_color)
        style.configure("Camera.TFrame", background="#2d2d2d", relief="raised", borderwidth=2)
        style.configure("Control.TFrame", background=bg_color)
        style.configure("Action.TButton", font=("Arial", 10, "bold"))
        
        # Main container
        main_container = tk.Frame(root, bg=bg_color)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(main_container, bg=bg_color)
        title_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="🚗 Parking Assist System", style="Title.TLabel")
        title_label.pack()
        
        # Control panel
        control_frame = ttk.Frame(main_container, style="Control.TFrame")
        control_frame.pack(fill="x", pady=(0, 20))
        
        # Camera selection frame
        camera_frame = tk.Frame(control_frame, bg=bg_color)
        camera_frame.pack(side="left", fill="x", expand=True)
        
        # Create horizontal camera selection
        for i in range(1, 5):
            cam_container = tk.Frame(camera_frame, bg=bg_color)
            cam_container.pack(side="left", padx=10)
            
            label = ttk.Label(cam_container, text=f"Camera {i}", style="Heading.TLabel")
            label.pack()
            
            # Camera dropdown
            setattr(self, f'cam{i}_options', detect_cameras())
            setattr(self, f'cam{i}_var', tk.StringVar(value=getattr(self, f'cam{i}_options')[1][0] if len(getattr(self, f'cam{i}_options')) > 1 and i == 1 else "None"))
            
            cam_menu = ttk.Combobox(cam_container, textvariable=getattr(self, f'cam{i}_var'), 
                                   values=[opt[0] for opt in getattr(self, f'cam{i}_options')],
                                   state="readonly", width=20)
            cam_menu.pack()
            setattr(self, f'cam{i}_menu', cam_menu)
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg=bg_color)
        button_frame.pack(side="right", padx=(20, 0))
        
        # Start button
        self.start_btn = tk.Button(button_frame, text="▶ START", command=self.start,
                                  bg=accent_color, fg="white", font=("Arial", 12, "bold"),
                                  padx=20, pady=10, bd=0, cursor="hand2")
        self.start_btn.pack(side="left", padx=5)
        
        # Stop button
        self.stop_btn = tk.Button(button_frame, text="⏹ STOP", command=self.stop,
                                 bg=danger_color, fg="white", font=("Arial", 12, "bold"),
                                 padx=20, pady=10, bd=0, cursor="hand2", state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        # Refresh button
        self.refresh_btn = tk.Button(button_frame, text="🔄 REFRESH", command=self.refresh_cameras,
                                    bg=button_bg, fg="white", font=("Arial", 12, "bold"),
                                    padx=20, pady=10, bd=0, cursor="hand2")
        self.refresh_btn.pack(side="left", padx=5)
        
        # Status bar
        self.status_frame = tk.Frame(main_container, bg="#333333", height=30)
        self.status_frame.pack(fill="x", pady=(0, 10))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="⚪ Ready", bg="#333333", fg="white",
                                    font=("Arial", 10), anchor="w", padx=10)
        self.status_label.pack(fill="both", expand=True)
        
        # Camera display grid
        display_frame = tk.Frame(main_container, bg=bg_color)
        display_frame.pack(fill="both", expand=True)
        
        # Configure grid weights
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=1)
        
        # Camera frames with better styling
        for i in range(1, 5):
            row = (i-1) // 2
            col = (i-1) % 2
            
            cam_frame = tk.Frame(display_frame, bg="#2d2d2d", relief="raised", bd=2)
            cam_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Camera title bar
            title_bar = tk.Frame(cam_frame, bg="#444444", height=30)
            title_bar.pack(fill="x")
            title_bar.pack_propagate(False)
            
            cam_title = tk.Label(title_bar, text=f"📹 Camera {i} - Detection Feed",
                               bg="#444444", fg="white", font=("Arial", 10, "bold"),
                               anchor="w", padx=10)
            cam_title.pack(fill="both", expand=True)
            
            # Camera display
            cam_label = tk.Label(cam_frame, bg="black", text=f"Camera {i}\nNo Signal",
                               fg="gray", font=("Arial", 12))
            cam_label.pack(fill="both", expand=True, padx=2, pady=2)
            
            setattr(self, f'cam{i}_frame', cam_frame)
            setattr(self, f'cam{i}_label', cam_label)
            setattr(self, f'cam{i}_title', cam_title)
        
        # Initialize variables
        self.running = False
        self.caps = {}
        self.frame_queue = queue.Queue(maxsize=8)
        self.last_detections = {}
        self.frame_count = {f'Cam {i}': 0 for i in range(1, 5)}
        self.camera_status = {f'Cam {i}': True for i in range(1, 5)}
        self.focal_lengths = {}
        self.last_frames = {}
        self.frame_times = {f'Cam {i}': [] for i in range(1, 5)}
        self.last_log_times = {f'Cam {i}': 0 for i in range(1, 5)}
        self.last_beep_time = 0
        
        # Bind hover effects
        self.add_button_effects()
    
    def add_button_effects(self):
        """Add hover effects to buttons"""
        def on_enter(e, btn, color):
            btn.config(bg=self.lighten_color(color))
        
        def on_leave(e, btn, color):
            btn.config(bg=color)
        
        # Start button
        self.start_btn.bind("<Enter>", lambda e: on_enter(e, self.start_btn, "#4CAF50"))
        self.start_btn.bind("<Leave>", lambda e: on_leave(e, self.start_btn, "#4CAF50"))
        
        # Stop button
        self.stop_btn.bind("<Enter>", lambda e: on_enter(e, self.stop_btn, "#f44336"))
        self.stop_btn.bind("<Leave>", lambda e: on_leave(e, self.stop_btn, "#f44336"))
        
        # Refresh button
        self.refresh_btn.bind("<Enter>", lambda e: on_enter(e, self.refresh_btn, "#2196F3"))
        self.refresh_btn.bind("<Leave>", lambda e: on_leave(e, self.refresh_btn, "#2196F3"))
    
    def lighten_color(self, color):
        """Make color lighter for hover effect"""
        color_map = {
            "#4CAF50": "#66BB6A",
            "#f44336": "#ef5350",
            "#2196F3": "#42A5F5"
        }
        return color_map.get(color, color)
    
    def update_status(self, message, color="white"):
        """Update status bar"""
        status_colors = {
            "green": "🟢",
            "red": "🔴",
            "yellow": "🟡",
            "white": "⚪"
        }
        prefix = status_colors.get(color, "⚪")
        self.status_label.config(text=f"{prefix} {message}")
    
    def refresh_cameras(self):
        """Refresh camera list"""
        self.update_status("Refreshing cameras...", "yellow")
        
        # Re-detect cameras
        for i in range(1, 5):
            options = detect_cameras()
            setattr(self, f'cam{i}_options', options)
            
            # Update combobox
            cam_menu = getattr(self, f'cam{i}_menu')
            cam_menu['values'] = [opt[0] for opt in options]
            
            # Keep current selection if still available
            current = getattr(self, f'cam{i}_var').get()
            if current not in [opt[0] for opt in options]:
                getattr(self, f'cam{i}_var').set("None")
        
        self.update_status("Camera list refreshed", "green")
        self.root.after(2000, lambda: self.update_status("Ready", "white"))
    
    def stop(self):
        """Stop the system"""
        self.running = False
        self.update_status("Stopping system...", "yellow")
        
        # Wait for threads to finish
        if hasattr(self, 'thread_capture'):
            self.thread_capture.join(timeout=2)
        if hasattr(self, 'thread_process'):
            self.thread_process.join(timeout=2)
        
        # Release cameras
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        
        # Update UI
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.refresh_btn.config(state="normal")
        
        # Clear displays
        for i in range(1, 5):
            cam_label = getattr(self, f'cam{i}_label')
            cam_label.config(image='', text=f"Camera {i}\nNo Signal")
            cam_title = getattr(self, f'cam{i}_title')
            cam_title.config(text=f"📹 Camera {i} - Detection Feed")
        
        self.update_status("System stopped", "red")

    def start(self):
        cam1_data = next((opt for opt in self.cam1_options if opt[0] == self.cam1_var.get()), None)
        cam2_data = next((opt for opt in self.cam2_options if opt[0] == self.cam2_var.get()), None)
        cam3_data = next((opt for opt in self.cam3_options if opt[0] == self.cam3_var.get()), None)
        cam4_data = next((opt for opt in self.cam4_options if opt[0] == self.cam4_var.get()), None)
        cameras = []
        if cam1_data and cam1_data[1] != -1:
            cameras.append({'index': cam1_data[1], 'name': 'Cam 1', 'ref_image': 'ref_car.png'})
        if cam2_data and cam2_data[1] != -1:
            cameras.append({'index': cam2_data[1], 'name': 'Cam 2', 'ref_image': 'ref_car.png'})
        if cam3_data and cam3_data[1] != -1:
            cameras.append({'index': cam3_data[1], 'name': 'Cam 3', 'ref_image': 'ref_car.png'})
        if cam4_data and cam4_data[1] != -1:
            cameras.append({'index': cam4_data[1], 'name': 'Cam 4', 'ref_image': 'ref_car.png'})
        if not cameras:
            messagebox.showerror("Error", "Please select at least one camera")
            return
        
        self.running = True
        self.update_status("Starting system...", "yellow")
        
        # Update button states
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.refresh_btn.config(state="disabled")
        
        # Start threads
        self.thread_capture = threading.Thread(target=self.capture_frames, args=(cameras,))
        self.thread_process = threading.Thread(target=self.run_parking_assist, args=(cameras,))
        self.thread_capture.daemon = True
        self.thread_process.daemon = True
        self.thread_capture.start()
        self.thread_process.start()
        
        self.update_status("System running", "green")

    def capture_frames(self, cameras):
        for cam in cameras:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                self.caps[cam['name']] = cv2.VideoCapture(cam['index'], backend)
                cap = self.caps[cam['name']]
                if cap.isOpened():
                    break
                cap.release()
            if not cap.isOpened():
                logging.error(f"Could not open camera {cam['name']} (index {cam['index']}).")
                self.camera_status[cam['name']] = False
                self.running = False
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logging.info(f"{cam['name']}: Set resolution {actual_width}x{actual_height}")

        while self.running:
            for cam in cameras:
                if not self.camera_status[cam['name']]:
                    continue
                cap = self.caps[cam['name']]
                start_time = time.time()
                ret, frame = cap.read()
                if ret:
                    # Ensure frame is in color (3 channels)
                    if len(frame.shape) == 2:  # If grayscale
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[2] == 4:  # If RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                    try:
                        self.frame_queue.put_nowait((cam['name'], frame))
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait((cam['name'], frame))
                        except queue.Empty:
                            pass
                else:
                    logging.warning(f"Failed to grab frame from {cam['name']}.")
                    self.camera_status[cam['name']] = False
                elapsed = time.time() - start_time
                logging.debug(f"{cam['name']} Frame capture time: {elapsed:.3f} s")

    def run_parking_assist(self, cameras):
        classNames = []
        try:
            with open('coco.names', 'r') as f:
                classNames = f.read().splitlines()
        except FileNotFoundError:
            logging.error("coco.names file not found.")
            messagebox.showerror("Error", "coco.names file not found. Please ensure it exists.")
            self.running = False
            return
        Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

        weightsPath = "frozen_inference_graph.pb"
        configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        try:
            net = cv2.dnn_DetectionModel(weightsPath, configPath)
            net.setInputSize(PROCESSING_WIDTH, PROCESSING_HEIGHT)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)
        except cv2.error as e:
            logging.error(f"Could not load detection model: {e}")
            messagebox.showerror("Error", f"Could not load detection model: {e}")
            self.running = False
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_folder = 'history'
        os.makedirs(history_folder, exist_ok=True)
        log_filename = os.path.join(history_folder, f'parking_log_{timestamp}.txt')
        log_file = open(log_filename, 'w')
        log_buffer = []
        log_file.write(f"--- Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        def FocalLength(measured_distance, real_width, width_in_rf_image):
            if width_in_rf_image <= 0:
                return 0
            focal_length = (width_in_rf_image * measured_distance) / real_width
            logging.info(f"Calculated Focal Length: {focal_length:.2f} pixels")
            return focal_length

        def Distance_finder(focal_length, real_width, width_in_frame):
            if width_in_frame <= 0 or focal_length <= 0:
                return 0
            distance = (real_width * focal_length) / width_in_frame
            logging.info(f"Distance Calc: real_width={real_width:.2f} cm, focal_length={focal_length:.2f} px, width_in_frame={width_in_frame:.2f} px, distance={distance:.2f} cm")
            return distance

        self.focal_lengths = {}
        for cam in cameras:
            self.focal_lengths[cam['name']] = default_focal_length
            ref_image_path = 'ref_car.png'
            if not os.path.exists(ref_image_path):
                warning_msg = f"Error: Reference image {ref_image_path} not found for {cam['name']}. Using default focal length {default_focal_length}."
                logging.error(warning_msg)
                log_buffer.append(warning_msg)
                ref_image = None
            else:
                ref_image = cv2.imread(ref_image_path)
                if ref_image is None:
                    warning_msg = f"Error: Could not read reference image {ref_image_path} for {cam['name']}. Using default focal length {default_focal_length}."
                    logging.error(warning_msg)
                    log_buffer.append(warning_msg)

            if ref_image is not None:
                ref_image = preprocess_frame(ref_image)
                try:
                    classIds, confs, bbox = net.detect(ref_image, confThreshold=thres)
                    ref_image_width = 0
                    if len(classIds) > 0:
                        for i, classId in enumerate(classIds):
                            if classNames[classId - 1] == 'car':
                                ref_image_width = bbox[i][2]
                                break
                    if ref_image_width > 0:
                        self.focal_lengths[cam['name']] = FocalLength(Known_distance, Known_width, ref_image_width)
                        log_entry = f"{cam['name']} Focal Length: {self.focal_lengths[cam['name']]:.2f} pixels"
                        logging.info(log_entry)
                        log_buffer.append(log_entry)
                    else:
                        warning_msg = f"Warning: No car detected in {ref_image_path} for {cam['name']}. Using default focal length {default_focal_length}."
                        logging.warning(warning_msg)
                        log_buffer.append(warning_msg)
                except Exception as e:
                    warning_msg = f"Error: Failed to process reference image for {cam['name']}: {e}. Using default focal length {default_focal_length}."
                    logging.error(warning_msg)
                    log_buffer.append(warning_msg)

        last_print_time = int(time.time())
        distances_in_second = {cam['name']: {} for cam in cameras}
        # Create blank frame with 3 channels (color)
        blank_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        last_log_time = time.time()
        camera_detections = {cam['name']: [] for cam in cameras}
        last_alert_objects = []  # Track last alerted objects locally

        while self.running:
            all_frames = {}
            updated = False
            too_close_detected = False
            close_objects = []
            
            try:
                while not self.frame_queue.empty():
                    cam_name, frame = self.frame_queue.get(timeout=0.01)
                    all_frames[cam_name] = frame
                    updated = True
            except queue.Empty:
                pass

            for cam in cameras:
                cam_name = cam['name']
                cam_num = int(cam_name.split()[1])
                current_time = time.time()
                if cam_name in all_frames:
                    self.frame_times[cam_name].append(current_time)
                    if len(self.frame_times[cam_name]) > 100:
                        self.frame_times[cam_name].pop(0)
                    if len(self.frame_times[cam_name]) > 1:
                        fps = len(self.frame_times[cam_name]) / (self.frame_times[cam_name][-1] - self.frame_times[cam_name][0])
                        if current_time - last_log_time > 5:
                            logging.info(f"{cam_name} FPS: {fps:.2f}")

                if not self.camera_status[cam_name]:
                    frame = blank_frame.copy()
                    cv2.putText(frame, f"Camera {cam_name} Offline", (10, CAMERA_HEIGHT//2), font, 1, RED, 1)
                    camera_detections[cam_name] = ["None, 0"]
                    # Update camera title to show offline
                    cam_title = getattr(self, f'cam{cam_num}_title')
                    cam_title.config(text=f"📹 Camera {cam_num} - OFFLINE")
                else:
                    frame = all_frames.get(cam_name, self.last_frames.get(cam_name, blank_frame.copy()))
                    # Ensure frame is in color
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    self.last_frames[cam_name] = frame.copy()
                    
                    # Update camera title to show active
                    cam_title = getattr(self, f'cam{cam_num}_title')
                    cam_title.config(text=f"📹 Camera {cam_num} - ACTIVE")

                self.frame_count[cam_name] += 1
                if self.frame_count[cam_name] % DETECTION_INTERVAL == 0 and self.camera_status[cam_name]:
                    processed_frame = preprocess_frame(frame)
                    try:
                        detect_start = time.time()
                        classIds, confs, bbox = net.detect(processed_frame, confThreshold=thres)
                        detect_end = time.time()
                        logging.info(f"{cam_name} Detection time: {detect_end - detect_start:.3f} s")
                        bbox = list(bbox)
                        confs = list(np.array(confs).reshape(1, -1)[0])
                        confs = list(map(float, confs))
                        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
                        self.last_detections[cam_name] = (classIds, confs, bbox, indices)
                    except Exception as e:
                        logging.error(f"Error during detection for {cam_name}: {e}")
                        camera_detections[cam_name] = ["None, 0"]
                        continue
                else:
                    classIds, confs, bbox, indices = self.last_detections.get(cam_name, ([], [], [], []))

                relevant_objects = []
                class_counters = {cls: 0 for cls in relevant_classes}
                camera_detections[cam_name] = []
                if len(classIds) != 0:
                    for i in indices:
                        box = bbox[i]
                        class_id = classIds[i] - 1
                        class_name = classNames[class_id]
                        if class_name not in relevant_classes:
                            continue
                        confidence = round(confs[i], 2)
                        color = Colors[class_id]
                        x, y, w, h = box[0], box[1], box[2], box[3]

                        class_counters[class_name] += 1
                        label = f"{class_name.capitalize()} {class_counters[class_name]}"
                        real_width = object_widths.get(class_name, default_object_width)
                        distance_to_object = Distance_finder(self.focal_lengths[cam_name], real_width, w)
                        distance_to_object = round(distance_to_object, 2)

                        camera_detections[cam_name].append(f"{label}, {distance_to_object}")

                        if distance_to_object > 0:
                            if class_name not in distances_in_second[cam_name]:
                                distances_in_second[cam_name][class_name] = []
                            distances_in_second[cam_name][class_name].append(distance_to_object)

                        relevant_objects.append(i)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                        cv2.putText(frame, f"{label} ({confidence})", (x + 10, y + 20), font, 1, color, 2)
                        cv2.putText(frame, f"Distance: {distance_to_object} cm", (x + 10, y + 40), font, 1, color, 2)
                        
                        if distance_to_object > 0 and distance_to_object < ALERT_DISTANCE:
                            cv2.putText(frame, "TOO CLOSE!", (x + 10, y + 60), font, 1.5, RED, 3)
                            too_close_detected = True
                            close_objects.append(class_name)

                if not camera_detections[cam_name]:
                    camera_detections[cam_name] = ["None, 0"]

                cv2.putText(frame, f"{len(relevant_objects)} Object", (50, 50), font, 1, RED, 2)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (UI_WIDTH, UI_HEIGHT))
                start_time = time.time()
                try:
                    from PIL import Image, ImageTk
                    # Convert to PIL Image and then to PhotoImage
                    img = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    cam_label = getattr(self, f'cam{cam_num}_label')
                    cam_label.configure(image=photo, text='')
                    cam_label.image = photo
                except ImportError:
                    # Fallback jika PIL tidak tersedia
                    photo = tk.PhotoImage(data=cv2.imencode('.png', frame_rgb)[1].tobytes())
                    cam_label = getattr(self, f'cam{cam_num}_label')
                    cam_label.configure(image=photo, text='')
                    cam_label.image = photo
                except tk.TclError:
                    logging.info("UI closed, stopping...")
                    self.running = False
                    break
                elapsed = time.time() - start_time
                logging.debug(f"{cam_name} UI update time: {elapsed:.3f} s")

            # Play alert sound with detected objects
            if too_close_detected and (current_time - self.last_beep_time > BEEP_COOLDOWN):
                # Check if objects have changed or cooldown has passed
                objects_changed = set(close_objects) != set(last_alert_objects)
                if objects_changed or (current_time - self.last_beep_time > BEEP_COOLDOWN * 2):
                    threading.Thread(target=play_alert_sound, args=(close_objects,), daemon=True).start()
                    self.last_beep_time = current_time
                    last_alert_objects = close_objects.copy()
                    logging.info(f"Alert sound played - Objects too close: {', '.join(close_objects)}")
                    self.update_status(f"⚠️ WARNING: {', '.join(close_objects)} too close!", "red")
            elif not too_close_detected:
                # Reset last alert objects when nothing is too close
                last_alert_objects = []
                self.update_status("System running - All clear", "green")

            current_time = time.time()
            if current_time - last_log_time >= 1:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}]"
                for cam in cameras:
                    cam_name = cam['name']
                    detection_str = camera_detections[cam_name][0] if camera_detections[cam_name] else "None, 0"
                    log_entry += f" {cam_name} = {detection_str} |"
                log_entry = log_entry.rstrip('|')
                logging.info(log_entry)
                log_buffer.append(log_entry)
                last_log_time = current_time

            if current_time > last_print_time:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                any_detections = False
                for cam in cameras:
                    cam_name = cam['name']
                    if distances_in_second[cam_name]:
                        any_detections = True
                        for class_name, distances in distances_in_second[cam_name].items():
                            avg_distance = round(sum(distances) / len(distances), 2)
                            class_counts = len(distances)
                            for i in range(1, class_counts + 1):
                                log_entry = f"{timestamp} | {cam_name} | {class_name.capitalize()} {i}, Distance: {avg_distance} cm"
                                logging.info(log_entry)
                                log_buffer.append(log_entry)
                if not any_detections:
                    log_entry = f"{timestamp} | No objects detected"
                    logging.info(log_entry)
                    log_buffer.append(log_entry)

                if len(log_buffer) >= 10:
                    log_file.write("\n".join(log_buffer) + "\n")
                    log_buffer.clear()
                distances_in_second = {cam['name']: {} for cam in cameras}
                last_print_time = current_time

            if updated:
                try:
                    self.root.update()
                except tk.TclError:
                    logging.info("UI closed, stopping...")
                    self.running = False
                    break

        log_file.write("\n".join(log_buffer) + "\n")
        log_file.write(f"--- Log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log_file.close()
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()
        self.running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingAssistUI(root)
    
    # Handle window close
    def on_closing():
        if app.running:
            app.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.running = False
        root.destroy()