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

# Setup logging
logging.basicConfig(filename='parking_assist.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants in centimeters
Known_distance = 200.0  # Reference distance of car in cm
Known_width = 180.0     # Average car width in cm
thres = 0.5             # Detection threshold
nms_threshold = 0.2     # Non-max suppression threshold
default_object_width = 50.0  # Default width for other objects in cm
default_focal_length = 1000.0  # Fallback focal length if calibration fails
DETECTION_INTERVAL = 7  # Detect every 7 frames for lower CPU load
CAMERA_WIDTH = 320      # Camera resolution
CAMERA_HEIGHT = 240
PROCESSING_WIDTH = 320  # Width for detection processing
PROCESSING_HEIGHT = 240 # Height for detection processing
UI_WIDTH = 320          # Reduced UI resolution
UI_HEIGHT = 240

# Dictionary of average widths for relevant object classes (in cm)
object_widths = {
    'car': 180.0,      # Average car
    'truck': 250.0,    # Average truck
    'bus': 300.0,      # Average bus
    'person': 50.0,    # Average person (shoulder width)
    'bicycle': 60.0,   # Average bicycle
    'motorcycle': 80.0,# Average motorcycle
    'traffic light': 30.0,  # Traffic light
    'stop sign': 75.0  # Stop sign
}

# Relevant classes for parking
relevant_classes = ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle', 'traffic light', 'stop sign']

# Colors in BGR format
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Font
font = cv2.FONT_HERSHEY_PLAIN

def detect_cameras(max_index=4):
    available_cameras = [("None", -1, "ref_car.png")]
    camera_names = {}
    
    try:
        result = subprocess.check_output(
            'wmic path Win32_PnPEntity where "Caption like \'%camera%\' or Caption like \'%webcam%\'" get Caption',
            shell=True
        ).decode('utf-8', errors='ignore').splitlines()
        devices = [line.strip() for line in result if line.strip() and not line.strip().startswith("Caption")]
        for i in range(len(devices)):
            if i < max_index:
                camera_names[i] = devices[i]
    except Exception as e:
        logging.error(f"Error retrieving camera names with wmic: {e}")

    for i in range(max_index):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                name = camera_names.get(i, f"Camera {i}")
                name = name.replace("(Integrated Camera)", "").replace("Camera", "").strip()
                if not name:
                    name = f"Camera {i}"
                available_cameras.append((name, i, "ref_car.png"))
                cap.release()
                break
            cap.release()

    if len(available_cameras) == 1:
        logging.warning("No cameras detected. Please check your camera connections.")
    
    return available_cameras

def preprocess_frame(frame):
    """Preprocess frame for detection: resize and optional noise reduction."""
    processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
    return processed_frame

class ParkingAssistUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Assist")
        self.root.geometry("640x480")  # Adjusted for 2x2 grid of 320x240

        self.top_frame = tk.Frame(root)
        self.top_frame.pack(fill="x", pady=5)

        tk.Label(self.top_frame, text="Pilih Kamera 1").pack(side="left", padx=5)
        self.cam1_options = detect_cameras()
        self.cam1_var = tk.StringVar(value=self.cam1_options[1][0] if len(self.cam1_options) > 1 else "None")
        self.cam1_menu = ttk.OptionMenu(self.top_frame, self.cam1_var, self.cam1_var.get(), *[opt[0] for opt in self.cam1_options])
        self.cam1_menu.pack(side="left", padx=5)

        tk.Label(self.top_frame, text="Pilih Kamera 2").pack(side="left", padx=5)
        self.cam2_options = detect_cameras()
        self.cam2_var = tk.StringVar(value="None")
        self.cam2_menu = ttk.OptionMenu(self.top_frame, self.cam2_var, self.cam2_var.get(), *[opt[0] for opt in self.cam2_options])
        self.cam2_menu.pack(side="left", padx=5)

        tk.Label(self.top_frame, text="Pilih Kamera 3").pack(side="left", padx=5)
        self.cam3_options = detect_cameras()
        self.cam3_var = tk.StringVar(value="None")
        self.cam3_menu = ttk.OptionMenu(self.top_frame, self.cam3_var, self.cam3_var.get(), *[opt[0] for opt in self.cam3_options])
        self.cam3_menu.pack(side="left", padx=5)

        tk.Label(self.top_frame, text="Pilih Kamera 4").pack(side="left", padx=5)
        self.cam4_options = detect_cameras()
        self.cam4_var = tk.StringVar(value="None")
        self.cam4_menu = ttk.OptionMenu(self.top_frame, self.cam4_var, self.cam4_var.get(), *[opt[0] for opt in self.cam4_options])
        self.cam4_menu.pack(side="left", padx=5)

        tk.Button(self.top_frame, text="Start", command=self.start).pack(side="right", padx=5)

        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(fill="both", expand=True)

        self.cam1_frame = tk.Frame(self.bottom_frame)
        self.cam1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        tk.Label(self.cam1_frame, text="Output Realtime Detection Camera 1").pack()
        self.cam1_label = tk.Label(self.cam1_frame)
        self.cam1_label.pack(fill="both", expand=True)

        self.cam2_frame = tk.Frame(self.bottom_frame)
        self.cam2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        tk.Label(self.cam2_frame, text="Output Realtime Detection Camera 2").pack()
        self.cam2_label = tk.Label(self.cam2_frame)
        self.cam2_label.pack(fill="both", expand=True)

        self.cam3_frame = tk.Frame(self.bottom_frame)
        self.cam3_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        tk.Label(self.cam3_frame, text="Output Realtime Detection Camera 3").pack()
        self.cam3_label = tk.Label(self.cam3_frame)
        self.cam3_label.pack(fill="both", expand=True)

        self.cam4_frame = tk.Frame(self.bottom_frame)
        self.cam4_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        tk.Label(self.cam4_frame, text="Output Realtime Detection Camera 4").pack()
        self.cam4_label = tk.Label(self.cam4_frame)
        self.cam4_label.pack(fill="both", expand=True)

        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_rowconfigure(1, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        self.running = False
        self.caps = {}
        self.frame_queue = queue.Queue(maxsize=8)  # Slightly larger queue
        self.last_detections = {}
        self.frame_count = {cam['name']: 0 for cam in [{'name': 'Cam 1'}, {'name': 'Cam 2'}, {'name': 'Cam 3'}, {'name': 'Cam 4'}]}
        self.camera_status = {cam['name']: True for cam in [{'name': 'Cam 1'}, {'name': 'Cam 2'}, {'name': 'Cam 3'}, {'name': 'Cam 4'}]}
        self.focal_lengths = {}
        self.last_frames = {}
        self.frame_times = {cam['name']: [] for cam in [{'name': 'Cam 1'}, {'name': 'Cam 2'}, {'name': 'Cam 3'}, {'name': 'Cam 4'}]}

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
        self.thread_capture = threading.Thread(target=self.capture_frames, args=(cameras,))
        self.thread_process = threading.Thread(target=self.run_parking_assist, args=(cameras,))
        self.thread_capture.daemon = True
        self.thread_process.daemon = True
        self.thread_capture.start()
        self.thread_process.start()

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
        blank_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        last_log_time = time.time()

        while self.running:
            all_frames = {}
            updated = False
            try:
                while not self.frame_queue.empty():
                    cam_name, frame = self.frame_queue.get(timeout=0.01)
                    all_frames[cam_name] = frame
                    updated = True
            except queue.Empty:
                pass

            for cam in cameras:
                cam_name = cam['name']
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
                else:
                    frame = all_frames.get(cam_name, self.last_frames.get(cam_name, blank_frame.copy()))
                    self.last_frames[cam_name] = frame.copy()

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
                        continue
                else:
                    classIds, confs, bbox, indices = self.last_detections.get(cam_name, ([], [], [], []))

                relevant_objects = []
                class_counters = {cls: 0 for cls in relevant_classes}
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

                        if distance_to_object > 0:
                            if class_name not in distances_in_second[cam_name]:
                                distances_in_second[cam_name][class_name] = []
                            distances_in_second[cam_name][class_name].append(distance_to_object)

                        relevant_objects.append(i)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
                        cv2.putText(frame, f"{label} ({confidence})", (x + 10, y + 20), font, 1, color, 2)
                        cv2.putText(frame, f"Distance: {distance_to_object} cm", (x + 10, y + 40), font, 1, color, 2)
                        if distance_to_object > 0 and distance_to_object < 50:
                            cv2.putText(frame, "Too Close!", (x + 10, y + 60), font, 1, RED, 2)

                cv2.putText(frame, f"{len(relevant_objects)} Object", (50, 50), font, 1, RED, 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (UI_WIDTH, UI_HEIGHT))
                start_time = time.time()
                try:
                    photo = tk.PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
                    if cam_name == 'Cam 1':
                        self.cam1_label.configure(image=photo)
                        self.cam1_label.image = photo
                    elif cam_name == 'Cam 2':
                        self.cam2_label.configure(image=photo)
                        self.cam2_label.image = photo
                    elif cam_name == 'Cam 3':
                        self.cam3_label.configure(image=photo)
                        self.cam3_label.image = photo
                    elif cam_name == 'Cam 4':
                        self.cam4_label.configure(image=photo)
                        self.cam4_label.image = photo
                except tk.TclError:
                    logging.info("UI closed, stopping...")
                    self.running = False
                    break
                elapsed = time.time() - start_time
                logging.debug(f"{cam_name} UI update time: {elapsed:.3f} s")

            current_time = time.time()
            if current_time - last_log_time > 5:
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
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.running = False
        root.destroy()