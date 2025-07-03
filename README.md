# 🚗 Real-Time Object Detection & Distance Estimation for Parking Assist

Proyek ini merupakan implementasi sistem deteksi objek secara real-time yang dilengkapi dengan estimasi jarak objek menggunakan model `SSD MobileNet V3` dari TensorFlow. Sistem ini sangat cocok digunakan dalam aplikasi seperti **Parking Assist**, pengawasan lalu lintas, dan sistem keamanan berbasis kamera.

---

## 📌 Fitur Utama

- 🔍 Deteksi objek real-time menggunakan webcam
- 📏 Estimasi jarak objek dari kamera dalam satuan sentimeter
- 📁 Logging otomatis hasil deteksi setiap detik ke folder `history`
- ⚠️ Peringatan visual untuk objek yang terlalu dekat
- 🎯 Mendukung berbagai kelas objek penting seperti:
  - Car
  - Truck
  - Bus
  - Person
  - Bicycle
  - Motorcycle
  - Traffic Light
  - Stop Sign

---


## 📁 Struktur Folder
Real-Time-Object-Detection/
├── history/ # Folder log hasil deteksi (dibuat otomatis)
├── .gitignore # File konfigurasi Git
├── coco.names # Daftar nama kelas COCO
├── frozen_inference_graph.pb # File model pra-latih
├── LICENSE # Lisensi proyek
├── objectdetector.py # Script utama pendeteksian
├── ref_car.png # Gambar referensi mobil (jarak 200cm)
├── requirements.txt # Daftar dependensi Python
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt # Konfigurasi model SSD

---

## ⚙️ Instalasi dan Penggunaan

### 1. Clone Repository

```bash
git clone <URL-REPO>
cd Real-Time-Object-Detection
