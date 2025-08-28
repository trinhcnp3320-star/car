import pandas as pd
import cv2
import re
import requests
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import easyocr
import os

# ======================================================================
# PHẦN 1: CẤU HÌNH VÀ BIẾN TOÀN CỤC
# ======================================================================

# Đường dẫn đến thư mục chứa tất cả kết quả (tự động tạo nếu chưa có)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn đến thư mục chứa ảnh biển số đã cắt
CROPS_DIR = RESULTS_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn đến thư mục chứa ảnh biển số đã tiền xử lý
PREPROC_DIR = RESULTS_DIR / "preproc"
PREPROC_DIR.mkdir(parents=True, exist_ok=True)

# Tải mô hình YOLOv8 pre-trained để phát hiện đối tượng
try:
    yolo_model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO: {e}")
    yolo_model = None

# Tải mô hình EasyOCR (hỗ trợ tiếng Anh và tiếng Việt)
try:
    ocr_reader = easyocr.Reader(['en', 'vi'], gpu=False) # Đặt gpu=True nếu bạn có GPU
except Exception as e:
    print(f"Lỗi khi tải mô hình EasyOCR: {e}")
    ocr_reader = None

# ======================================================================
# PHẦN 2: CÁC HÀM XỬ LÝ CHÍNH
# ======================================================================

def detect_plates(image_path: str):
    """Phát hiện biển số trong ảnh bằng YOLO và cắt các biển số ra."""
    if not yolo_model:
        print("Lỗi: Mô hình YOLO không khả dụng.")
        return []

    try:
        results = yolo_model(image_path, classes=[2, 3, 5, 7]) # Lọc các class liên quan đến xe
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                img = cv2.imread(image_path)
                if img is None:
                    continue

                crop = img[y1:y2, x1:x2]
                
                # Lưu ảnh đã cắt
                filename = f"{Path(image_path).stem}_{x1}_{y1}.png"
                crop_path = CROPS_DIR / filename
                cv2.imwrite(str(crop_path), crop)
                
                detections.append({
                    "image_path": image_path,
                    "crop_path": str(crop_path),
                    "bbox": f"{x1},{y1},{x2},{y2}",
                    "confidence_yolo": conf,
                    "class_yolo": yolo_model.names[cls]
                })
        return detections
    except Exception as e:
        print(f"Lỗi khi phát hiện biển số: {e}")
        return []

def preprocess_plate(img):
    """Hàm tiền xử lý ảnh biển số: resize, tăng độ tương phản và nhị phân hóa."""
    try:
        # Resize để chuẩn hóa đầu vào
        img = cv2.resize(img, (200, 50))
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Tăng độ tương phản
        enhanced = cv2.equalizeHist(gray)
        # Nhị phân hóa (chuyển sang đen trắng)
        _, bin_img = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return enhanced, bin_img
    except Exception as e:
        print(f"Lỗi tiền xử lý ảnh: {e}")
        return None, None

def recognize_text(image):
    """Nhận dạng văn bản từ ảnh bằng EasyOCR."""
    if not ocr_reader:
        print("Lỗi: Mô hình EasyOCR không khả dụng.")
        return "", 0.0
    
    try:
        results = ocr_reader.readtext(image)
        if results:
            # Lấy kết quả với độ tin cậy cao nhất
            best_result = max(results, key=lambda x: x[2])
            text, prob = best_result[1], best_result[2]
            return text, prob
        return "", 0.0
    except Exception as e:
        print(f"Lỗi khi nhận dạng văn bản: {e}")
        return "", 0.0

def normalize_plate(raw_text: str) -> str:
    """Chuẩn hóa định dạng biển số xe Việt Nam."""
    if not raw_text:
        return ""
    
    CONFUSION_MAP = {'O': '0', 'o': '0', 'I': '1', 'l': '1', '|': '1', 'S': '5', 'B': '8', 'Z': '2', 'D': '0'}
    
    # Làm sạch ký tự và chuyển chữ hoa
    text = "".join(CONFUSION_MAP.get(ch, ch) for ch in raw_text.upper())
    text = re.sub(r"[^A-Z0-9]", "", text)
    
    # Chèn dấu gạch ngang sau mã tỉnh (2 hoặc 3 số)
    m = re.match(r"^(\d{2,3})([A-Z])(\d+)$", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}{m.group(3)}"
    return text

def send_to_backend(data: dict):
    """Gửi dữ liệu kết quả đến một API backend."""
    backend_url = "http://your-backend-api-url/api/process" # Thay thế bằng URL của bạn
    try:
        response = requests.post(backend_url, json=data)
        if response.status_code == 200:
            print("✔️ Gửi dữ liệu thành công đến backend.")
        else:
            print(f"❌ Gửi dữ liệu thất bại. Mã trạng thái: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi kết nối đến backend: {e}")

# ======================================================================
# PHẦN 3: LƯU TRỮ VÀ XUẤT KẾT QUẢ
# ======================================================================

def run_pipeline(image_list: list):
    """Chạy toàn bộ quy trình từ phát hiện đến chuẩn hóa và gửi dữ liệu."""
    all_results = []
    
    # Bước 1: Phát hiện biển số
    print("🚗 Bước 1: Đang phát hiện biển số...")
    detection_rows = []
    for img_path in tqdm(image_list, desc="Phát hiện biển số"):
        detection_rows.extend(detect_plates(img_path))
    
    if not detection_rows:
        print("Không tìm thấy biển số nào.")
        return
        
    df_det = pd.DataFrame(detection_rows)
    df_det.to_csv(RESULTS_DIR / "detections.csv", index=False)
    print(f"✅ Đã phát hiện và lưu {len(detection_rows)} biển số.")
    
    # Bước 2: Tiền xử lý ảnh biển số
    print("\n🖼️ Bước 2: Đang tiền xử lý ảnh...")
    preproc_rows = []
    for row in tqdm(df_det.to_dict(orient="records"), desc="Tiền xử lý ảnh"):
        crop_path = Path(row["crop_path"])
        img = cv2.imread(str(crop_path))
        if img is None:
            continue
        
        enhanced, bin_img = preprocess_plate(img)
        if enhanced is None or bin_img is None:
            continue
            
        out_enh = PREPROC_DIR / f"{crop_path.stem}_enh.png"
        out_bin = PREPROC_DIR / f"{crop_path.stem}_bin.png"
        cv2.imwrite(str(out_enh), enhanced)
        cv2.imwrite(str(out_bin), bin_img)
        
        preproc_rows.append({
            **row,
            "enhanced_path": str(out_enh),
            "binary_path": str(out_bin),
        })
    df_preproc = pd.DataFrame(preproc_rows)
    df_preproc.to_csv(RESULTS_DIR / "preproc.csv", index=False)
    print(f"✅ Đã tiền xử lý {len(preproc_rows)} ảnh.")

    # Bước 3: Nhận dạng văn bản OCR
    print("\n📝 Bước 3: Đang nhận dạng ký tự (OCR)...")
    ocr_rows = []
    for row in tqdm(df_preproc.to_dict(orient="records"), desc="Nhận dạng OCR"):
        bin_img_path = row["binary_path"]
        img = cv2.imread(bin_img_path)
        if img is None:
            continue
            
        text_raw, conf_ocr = recognize_text(img)
        
        ocr_rows.append({
            **row,
            "text_raw": text_raw,
            "conf_ocr": conf_ocr,
        })
    df_ocr = pd.DataFrame(ocr_rows)
    df_ocr.to_csv(RESULTS_DIR / "ocr.csv", index=False)
    print(f"✅ Đã nhận dạng {len(ocr_rows)} biển số.")

    # Bước 4: Chuẩn hóa dữ liệu
    print("\n🧭 Bước 4: Đang chuẩn hóa dữ liệu...")
    df_ocr["text_norm"] = df_ocr["text_raw"].apply(normalize_plate)
    df_ocr.to_csv(RESULTS_DIR / "normalized.csv", index=False)
    print(f"✅ Đã chuẩn hóa {len(df_ocr)} kết quả.")
    
    # Bước 5: Gửi kết quả cho backend
    print("\n⬆️ Bước 5: Đang gửi dữ liệu đến backend...")
    for index, row in tqdm(df_ocr.iterrows(), total=len(df_ocr), desc="Gửi dữ liệu"):
        # Tạo một dictionary với các dữ liệu cần thiết để gửi
        data_to_send = {
            "license_plate": row["text_norm"],
            "raw_text": row["text_raw"],
            "confidence": row["conf_ocr"],
            "image_path": row["image_path"],
            "bbox": row["bbox"],
        }
        send_to_backend(data_to_send)
    print("✅ Hoàn tất.")

# ======================================================================
# CHẠY THỬ NGHIỆM
# ======================================================================

if __name__ == '__main__':
    # THAY THẾ DANH SÁCH NÀY BẰNG CÁC ĐƯỜNG DẪN ĐẾN ẢNH CỦA BẠN
    image_list = ["path/to/your/image1.jpg", "path/to/your/image2.png"]
    
    # Tạo một thư mục chứa ảnh demo nếu chưa có
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # Thêm một đường dẫn ảnh giả lập để chạy thử
    dummy_image_path = demo_dir / "car_demo.jpg"
    if not dummy_image_path.exists():
        print("Tạo một file ảnh giả lập để chạy thử...")
        # Bạn cần phải có ảnh để thay thế vào đây. 
        # Tải một ảnh demo về và đặt tên là car_demo.jpg trong thư mục demo_images
        # Dùng google_search để tìm ảnh.
    
    image_list = [str(dummy_image_path)]
    
    run_pipeline(image_list)
