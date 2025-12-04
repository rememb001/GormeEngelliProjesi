import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n_ncnn_model')
cap = cv2.VideoCapture('rtsp://192.168.1.186:8080/h264_ulaw.sdp')

print("🚗👤 İNSAN & ARABA SAYACI - 2 FPS")
print("⏹️  'q' ile çık")

last_time = 0
scan_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    if current_time - last_time >= 0.5:  # 2 FPS
        results = model.track(frame, conf=0.3, verbose=False, tracker="bytetrack.yaml")
        
        person_count = 0
        car_count = 0
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls)
                
                if class_id == 0:  # İnsan
                    person_count += 1
                    color = (0, 255, 0)  # Yeşil
                elif class_id == 2:  # Araba
                    car_count += 1
                    color = (255, 0, 0)  # Mavi
                else:
                    continue
                
                # Kutu çiz
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Etiket
                label = 'INSAN' if class_id == 0 else 'ARABA'
                if box.id is not None:
                    label += f' ID:{int(box.id)}'
                
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        scan_count += 1
        print(f"🔍 Tarama {scan_count}: 👤 {person_count} insan, 🚗 {car_count} araba")
        
        # Basit bilgi
        cv2.putText(frame, f'INSAN: {person_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'ARABA: {car_count}', (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        last_time = current_time
    
    cv2.imshow('INSAN & ARABA TAKIP', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()