import cv2
from ultralytics import YOLO
import time
import numpy as np
import os

# ----------------------------------------------------
# 🔧 AYARLAR (THRESHOLDS) - SADECE BURAYI DEĞİŞTİRİN
# ----------------------------------------------------
CROSSWALK_CONFIG = {
    # Görüntü İşleme Alanı (ROI) - Yaya Geçidi sadece bu alanda aranır.
    'ROI_HEIGHT_RATIO': 0.50,         # Görüntünün alt %50'lik kısmı 

    # Renk Filtreleme (Hassas Ayarlar)
    'WHITE_LOWER_HSV': np.array([0, 0, 180]),     
    'WHITE_UPPER_HSV': np.array([179, 50, 255]),  
    'YELLOW_LOWER_HSV': np.array([20, 100, 150]), 
    'YELLOW_UPPER_HSV': np.array([40, 255, 255]), 

    # Morfolojik İşlemler
    'KERNEL_SIZE': (7, 7),          
    'OPENING_ITERATIONS': 2,        

    # Yaya Geçidi Kriterleri (Kontur Analizi)
    'MIN_CONTOUR_AREA': 1000,       
    'MIN_WIDTH_RATIO': 1.0,         
    'REQUIRED_PARALLEL_LINES': 3,   
    'MAX_CLUSTER_HEIGHT': 150       
}
# ----------------------------------------------------

# --- İLK YÜKLEMELER ---
print("YOLO modeli yükleniyor...")

# YOLO modeli seçenekleri - Raspberry Pi için optimize edilmiş
try:
    # İlk önce hafif model deneyelim
    model = YOLO('yolov8n.pt')  # Standart model
    print("✅ YOLOv8n.pt modeli yüklendi")
except:
    try:
        # Alternatif olarak küçük model
        model = YOLO('yolov8s.pt')
        print("✅ YOLOv8s.pt modeli yüklendi")
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        print("⚠️  Model indiriliyor...")
        # Modeli otomatik indir
        from ultralytics import download
        model = YOLO('yolov8n.pt')
        print("✅ Model indirildi ve yüklendi")

# USB KAMERA BAĞLANTISI
print("Kamera bağlantısı kuruluyor...")
cap = cv2.VideoCapture(0)  # USB kamera için

# Alternatif kamera cihazları
if not cap.isOpened():
    print("video0 açılamadı, alternatif deniyor...")
    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ video{i} açıldı")
            break

if not cap.isOpened():
    print("❌ HATA: Hiçbir kamera açılamadı!")
    print("Deneyebileceğiniz çözümler:")
    print("1. Kamera bağlantısını kontrol edin")
    print("2. sudo chmod 666 /dev/video*")
    print("3. lsusb komutu ile kameranın göründüğünden emin olun")
    exit()

# Kamerayı yapılandıralım (Raspberry Pi için optimize)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Raspberry Pi için düşük çözünürlük
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Daha iyi performans
cap.set(cv2.CAP_PROP_FPS, 10)            # Düşük FPS

print("🚗👤 İNSAN & ARABA SAYACI + HASSAS YAYA GEÇİDİ TESPİTİ")
print("📷 Kaynak: USB Kamera")
print("📊 ÇIKTI: Her frame için konsola yazdırma AKTİF")
print("⏹️ 'q' ile çık | 's' ekran görüntüsü | 'p' duraklat")

# Pencereyi önceden oluşturalım (setWindowTitle hatasını önlemek için)
cv2.namedWindow('INSAN & ARABA TAKIP + HASSAS FILTRE', cv2.WINDOW_NORMAL)
cv2.resizeWindow('INSAN & ARABA TAKIP + HASSAS FILTRE', 800, 600)

last_time = 0
scan_count = 0
paused = False

def detect_crosswalk_color_optimized(frame, config):
    """
    Beyaz ve Sarı işaretleri toplayan, güçlü morfolojik filtreleme kullanan algoritma.
    """
    height, width = frame.shape[:2]
    
    # 1. Yaya Geçidi için ROI (Region of Interest) belirle
    roi_y_start = int(height * (1.0 - config['ROI_HEIGHT_RATIO']))
    roi = frame[roi_y_start:height, 0:width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 2. İşaret Rengi Maskeleri
    white_mask = cv2.inRange(hsv, config['WHITE_LOWER_HSV'], config['WHITE_UPPER_HSV'])
    yellow_mask = cv2.inRange(hsv, config['YELLOW_LOWER_HSV'], config['YELLOW_UPPER_HSV'])
    final_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 3. Gürültü azaltma ve çizgileri birleştirme (Morfolojik İşlemler)
    kernel = np.ones(config['KERNEL_SIZE'], np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=config['OPENING_ITERATIONS']) 
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1) 
    
    # 4. Kontur Tespiti
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > config['MIN_CONTOUR_AREA']:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Geometri filtresi: Yüksek yataylık zorunluluğu
            if (w / h) >= config['MIN_WIDTH_RATIO']:
                # Kontur tam görüntü koordinatlarına ekle
                filtered_contours.append(((x, y + roi_y_start, w, h))) 

    # 5. Kümelenme ve Sayı Kontrolü
    if len(filtered_contours) < config['REQUIRED_PARALLEL_LINES']:
        return False, filtered_contours, roi_y_start

    y_coords = sorted([cont[1] for cont in filtered_contours])

    y_min = y_coords[0]
    y_max = y_coords[-1]
    
    crosswalk_detected = (y_max - y_min) < config['MAX_CLUSTER_HEIGHT']
    
    return crosswalk_detected, filtered_contours, roi_y_start

# --- Ana Döngü ---
print("⏳ Başlatılıyor... İlk kare için bekleniyor...")

# Konsol temizleme (isteğe bağlı)
os.system('clear' if os.name == 'posix' else 'cls')

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Kare alınamadı. Kamerayı kontrol edin.")
                time.sleep(1)
                continue
            
            # Frame boyutunu kontrol et ve ayarla
            if frame.shape[1] > 800:  # Çok büyükse küçült
                frame = cv2.resize(frame, (640, 480))
        
        current_time = time.time()
        
        if not paused and (current_time - last_time >= 0.5):  # 2 FPS
            height, width = frame.shape[:2]
            
            # 1. Yaya Geçidi tespiti
            crosswalk_detected, crosswalk_contours, roi_y_start = detect_crosswalk_color_optimized(frame, CROSSWALK_CONFIG)
            
            # 2. YOLO ile nesne tespiti (Tüm Frame üzerinde çalışır)
            try:
                # Raspberry Pi için optimize edilmiş YOLO ayarları
                results = model(frame, conf=0.25, verbose=False, device='cpu', imgsz=320)
            except Exception as e:
                print(f"⚠️ YOLO hatası: {e}")
                results = [type('obj', (object,), {'boxes': None})()]
            
            person_count = 0
            car_count = 0
            
            # --- YOLO Çizimleri ---
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    if class_id in [0, 2]:  # 0: person, 2: car
                        if class_id == 0: 
                            person_count += 1
                            color = (0, 255, 0)
                            label_text = 'INSAN'
                        else: 
                            car_count += 1
                            color = (255, 0, 0)
                            label_text = 'ARABA'
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = label_text
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 3. Yaya Geçidi Çizimleri ve Görselleştirme
            if crosswalk_contours:
                for cont in crosswalk_contours:
                    x, y, w, h = cont
                    # Algılanan her konturu işaretle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255) if crosswalk_detected else (0, 165, 255), 2)
                
                if crosswalk_detected:
                    # Yaya geçidini vurgula
                    y_coords = [c[1] for c in crosswalk_contours]
                    x_coords = [c[0] for c in crosswalk_contours] + [c[0] + c[2] for c in crosswalk_contours]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x_min - 5, y_min - 10), (x_max + 5, y_max + 10), (0, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            scan_count += 1
            
            # --- Bilgi Çıktıları ---
            info_y = 30
            line_spacing = 25
            
            # Arka plan ekleyelim (okunabilirlik için)
            cv2.rectangle(frame, (5, 5), (300, info_y + line_spacing * 3 + 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (300, info_y + line_spacing * 3 + 10), (255, 255, 255), 1)
            
            cv2.putText(frame, f'INSAN: {person_count}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'ARABA: {car_count}', (10, info_y + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            crosswalk_color = (0, 255, 255) if crosswalk_detected else (0, 0, 255)
            crosswalk_text = 'YAYA GECIDI: VAR' if crosswalk_detected else 'YAYA GECIDI: YOK'
            cv2.putText(frame, crosswalk_text, (10, info_y + line_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, crosswalk_color, 2)
            
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            cv2.putText(frame, f'FPS: {fps:.1f}', (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 4. Yaya Geçidi Arama Alanını Çiz (SADECE YAYA GEÇİDİ İÇİN ROI)
            cv2.rectangle(frame, (0, roi_y_start), (width, height), (0, 255, 255), 1)
            cv2.putText(frame, 'YAYA GECIDI ARAMA BOLGESI', (10, max(roi_y_start - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Durum bilgisi
            status_text = "DURAKLATILDI" if paused else f"TARAMA: {scan_count}"
            cv2.putText(frame, status_text, (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # HER FRAME İÇİN KONSOLA YAZDIR (10 frame'de 1 yerine)
            print(f"🔍 Frame {scan_count}: 👤 {person_count} insan, 🚗 {car_count} araba, 🚸 Yaya geçidi: {'EVET' if crosswalk_detected else 'HAYIR'}, 📐 ROI: {CROSSWALK_CONFIG['ROI_HEIGHT_RATIO']:.1f}, ⚡ FPS: {fps:.1f}")
            
            last_time = current_time
        
        # Pencere başlığını güncelle (pencere oluşturulduktan sonra)
        try:
            cv2.setWindowTitle('INSAN & ARABA TAKIP + HASSAS FILTRE', 
                              f'USB Kamera | 👤:{person_count} 🚗:{car_count} | FPS:{fps:.1f}' if 'fps' in locals() else 'USB Kamera | Yükleniyor...')
        except:
            pass
        
        cv2.imshow('INSAN & ARABA TAKIP + HASSAS FILTRE', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Çıkış yapılıyor...")
            break
        elif key == ord('s'):  # Ekran görüntüsü
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Ekran görüntüsü kaydedildi: {screenshot_path}")
        elif key == ord('p'):  # Duraklat/devam
            paused = not paused
            print(f"⏸️  {'Duraklatıldı' if paused else 'Devam ediliyor'}")
        elif key == ord('+'):  # ROI boyutunu artır
            CROSSWALK_CONFIG['ROI_HEIGHT_RATIO'] = min(0.8, CROSSWALK_CONFIG['ROI_HEIGHT_RATIO'] + 0.1)
            print(f"📐 ROI yükseklik oranı: {CROSSWALK_CONFIG['ROI_HEIGHT_RATIO']:.1f}")
        elif key == ord('-'):  # ROI boyutunu azalt
            CROSSWALK_CONFIG['ROI_HEIGHT_RATIO'] = max(0.1, CROSSWALK_CONFIG['ROI_HEIGHT_RATIO'] - 0.1)
            print(f"📐 ROI yükseklik oranı: {CROSSWALK_CONFIG['ROI_HEIGHT_RATIO']:.1f}")
        elif key == ord('c'):  # Konsolu temizle
            os.system('clear' if os.name == 'posix' else 'cls')
            print("🧹 Konsol temizlendi")
            print("🚗👤 İNSAN & ARABA SAYACI + HASSAS YAYA GEÇİDİ TESPİTİ")
            print(f"📷 Kaynak: USB Kamera | 📊 Frame: {scan_count}")

except KeyboardInterrupt:
    print("\n⏹️  KeyboardInterrupt: Program durduruluyor...")
except Exception as e:
    print(f"\n❌ Beklenmeyen hata: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Kaynaklar serbest bırakıldı. Program sonlandı.")
    print(f"📊 Toplam işlenen frame sayısı: {scan_count}")
