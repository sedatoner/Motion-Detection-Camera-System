import cv2 
import datetime
import os
import csv
import imutils

# klasörleri yoksa oluştur, bir kereye mahsus
os.makedirs("captures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# log dosyası kontrol, başlık at gitsin
log_path = "logs/motion_log.csv"
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image Path"])

# kamerayı bağla (dahiliyse 0 zaten)
cam = cv2.VideoCapture(0)
first_frame = None
frame_count = 0  # kaç kare geçti sayıyoruz

while True:
    ret, frame = cam.read()
    if not ret:
        break  # kamera vermedi mi? çık

    # ekran boyutunu küçült, hız kazancı
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gri yap neden çünkü RGB de 3 kat daha fazla veri var bu da 3 kat daha fazla bellek ihtiyacı.
    #grayscale "değişim" takibinde daha efektif RGB ye göre RGB de uzun vadeli sıkıntılar çıkabiliyor grayscale daha kararlı
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # parazit azalt

    frame_count += 1

    # referans kareyi belirli aralıklarla güncelle. 
    #sabit kalmasın çünkü uzun vadede her şeyi hareket olarak algılamaya başlıyor her şey sabit olsa bile
    if first_frame is None or frame_count % 30 == 0:
        first_frame = gray  # yeni referans
        continue  # bu kareyi kıyaslama, sıradaki

    # hareket kontrolü
    delta = cv2.absdiff(first_frame, gray)  # farkı al
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]  # eşik
    thresh = cv2.dilate(thresh, None, iterations=2)  # boşlukları doldur
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # küçük hareketleri ayıkla
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # kutu çiz gitsin

    # hareket varsa kaydet ve logla
    if motion_detected:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        img_path = f"captures/motion_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)  # görüntüyü diske at

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, img_path])  # loga da yaz

        print(f"[{timestamp}] Hareket algılandı. Görüntü kaydedildi.")

    cv2.imshow("Canlı Görüntü", frame)  # gösterim penceresi
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  # q'ya bas exit

cam.release()
cv2.destroyAllWindows()
