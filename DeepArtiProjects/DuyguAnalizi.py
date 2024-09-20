import face_recognition  # Yüz tanıma için kullanılan face_recognition kütüphanesi içe aktarılıyor
from fer import FER  # Duygu analizi için FER kütüphanesi içe aktarılıyor
import matplotlib.pyplot as plt  # Grafik çizmek için matplotlib kütüphanesinden pyplot modülü içe aktarılıyor
import cv2  # OpenCV kütüphanesi görüntü işleme görevleri için içe aktarılıyor

# Yüz görüntüsünü yükleyelim
image_path = 'Woman.jpeg'  # Kullanacağımız yüz görüntüsünün dosya yolu belirleniyor
image = face_recognition.load_image_file(image_path)  # Belirtilen dosya yolundan görüntü yükleniyor

# Yüz tanıma işlemi başlıyor
face_locations = face_recognition.face_locations(image)  # Yüz tanıma işlemi yapılarak görüntüdeki yüzlerin konumları alınıyor
print(f"Bulunan yüz sayısı: {len(face_locations)}")  # Tespit edilen yüzlerin sayısı ekrana yazdırılıyor

# Yüz ifadeleri için FER kütüphanesi ile duygusal analiz yapılıyor
detector = FER(mtcnn=True)  # Duygu tespiti için FER dedektörü oluşturuluyor MTCNN yüz algılama kullanılıyor
emotion_result = detector.detect_emotions(image)  # Yüz ifadeleri üzerinde duygu analizi gerçekleştiriliyor

if emotion_result:  # Eğer duygusal analiz sonucunda bir sonuç varsa
    # Bulunan duygusal durumları yazdır
    print("Yüz ifadeleri ve duygusal analiz sonuçları:")  # Duygusal durumların listeleneceği mesaj yazdırılıyor
    for result in emotion_result:  # Tespit edilen her yüz için döngü başlatılıyor
        print(result['emotions'])  # Her bir yüz ifadesinin duygusal durumları ekrana yazdırılıyor

# Görüntüyü çizerek yüzleri işaretleyelim
for top, right, bottom, left in face_locations:  # Tespit edilen yüzlerin üst sol alt sağ köşe koordinatları döngü ile alınıyor
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # Yüzü dikdörtgen ile işaretlemek için cv2.rectangle fonksiyonu kullanılıyor

plt.imshow(image)  # Yüz görüntüsü matplotlib ile çiziliyor
plt.axis('off')  # Eksenlerin gösterimini kapatıyoruz
plt.show()  # Görüntü gösteriliyor

from transformers import pipeline  # Hugging Face Transformers kütüphanesinden pipeline fonksiyonu içe aktarılıyor

# BERT ile metin duygu analizi yapmak için bir pipeline oluşturuluyor
classifier = pipeline('sentiment-analysis')  # Metin duygu analizi için sınıflandırıcı oluşturuluyor

# Test edilecek bir metin belirleyelim
text = "I am feeling great today!"  # Duygu analizi yapılacak metin belirleniyor
result = classifier(text)  # Belirtilen metin üzerinde duygu analizi yapılıyor

print(f"Metin: {text}")  # Analiz edilen metin ekrana yazdırılıyor
print(f"Duygu Analizi Sonucu: {result}")  # Metin üzerindeki duygu analizi sonucu ekrana yazdırılıyor

# Yüz ifadesinden alınan duygu
face_emotion = emotion_result[0]['emotions'] if emotion_result else None  # Eğer yüz ifadesi tespit edildiyse ilk yüzün duygusu alınıyor

# Metin üzerinden alınan duygu
text_emotion = result[0]['label']  # Metin analizinden elde edilen duygusal etiket alınıyor

# Multimodal analiz: yüz ifadesi ve metin arasındaki uyumsuzluğu kontrol et
def compare_emotions(face_emotion, text_emotion):  # Yüz ifadesi ve metin duygusu karşılaştırması için fonksiyon tanımlanıyor
    if not face_emotion:  # Eğer yüz ifadesi yoksa
        return "Yüz ifadesi tespit edilemedi."  # Uyarı mesajı döndürülüyor

    # Yüz ifadesindeki en baskın duyguyu bul
    dominant_face_emotion = max(face_emotion, key=face_emotion.get)  # Yüz ifadesindeki en baskın duygu belirleniyor

    print(f"Yüz İfadesi Duygusu: {dominant_face_emotion}")  # Yüz ifadesinin baskın duygusu ekrana yazdırılıyor
    print(f"Metin Duygusu: {text_emotion}")  # Metin üzerindeki duygu ekrana yazdırılıyor

    # Yüz ifadesi ve metin duygusunun uyumunu kontrol et
    if dominant_face_emotion.lower() in text_emotion.lower():  # Yüz ifadesinin duygusu metin duygusu içinde var mı kontrol ediliyor
        return "Yüz ifadesi ile metin duygusu uyumlu."  # Eğer uyumluysa mesaj döndürülüyor
    else:  # Eğer uyumsuzsa
        return "Yüz ifadesi ile metin duygusu uyumsuz."  # Uyuşmazlık durumunda mesaj döndürülüyor

# Sonucu yazdır
result = compare_emotions(face_emotion, text_emotion)  # Yüz ve metin duyguları karşılaştırılıyor
print(result)  # Karşılaştırma sonucunu ekrana yazdır

import matplotlib.pyplot as plt  # Tekrar matplotlib kütüphanesi içe aktarılıyor
from PIL import Image  # Görüntü işleme için PIL kütüphanesi içe aktarılıyor

# Sonuçları görselleştirecek bir fonksiyon tanımlanıyor
def visualize_results(image_path, face_emotion, text_emotion, comparison_result):  # Görselleştirme için fonksiyon tanımlanıyor
    # Yüz resmini yükleyelim
    image = Image.open(image_path)  # Belirtilen dosya yolundan yüz resmi yükleniyor

    # Görselleştirme için hazırlık
    plt.figure(figsize=(10, 5))  # Görüntü boyutu ayarlanıyor

    # Orijinal yüz resmi
    plt.subplot(1, 2, 1)  # İlk alt grafikte yüz resmi gösterilecek
    plt.imshow(image)  # Yüz resmi çiziliyor
    plt.title("Yüz Görüntüsü")  # Başlık ekleniyor
    plt.axis('off')  # Eksenlerin gösterimi kapatılıyor

    # Sonuçları gösterecek alt grafik
    plt.subplot(1, 2, 2)  # İkinci alt grafikte sonuçlar gösterilecek
    plt.text(0.5, 0.8, f"Yüz İfadesi: {max(face_emotion, key=face_emotion.get)}", fontsize=12, ha='center')  # Yüz ifadesi yazdırılıyor
    plt.text(0.5, 0.6, f"Metin Duygusu: {text_emotion}", fontsize=12, ha='center')  # Metin duygusu yazdırılıyor
    plt.text(0.5, 0.4, f"Sonuç: {comparison_result}", fontsize=12, ha='center')  # Karşılaştırma sonucu yazdırılıyor
    plt.axis('off')  # Eksenlerin gösterimi kapatılıyor

    plt.show()  # Görselleştiriliyor

# Sonuçları görselleştir
visualize_results(image_path, face_emotion, text_emotion, result)  # Sonuçların görselleştirilmesi için fonksiyon çağrılıyor
