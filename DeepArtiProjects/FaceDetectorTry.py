import tensorflow as tf  # TensorFlow kütüphanesi içe aktarılıyor
import numpy as np  # Numpy kütüphanesi matematiksel işlemler için içe aktarılıyor
from tensorflow.keras.preprocessing import image  # Keras görüntü işleme modülü içe aktarılıyor
import matplotlib.pyplot as plt  # Görselleştirme için matplotlib kütüphanesinden pyplot modülü içe aktarılıyor
from sklearn.datasets import fetch_lfw_people  # LFW yüz veri setini indirmek için sklearn kütüphanesinden fetch_lfw_people içe aktarılıyor

# Modeli yükle
model = tf.keras.models.load_model('face_recognition_model.h5')  # Daha önce kaydedilen yüz tanıma modeli dosyadan yükleniyor

# LFW veri setinden hedef isimleri al
faces = fetch_lfw_people(min_faces_per_person=60, resize=0.4)  # LFW veri seti indirilirken her kişiden en az 60 yüz görüntüsü olmasına dikkat ediliyor
target_names = faces.target_names  # Hedef sınıf isimleri (yüzlerin isimleri) alınıyor

# Resmi işleme
img_path = 'George_Bush.jpeg'  # Tahmin edilecek yüzün bulunduğu resim dosyasının yolu belirtiliyor
img = image.load_img(img_path, target_size=(50, 37), color_mode='grayscale')  # Resim belirtilen boyutta (50x37) gri tonlamalı olarak yükleniyor
img_array = image.img_to_array(img)  # Resim numpy dizisine dönüştürülüyor
img_array = np.expand_dims(img_array, axis=0)  # Boyut modelin giriş şekline uygun hale getiriliyor
img_array /= 255.0  # Piksel değerleri 0 ile 1 arasına normalize ediliyor

# Model ile tahmin yapma
predictions = model.predict(img_array)  # Model kullanılarak tahmin yapılıyor
predicted_class = np.argmax(predictions)  # Tahmin sonuçlarından en yüksek değere sahip sınıf alınıyor

# Tahmin edilen sınıfı yazdır
print(f"Tahmin edilen kişi: {target_names[predicted_class]}")  # Tahmin edilen kişinin adı ekrana yazdırılıyor
