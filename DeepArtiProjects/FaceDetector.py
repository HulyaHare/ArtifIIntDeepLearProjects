import ssl  # SSL kütüphanesi içe aktarılıyor
ssl._create_default_https_context = ssl._create_unverified_context  # HTTPS bağlantı hatalarının önüne geçmek için SSL ayarları yapılıyor
from sklearn.datasets import fetch_lfw_people  # LFW yüz veri setini indirmek için sklearn kütüphanesinden fetch_lfw_people içe aktarılıyor
import numpy as np  # Numpy kütüphanesi matematiksel işlemler için içe aktarılıyor
import matplotlib.pyplot as plt  # Görselleştirme için matplotlib kütüphanesinden pyplot modülü içe aktarılıyor

# LFW veri setini indir
faces = fetch_lfw_people(min_faces_per_person=60, resize=0.4)  # LFW veri seti indirilirken her kişiden en az 60 yüz görüntüsü olmasına ve görüntülerin boyutunun %40 küçültülmesine dikkat ediliyor

# Veri seti boyutu
print(f"Veri Seti Boyutu: {faces.images.shape}")  # Veri setindeki yüz görüntülerinin boyutu ekrana yazdırılıyor

# Örnek görselleri göster
fig, ax = plt.subplots(3, 5, figsize=(15, 9))  # 3 satır ve 5 sütun içeren bir alt grafik oluşturuluyor
for i, axi in enumerate(ax.flat):  # Her bir alt grafikte gösterilecek yüz için döngü başlatılıyor
    axi.imshow(faces.images[i], cmap='gray')  # Yüz görüntüsü gri tonlamalı olarak çiziliyor
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])  # Eksenler kapatılıyor ve her bir yüzün adı yazdırılıyor
plt.show()  # Oluşturulan grafik gösteriliyor

import tensorflow as tf  # TensorFlow kütüphanesi içe aktarılıyor
from tensorflow.keras.models import Sequential  # Keras modelleri için Sequential sınıfı içe aktarılıyor
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # CNN katmanları içe aktarılıyor

# Veri setini hazırlayalım
X = faces.images  # Yüz görüntüleri X değişkenine atanıyor
y = faces.target  # Yüzlerin hedef sınıfları y değişkenine atanıyor
X = X[..., np.newaxis]  # Tek renk kanalı ekleniyor çünkü görüntüler tek kanallı (gri tonlamalı)

# Veriyi normalize et
X = X / 255.0  # Görüntü piksel değerleri 0 ile 1 arasına normalize ediliyor

# Model oluşturma
model = Sequential()  # Boş bir model oluşturuluyor

# Convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 37, 1)))  # İlk konvolüsyon katmanı ekleniyor 32 filtre ve 3x3 boyutlarında
model.add(MaxPooling2D(pool_size=(2, 2)))  # Maksimum havuzlama katmanı ekleniyor

# İkinci Convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # İkinci konvolüsyon katmanı ekleniyor 64 filtre ile
model.add(MaxPooling2D(pool_size=(2, 2)))  # İkinci maksimum havuzlama katmanı ekleniyor

# Fully connected layer
model.add(Flatten())  # Veriyi düzleştirmek için Flatten katmanı ekleniyor
model.add(Dense(128, activation='relu'))  # Tam bağlantılı (fully connected) katman ekleniyor 128 nöron ile

# Çıkış katmanı
model.add(Dense(len(faces.target_names), activation='softmax'))  # Sınıflar kadar nöron olan çıkış katmanı ekleniyor softmax aktivasyon fonksiyonu ile

# Modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Model derleniyor Adam optimizasyon algoritması kullanılarak kayıp fonksiyonu ve metrik belirleniyor

# Model özetini yazdır
model.summary()  # Modelin yapısı ve katmanları hakkında özet bilgi yazdırılıyor

from sklearn.model_selection import train_test_split  # Eğitim ve test setlerine ayırmak için train_test_split içe aktarılıyor

# Eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veriler eğitim ve test setlerine ayrılıyor %20 test için ayrılıyor

# Modeli eğit
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))  # Model belirtilen eğitim verisi ile eğitiliyor doğrulama verisi olarak test seti kullanılıyor

# Test seti üzerinde değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)  # Test seti üzerinde modelin performansı değerlendiriliyor
print(f"Test seti doğruluğu: {test_acc}")  # Test setindeki doğruluk oranı ekrana yazdırılıyor

predictions = model.predict(X_test)  # Test seti için tahminler yapılıyor

# İlk 5 tahmini görselleştir
fig, ax = plt.subplots(1, 5, figsize=(20, 5))  # 1 satır ve 5 sütun içeren bir alt grafik oluşturuluyor
for i, axi in enumerate(ax.flat):  # Her tahmin için döngü başlatılıyor
    axi.imshow(X_test[i].reshape(50, 37), cmap='gray')  # Test setindeki görüntü gri tonlamalı olarak çiziliyor
    axi.set(xticks=[], yticks=[])  # Eksenler kapatılıyor
    pred = np.argmax(predictions[i])  # Tahmin sonuçlarından en yüksek değere sahip sınıf alınıyor
    axi.set_title(f"Gerçek: {faces.target_names[y_test[i]]}\nTahmin: {faces.target_names[pred]}")  # Gerçek ve tahmin edilen sınıf adları başlık olarak yazdırılıyor
plt.show()  # Grafik gösteriliyor

# Modeli kaydetme
model.save('face_recognition_model.h5')  # Eğitimli model belirtilen dosya adına kaydediliyor
