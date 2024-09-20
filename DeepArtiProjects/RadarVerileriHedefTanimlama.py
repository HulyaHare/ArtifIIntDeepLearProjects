import numpy as np  # Numpy kütüphanesi matematiksel işlemler için içe aktarılıyor
import matplotlib.pyplot as plt  # Görselleştirme için matplotlib kütüphanesinden pyplot modülü içe aktarılıyor
import seaborn as sns  # İleri düzey görselleştirme için seaborn kütüphanesi içe aktarılıyor
import tensorflow as tf  # TensorFlow kütüphanesi içe aktarılıyor
from tensorflow.keras.models import Sequential  # Keras'tan Sequential model sınıfı içe aktarılıyor
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization  # CNN katmanları içe aktarılıyor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Eğitim sırasında model optimizasyonu için callback'ler içe aktarılıyor
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için içe aktarılıyor
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve  # Performans değerlendirmeleri için metrikler içe aktarılıyor

# Radar sinyali simülasyonu
def generate_radar_signal(num_samples, target_type, noise_level=0.5):  # Radar sinyali üretimi için fonksiyon tanımlanıyor
    np.random.seed(42)  # Rastgele sayıların tutarlılığı için tohum ayarlanıyor

    if target_type == "aircraft":  # Hedef tipi uçak ise
        frequency = 5  # Uçak için sinyal frekansı
    elif target_type == "helicopter":  # Hedef tipi helikopter ise
        frequency = 10  # Helikopter için sinyal frekansı
    elif target_type == "uav":  # Hedef tipi İHA ise
        frequency = 20  # İHA için sinyal frekansı
    else:  # Diğer hedefler için
        frequency = 1  # Varsayılan frekans

    t = np.linspace(0, 1, num_samples)  # Zaman vektörü oluşturuluyor
    # Gürültü seviyesi eklenmiş sinyal üretimi
    signal = np.sin(2 * np.pi * frequency * t) + noise_level * np.random.randn(num_samples)  # Sinüs dalgası ve gürültü ekleniyor
    return signal  # Üretilen sinyal döndürülüyor

# Radar sinyali görselleştirme
def plot_signals(signals, titles, num_signals):  # Sinyalleri görselleştirmek için fonksiyon tanımlanıyor
    plt.figure(figsize=(10, 6))  # Grafik boyutu ayarlanıyor
    for i in range(num_signals):  # Belirtilen sinyal sayısı kadar döngü başlatılıyor
        plt.subplot(num_signals, 1, i + 1)  # Alt grafik oluşturuluyor
        plt.plot(signals[i])  # Sinyal çiziliyor
        plt.title(titles[i])  # Başlık ekleniyor
    plt.tight_layout()  # Alt grafikler arasındaki boşluk ayarlanıyor
    plt.show()  # Grafik gösteriliyor

# Radar sinyalleri ve gürültü seviyeleri ile veri artırımı
def augment_data(num_samples, target_type, n_augment=5):  # Veri artırımı için fonksiyon tanımlanıyor
    signals = []  # Sinyalleri depolamak için boş liste oluşturuluyor
    for _ in range(n_augment):  # Belirtilen sayıda sinyal üretimi
        noise_level = np.random.uniform(0.1, 0.8)  # Farklı gürültü seviyeleri belirleniyor
        signal = generate_radar_signal(num_samples, target_type, noise_level)  # Radar sinyali üretiliyor
        signals.append(signal)  # Üretilen sinyal listeye ekleniyor
    return signals  # Üretilen sinyaller döndürülüyor

# Sinyalleri ve artırılmış sinyalleri oluşturma
num_samples = 1000  # Her sinyal için örnek sayısı
num_signals = 5  # Üretilecek sinyal sayısı

signal_aircraft = augment_data(num_samples, "aircraft")  # Uçak sinyalleri oluşturuluyor
signal_helicopter = augment_data(num_samples, "helicopter")  # Helikopter sinyalleri oluşturuluyor
signal_uav = augment_data(num_samples, "uav")  # İHA sinyalleri oluşturuluyor

signals = signal_aircraft + signal_helicopter + signal_uav  # Tüm sinyaller birleştiriliyor
titles = ["Aircraft Signal " + str(i + 1) for i in range(len(signal_aircraft))] + \
         ["Helicopter Signal " + str(i + 1) for i in range(len(signal_helicopter))] + \
         ["UAV Signal " + str(i + 1) for i in range(len(signal_uav))]  # Başlıklar oluşturuluyor

plot_signals(signals, titles, num_signals=3)  # İlk üç sinyali görselleştiriyoruz

# Veri seti oluşturma
signals = []  # Boş sinyal listesi
labels = []  # Boş etiket listesi

# Uçak, Helikopter ve İHA sınıfları için veri üretimi
for _ in range(500):  # Her sınıftan 500 sinyal
    signals += augment_data(num_samples, "aircraft")  # Uçak sinyalleri ekleniyor
    labels += [0] * 5  # Aircraft sınıfı 0 ile etiketleniyor
    signals += augment_data(num_samples, "helicopter")  # Helikopter sinyalleri ekleniyor
    labels += [1] * 5  # Helicopter sınıfı 1 ile etiketleniyor
    signals += augment_data(num_samples, "uav")  # İHA sinyalleri ekleniyor
    labels += [2] * 5  # UAV sınıfı 2 ile etiketleniyor

signals = np.array(signals)  # Sinyaller numpy dizisine dönüştürülüyor
labels = np.array(labels)  # Etiketler numpy dizisine dönüştürülüyor

# Veriyi normalleştir (standartlaştır)
signals = (signals - np.min(signals)) / (np.max(signals) - np.min(signals))  # Sinyaller 0 ile 1 arasına normalize ediliyor

# Eğitim ve test seti olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(signals, labels, test_size=0.3, random_state=42)  # Veriler eğitim ve test setlerine ayrılıyor %30 test için ayrılıyor

# CNN modeli oluşturma
model = Sequential()  # Boş bir Sequential model oluşturuluyor

# Convolutional katmanlar
model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=(num_samples, 1)))  # İlk konvolüsyon katmanı ekleniyor
model.add(BatchNormalization())  # Batch normalization ekleyerek öğrenmeyi iyileştiriyoruz
model.add(MaxPooling1D(pool_size=2))  # Maksimum havuzlama katmanı ekleniyor

model.add(Conv1D(256, kernel_size=3, activation='relu'))  # İkinci konvolüsyon katmanı ekleniyor
model.add(BatchNormalization())  # Batch normalization ekleniyor
model.add(MaxPooling1D(pool_size=2))  # İkinci maksimum havuzlama katmanı ekleniyor

model.add(Conv1D(512, kernel_size=3, activation='relu'))  # Üçüncü konvolüsyon katmanı ekleniyor
model.add(MaxPooling1D(pool_size=2))  # Üçüncü maksimum havuzlama katmanı ekleniyor

# Tam bağlantılı katmanlar
model.add(Flatten())  # Veriyi düzleştirmek için Flatten katmanı ekleniyor
model.add(Dense(256, activation='relu'))  # Tam bağlantılı katman ekleniyor
model.add(Dropout(0.5))  # Dropout ekleyerek overfitting'i önlemeye çalışıyoruz
model.add(Dense(128, activation='relu'))  # İkinci tam bağlantılı katman ekleniyor
model.add(Dense(3, activation='softmax'))  # 3 sınıf (Aircraft, Helicopter, UAV) için çıkış katmanı ekleniyor

# Optimizer ve learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adam optimizasyon algoritması ile öğrenme oranı belirleniyor
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)  # Öğrenme oranını azaltacak scheduler tanımlanıyor

# Modeli derleme
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Model derleniyor kayıp fonksiyonu ve metrik belirleniyor

# Erken durdurma (Early Stopping)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Erken durdurma için callback tanımlanıyor

# Veriyi CNN için uygun hale getirme
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Eğitim verisi 3D şekle dönüştürülüyor
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Test verisi 3D şekle dönüştürülüyor

# Modeli eğitme
history = model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_test_cnn, y_test),
                    callbacks=[early_stopping, reduce_lr])  # Model eğitim verisi ile belirtilen epoch sayısında eğitiliyor

# Modelin test seti üzerindeki performansı
test_loss, test_acc = model.evaluate(X_test_cnn, y_test)  # Test seti üzerinde modelin performansı değerlendiriliyor
print(f"Test Doğruluk Oranı: {test_acc * 100:.2f}%")  # Test doğruluk oranı ekrana yazdırılıyor

# Confusion Matrix ve ROC AUC ile performans değerlendirme
y_pred = np.argmax(model.predict(X_test_cnn), axis=1)  # Test seti üzerindeki tahminler alınıyor

conf_matrix = confusion_matrix(y_test, y_pred)  # Karışıklık matrisi oluşturuluyor
plt.figure(figsize=(8, 6))  # Grafik boyutu ayarlanıyor
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Aircraft", "Helicopter", "UAV"],
            yticklabels=["Aircraft", "Helicopter", "UAV"])  # Karışıklık matrisini çizdirme
plt.ylabel('Gerçek Değerler')  # Y ekseni başlığı
plt.xlabel('Tahmin Değerleri')  # X ekseni başlığı
plt.title('Confusion Matrix')  # Grafik başlığı
plt.show()  # Grafik gösteriliyor

# ROC ve AUC hesaplama
y_pred_prob = model.predict(X_test_cnn)  # Test seti üzerindeki tahmin olasılıkları alınıyor
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')  # ROC AUC skoru hesaplanıyor

# ROC eğrisi çizdirme
fpr = dict()  # False positive rate sözlüğü
tpr = dict()  # True positive rate sözlüğü
roc_auc = dict()  # ROC AUC değerleri için sözlük
for i in range(3):  # Her sınıf için döngü
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_prob[:, i], pos_label=i)  # ROC eğrisi için değerler hesaplanıyor
    roc_auc[i] = roc_auc_score((y_test == i).astype(int), y_pred_prob[:, i])  # Her sınıf için AUC hesaplanıyor

plt.figure(figsize=(10, 6))  # Grafik boyutu ayarlanıyor
plt.plot(fpr[0], tpr[0], label=f'Aircraft ROC curve (area = {roc_auc[0]:.2f})')  # Uçak ROC eğrisi
plt.plot(fpr[1], tpr[1], label=f'Helicopter ROC curve (area = {roc_auc[1]:.2f})')  # Helikopter ROC eğrisi
plt.plot(fpr[2], tpr[2], label=f'UAV ROC curve (area = {roc_auc[2]:.2f})')  # İHA ROC eğrisi
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal çizgi
plt.xlim([0.0, 1.0])  # X ekseni sınırları
plt.ylim([0.0, 1.05])  # Y ekseni sınırları
plt.xlabel('False Positive Rate')  # X ekseni başlığı
plt.ylabel('True Positive Rate')  # Y ekseni başlığı
plt.title('Receiver Operating Characteristic (ROC) Curves')  # Grafik başlığı
plt.legend(loc="lower right")  # Lejant konumu
plt.show()  # Grafik gösteriliyor

# Detaylı sınıflandırma raporu
print(classification_report(y_test, y_pred, target_names=["Aircraft", "Helicopter", "UAV"]))  # Sınıflandırma raporu yazdırılıyor

# Örnek tahmin
new_signal = generate_radar_signal(num_samples, "uav").reshape(1, num_samples, 1)  # Yeni sinyal üretiliyor ve 3D forma dönüştürülüyor
prediction = model.predict(new_signal)  # Yeni sinyal ile tahmin yapılıyor
predicted_class = np.argmax(prediction)  # Tahmin sonucu sınıf belirleniyor

classes = ["Aircraft", "Helicopter", "UAV"]  # Sınıf isimleri
print(f"Tahmin Edilen Sınıf: {classes[predicted_class]}")  # Tahmin edilen sınıf ekrana yazdırılıyor
