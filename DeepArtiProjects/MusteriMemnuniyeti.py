import mysql.connector  # MySQL veritabanı ile bağlantı için mysql.connector kütüphanesi içe aktarılıyor
import pandas as pd  # Veri işleme için pandas kütüphanesi içe aktarılıyor
import numpy as np  # Matematiksel işlemler için numpy kütüphanesi içe aktarılıyor
from sklearn.preprocessing import StandardScaler  # Veriyi ölçeklemek için StandardScaler sınıfı içe aktarılıyor
from sklearn.ensemble import RandomForestClassifier  # Random Forest sınıflandırıcısı için içe aktarılıyor
from sklearn.model_selection import train_test_split, GridSearchCV  # Veriyi ayırmak ve hiperparametre optimizasyonu için içe aktarılıyor
from sklearn.metrics import accuracy_score  # Model doğruluğunu değerlendirmek için accuracy_score içe aktarılıyor

# MySQL veri tabanı bağlantısı
conn = mysql.connector.connect(  # MySQL veritabanına bağlantı kuruluyor
    host="localhost",  # Veritabanının bulunduğu ana bilgisayar
    user="root",  # Kullanıcı adı
    password="okm951esx753",  # Kullanıcı şifresi
    database="customer_db"  # Bağlanılacak veritabanı adı
)

cursor = conn.cursor()  # SQL sorguları için bir cursor oluşturuluyor

# prediction_results tablosu oluştur
cursor.execute(""" 
CREATE TABLE IF NOT EXISTS prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY, 
    customer_id INT, 
    actual_satisfaction INT, 
    predicted_satisfaction INT 
)
""")
conn.commit()  # Yapılan değişiklikler veritabanına kaydediliyor

# Örnek veri seti oluştur
np.random.seed(42)  # Rastgele sayı üretiminde tutarlılık sağlamak için tohum ayarlanıyor
data = {
    'age': np.random.randint(18, 70, size=100),  # 18 ile 70 arasında rastgele yaşlar
    'income': np.random.randint(20000, 150000, size=100),  # 20000 ile 150000 arasında rastgele gelirler
    'spending_score': np.random.randint(1, 100, size=100),  # 1 ile 100 arasında rastgele harcama puanları
    'satisfaction': np.random.randint(0, 2, size=100)  # 0 veya 1 olarak rastgele memnuniyet durumu
}
df = pd.DataFrame(data)  # Oluşturulan veriler pandas DataFrame formatına dönüştürülüyor

# SQL'e veri ekleme
for i, row in df.iterrows():  # Her bir satır için döngü başlatılıyor
    sql = "INSERT INTO customer_data (age, income, spending_score, satisfaction) VALUES (%s, %s, %s, %s)"  # SQL sorgusu tanımlanıyor
    cursor.execute(sql, tuple(row))  # Veriler SQL sorgusuna ekleniyor

conn.commit()  # Yapılan değişiklikler veritabanına kaydediliyor

# SQL'den veriyi çekelim
query = "SELECT age, income, spending_score, satisfaction FROM customer_data"  # SQL sorgusu ile veri çekiliyor
customer_data = pd.read_sql(query, conn)  # Çekilen veri pandas DataFrame olarak alınıyor

# Girdi ve çıktı değişkenlerini ayıralım
X = customer_data[['age', 'income', 'spending_score']]  # Girdi değişkenleri (özellikler) ayrılıyor
y = customer_data['satisfaction']  # Çıktı değişkeni (etiket) ayrılıyor

# Veriyi normalleştirelim (ölçekleyelim)
scaler = StandardScaler()  # StandardScaler nesnesi oluşturuluyor
X = scaler.fit_transform(X)  # Girdi verileri normalize ediliyor

# Eğitim ve test setlerini ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veriler eğitim ve test setlerine ayrılıyor %20 test için ayrılıyor

# RandomForest Modeli ile başarı oranını artırma
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)  # Random Forest modeli oluşturuluyor
rf_model.fit(X_train, y_train)  # Model eğitim verisi ile eğitiliyor

# Test seti üzerinde tahmin yapalım
y_pred_rf = rf_model.predict(X_test)  # Test seti üzerinde tahminler yapılıyor

# Başarı oranını hesaplayalım
accuracy_rf = accuracy_score(y_test, y_pred_rf)  # Doğruluk oranı hesaplanıyor
print(f"Random Forest Modeli Başarı Oranı: {accuracy_rf * 100:.2f}%")  # Başarı oranı ekrana yazdırılıyor

# Logistic Regression Hiperparametre Optimizasyonu
from sklearn.linear_model import LogisticRegression  # Logistic Regression sınıfı içe aktarılıyor

# Hiperparametre optimizasyonu
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}  # Hiperparametre arama alanı tanımlanıyor
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)  # GridSearchCV ile hiperparametre optimizasyonu için nesne oluşturuluyor
grid.fit(X_train, y_train)  # Eğitim verisi ile model eğitiliyor

# En iyi parametrelerle yeniden model eğitimi
logistic_model = grid.best_estimator_  # En iyi parametrelerle oluşturulan model alınıyor
logistic_model.fit(X_train, y_train)  # Model eğitim verisi ile yeniden eğitiliyor

# Test seti üzerinde tahmin yapalım
y_pred_logistic = logistic_model.predict(X_test)  # Test seti üzerinde tahminler yapılıyor

# Başarı oranını hesaplayalım
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)  # Doğruluk oranı hesaplanıyor
print(f"Logistic Regression Modeli Başarı Oranı: {accuracy_logistic * 100:.2f}%")  # Başarı oranı ekrana yazdırılıyor

# Derin öğrenme modeli için gerekli kütüphaneler
from tensorflow.keras.models import Sequential  # Keras'tan Sequential model sınıfı içe aktarılıyor
from tensorflow.keras.layers import Dense, Dropout  # Keras katmanları içe aktarılıyor

# Derin öğrenme modeli oluşturma
deep_model = Sequential()  # Boş bir Sequential model oluşturuluyor
deep_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # İlk katman 128 nöron ve ReLU aktivasyonu ile ekleniyor
deep_model.add(Dropout(0.3))  # Overfitting'i önlemek için Dropout katmanı ekleniyor
deep_model.add(Dense(64, activation='relu'))  # İkinci katman 64 nöron ve ReLU aktivasyonu ile ekleniyor
deep_model.add(Dropout(0.3))  # Tekrar Dropout katmanı ekleniyor
deep_model.add(Dense(32, activation='relu'))  # Üçüncü katman 32 nöron ve ReLU aktivasyonu ile ekleniyor
deep_model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı tek nöron ve sigmoid aktivasyonu ile ekleniyor

# Modeli derleyelim
deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Model derleniyor kayıp fonksiyonu ve optimizasyon algoritması belirleniyor

# Modeli eğitelim
deep_model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)  # Model eğitim verisi ile belirtilen epoch sayısında eğitiliyor

# Test seti üzerinde tahmin yapalım
_, deep_accuracy = deep_model.evaluate(X_test, y_test)  # Test seti üzerinde modelin performansı değerlendiriliyor
print(f"İyileştirilmiş Derin Öğrenme Modeli Başarı Oranı: {deep_accuracy * 100:.2f}%")  # Başarı oranı ekrana yazdırılıyor

# Logistic Regression sonuçları SQL'e kaydedelim
for i in range(len(y_pred_logistic)):  # Tahmin sonuçları için döngü başlatılıyor
    sql = "INSERT INTO prediction_results (customer_id, actual_satisfaction, predicted_satisfaction) VALUES (%s, %s, %s)"  # SQL sorgusu tanımlanıyor
    val = (int(i+1), int(y_test.values[i]), int(y_pred_logistic[i]))  # Veritabanına eklenmek üzere değerler hazırlanıyor
    cursor.execute(sql, val)  # Değerler SQL sorgusuna ekleniyor

conn.commit()  # Yapılan değişiklikler veritabanına kaydediliyor
print("Tahmin sonuçları SQL veri tabanına kaydedildi.")  # Başarı mesajı yazdırılıyor

# Örnek bir veri oluşturup model üzerinde deneyelim
new_customer = pd.DataFrame({  # Yeni bir müşteri için DataFrame oluşturuluyor
    'age': [30],  # 30 yaşında bir müşteri
    'income': [60000],  # Geliri 60000
    'spending_score': [75]  # Harcama skoru 75
})

# Veriyi normalleştirelim
new_customer_scaled = scaler.transform(new_customer)  # Yeni müşteri verisi normalleştiriliyor

# RandomForest ile tahmin yapalım
rf_prediction = rf_model.predict(new_customer_scaled)  # Random Forest modeli ile tahmin yapılıyor
print(f"RandomForest ile tahmin edilen memnuniyet: {rf_prediction[0]}")  # Tahmin sonucu yazdırılıyor

# Logistic Regression ile tahmin yapalım
logistic_prediction = logistic_model.predict(new_customer_scaled)  # Logistic Regression ile tahmin yapılıyor
print(f"Logistic Regression ile tahmin edilen memnuniyet: {logistic_prediction[0]}")  # Tahmin sonucu yazdırılıyor

# Derin öğrenme ile tahmin yapalım
deep_prediction = deep_model.predict(new_customer_scaled)  # Derin öğrenme modeli ile tahmin yapılıyor
print(f"Derin Öğrenme ile tahmin edilen memnuniyet: {deep_prediction[0][0]}")  # Tahmin sonucu yazdırılıyor
