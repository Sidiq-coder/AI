import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Baca dataset dari file lokal atau URL
url = 'https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv'
df = pd.read_csv(url)

# 2. Tampilkan kolom-kolom
print("Kolom-kolom dataset:", df.columns.tolist())

# 3. Pisahkan fitur dan label
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Normalisasi fitur (penting untuk k-NN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 6. Buat dan latih model k-NN
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# 7. Prediksi
y_pred = model.predict(X_test)

# 8. Evaluasi
print(f"\nEvaluasi Model k-NN (k={k}):")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (k={k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
