### **PERSIAPAN DATA**

import os
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use pd.read_csv to read the CSV file
df = pd.read_csv('Curah Hujan 2020-2024.csv')
print(df.head())

# Cek informasi umum dari DataFrame (tipe data, non-null count, dsb)
print(df.info())

# Cek kolom yang tersedia
print(df.columns)

# Cek ringkasan statistik dari DataFrame
print(df.describe())

df = pd.read_csv('Curah Hujan 2020-2024.csv')
df.head()


# Cek apakah ada data kosong
print("Jumlah data kosong per kolom:")
print(df.isnull().sum())

# Visualisasi jumlah data kosong dengan heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Visualisasi Data Kosong')
plt.show()

df['RR'] = pd.to_numeric(df['RR'], errors='coerce')

# Cek apakah ada data kosong setelah konversi RR
print("Jumlah data kosong di kolom RR setelah konversi:")
print(df.isnull().sum())

# Menangani nilai yang hilang (misalnya, dengan interpolation atau menghapus)
df.fillna(method='ffill', inplace=True)  # mengisi nilai kosong dengan nilai sebelumnya

df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')

# Tambahkan kolom bulan dan tahun untuk agregasi
df['Bulan'] = df['Tanggal'].dt.month
df['Tahun'] = df['Tanggal'].dt.year
print(df[['Tanggal', 'Bulan', 'Tahun']].head())

df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
df.fillna(method='ffill', inplace=True)
df_rain = df.groupby(['Tahun', 'Bulan'])['RR'].mean().reset_index()
df_pivot = df_rain.pivot(index='Tahun', columns='Bulan', values='RR')
df_pivot = df_pivot.fillna(0)
print(df_pivot)

### **PEMILIHAN DATA**

print(df.columns)
print(df.info())

# Ubah kolom tanggal ke format datetime
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
print(df['Tanggal'].head())

# Tambahkan kolom bulan dan tahun untuk agregasi
df['Bulan'] = df['Tanggal'].dt.month
df['Tahun'] = df['Tanggal'].dt.year
print(df[['Tanggal', 'Bulan', 'Tahun']].head())

# Cek apakah ada data kosong setelah memperbarui DataFrame
print("Jumlah data kosong per kolom:")
print(df.isnull().sum())

# Gunakan errors='coerce' untuk mengubah nilai non-numerik menjadi NaN
df['RR'] = pd.to_numeric(df['RR'], errors='coerce')

# Cek apakah ada data kosong setelah konversi RR
print("Jumlah data kosong di kolom RR setelah konversi:")
print(df.isnull().sum())

# Agregasi rata-rata curah hujan bulanan per tahun
df_rain = df.groupby(['Tahun', 'Bulan'])['RR'].mean().reset_index()
df_pivot = df_rain.pivot(index='Tahun', columns='Bulan', values='RR')
df_pivot = df_pivot.fillna(0)  # Isi NaN dengan nol
print(df_pivot)

### **MODEL**

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_samples
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

# Skala data sebelum menerapkan KMeans
scaler = StandardScaler()
# Cek apakah kolom 'Cluster' ada di df_pivot sebelum menghapusnya
if 'Cluster' in df_pivot.columns:
    df_scaled = scaler.fit_transform(df_pivot.drop('Cluster', axis=1))
else:
    # Jika 'Cluster' belum ada, scaling seluruh df_pivot (ini kasus sebelum clustering pertama kali)
    df_scaled = scaler.fit_transform(df_pivot)
print(df_scaled.shape)

# Menentukan WCSS (Within-Cluster Sum of Squares) untuk berbagai jumlah cluster
wcss = []
max_clusters = min(11, df_scaled.shape[0] + 1)
for i in range(1, max_clusters):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)

# Plot Elbow Method untuk menentukan jumlah cluster yang optimal
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_clusters), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Misalkan kita pilih 3 cluster (dari hasil metode elbow yang telah dianalisis)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')

# Perform clustering pada data yang telah diskalakan
cluster_labels = kmeans.fit_predict(df_scaled)

# Tambahkan label cluster sebagai kolom baru di df_pivot
df_pivot['Cluster'] = cluster_labels
print(df_pivot[['Cluster']])

# Hitung rata-rata curah hujan per bulan di setiap cluster
cluster_summary = df_pivot.groupby('Cluster').mean()
print(cluster_summary)

from sklearn.metrics import silhouette_score
# Hitung dan tampilkan Silhouette Score untuk mengevaluasi kualitas clustering
score = silhouette_score(df_scaled, cluster_labels)
print(f'Silhouette Score: {score}')

from sklearn.metrics import silhouette_score

score = silhouette_score(df_scaled, cluster_labels)
print(f'Silhouette Score: {score}')


# Plot rata-rata curah hujan per bulan untuk masing-masing cluster
cluster_summary.T.plot(kind='bar', figsize=(12, 6))
plt.title('Rata-rata Curah Hujan per Bulan untuk Setiap Cluster')
plt.xlabel('Bulan')
plt.ylabel('Curah Hujan (mm)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Mengurangi dimensi data menjadi 2D untuk visualisasi
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Buat DataFrame untuk hasil PCA
df_pca_df = pd.DataFrame(data=df_pca, columns=['Komponen 1', 'Komponen 2'])
df_pca_df['Cluster'] = cluster_labels

# Visualisasi dengan scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca_df, x='Komponen 1', y='Komponen 2', hue='Cluster', palette='Set1', s=100)
plt.title('Visualisasi Hasil Clustering K-Means')
plt.xlabel('Komponen 1')
plt.ylabel('Komponen 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Visualisasi rata-rata curah hujan per bulan untuk setiap cluster menggunakan heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='YlGnBu', fmt=".1f")
plt.title('Heatmap Rata-rata Curah Hujan per Bulan untuk Setiap Cluster')
plt.xlabel('Cluster')
plt.ylabel('Bulan')
plt.show()

# Hitung silhouette score untuk setiap titik data
silhouette_vals = silhouette_samples(df_scaled, cluster_labels)
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(3):  # Gantilah 3 dengan jumlah cluster yang Anda pilih
    # Ambil silhouette score untuk cluster i
    ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    ith_cluster_silhouette_vals.sort()

    # Hitung posisi y untuk cluster
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals)

    y_lower = y_upper + 10

plt.title('Silhouette Plot untuk K-Means Clustering')
plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster')
plt.axvline(x=0, linestyle='--', color='red')  # Garis referensi
plt.show()

df_combined = df_pivot.reset_index()
df_combined['Cluster'] = cluster_labels

sns.pairplot(df_combined, hue='Cluster', palette='Set1')
plt.title('Pair Plot berdasarkan Cluster')
plt.show()
