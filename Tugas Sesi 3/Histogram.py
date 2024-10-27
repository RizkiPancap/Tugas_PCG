import gdown
import numpy as np
import imageio
import matplotlib.pyplot as plt

# Masukkan ID file dari Google Drive
file_id = '1fHaLYdnilJlaImhUy2PNbhfrDJvkyOYU'  # Ganti dengan ID file Anda
url = f'https://drive.google.com/uc?id={file_id}'

# Nama file output
output = 'Bike.png'

# Unduh file dari Google Drive
gdown.download(url, output, quiet=False)

# Membaca gambar yang diunduh
img = imageio.imread(output)

# Mengonversi gambar RGB ke Grayscale
def convert_to_grayscale(image):
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale.astype(np.uint8)

# Menghitung histogram
def calculate_histogram(grayscale_img):
    histogram, bins = np.histogram(grayscale_img, bins=256, range=(0, 255))
    return histogram

# Konversi gambar ke grayscale
grayscale_img = convert_to_grayscale(img)

# Hitung histogram dari gambar grayscale
histogram = calculate_histogram(grayscale_img)

# Tampilkan histogram menggunakan Matplotlib
plt.figure(figsize=(10, 6))
plt.bar(range(256), histogram, width=1, edgecolor='black')
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Number of Pixels')
plt.show()

# Jumlah total piksel untuk setiap intensitas (0-255)
for i, count in enumerate(histogram):
    print(f"Intensitas {i}: {count} piksel")