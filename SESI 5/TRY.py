import cv2
import matplotlib.pyplot as plt


# 1. Persiapan dan Pembacaan Citra
# Pertama, kita akan membaca citra menggunakan OpenCV dan mengkonversi citra berwarna menjadi grayscale.

  
# Membaca citra berwarna
image_color = cv2.imread('Pudel.jpg')
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB untuk plotting

# Mengkonversi citra berwarna menjadi grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

# Tampilkan citra asli (berwarna dan grayscale)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_color)
plt.title('Citra Berwarna')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.title('Citra Grayscale')
plt.axis('off')
plt.show()


# 2. Low-Pass Filter (Blur)
# Low-pass filter digunakan untuk menghaluskan citra dengan mengurangi noise dan detail halus.


# Low-pass filter untuk citra grayscale
blurred_gray = cv2.GaussianBlur(image_gray, (9, 9), 0)

# Low-pass filter untuk citra berwarna
blurred_color = cv2.GaussianBlur(image_color, (9, 9), 0)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(blurred_gray, cmap='gray')
plt.title('Citra Grayscale - Low Pass (Blur)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_color)
plt.title('Citra Berwarna - Low Pass (Blur)')
plt.axis('off')
plt.show()


# 3. High-Pass Filter (Edge Detection)
# High-pass filter digunakan untuk mendeteksi tepi dengan menonjolkan perbedaan intensitas.


# High-pass filter (Sobel) untuk citra grayscale
sobel_gray = cv2.Sobel(image_gray, cv2.CV_64F, 1, 1, ksize=5)
sobel_gray = cv2.convertScaleAbs(sobel_gray)  # Konversi ke uint8

# High-pass filter (Sobel) untuk citra berwarna - diterapkan pada setiap channel
sobel_color = image_color.copy()
for i in range(3):  # Untuk setiap channel warna
    sobel_color[:, :, i] = cv2.Sobel(image_color[:, :, i], cv2.CV_64F, 1, 1, ksize=5)
sobel_color = cv2.convertScaleAbs(sobel_color)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sobel_gray, cmap='gray')
plt.title('Citra Grayscale - High Pass (Edge Detection)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_color)
plt.title('Citra Berwarna - High Pass (Edge Detection)')
plt.axis('off')
plt.show()


# 4. High-Boost Filter (Enhancement)
# High-boost filter meningkatkan ketajaman dengan menggabungkan citra asli dengan citra hasil high-pass.


# Parameter boosting
boost_factor = 1.5

# High-boost filter untuk citra grayscale
blurred_gray = cv2.GaussianBlur(image_gray, (9, 9), 0)
high_boost_gray = cv2.addWeighted(image_gray, boost_factor, blurred_gray, -0.5, 0)

# High-boost filter untuk citra berwarna
blurred_color = cv2.GaussianBlur(image_color, (9, 9), 0)
high_boost_color = cv2.addWeighted(image_color, boost_factor, blurred_color, -0.5, 0)

# Tampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(high_boost_gray, cmap='gray')
plt.title('Citra Grayscale - High Boost')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(high_boost_color)
plt.title('Citra Berwarna - High Boost')
plt.axis('off')
plt.show()