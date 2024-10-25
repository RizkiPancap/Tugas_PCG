import numpy as np
import imageio
import matplotlib.pyplot as plt

# Membaca gambar
img = imageio.imread('daun_pepaya.png')

# Fungsi untuk konversi grayscale
def convert_to_grayscale(image):
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale.astype(np.uint8)

# Fungsi untuk konversi biner (thresholding)
def convert_to_binary(image, threshold=128):
    grayscale = convert_to_grayscale(image)
    binary = np.where(grayscale > threshold, 255, 0)
    return binary

# Konversi gambar ke biner
binary_img = convert_to_binary(img, threshold=128)

# Tampilkan gambar biner
plt.imshow(binary_img, cmap='gray')
plt.title('Binary Image (Threshold)')
plt.axis('off')
plt.show()
