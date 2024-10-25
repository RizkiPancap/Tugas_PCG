import numpy as np
import imageio
import matplotlib.pyplot as plt

# Membaca gambar
img = imageio.imread('daun_pepaya.png')

# Fungsi untuk konversi grayscale
def convert_to_grayscale(image):
    # Konversi menggunakan rumus standar grayscale
    grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return grayscale.astype(np.uint8)

# Konversi gambar ke grayscale
grayscale_img = convert_to_grayscale(img)

# Tampilkan gambar grayscale
plt.imshow(grayscale_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()
