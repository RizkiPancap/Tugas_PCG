import numpy as np
import imageio
import matplotlib.pyplot as plt

# Membaca gambar
img = imageio.imread('daun_pepaya.png')

# Fungsi untuk mendapatkan channel warna G (Green)
def get_green_channel(image):
    G = image.copy()
    G[:, :, 0] = 0  # Set Red to 0
    G[:, :, 2] = 0  # Set Blue to 0
    return G

# Dapatkan channel warna G
green_channel = get_green_channel(img)

# Tampilkan channel warna hijau
plt.imshow(green_channel)
plt.title('Green Channel')
plt.axis('off')
plt.show()
