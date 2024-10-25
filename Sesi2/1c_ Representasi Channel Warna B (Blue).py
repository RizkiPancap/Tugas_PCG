import numpy as np
import imageio
import matplotlib.pyplot as plt

# Membaca gambar
img = imageio.imread('daun_pepaya.png')

# Fungsi untuk mendapatkan channel warna B (Blue)
def get_blue_channel(image):
    B = image.copy()
    B[:, :, 0] = 0  # Set Red to 0
    B[:, :, 1] = 0  # Set Green to 0
    return B

# Dapatkan channel warna B
blue_channel = get_blue_channel(img)

# Tampilkan channel warna biru
plt.imshow(blue_channel)
plt.title('Blue Channel')
plt.axis('off')
plt.show()
