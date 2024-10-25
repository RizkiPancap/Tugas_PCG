import numpy as np
import imageio
import matplotlib.pyplot as plt

# Membaca gambar
img = imageio.imread('D:\Kampus\SEMESTER 5\Pengolahan Citra Digital\Sesi2\daun_pepaya.png')

# Fungsi untuk mendapatkan channel warna R (Red)
def get_red_channel(image):
    R = image.copy()
    R[:, :, 1] = 0  # Set Green to 0
    R[:, :, 2] = 0  # Set Blue to 0
    return R

# Dapatkan channel warna R
red_channel = get_red_channel(img)

# Tampilkan channel warna merah
plt.imshow(red_channel)
plt.title('Red Channel')
plt.axis('off')
plt.show()
