import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Membaca citra dan mengonversi menjadi grayscale
image_path = '/content/Pudel.jpg'
image = iio.imread(image_path, pilmode='L') / 255.0  # Normalize to [0, 1] range

# Ekualisasi Histogram
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 1])
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize to [0,1]
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf)
    return img_equalized.reshape(img.shape)

# Filter Low-Pass (Gaussian Blur)
blurred_image = ndimage.gaussian_filter(image, sigma=2)

# Filter High-Pass (Sobel Filter for Edge Detection)
sobel_image_x = ndimage.sobel(image, axis=0)
sobel_image_y = ndimage.sobel(image, axis=1)
sobel_image = np.hypot(sobel_image_x, sobel_image_y)

# Filter High-Boost
boost_factor = 1.5
high_boost_image = image + boost_factor * (image - blurred_image)
high_boost_image = np.clip(high_boost_image, 0, 1)  # Ensure within valid range

# Ekualisasi histogram pada citra asli
image_equalized = histogram_equalization(image)

# Tampilkan hasil
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Citra Asli')
axes[0].axis('off')

axes[1].imshow(blurred_image, cmap='gray')
axes[1].set_title('Low Pass (Gaussian Blur)')
axes[1].axis('off')

axes[2].imshow(sobel_image, cmap='gray')
axes[2].set_title('High Pass (Sobel)')
axes[2].axis('off')

axes[3].imshow(high_boost_image, cmap='gray')
axes[3].set_title('High Boost Filter')
axes[3].axis('off')

axes[4].imshow(image_equalized, cmap='gray')
axes[4].set_title('Ekualisasi Histogram')
axes[4].axis('off')

# Hide the last subplot as it
