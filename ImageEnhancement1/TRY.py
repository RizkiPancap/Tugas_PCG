import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Membaca citra
image_path = 'Pudel.jpg'
image = imageio.imread(image_path, as_gray=True)

# Ekualisasi Histogram
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 1])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_equalized = cdf[img.astype('uint8')]
    return img_equalized

# Filter Low-Pass (Gaussian Blur)
blurred_image = ndimage.gaussian_filter(image, sigma=2)

# Filter High-Pass (Sobel Filter for Edge Detection)
sobel_image = ndimage.sobel(image)

# Filter High-Boost
boost_factor = 1.5
high_boost_image = image + boost_factor * (image - blurred_image)

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

plt.tight_layout()
plt.show()