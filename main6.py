import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
matplotlib.use('TkAgg')

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def show_image(title, img):
    plt.figure()
    # αυτόματο scaling στο [0,1] ή [0,255] με βάση το dtype κάθε εικόνας
    if img.dtype == np.uint8:
        vmin, vmax = 0, 255
    else:
        vmin, vmax = float(np.min(img)), float(np.max(img))

    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Φόρτωση εικόνας στόχου
target = cv2.imread('images/lena.png', cv2.IMREAD_GRAYSCALE)

# Φόρτωση παραμορφωμένων εικόνων
distorted_images = {
    "Distorted Image #1": cv2.imread("images/distorted_img_1.png", cv2.IMREAD_GRAYSCALE),
    "Distorted Image #2": cv2.imread("images/distorted_img_2.png", cv2.IMREAD_GRAYSCALE),
    "Distorted Image #3": cv2.imread("images/distorted_img_3.png", cv2.IMREAD_GRAYSCALE),
    "Distorted Image #4": cv2.imread("images/distorted_img_4.png", cv2.IMREAD_GRAYSCALE),
    "Distorted Image #5": cv2.imread("images/distorted_img_5.png", cv2.IMREAD_GRAYSCALE),
    "Distorted Image #6": cv2.imread("images/distorted_img_6.png", cv2.IMREAD_GRAYSCALE),
    "Distorted Image #7": cv2.imread("images/distorted_img_7.png", cv2.IMREAD_GRAYSCALE),
}


distorted_img_1 = distorted_images["Distorted Image #6"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - distorted_img_1) ** 2)
psnr_distorted = psnr(target, distorted_img_1)
ssim_distorted = ssim(target, distorted_img_1)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)

restored = cv2.GaussianBlur(distorted_img_1,[5,5],0)



show_image("Gaussian Blurr Output", restored)

print("After Restoration:")
mse_score = np.mean((target - restored) ** 2)
psnr_score = psnr(target, restored)
ssim_score = ssim(target, restored)
#
print("MSE:", mse_score)
print("PSNR:", psnr_score)
print("SSIM:", ssim_score)