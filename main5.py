import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('TkAgg')


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



## Αποκατάσταση: Distorted Image #4

input_img = distorted_images["Distorted Image #5"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - input_img) ** 2)
psnr_distorted = psnr(target, input_img)
ssim_distorted = ssim(target, input_img)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)



target_hist , _ = np.histogram(target.flatten(), bins=256, range=(0, 256))
distorted_hist , _ = np.histogram(input_img.flatten(), bins=256, range=(0, 256))


plt.figure(figsize=(10, 4))
plt.plot(distorted_hist, label='Distorted Image', color='gray')
# plt.plot(hist_restored, label='Restored Image', color='green', linestyle='--')
plt.plot(target_hist, label='Original Lena', color='red', linestyle=':')
plt.title("Overlay of Histograms")
plt.xlabel("Intensity")
plt.ylabel("Normalized Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


restored = cv2.equalizeHist(input_img)
show_image("equalized",restored)