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

def histogram_specification(source_img, reference_hist):
    # Βήμα 1: Υπολογισμός ιστογράμματος της εισόδου
    src_hist, _ = np.histogram(source_img.flatten(), bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]  # Κανονικοποίηση στο [0, 1]

    # Βήμα 2: Υπολογισμός CDF της reference κατανομής
    ref_cdf = np.cumsum(reference_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Βήμα 3: Δημιουργία mapping s_k -> z_q μέσω πλησιέστερης τιμής CDF
    mapping = np.interp(src_cdf, ref_cdf, np.arange(256))

    # Βήμα 4: Εφαρμογή mapping στα pixels της εικόνας
    matched = np.interp(source_img.flatten(), np.arange(256), mapping)
    return matched.reshape(source_img.shape).astype(np.uint8)

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

input_img = distorted_images["Distorted Image #4"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - input_img) ** 2)
psnr_distorted = psnr(target, input_img)
ssim_distorted = ssim(target, input_img)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)

distorted_hist , _ = np.histogram(input_img.flatten(), bins=256, range=(0, 256))
target_hist , _ = np.histogram(target.flatten(), bins=256, range=(0, 256))
restored = histogram_specification(input_img,target_hist)


# === Εφαρμογή τεχνικών αποκατάστασης ===



restored_hist , _ = np.histogram(restored.flatten(), bins=256, range=(0, 256))

plt.figure(figsize=(10, 4))
plt.plot(distorted_hist, label='Distorted Image', color='gray')
plt.plot(restored_hist, label='Restored Image', color='green', linestyle='--')
plt.plot(target_hist, label='Original Lena', color='red', linestyle=':')
plt.title("Overlay of Histograms")
plt.xlabel("Intensity")
plt.ylabel("Normalized Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# show_image("equalized",restored)
#
#
#
# #
# # # === Οπτική αξιολόγηση ===
# # show_image("Distorted Image #4", input_img)
# # show_image("Restored Image", restored)
# #
# # # === Ποσοτική αξιολόγηση μετά την αποκατάσταση (με χρήση της lena.jpg ως ground truth) ===
# mse_score = np.mean((target - restored) ** 2)
# psnr_score = psnr(target, restored)
# ssim_score = ssim(target, restored)
# #
# print("MSE:", mse_score)
# print("PSNR:", psnr_score)
# print("SSIM:", ssim_score)