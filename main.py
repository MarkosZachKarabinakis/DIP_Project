import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

# Επιβεβαίωση μεγέθους εικόνων
print("Target:", target.shape)
for k, img in distorted_images.items():
    print(k, "->", img.shape)

# Εμφάνιση εικόνας στόχου
plt.figure()
plt.imshow(target, cmap='gray')
plt.title("Original-target image")
plt.axis('off')
plt.show()

# Εμφάνιση όλων των παραμορφώσεων
# for title, img in distorted_images.items():
#     show_image(title, img)


## Αποκατάσταση: Distorted Image #1

distorted_img_1 = distorted_images["Distorted Image #1"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - distorted_img_1) ** 2)
psnr_distorted = psnr(target, distorted_img_1)
ssim_distorted = ssim(target, distorted_img_1)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)

# === Εφαρμογή τεχνικών αποκατάστασης ===

## Θα αρχισουμε παρατηρωντας το ιστογραμμα των εικονων μας
target_hist, _ = np.histogram(target.flatten(), bins=256, range=(0, 256))
distorted_img_1_hist , _ = np.histogram(distorted_img_1.flatten(), bins=256, range=(0, 256))


plt.figure(figsize=(10, 4))
plt.plot(distorted_img_1_hist, label='Distorted Image', color='gray')
# plt.plot(hist_restored, label='Restored Image', color='green', linestyle='--')
plt.plot(target_hist, label='Original Lena', color='red', linestyle=':')
plt.title("Overlay of Histograms")
plt.xlabel("Intensity")
plt.ylabel("Normalized Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


## παρατηρουμε οτι εχουμε high contrast

## πρωτο attempt , δεν ειναι σωστο
#restored = np.log1p(distorted_img_1)

## δευτερο attempt , histogram matching / specification


# === 2. Συνάρτηση: Καθορισμός Ιστογράμματος μέσω CDF Matching ===
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


restored = histogram_specification(distorted_img_1, target_hist)

plt.figure()
plt.imshow(restored, cmap='gray')
plt.title("Restored image 1")
plt.axis('off')
plt.show()


# restored = ...

# === Οπτική αξιολόγηση ===
show_image("Distorted Image #1", distorted_img_1)
show_image("Restored Image", restored)

# === Ποσοτική αξιολόγηση μετά την αποκατάσταση (με χρήση της lena.jpg ως ground truth) ===
mse_score = np.mean((target - restored) ** 2)
psnr_score = psnr(target, restored)
ssim_score = ssim(target, restored)

print("MSE:", mse_score)
print("PSNR:", psnr_score)
print("SSIM:", ssim_score)