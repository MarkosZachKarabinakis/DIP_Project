import cv2
import numpy as np
import matplotlib
import scipy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
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

## Αποκατάσταση: Distorted Image #2

input_img = distorted_images["Distorted Image #2"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - input_img) ** 2)
psnr_distorted = psnr(target, input_img)
ssim_distorted = ssim(target, input_img)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)
print("--------------------")

# === Εφαρμογή τεχνικών αποκατάστασης ===

# παρατηρουμε οτι ειναι αρκετα θολωμενη οποτε δοκιμαζουμε unsharp masking

# blurred img
blurred = cv2.GaussianBlur(input_img, (7, 7),0)
show_image("Blurred Image", blurred)

mask = np.subtract(input_img,blurred)
show_image("Mask", mask)
restored = input_img + 3*mask

# dokimasa me diafora k size 3,3  11,11 kai diafora k * mask , o parapano syndyasmos itan o kalyteros poy brika

show_image("Restored", restored)


# === Ποσοτική αξιολόγηση μετά την αποκατάσταση (με χρήση της lena.jpg ως ground truth) ===
mse_score = np.mean((target - restored) ** 2)
psnr_score = psnr(target, restored)
ssim_score = ssim(target, restored)

print("MSE:", mse_score)
print("PSNR:", psnr_score)
print("SSIM:", ssim_score)