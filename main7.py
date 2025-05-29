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

## Αποκατάσταση: Distorted Image #7

input_img = distorted_images["Distorted Image #7"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - input_img) ** 2)
psnr_distorted = psnr(target, input_img)
ssim_distorted = ssim(target, input_img)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)


# === Εφαρμογή τεχνικών αποκατάστασης ===
# Για την εικονα 7 , παρατηρουμε παλι οτι εχουμε θορυβο , συγκεκριμενα salt & pepper
# Απο θεωρια ξερουμε οτι μια καλη αντιμετωπιση για αυτον τον θορυβο ειναι το median blurr
# Defining median blurr
def simple_median_blurr(image, window_size=3):  # βαζουμε default window size 3
    pad = window_size // 2

    # pad με αντιγραφη των edges ουσιαστικα
    padded = np.pad(image, pad, mode='edge')

    # νεα εικονα για να μην επηρεαζονται οι τιμες οσο κανουμε την διαδικασια
    result = np.zeros_like(image)

    # για καθε πιξελ , βρισκουμε το window που αντιστοιχει σε αυτο στη νεα padded εικονα βαση
    # και του window size που ορισαμε , και κανουμε στο window αυτο median
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:window_size + i, j:window_size + j]
            result[i, j] = np.median(window)


restored = simple_median_blurr(input_img)

# === Οπτική αξιολόγηση ===
show_image("Distorted Image #7", input_img)
show_image("Restored Image", restored)

# === Ποσοτική αξιολόγηση μετά την αποκατάσταση (με χρήση της lena.jpg ως ground truth) ===
mse_score = np.mean((target - restored) ** 2)
psnr_score = psnr(target, restored)
ssim_score = ssim(target, restored)

print("MSE:", mse_score)
print("PSNR:", psnr_score)
print("SSIM:", ssim_score)