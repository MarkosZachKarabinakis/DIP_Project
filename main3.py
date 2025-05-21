import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.signal import convolve2d
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

## Αποκατάσταση: Distorted Image #3

input_img = distorted_images["Distorted Image #3"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - input_img) ** 2)
psnr_distorted = psnr(target, input_img)
ssim_distorted = ssim(target, input_img)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)
print("-------------------")

# === Εφαρμογή τεχνικών αποκατάστασης ===

# θα κανουμε laplacian sharpening
laplacian_kernel = np.array([
    [-1, -1,  -1],
    [-1, 9, -1],
    [-1, -1,  -1]
])

output = convolve2d(input_img, laplacian_kernel, mode='same')

restored = np.clip(output, 0, 255) # gia ta wrap around effect

target = target.astype(np.uint8) # an einai float , mpainoyn kai arnitikoi arithmoi kai 'xalane' tous deiktes mas , px ton MSE
restored = restored.astype(np.uint8) # den allazi kati ousiastiko stin eikona mas
show_image("Convolution Output", restored)

# # Apply gamma correction
# gamma = 0.8  # < 1 brightens, > 1 darkens
# img_gamma = np.power(output, gamma)
# to parapano to ebgala giati den bgainei oreo


# show_image("Brightened Output", img_gamma)

output_hist,_ = np.histogram(output.flatten(), bins=256, range=(0, 256))
# gamma_hist , _ = np.histogram(img_gamma.flatten(), bins=256, range=(0, 256))
target_hist,_ = np.histogram(target.flatten(), bins=256, range=(0, 256))


plt.figure(figsize=(10, 4))
plt.plot(output_hist, label='Processed Img', color='gray')
# plt.plot(gamma_hist, label='Gamma', color='blue')
plt.plot(target_hist, label='Original Lena', color='red', linestyle=':')
plt.title("Overlay of Histograms")
plt.xlabel("Intensity")
plt.ylabel("Normalized Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# restored = ...

# show_image("Restored Image", restored)

# === Ποσοτική αξιολόγηση μετά την αποκατάσταση (με χρήση της lena.jpg ως ground truth) ===

# metatropi

mse_score = np.mean((target - restored) ** 2)
psnr_score = psnr(target, restored)
ssim_score = ssim(target, restored)

print("MSE:", mse_score)
print("PSNR:", psnr_score)
print("SSIM:", ssim_score)
print("---------------")


