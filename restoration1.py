import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

# Επιβεβαίωση μεγέθους εικόνων
print("Target:", target.shape)
for k, img in distorted_images.items():
    print(k, "->", img.shape)


## Αποκατάσταση: Distorted Image #1

input_img = distorted_images["Distorted Image #1"].copy()

# === Πριν την αποκατάσταση: Αξιολόγηση "distorted" ===
mse_distorted = np.mean((target - input_img) ** 2)
psnr_distorted = psnr(target, input_img)
ssim_distorted = ssim(target, input_img)
print("Before Restoration:")
print("MSE:", mse_distorted)
print("PSNR:", psnr_distorted)
print("SSIM:", ssim_distorted)


###################################################################################################################################

# === Εφαρμογή τεχνικών αποκατάστασης ===







## Προκειται για εικονα που εχει distortion -> contrast stretch .
## Για να το διορθωσουμε θα κανουμε log / power trasnformations .
## Αρχικα ομως πρεπει να δουμε το ιστογραμμα της distorted εικονας μας

#calculate histogram
counts, bins = np.histogram(input_img, range(257))
# plot histogram centered on values 0..255
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.xlim([-0.5, 255.5])
plt.show()


# calculate histogram
# counts, bins = np.histogram(target, range(257))
# # plot histogram centered on values 0..255
# plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
# plt.xlim([-0.5, 255.5])
# plt.show()

# Αναθεωρουμε και παμε για histogram matching


# Το παρακάτω block σας βοηθά να αντιμετωπίσετε την περίπτωση παραμόρφωσης με ανεπαρκή ισοστάθμιση ιστογράμματος.
# Μην το εκτελέσετε όπως είναι — χρησιμοποιήστε το ως οδηγό και προσαρμόστε το σωστά στο κελί που αντιστοιχεί
# στο κατάλληλο distortion.

# === 1. Υπολογισμός reference ιστογράμματος από την αρχική εικόνα ===
# Χρησιμοποιείται ως "στόχος" για την ανακατανομή των εντάσεων
hist_ref, _ = np.histogram(target.flatten(), bins=256, range=(0, 256))
hist_ref = hist_ref.astype(np.float64) / hist_ref.sum()  # Κανονικοποίηση ώστε να αναπαριστά πιθανότητες

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

# === 3. Παράδειγμα χρήσης (συμπληρώστε με τον σωστό αριθμό εικόνας) ===
input_img = distorted_images["Distorted Image #1"].copy()
restored = histogram_specification(input_img, hist_ref)

# === 4. Οπτική αξιολόγηση (προαιρετικά) ===
show_image("Distorted Image #1", input_img)
show_image("Restored Image", restored)

# === 5. Συγκριτική απεικόνιση ιστογραμμάτων (προαιρετικά) ===
hist_input, _ = np.histogram(input_img.flatten(), bins=256, range=(0, 256))
hist_input = hist_input.astype(np.float64) / hist_input.sum()

hist_restored, _ = np.histogram(restored.flatten(), bins=256, range=(0, 256))
hist_restored = hist_restored.astype(np.float64) / hist_restored.sum()

hist_target, _ = np.histogram(target.flatten(), bins=256, range=(0, 256))
hist_target = hist_target.astype(np.float64) / hist_target.sum()

plt.figure(figsize=(10, 4))
plt.plot(hist_input, label='Distorted Image', color='gray')
plt.plot(hist_restored, label='Restored Image', color='green', linestyle='--')
plt.plot(hist_target, label='Original Lena', color='red', linestyle=':')
plt.title("Overlay of Histograms")
plt.xlabel("Intensity")
plt.ylabel("Normalized Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






# restored = ...


###################################################################################################################################
# === Οπτική αξιολόγηση ===
# show_image("Distorted Image #1", input_img)
# show_image("Restored Image", restored)

# === Ποσοτική αξιολόγηση μετά την αποκατάσταση (με χρήση της lena.jpg ως ground truth) ===
# mse_score = np.mean((target - restored) ** 2)
# psnr_score = psnr(target, restored)
# ssim_score = ssim(target, restored)

# print("MSE:", mse_score)
# print("PSNR:", psnr_score)
# print("SSIM:", ssim_score)