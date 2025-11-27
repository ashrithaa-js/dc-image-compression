import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Read grayscale image
image_path = r"A:\PSG TECH\Package\SEMESTER 5\DC\dataset\SOCOFing\SOCOFing\Real\1__M_Left_index_finger.bmp"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

img = img.astype(np.float64)
h, w = img.shape

# 2. Center the image (PCA preprocessing)
mean_vector = np.zeros(w)
X_centered = np.zeros_like(img)

# Compute mean of each column
for col in range(w):
    sum_col = 0
    for row in range(h):
        sum_col += img[row, col]
    mean_vector[col] = sum_col / h

# Subtract mean from each pixel
for row in range(h):
    for col in range(w):
        X_centered[row, col] = img[row, col] - mean_vector[col]

# 3. Covariance matrix computation
cov_matrix = np.zeros((w, w))
for i in range(w):
    for j in range(w):
        cov_sum = 0
        for row in range(h):
            cov_sum += X_centered[row, i] * X_centered[row, j]
        cov_matrix[i, j] = cov_sum / (h - 1)

# 4. Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigvals)[::-1]

sorted_eigvals = np.zeros_like(eigvals)
sorted_eigvecs = np.zeros_like(eigvecs)
for i, idx_val in enumerate(sorted_indices):
    sorted_eigvals[i] = eigvals[idx_val]
    sorted_eigvecs[:, i] = eigvecs[:, idx_val]

# 5. Choose top-k components (analogous to thresholds in DWT)
top_ks = [5, 10, 20, 50]

plt.figure(figsize=(18, 6))
plt.subplot(1, len(top_ks)+2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

print("PCA Compression Results:\n")

# 6. Compression and Reconstruction
for i, k in enumerate(top_ks, start=2):
    # Select top-k eigenvectors
    top_eigvecs = sorted_eigvecs[:, :k]

    # Project image onto top-k eigenvectors
    projected = np.zeros((h, k))
    for row in range(h):
        for col in range(k):
            projected[row, col] = np.dot(X_centered[row, :], top_eigvecs[:, col])

    # Reconstruct image from projection
    reconstructed = np.zeros_like(img)
    for row in range(h):
        for col in range(w):
            value = 0
            for l in range(k):
                value += projected[row, l] * top_eigvecs[col, l]
            reconstructed[row, col] = value + mean_vector[col]

    # Clip values to valid pixel range
    reconstructed = np.clip(reconstructed, 0, 255)
    reconstructed = np.uint8(reconstructed)

    # 7. Compute compression metrics
    non_zero = h*k + k*w  # approximate storage size
    compression_ratio = img.size / non_zero
    data_reduction = 100 - (img.size / non_zero) * 100

    mse = 0
    for row in range(h):
        for col in range(w):
            mse += (img[row, col] - reconstructed[row, col])**2
    mse /= (h * w)

    psnr = 10 * np.log10(255*255 / mse)

    print(f"Top-{k} Components:")
    print(f"  Compression ratio = {compression_ratio:.2f}")
    print(f"  Data reduction    = {data_reduction:.2f}%")
    print(f"  PSNR              = {psnr:.2f} dB\n")

    # 8. Display reconstructed image
    plt.subplot(1, len(top_ks)+2, i)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f"k={k}\nCR={compression_ratio:.2f}\nPSNR={psnr:.2f}")
    plt.axis('off')

# 9. Show approximation block using last top-k
approx_block = np.zeros((k, h))
for i in range(k):
    for j in range(h):
        approx_block[i, j] = np.dot(top_eigvecs[:, i], X_centered[j, :])

approx_block = np.clip(approx_block.T, 0, 255)
approx_block = np.uint8(approx_block)

plt.subplot(1, len(top_ks)+2, len(top_ks)+2)
plt.imshow(approx_block, cmap='gray')
plt.title(f"Approximation\n(top-{top_ks[-1]} components)")
plt.axis('off')

plt.tight_layout()
plt.show()
