import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Read grayscale image
image_path = r"A:\PSG TECH\Package\SEMESTER 5\DC\dataset\SOCOFing\SOCOFing\Real\1__M_Left_index_finger.bmp"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

img = img.astype(np.float64)
h, w = img.shape

# Pad to make even dimensions
if h % 2 != 0:
    img = np.pad(img, ((0, 1), (0, 0)), mode='symmetric')
if w % 2 != 0:
    img = np.pad(img, ((0, 0), (0, 1)), mode='symmetric')


#1-Level Haar DWT
def dwt2D(matrix):
    rows, cols = matrix.shape
    output = np.zeros_like(matrix)
    temp = np.zeros_like(matrix)

    # Row transform
    for i in range(rows):
        for j in range(0, cols, 2):
            avg = (matrix[i, j] + matrix[i, j+1]) / np.sqrt(2)
            diff = (matrix[i, j] - matrix[i, j+1]) / np.sqrt(2)
            temp[i, j//2] = avg
            temp[i, j//2 + cols//2] = diff

    # Column transform
    for j in range(cols):
        for i in range(0, rows, 2):
            avg = (temp[i, j] + temp[i+1, j]) / np.sqrt(2)
            diff = (temp[i, j] - temp[i+1, j]) / np.sqrt(2)
            output[i//2, j] = avg
            output[i//2 + rows//2, j] = diff

    return output

#1-Level Inverse Haar DWT
def idwt2D(coeffs):
    rows, cols = coeffs.shape
    temp = np.zeros_like(coeffs)

    # Inverse column transform
    for j in range(cols):
        for i in range(rows//2):
            avg = coeffs[i, j]
            diff = coeffs[i + rows//2, j]
            temp[2*i, j] = (avg + diff) / np.sqrt(2)
            temp[2*i + 1, j] = (avg - diff) / np.sqrt(2)

    # Inverse row transform
    output = np.zeros_like(temp)
    for i in range(rows):
        for j in range(cols//2):
            avg = temp[i, j]
            diff = temp[i, j + cols//2]
            output[i, 2*j] = (avg + diff) / np.sqrt(2)
            output[i, 2*j + 1] = (avg - diff) / np.sqrt(2)

    return output

#Multi-Level DWT
def multi_level_dwt(image, levels=2):
    coeffs = image.copy()
    size_list = []
    for level in range(levels):
        rows, cols = coeffs.shape
        rows //= (2**level)
        cols //= (2**level)
        coeffs[:rows, :cols] = dwt2D(coeffs[:rows, :cols])
        size_list.append((rows, cols))
    return coeffs, size_list

# 5. Multi-Level Inverse DWT
def multi_level_idwt(coeffs, size_list):
    for rows, cols in reversed(size_list):
        coeffs[:rows, :cols] = idwt2D(coeffs[:rows, :cols])
    return coeffs

#Apply Multi-Level DWT
levels = 1  # Change to 2 or 3 for multi-level
coeffs, size_list = multi_level_dwt(img, levels=levels)

#Compression via Multiple Thresholds
thresholds = [5, 10, 20, 50]

plt.figure(figsize=(18, 6))
plt.subplot(1, len(thresholds)+2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

print("DWT Compression Results:\n")

for i, t in enumerate(thresholds, start=2):
    coeffs_copy = coeffs.copy()
    coeffs_copy[np.abs(coeffs_copy) < t] = 0
    reconstructed = multi_level_idwt(coeffs_copy.copy(), size_list)
    reconstructed = reconstructed[:h, :w]
    reconstructed = np.uint8(np.clip(reconstructed, 0, 255))

    non_zero = np.count_nonzero(coeffs_copy)
    compression_ratio = img.size / non_zero
    data_reduction = 100 - (img.size / non_zero) * 100
    mse = np.mean((img[:h, :w] - reconstructed)**2)
    psnr = 10 * np.log10(255*255 / mse)

    print(f"Threshold {t}:")
    print(f"  Compression ratio = {compression_ratio:.2f}")
    print(f"  Data reduction    = {data_reduction:.2f}%")
    print(f"  PSNR              = {psnr:.2f} dB\n")

    plt.subplot(1, len(thresholds)+2, i)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f"T={t}\nCR={compression_ratio:.2f}\nPSNR={psnr:.2f}")
    plt.axis('off')

    # Save image
    #Image.fromarray(reconstructed).save(f"fingerprint_DWT_T{t}.bmp")

# Multi-level approximation
plt.subplot(1, len(thresholds)+2, len(thresholds)+2)
approx_block = coeffs[:h//(2**levels), :w//(2**levels)]
plt.imshow(approx_block, cmap='gray')
plt.title(f"Approximation\n(Level {levels})")
plt.axis('off')

plt.tight_layout()
plt.show()
