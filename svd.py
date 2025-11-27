import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === Load and preprocess image ===
img = Image.open(r"A:\PSG TECH\Package\SEMESTER 5\DC\dataset\SOCOFing\SOCOFing\Real\1__M_Left_index_finger.BMP").convert("L")
A = np.array(img, dtype=float)

# === Compute SVD manually ===
ATA = np.dot(A.T, A)
eig_vals, V = np.linalg.eigh(ATA)
idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
V = V[:, idx]
S = np.sqrt(np.maximum(eig_vals, 0))
U = np.dot(A, V) / np.where(S != 0, S, 1)

print("THE SINGULAR VALUES ARE: ",S)
# === Plot singular values ===
plt.figure(figsize=(7, 4))
plt.plot(S, 'r')
plt.title('Singular Values of the Fingerprint Image')
plt.xlabel('Index')
plt.ylabel('Singular Value Magnitude')
plt.grid(True)
plt.show()

# === Manual full reconstruction (uncompressed, from SVD) ===
A_reconstructed_full = np.dot(U, np.dot(np.diag(S), V.T))
A_reconstructed_full = np.clip(A_reconstructed_full, 0, 255).astype(np.uint8)
Image.fromarray(A_reconstructed_full).save("reconstructed_full.BMP")

# === Reconstruct image using top k singular values ===
def reconstruct_image(k):
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:, :k]
    return np.dot(U_k, np.dot(S_k, V_k.T))

# === Display and analyze compression ===
k_values = [5, 10, 20, 30, 50]
plt.figure(figsize=(15, 8))
m, n = A.shape
original_size = m * n

# Original (uncompressed)
plt.subplot(2, 4, 1)
plt.imshow(A, cmap='gray')
plt.title("Original (Uncompressed)")
plt.axis('off')

# Fully reconstructed (manual full SVD)
plt.subplot(2, 4, 2)
plt.imshow(A_reconstructed_full, cmap='gray')
plt.title("Reconstructed (Full SVD)")
plt.axis('off')

print("Compression Results:\n")

# Compressed versions
for i, k in enumerate(k_values, start=3):
    Ak = reconstruct_image(k)
    Ak_img = np.clip(Ak, 0, 255).astype(np.uint8)
    out_name = f"compressed_k{k}.BMP"
    Image.fromarray(Ak_img).save(out_name)

    compressed_size = k * (m + n + 1)
    compression_ratio = (compressed_size / original_size) * 100
    compression_ratio = min(compression_ratio, 100)
    data_reduction = 100 - compression_ratio

    print(f"Image: {out_name}")
    print(f"  k = {k}")
    print(f"  Compression ratio = {compression_ratio:.2f}% of original data")
    print(f"  Data reduction    = {data_reduction:.2f}%\n")

    plt.subplot(2, 4, i)
    plt.imshow(Ak_img, cmap='gray')
    plt.title(f"Compressed (k={k})")
    plt.axis('off')

plt.tight_layout()
plt.show()
