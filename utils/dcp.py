# utils/dcp.py
import cv2
import numpy as np

def dark_channel(img, size=25):  # increased patch size
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def atmospheric_light(img, dark):
    h, w = dark.shape
    num_pixels = h * w
    num_bright = int(max(num_pixels * 0.001, 1))

    indices = np.argsort(dark.ravel())[-num_bright:]
    brightest = img.reshape(-1, 3)[indices]

    # better than mean → pick brightest pixel
    A = brightest[np.argmax(np.sum(brightest, axis=1))]
    return A

def transmission(img, A, omega=0.99, size=25):
    norm_img = img / (A + 1e-8)  # avoid division issues
    dark = dark_channel(norm_img, size)
    t = 1 - omega * dark

    # 🔥 force stronger dehazing
    t = np.clip(t * 0.7, 0.1, 1)
    return t

def recover(img, t, A, t0=0.1):
    t = np.clip(t, t0, 1)
    J = (img - A) / t[..., None] + A

    # 🔥 underwater color compensation (important)
    J[:, :, 0] = J[:, :, 0] * 1.5  # boost red

    return np.clip(J, 0, 1)

def dcp_restore(img):
    img = img.astype(np.float32) / 255.0
    dark = dark_channel(img)
    A = atmospheric_light(img, dark)
    t = transmission(img, A)
    J = recover(img, t, A)
    return (J * 255).astype(np.uint8), t, A