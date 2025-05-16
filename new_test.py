import pygame
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import copy
import platform
import asyncio

# ---------------------------
# Helper Functions for Dialogs
# ---------------------------
def browse_image():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', '1')
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    root.destroy()
    return file_path

def save_image_dialog(img):
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', '1')
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")]
    )
    root.destroy()
    if file_path:
        cv2.imwrite(file_path, img)

# ---------------------------
# Image Processing Functions
# ---------------------------
def sharpen(img, amount):
    if amount == 0:
        return img
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(sharpened, amount / 10.0, img, 1.0 - amount / 10.0, 0)

def add_noise(img, amount):
    if amount == 0:
        return img
    row, col, ch = img.shape
    mean = 0
    var = (amount / 10.0) * 255
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.int16)
    noisy = img.astype(np.int16) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def adjust_image(img, brightness=1.0, contrast=1.0, saturation=1.0, hue_shift=0.0):
    adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=(brightness - 1) * 50)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
    hsv[:,:,1] *= saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------------------------
# Effects (All 67)
# ---------------------------
def effect_pencil_sketch(img, intensity):
    gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blended = cv2.addWeighted(gray, intensity, orig_gray, 1 - intensity, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)

def effect_color_pencil_sketch(img, intensity):
    gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv2.addWeighted(color, intensity, img, 1 - intensity, 0)

def effect_stylization(img, intensity):
    style = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
    return cv2.addWeighted(style, intensity, img, 1 - intensity, 0)

def effect_cartoon(img, intensity):
    color = img.copy()
    for i in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_color)
    return cv2.addWeighted(cartoon, intensity, img, 1 - intensity, 0)

def effect_detail_enhance(img, intensity):
    detail = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    return cv2.addWeighted(detail, intensity, img, 1 - intensity, 0)

def effect_edge_sketch(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_inv = cv2.bitwise_not(edges)
    edges_inv_color = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(edges_inv_color, intensity, img, 1 - intensity, 0)

def effect_sepia(img, intensity):
    img_float = np.array(img, dtype=np.float64)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img_float, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return cv2.addWeighted(sepia_img, intensity, img, 1 - intensity, 0)

def effect_watercolor(img, intensity):
    watercolor = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    return cv2.addWeighted(watercolor, intensity, img, 1 - intensity, 0)

def effect_emboss(img, intensity):
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    embossed = cv2.filter2D(gray, -1, kernel) + 128
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)
    embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(embossed, intensity, img, 1 - intensity, 0)

def effect_oil_painting(img, intensity):
    try:
        oil = cv2.xphoto.oilPainting(img, 7, 1)
    except Exception:
        oil = img.copy()
    return cv2.addWeighted(oil, intensity, img, 1 - intensity, 0)

def effect_vintage(img, intensity):
    sepia = effect_sepia(img, 1.0)
    vignette = apply_vignette(sepia, 0.5)
    return cv2.addWeighted(vignette, intensity, img, 1 - intensity, 0)

def effect_hdr(img, intensity):
    hdr = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    return cv2.addWeighted(hdr, intensity, img, 1 - intensity, 0)

def effect_glitch(img, intensity):
    rows, cols, _ = img.shape
    shift = int(10 * intensity)
    b, g, r = cv2.split(img)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    return cv2.merge((b, g, r))

def effect_tilt_shift(img, intensity):
    rows, cols = img.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.float32)
    center = rows // 2
    width = int(rows * (1 - intensity * 0.5))
    mask[center - width//2:center + width//2, :] = 1.0
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = cv2.merge([mask, mask, mask])
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    result = img * mask + blur * (1 - mask)
    return result.astype(np.uint8)

def effect_duotone(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = np.array([50, 20, 20], dtype=np.float32)
    light = np.array([255, 200, 180], dtype=np.float32)
    normalized = gray.astype(np.float32) / 255.0
    duotone = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        duotone[:,:,i] = dark[i] + normalized * (light[i] - dark[i])
    return np.clip(duotone, 0, 255).astype(np.uint8)

def apply_vignette(img, level):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * level)
    kernel_y = cv2.getGaussianKernel(rows, rows * level)
    mask = kernel_y * kernel_x.T
    mask = mask / mask.max()
    vignette = np.empty_like(img)
    for i in range(3):
        vignette[:,:,i] = img[:,:,i] * mask
    return vignette

def effect_vignette(img, intensity):
    return apply_vignette(img, intensity * 0.5)

def effect_pop(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= (1.0 + intensity)
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_thermal_vision(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    return cv2.addWeighted(colored, intensity, img, 1 - intensity, 0)

def effect_blue_neon_glow(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = 240
    hsv[:,:,1] *= 1.5
    hsv[:,:,2] *= 1.2
    tinted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blurred = cv2.GaussianBlur(tinted, (0,0), sigmaX=10 + 20*intensity)
    return cv2.addWeighted(tinted, 1 - intensity, blurred, intensity, 0)

def effect_infrared_scan(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(colored, intensity, img, 1 - intensity, 0)

def effect_cyberpunk_chromatic(img, intensity):
    b, g, r = cv2.split(img)
    shift = int(5 * intensity)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    return cv2.merge((b, g, r))

def effect_solarize(img, intensity):
    threshold = 128 * (1 - intensity)
    return np.where(img > threshold, 255 - img, img).astype(np.uint8)

def effect_night_vision(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = 60
    hsv[:,:,1] = 255 * intensity
    hsv[:,:,2] *= 1.5
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_x_ray(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    blue_edges = cv2.merge([edges, np.zeros_like(edges), np.zeros_like(edges)])
    return cv2.addWeighted(img, 1 - intensity, blue_edges, intensity, 0)

def effect_infra_glow(img, intensity):
    thermal = effect_thermal_vision(img, 1.0)
    blurred = cv2.GaussianBlur(thermal, (0,0), sigmaX=10 + 20*intensity)
    return cv2.addWeighted(thermal, 1 - intensity, blurred, intensity, 0)

def effect_cross_process(img, intensity):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:,0,0] = np.clip(1.2 * np.arange(256) - 30, 0, 255).astype(np.uint8)
    lut[:,0,1] = np.clip(1.0 * np.arange(256), 0, 255).astype(np.uint8)
    lut[:,0,2] = np.clip(0.8 * np.arange(256) + 20, 0, 255).astype(np.uint8)
    processed = cv2.LUT(img, lut)
    return cv2.addWeighted(processed, intensity, img, 1 - intensity, 0)

def effect_lomo_chrome(img, intensity):
    vignette = apply_vignette(img, intensity * 0.5)
    hsv = cv2.cvtColor(vignette, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= 1.2
    hsv[:,:,2] *= 1.1
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_film_grain(img, intensity):
    noise = np.random.normal(0, 25 * intensity, img.shape).astype(np.int16)
    noisy_img = img.astype(np.int16) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def effect_fisheye_distortion(img, intensity):
    h, w = img.shape[:2]
    K = np.eye(3)
    k1 = intensity * 0.5
    D = np.array([k1, 0, 0, 0])
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distorted_points = np.dstack((x.ravel(), y.ravel())).astype(np.float32)
    distorted_points = distorted_points.reshape(-1, 1, 2)
    undistorted_points, _ = cv2.fisheye.undistortPoints(distorted_points, K, D=D, R=np.eye(3), P=K)
    undistorted_points = undistorted_points.reshape(h, w, 2)
    map_x = undistorted_points[:,:,0]
    map_y = undistorted_points[:,:,1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def effect_barrel_warp(img, intensity):
    h, w = img.shape[:2]
    K = np.eye(3)
    k1 = intensity * 0.3
    D = np.array([k1, 0, 0, 0])
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    distorted_points = np.dstack((x.ravel(), y.ravel())).astype(np.float32)
    distorted_points = distorted_points.reshape(-1, 1, 2)
    undistorted_points, _ = cv2.fisheye.undistortPoints(distorted_points, K, D=D, R=np.eye(3), P=K)
    undistorted_points = undistorted_points.reshape(h, w, 2)
    map_x = undistorted_points[:,:,0]
    map_y = undistorted_points[:,:,1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def effect_prisma_art(img, intensity):
    style = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return cv2.addWeighted(style, intensity, img, 1 - intensity, 0)

def effect_noir_grain(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    noise = np.random.normal(0, 25 * intensity, img.shape).astype(np.int16)
    noisy_img = gray.astype(np.int16) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def effect_crimson_tint(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = 0
    hsv[:,:,1] *= 1.2
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_teal_orange(img, intensity):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:,0,0] = np.clip(0.8 * np.arange(256) + 20, 0, 255).astype(np.uint8)
    lut[:,0,1] = np.clip(1.0 * np.arange(256), 0, 255).astype(np.uint8)
    lut[:,0,2] = np.clip(1.2 * np.arange(256) - 30, 0, 255).astype(np.uint8)
    processed = cv2.LUT(img, lut)
    return cv2.addWeighted(processed, intensity, img, 1 - intensity, 0)

def effect_bleach_bypass(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= 0.5
    hsv[:,:,2] *= 1.5
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_mosaic_pixelate(img, intensity):
    block_size = int(10 * intensity) + 1
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // block_size, h // block_size))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def effect_kaleidoscope(img, intensity):
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    angle = 360 * intensity / 6
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return cv2.addWeighted(rotated, intensity, img, 1 - intensity, 0)

def effect_cartoon_outline(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(edges_color, intensity, img, 1 - intensity, 0)

def effect_hdr_fusion(img, intensity):
    hdr = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    return cv2.addWeighted(hdr, intensity, img, 1 - intensity, 0)

def effect_light_leak(img, intensity):
    h, w = img.shape[:2]
    gradient = np.zeros_like(img, dtype=np.float32)
    gradient[:,:,2] = np.linspace(0, 255 * intensity, h).reshape(-1, 1)
    return np.clip(img.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

def effect_double_exposure(img, intensity):
    h, w = img.shape[:2]
    gradient = np.zeros_like(img, dtype=np.float32)
    gradient[:,:,1] = np.linspace(0, 255, h).reshape(-1, 1)
    blended = cv2.addWeighted(img, 1 - intensity, gradient.astype(np.uint8), intensity, 0)
    return blended

def effect_reflect_mirror(img, intensity):
    return cv2.flip(img, 1)

def effect_depth_blur(img, intensity):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = h // 2
    mask[center - h//4:center + h//4, :] = 1.0
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = cv2.merge([mask, mask, mask])
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    return (img * mask + blur * (1 - mask)).astype(np.uint8)

def effect_miniature_tilt(img, intensity):
    return effect_tilt_shift(img, intensity)

def effect_pop_art_halftone(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dots = (gray // (intensity * 50 + 1)) * (intensity * 50 + 1)
    return cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)

def effect_vintage_fade(img, intensity):
    sepia = effect_sepia(img, 1.0)
    vignette = apply_vignette(sepia, intensity * 0.5)
    noise = np.random.normal(0, 25 * intensity, img.shape).astype(np.int16)
    noisy_img = vignette.astype(np.int16) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def effect_luminous_bloom(img, intensity):
    bright = cv2.convertScaleAbs(img, beta=50 * intensity)
    blurred = cv2.GaussianBlur(bright, (0,0), sigmaX=10 + 20*intensity)
    return cv2.addWeighted(img, 1 - intensity, blurred, intensity, 0)

def effect_acid_trip(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] += 180 * intensity
    hsv[:,:,0] %= 360
    hsv[:,:,1] *= 1.5
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_chromakey_green(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    result = img.copy()
    result[green_mask != 0] = [0, 0, 255]
    return cv2.addWeighted(result, intensity, img, 1 - intensity, 0)

def effect_posterize(img, intensity):
    levels = int(256 / (1 + intensity * 4))
    return (img // levels * levels).astype(np.uint8)

def effect_vhs_glitch(img, intensity):
    noise = np.random.normal(0, 25 * intensity, img.shape).astype(np.int16)
    noisy_img = img.astype(np.int16) + noise
    scan_lines = np.zeros_like(img)
    scan_lines[::4] = 10 * intensity
    return np.clip(noisy_img + scan_lines, 0, 255).astype(np.uint8)

def effect_roll_film_scratch(img, intensity):
    scratches = np.zeros_like(img)
    for _ in range(int(5 * intensity)):
        x = np.random.randint(0, img.shape[1])
        scratches[:, x:x+2] = 255
    return cv2.addWeighted(img, 1 - intensity, scratches, intensity, 0)

def effect_prism_shift(img, intensity):
    b, g, r = cv2.split(img)
    b = np.roll(b, int(3 * intensity), axis=0)
    r = np.roll(r, int(-3 * intensity), axis=0)
    return cv2.merge((b, g, r))

def effect_frosted_glass(img, intensity):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=10 + 20*intensity)
    noise = np.random.normal(0, 25 * intensity, img.shape).astype(np.int16)
    noisy_img = blurred.astype(np.int16) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def effect_color_splash(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = cv2.inRange(img, (0, 50, 50), (50, 255, 255))
    result = gray.copy()
    result[mask != 0] = img[mask != 0]
    return cv2.addWeighted(result, intensity, img, 1 - intensity, 0)

def effect_blacklight_glow(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = 270
    hsv[:,:,1] *= 1.5
    hsv[:,:,2] *= 1.2
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_holographic(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] += np.sin(np.linspace(0, np.pi, img.shape[1]) * intensity * 10) * 30
    hsv[:,:,0] %= 360
    hsv[:,:,2] *= 1.3
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_wireframe_overlay(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(edges_color, intensity, img, 1 - intensity, 0)

def effect_cel_shaded(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    levels = 8
    posterized = (img // (256 // levels) * (256 // levels)).astype(np.uint8)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cel = cv2.bitwise_and(posterized, edges_color)
    return cv2.addWeighted(cel, intensity, img, 1 - intensity, 0)

def effect_sketchy_lines(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_inv = cv2.bitwise_not(edges)
    sketch = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    noise = np.random.normal(0, 25 * intensity, img.shape).astype(np.int16)
    noisy_sketch = sketch.astype(np.int16) + noise
    return np.clip(noisy_sketch, 0, 255).astype(np.uint8)

def effect_metallic_sheen(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradient = np.linspace(0, 255, img.shape[1]).astype(np.uint8)
    gradient = cv2.merge([gradient, gradient, gradient])
    metallic = cv2.addWeighted(gray, 0.5, gradient, 0.5 * intensity, 0)
    return cv2.cvtColor(metallic, cv2.COLOR_GRAY2BGR)

def effect_radiance_boost(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] *= 1.0 + intensity
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_subsurface_scatter(img, intensity):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=10 + 20*intensity)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] *= 1.2
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def effect_water_drop_refraction(img, intensity):
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h)).astype(np.float32)
    for _ in range(int(10 * intensity)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(10, 50)
        dist = np.sqrt((map_x - cx)**2 + (map_y - cy)**2)
        mask = dist < radius
        map_x[mask] += (map_x[mask] - cx) * 0.1 * intensity
        map_y[mask] += (map_y[mask] - cy) * 0.1 * intensity
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def effect_soft_focus(img, intensity):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=10 + 20*intensity)
    return cv2.addWeighted(blurred, intensity, img, 1 - intensity, 0)

def effect_aurora_lights(img, intensity):
    h, w = img.shape[:2]
    gradient = np.zeros_like(img, dtype=np.float32)
    gradient[:,:,1] = np.sin(np.linspace(0, np.pi, w) * 5) * 255 * intensity
    gradient[:,:,2] = np.cos(np.linspace(0, np.pi, w) * 5) * 255 * intensity
    return np.clip(img.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

def effect_spectrum_shift(img, intensity):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] += 90 * intensity
    hsv[:,:,0] %= 360
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------------------------
# Effects Registry
# ---------------------------
effects = [
    effect_pencil_sketch, effect_color_pencil_sketch, effect_stylization, effect_cartoon,
    effect_detail_enhance, effect_edge_sketch, effect_sepia, effect_watercolor, effect_emboss,
    effect_oil_painting, effect_vintage, effect_hdr, effect_glitch, effect_tilt_shift,
    effect_duotone, effect_vignette, effect_pop, effect_thermal_vision, effect_blue_neon_glow,
    effect_infrared_scan, effect_cyberpunk_chromatic, effect_solarize, effect_night_vision,
    effect_x_ray, effect_infra_glow, effect_cross_process, effect_lomo_chrome, effect_film_grain,
    effect_fisheye_distortion, effect_barrel_warp, effect_prisma_art, effect_noir_grain,
    effect_crimson_tint, effect_teal_orange, effect_bleach_bypass, effect_mosaic_pixelate,
    effect_kaleidoscope, effect_cartoon_outline, effect_hdr_fusion, effect_light_leak,
    effect_double_exposure, effect_reflect_mirror, effect_depth_blur, effect_miniature_tilt,
    effect_pop_art_halftone, effect_vintage_fade, effect_luminous_bloom, effect_acid_trip,
    effect_chromakey_green, effect_posterize, effect_vhs_glitch, effect_roll_film_scratch,
    effect_prism_shift, effect_frosted_glass, effect_color_splash, effect_blacklight_glow,
    effect_holographic, effect_wireframe_overlay, effect_cel_shaded, effect_sketchy_lines,
    effect_metallic_sheen, effect_radiance_boost, effect_subsurface_scatter,
    effect_water_drop_refraction, effect_soft_focus, effect_aurora_lights, effect_spectrum_shift
]

effect_names = [
    "Pencil Sketch", "Color Sketch", "Stylization", "Cartoon", "Detail Enhance",
    "Edge Sketch", "Sepia", "Watercolor", "Emboss", "Oil Painting", "Vintage", "HDR",
    "Glitch", "Tilt Shift", "Duotone", "Vignette", "Pop Effect", "Thermal Vision",
    "Blue Neon Glow", "Infrared Scan", "Cyberpunk Chromatic", "Solarize", "Night Vision",
    "X-Ray View", "Infra Glow", "Cross Process", "Lomo Chrome", "Film Grain",
    "Fisheye Distortion", "Barrel Warp", "Prisma Art", "Noir Grain", "Crimson Tint",
    "Teal & Orange", "Bleach Bypass", "Mosaic Pixelate", "Kaleidoscope", "Cartoon Outline",
    "HDR Fusion", "Light Leak", "Double Exposure", "Reflect Mirror", "Depth Blur",
    "Miniature Tilt", "Pop Art Halftone", "Vintage Fade", "Luminous Bloom", "Acid Trip",
    "Chromakey Green", "Posterize", "VHS Glitch", "Roll Film Scratch", "Prism Shift",
    "Frosted Glass", "Color Splash", "Blacklight Glow", "Holographic", "Wireframe Overlay",
    "Cel-Shaded", "Sketchy Lines", "Metallic Sheen", "Radiance Boost", "Subsurface Scatter",
    "Water Drop Refraction", "Soft Focus", "Aurora Lights", "Spectrum Shift"
]

# ---------------------------
# UI Components
# ---------------------------
class Slider:
    def __init__(self, name, x, y, width, height, min_val, max_val, default):
        self.name = name
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_rect = pygame.Rect(x + int((default - min_val) / (max_val - min_val) * width) - 5, y - 5, 10, height + 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.default_value = default
        self.prev_value = default
        self.dragging = False
        self.changed = False
        self.tooltip = f"Adjust {name.lower()}"

    def draw(self, screen, font, theme):
        track_color = theme["slider_track"] if not self.dragging else theme["slider_active"]
        pygame.draw.rect(screen, track_color, self.rect, border_radius=8)
        pygame.draw.rect(screen, theme["slider_handle"], self.handle_rect, border_radius=5)
        text = font.render(f"{self.name}: {self.value:.1f}", True, theme["text"])
        screen.blit(text, (self.rect.x, self.rect.y - 25))

    def update(self, mouse_pos):
        if self.dragging:
            new_x = max(self.rect.x, min(mouse_pos[0], self.rect.x + self.rect.width))
            self.handle_rect.x = new_x - 5
            ratio = (new_x - self.rect.x) / self.rect.width
            new_value = self.min_val + ratio * (self.max_val - self.min_val)
            if abs(new_value - self.value) > 0.01:
                self.value = new_value
                self.changed = True

    def check_change(self):
        if self.changed or abs(self.value - self.prev_value) > 0.01:
            self.prev_value = self.value
            self.changed = False
            return True
        return False

    def reset(self):
        self.value = self.default_value
        self.prev_value = self.default_value
        self.changed = False
        self.handle_rect.x = self.rect.x + int((self.default_value - self.min_val) / (self.max_val - self.min_val) * self.rect.width) - 5

class Button:
    def __init__(self, text, x, y, width, height, tooltip=""):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.tooltip = tooltip
        self.hovered = False
        self.scale = 1.0

    def draw(self, screen, font, theme):
        gradient = pygame.Surface((self.rect.width, self.rect.height))
        color1 = theme["button_normal"] if not self.hovered else theme["button_hover"]
        color2 = [max(0, c - 20) for c in color1]
        for y in range(self.rect.height):
            ratio = y / self.rect.height
            color = [int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)]
            pygame.draw.line(gradient, color, (0, y), (self.rect.width, y))
        
        scaled_size = (int(self.rect.width * self.scale), int(self.rect.height * self.scale))
        scaled_surf = pygame.transform.smoothscale(gradient, scaled_size)
        scaled_rect = scaled_surf.get_rect(center=self.rect.center)
        screen.blit(scaled_surf, scaled_rect)
        
        pygame.draw.rect(screen, theme["border"], scaled_rect, 2, border_radius=8)
        
        text_surf = font.render(self.text, True, theme["text"])
        text_rect = text_surf.get_rect(center=scaled_rect.center)
        screen.blit(text_surf, text_rect)
        
        if self.hovered and self.scale < 1.05:
            self.scale += 0.005
        elif not self.hovered and self.scale > 1.0:
            self.scale -= 0.005

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

# ---------------------------
# Main Application
# ---------------------------
class ImageEditor:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Image Editor")
        self.font = pygame.font.SysFont("Arial", 16)
        self.clock = pygame.time.Clock()
        
        self.themes = {
            "dark": {
                "background": (30, 30, 30),
                "panel": (40, 40, 40),
                "text": (200, 200, 200),
                "button_normal": (60, 60, 60),
                "button_hover": (80, 80, 80),
                "slider_track": (50, 50, 50),
                "slider_handle": (100, 100, 100),
                "slider_active": (70, 70, 70),
                "border": (100, 100, 100),
                "highlight": (0, 120, 255)
            },
            "light": {
                "background": (220, 220, 220),
                "panel": (200, 200, 200),
                "text": (30, 30, 30),
                "button_normal": (180, 180, 180),
                "button_hover": (160, 160, 160),
                "slider_track": (190, 190, 190),
                "slider_handle": (140, 140, 140),
                "slider_active": (170, 170, 170),
                "border": (120, 120, 120),
                "highlight": (0, 100, 200)
            }
        }
        self.theme = "dark"
        
        self.original_img = None
        self.current_img = None
        self.display_img = None
        self.last_processed_img = None
        self.zoom = 1.0
        self.pan_offset = [0, 0]
        self.history = []
        self.redo_stack = []
        self.needs_slider_update = False
        
        self.left_panel_width = 300
        self.right_panel_width = 200
        self.toolbar_height = 50
        self.status_bar_height = 30
        self.left_scroll_offset = 0
        self.right_scroll_offset = 0
        
        self.sliders = [
            Slider("Intensity", 10, 0, 180, 10, 0.0, 1.0, 1.0),
            Slider("Brightness", 10, 0, 180, 10, 0.5, 1.5, 1.0),
            Slider("Contrast", 10, 0, 180, 10, 0.5, 1.5, 1.0),
            Slider("Saturation", 10, 0, 180, 10, 0.5, 1.5, 1.0),
            Slider("Hue", 10, 0, 180, 10, -180.0, 180.0, 0.0),
            Slider("Sharpness", 10, 0, 180, 10, 0.0, 10.0, 0.0),
            Slider("Noise", 10, 0, 180, 10, 0.0, 10.0, 0.0)
        ]
        
        self.action_buttons = [
            Button("Load", 0, 0, 180, 30, "Load an image"),
            Button("Save", 0, 0, 180, 30, "Save the image"),
            Button("Undo", 0, 0, 180, 30, "Undo last action"),
            Button("Redo", 0, 0, 180, 30, "Redo last action"),
            Button("Reset", 0, 0, 180, 30, "Reset all changes")
        ]
        
        self.toolbar_buttons = [
            Button("Zoom In", 10, 10, 80, 30, "Increase zoom"),
            Button("Zoom Out", 100, 10, 80, 30, "Decrease zoom"),
            Button("Theme", 190, 10, 80, 30, "Toggle dark/light theme"),
            Button("Help", self.screen.get_width() - 90, 10, 80, 30, "Show help")
        ]
        
        self.effect_buttons = []
        self.update_panel_layout()
        
        self.show_welcome = True
        self.welcome_alpha = 255
        self.welcome_button = Button("Start Editing", 590, 500, 100, 40, "Begin editing")
        
        self.context_menu = [
            Button("Save", 0, 0, 100, 30, "Save the image"),
            Button("Reset", 0, 30, 100, 30, "Reset all changes"),
            Button("Undo", 0, 60, 100, 30, "Undo last action")
        ]
        self.context_menu_active = False
        self.context_menu_pos = (0, 0)
        
        self.status_message = "Welcome to Image Editor"
        self.status_timer = 0
        
        self.show_help = False

    def update_panel_layout(self):
        # Left panel: Effect buttons
        y_offset = self.toolbar_height + 10
        self.effect_buttons = [
            (Button(name, 10, y_offset + i * 40, 280, 30, f"Apply {name} effect"), idx)
            for i, (name, idx) in enumerate(zip(effect_names, range(len(effect_names))))
        ]
        
        # Right panel: Sliders and action buttons
        right_x = self.screen.get_width() - self.right_panel_width
        y_offset = self.toolbar_height + 10
        for slider in self.sliders:
            slider.rect.x = right_x + 10
            slider.rect.y = y_offset + 25
            slider.handle_rect.x = right_x + 10 + int((slider.value - slider.min_val) / (slider.max_val - slider.min_val) * 180)
            slider.handle_rect.y = y_offset + 20
            y_offset += 50
        
        y_offset += 10  # Space between sliders and buttons
        for button in self.action_buttons:
            button.rect.x = right_x + 10
            button.rect.y = y_offset
            y_offset += 40

    def draw_gradient(self, surface, rect, color1, color2, vertical=True):
        gradient = pygame.Surface((rect.width, rect.height))
        if vertical:
            for y in range(rect.height):
                ratio = y / rect.height
                color = [int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)]
                pygame.draw.line(gradient, color, (0, y), (rect.width, y))
        else:
            for x in range(rect.width):
                ratio = x / rect.width
                color = [int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)]
                pygame.draw.line(gradient, color, (x, 0), (x, rect.height))
        surface.blit(gradient, rect)

    def draw(self):
        theme = self.themes[self.theme]
        self.screen.fill(theme["background"])
        
        # Toolbar
        pygame.draw.rect(self.screen, theme["panel"], (0, 0, self.screen.get_width(), self.toolbar_height))
        for button in self.toolbar_buttons:
            button.draw(self.screen, self.font, theme)
        
        # Left panel
        pygame.draw.rect(self.screen, theme["panel"], (0, self.toolbar_height, self.left_panel_width, self.screen.get_height() - self.status_bar_height))
        clip_rect = pygame.Rect(0, self.toolbar_height, self.left_panel_width, self.screen.get_height() - self.status_bar_height)
        self.screen.set_clip(clip_rect)
        for button, _ in self.effect_buttons:
            button.rect.y -= self.left_scroll_offset
            button.draw(self.screen, self.font, theme)
            button.rect.y += self.left_scroll_offset
        self.screen.set_clip(None)
        
        # Right panel
        right_x = self.screen.get_width() - self.right_panel_width
        pygame.draw.rect(self.screen, theme["panel"], (right_x, self.toolbar_height, self.right_panel_width, self.screen.get_height() - self.status_bar_height))
        clip_rect = pygame.Rect(right_x, self.toolbar_height, self.right_panel_width, self.screen.get_height() - self.status_bar_height)
        self.screen.set_clip(clip_rect)
        for slider in self.sliders:
            slider.rect.y -= self.right_scroll_offset
            slider.handle_rect.y -= self.right_scroll_offset
            slider.draw(self.screen, self.font, theme)
            slider.rect.y += self.right_scroll_offset
            slider.handle_rect.y += self.right_scroll_offset
        for button in self.action_buttons:
            button.rect.y -= self.right_scroll_offset
            button.draw(self.screen, self.font, theme)
            button.rect.y += self.right_scroll_offset
        self.screen.set_clip(None)
        
        # Image (middle section)
        if self.display_img:
            img_rect = self.display_img.get_rect()
            img_rect.center = (
                self.left_panel_width + (self.screen.get_width() - self.left_panel_width - self.right_panel_width) / 2 + self.pan_offset[0],
                self.toolbar_height + (self.screen.get_height() - self.toolbar_height - self.status_bar_height) / 2 + self.pan_offset[1]
            )
            frame_rect = img_rect.inflate(10, 10)
            pygame.draw.rect(self.screen, theme["border"], frame_rect, 2, border_radius=8)
            self.screen.blit(self.display_img, img_rect)
        
        # Status bar
        pygame.draw.rect(self.screen, theme["panel"], (0, self.screen.get_height() - self.status_bar_height, self.screen.get_width(), self.status_bar_height))
        status_text = self.font.render(self.status_message, True, theme["text"])
        self.screen.blit(status_text, (10, self.screen.get_height() - self.status_bar_height + 5))
        
        # Welcome screen
        if self.show_welcome:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, self.welcome_alpha))
            self.screen.blit(overlay, (0, 0))
            title = self.font.render("Image Editor Pro", True, theme["text"])
            self.screen.blit(title, (self.screen.get_width() // 2 - title.get_width() // 2, 300))
            self.welcome_button.draw(self.screen, self.font, theme)
        
        # Context menu
        if self.context_menu_active:
            for button in self.context_menu:
                button.draw(self.screen, self.font, theme)
        
        # Help overlay
        if self.show_help:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            self.screen.blit(overlay, (0, 0))
            help_text = [
                "Image Editor Help",
                "Shortcuts:",
                "Ctrl+O: Load Image",
                "Ctrl+S: Save Image",
                "Ctrl+Z: Undo",
                "Ctrl+Y: Redo",
                "Ctrl+R: Reset",
                "Mouse Wheel: Zoom or Scroll",
                "Right-Click: Context Menu",
                "Click 'Help' to close"
            ]
            for i, line in enumerate(help_text):
                text = self.font.render(line, True, theme["text"])
                self.screen.blit(text, (self.screen.get_width() // 2 - text.get_width() // 2, 200 + i * 30))

    async def main(self):
        self.setup()
        while True:
            self.update_loop()
            await asyncio.sleep(1.0 / 60)

    def setup(self):
        self.original_img = None
        self.current_img = None
        self.last_processed_img = None
        self.display_img = None
        self.history = []
        self.redo_stack = []
        self.needs_slider_update = False

    def apply_slider_adjustments(self):
        if self.current_img is None:
            return
        img = self.current_img.copy()
        img = adjust_image(
            img,
            brightness=self.sliders[1].value,
            contrast=self.sliders[2].value,
            saturation=self.sliders[3].value,
            hue_shift=self.sliders[4].value
        )
        img = sharpen(img, self.sliders[5].value)
        img = add_noise(img, self.sliders[6].value)
        self.last_processed_img = img
        self.current_img = img.copy()
        self.history.append(self.current_img.copy())
        if len(self.history) > 10:
            self.history.pop(0)
        self.status_message = "Slider Adjustments Applied"
        self.status_timer = 120

    def update_loop(self):
        theme = self.themes[self.theme]
        mouse_pos = pygame.mouse.get_pos()
        
        # Hover checks
        for button in self.toolbar_buttons + self.action_buttons + [b for b, _ in self.effect_buttons]:
            button.check_hover(mouse_pos)
        if self.show_welcome:
            self.welcome_button.check_hover(mouse_pos)
        if self.context_menu_active:
            for button in self.context_menu:
                button.check_hover(mouse_pos)
        
        # Slider updates
        for slider in self.sliders:
            if slider.dragging:
                slider.update(mouse_pos)
                if slider.check_change():
                    self.needs_slider_update = True
        
        # Status timer
        if self.status_timer > 0:
            self.status_timer -= 1
            if self.status_timer == 0:
                self.status_message = "Ready"
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.toolbar_buttons[3].rect.x = self.screen.get_width() - 90
                self.update_panel_layout()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.show_welcome:
                        if self.welcome_button.rect.collidepoint(event.pos):
                            self.show_welcome = False
                        continue
                    
                    if self.context_menu_active:
                        for i, button in enumerate(self.context_menu):
                            if button.rect.collidepoint(event.pos):
                                if i == 0:
                                    save_image_dialog(self.current_img)
                                    self.status_message = "Image Saved"
                                    self.status_timer = 120
                                elif i == 1:
                                    self.current_img = self.original_img.copy()
                                    self.last_processed_img = None
                                    for slider in self.sliders:
                                        slider.reset()
                                    self.history.append(self.current_img.copy())
                                    self.redo_stack.clear()
                                    self.needs_slider_update = False
                                    self.status_message = "Image Reset"
                                    self.status_timer = 120
                                elif i == 2 and self.history:
                                    self.redo_stack.append(self.current_img.copy())
                                    self.current_img = self.history.pop()
                                    self.last_processed_img = None
                                    for slider in self.sliders:
                                        slider.reset()
                                    self.needs_slider_update = False
                                    self.status_message = "Undo Applied"
                                    self.status_timer = 120
                        self.context_menu_active = False
                        continue
                    
                    # Sliders (right panel)
                    for slider in self.sliders:
                        adjusted_rect = slider.rect.move(0, -self.right_scroll_offset)
                        if adjusted_rect.collidepoint(event.pos):
                            slider.dragging = True
                    
                    # Action buttons (right panel)
                    for button in self.action_buttons:
                        adjusted_rect = button.rect.move(0, -self.right_scroll_offset)
                        if adjusted_rect.collidepoint(event.pos):
                            if button.text == "Load":
                                file_path = browse_image()
                                if file_path:
                                    self.original_img = cv2.imread(file_path)
                                    if self.original_img is not None:
                                        self.current_img = self.original_img.copy()
                                        self.last_processed_img = None
                                        for slider in self.sliders:
                                            slider.reset()
                                        self.history = [self.current_img.copy()]
                                        self.redo_stack = []
                                        self.needs_slider_update = False
                                        self.status_message = "Image Loaded"
                                        self.status_timer = 120
                            elif button.text == "Save":
                                if self.current_img is not None:
                                    save_image_dialog(self.current_img)
                                    self.status_message = "Image Saved"
                                    self.status_timer = 120
                            elif button.text == "Undo" and len(self.history) > 1:
                                self.redo_stack.append(self.current_img.copy())
                                self.current_img = self.history.pop()
                                self.last_processed_img = None
                                for slider in self.sliders:
                                    slider.reset()
                                self.needs_slider_update = False
                                self.status_message = "Undo Applied"
                                self.status_timer = 120
                            elif button.text == "Redo" and self.redo_stack:
                                self.history.append(self.current_img.copy())
                                self.current_img = self.redo_stack.pop()
                                self.last_processed_img = None
                                for slider in self.sliders:
                                    slider.reset()
                                self.needs_slider_update = False
                                self.status_message = "Redo Applied"
                                self.status_timer = 120
                            elif button.text == "Reset" and self.original_img is not None:
                                self.current_img = self.original_img.copy()
                                self.last_processed_img = None
                                for slider in self.sliders:
                                    slider.reset()
                                self.history.append(self.current_img.copy())
                                self.redo_stack.clear()
                                self.needs_slider_update = False
                                self.status_message = "Image Reset"
                                self.status_timer = 120
                    
                    # Toolbar buttons
                    for button in self.toolbar_buttons:
                        if button.rect.collidepoint(event.pos):
                            if button.text == "Zoom In":
                                self.zoom = min(self.zoom + 0.1, 3.0)
                            elif button.text == "Zoom Out":
                                self.zoom = max(self.zoom - 0.1, 0.5)
                            elif button.text == "Theme":
                                self.theme = "light" if self.theme == "dark" else "dark"
                                self.status_message = f"{self.theme.capitalize()} Theme Applied"
                                self.status_timer = 120
                            elif button.text == "Help":
                                self.show_help = not self.show_help
                    
                    # Effect buttons (left panel)
                    for button, idx in self.effect_buttons:
                        adjusted_rect = button.rect.move(0, -self.left_scroll_offset)
                        if adjusted_rect.collidepoint(event.pos):
                            if self.current_img is not None:
                                effect = effects[idx]
                                intensity = self.sliders[0].value
                                self.current_img = effect(self.current_img, intensity)
                                self.last_processed_img = None
                                for slider in self.sliders[1:]:
                                    slider.reset()
                                self.history.append(self.current_img.copy())
                                self.redo_stack.clear()
                                if len(self.history) > 10:
                                    self.history.pop(0)
                                self.needs_slider_update = False
                                self.status_message = f"Applied {button.text}"
                                self.status_timer = 120
                
                elif event.button == 3:
                    if self.current_img is not None:
                        img_rect = self.display_img.get_rect()
                        img_rect.center = (
                            self.left_panel_width + (self.screen.get_width() - self.left_panel_width - self.right_panel_width) / 2 + self.pan_offset[0],
                            self.toolbar_height + (self.screen.get_height() - self.toolbar_height - self.status_bar_height) / 2 + self.pan_offset[1]
                        )
                        if img_rect.collidepoint(event.pos):
                            self.context_menu_active = True
                            self.context_menu_pos = event.pos
                            for i, button in enumerate(self.context_menu):
                                button.rect.topleft = (event.pos[0], event.pos[1] + i * 30)
                
                elif event.button == 4:
                    if event.pos[0] <= self.left_panel_width:
                        max_scroll = max(0, (len(self.effect_buttons) * 40 + 10) - (self.screen.get_height() - self.toolbar_height - self.status_bar_height))
                        self.left_scroll_offset = min(self.left_scroll_offset + 20, max_scroll)
                    elif event.pos[0] >= self.screen.get_width() - self.right_panel_width:
                        max_scroll = max(0, (len(self.sliders) * 50 + len(self.action_buttons) * 40 + 20) - (self.screen.get_height() - self.toolbar_height - self.status_bar_height))
                        self.right_scroll_offset = min(self.right_scroll_offset + 20, max_scroll)
                    else:
                        self.zoom = min(self.zoom + 0.1, 3.0)
                
                elif event.button == 5:
                    if event.pos[0] <= self.left_panel_width:
                        self.left_scroll_offset = max(0, self.left_scroll_offset - 20)
                    elif event.pos[0] >= self.screen.get_width() - self.right_panel_width:
                        self.right_scroll_offset = max(0, self.right_scroll_offset - 20)
                    else:
                        self.zoom = max(self.zoom - 0.1, 0.5)
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    for slider in self.sliders:
                        if slider.dragging and slider.check_change():
                            self.needs_slider_update = True
                        slider.dragging = False
            
            if event.type == pygame.MOUSEMOTION:
                if event.buttons[0] and self.current_img is not None:
                    img_rect = self.display_img.get_rect()
                    img_rect.center = (
                        self.left_panel_width + (self.screen.get_width() - self.left_panel_width - self.right_panel_width) / 2 + self.pan_offset[0],
                        self.toolbar_height + (self.screen.get_height() - self.toolbar_height - self.status_bar_height) / 2 + self.pan_offset[1]
                    )
                    if img_rect.collidepoint(event.pos):
                        self.pan_offset[0] += event.rel[0]
                        self.pan_offset[1] += event.rel[1]
            
            if event.type == pygame.KEYDOWN:
                if event.mod & pygame.KMOD_CTRL:
                    if event.key == pygame.K_o:
                        file_path = browse_image()
                        if file_path:
                            self.original_img = cv2.imread(file_path)
                            if self.original_img is not None:
                                self.current_img = self.original_img.copy()
                                self.last_processed_img = None
                                for slider in self.sliders:
                                    slider.reset()
                                self.history = [self.current_img.copy()]
                                self.redo_stack = []
                                self.needs_slider_update = False
                                self.status_message = "Image Loaded"
                                self.status_timer = 120
                    elif event.key == pygame.K_s and self.current_img is not None:
                        save_image_dialog(self.current_img)
                        self.status_message = "Image Saved"
                        self.status_timer = 120
                    elif event.key == pygame.K_z and len(self.history) > 1:
                        self.redo_stack.append(self.current_img.copy())
                        self.current_img = self.history.pop()
                        self.last_processed_img = None
                        for slider in self.sliders:
                            slider.reset()
                        self.needs_slider_update = False
                        self.status_message = "Undo Applied"
                        self.status_timer = 120
                    elif event.key == pygame.K_y and self.redo_stack:
                        self.history.append(self.current_img.copy())
                        self.current_img = self.redo_stack.pop()
                        self.last_processed_img = None
                        for slider in self.sliders:
                            slider.reset()
                        self.needs_slider_update = False
                        self.status_message = "Redo Applied"
                        self.status_timer = 120
                    elif event.key == pygame.K_r and self.original_img is not None:
                        self.current_img = self.original_img.copy()
                        self.last_processed_img = None
                        for slider in self.sliders:
                            slider.reset()
                        self.history.append(self.current_img.copy())
                        self.redo_stack.clear()
                        self.needs_slider_update = False
                        self.status_message = "Image Reset"
                        self.status_timer = 120
        
        # Image update
        if self.current_img is not None:
            if self.needs_slider_update:
                self.apply_slider_adjustments()
                self.needs_slider_update = False
            
            img = self.last_processed_img if self.last_processed_img is not None else self.current_img.copy()
            h, w = img.shape[:2]
            new_size = (int(w * self.zoom), int(h * self.zoom))
            img = cv2.resize(img, new_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (1, 0, 2))
            self.display_img = pygame.surfarray.make_surface(img)
        
        # Welcome fade
        if not self.show_welcome and self.welcome_alpha > 0:
            self.welcome_alpha = max(0, self.welcome_alpha - 5)
        
        self.draw()
        pygame.display.flip()
        self.clock.tick(60)

if platform.system() == "Emscripten":
    asyncio.ensure_future(ImageEditor().main())
else:
    if __name__ == "__main__":
        asyncio.run(ImageEditor().main())