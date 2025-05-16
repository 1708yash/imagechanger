import pygame
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import asyncio
import platform

# ---------------------------
# Helper Functions for Dialogs
# ---------------------------
def browse_image():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', '1')
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
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
# Image Effects (32 Filters)
# ---------------------------
def effect_pencil_sketch(img, intensity):
    gray, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blended = cv2.addWeighted(gray, intensity, orig_gray, 1 - intensity, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)

def effect_color_pencil_sketch(img, intensity):
    _, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv2.addWeighted(color, intensity, img, 1 - intensity, 0)

def effect_stylization(img, intensity):
    style = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
    return cv2.addWeighted(style, intensity, img, 1 - intensity, 0)

def effect_cartoon(img, intensity):
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
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
    img_float = img.astype(np.float64)
    sepia_filter = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img_float, sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return cv2.addWeighted(sepia_img, intensity, img, 1 - intensity, 0)

def effect_watercolor(img, intensity):
    watercolor = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    return cv2.addWeighted(watercolor, intensity, img, 1 - intensity, 0)

def effect_emboss(img, intensity):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
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
    b, g, r = cv2.split(img)
    shift = int(10 * intensity)
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

def effect_negative(img, intensity):
    neg = cv2.bitwise_not(img)
    return cv2.addWeighted(neg, intensity, img, 1 - intensity, 0)

def effect_thermal(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, intensity, img, 1 - intensity, 0)

def effect_pixelate(img, intensity):
    h, w = img.shape[:2]
    k = int(1 + (50 * (1 - intensity)))
    small = cv2.resize(img, (w//k, h//k), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def effect_invert_gray(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(inv_bgr, intensity, img, 1 - intensity, 0)

def effect_color_quantization(img, intensity):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    K = int(2 + intensity * 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quant = res.reshape(img.shape)
    return cv2.addWeighted(quant, intensity, img, 1 - intensity, 0)

def effect_vortex(img, intensity):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r = np.sqrt(dx*dx + dy*dy)
            theta = np.arctan2(dy, dx) + intensity * 2
            map_x[y,x] = center[0] + r * np.cos(theta)
            map_y[y,x] = center[1] + r * np.sin(theta)
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return warped

def effect_sketch_colorized(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    color = cv2.bitwise_and(img, img, mask=edges)
    return cv2.addWeighted(color, intensity, img, 1 - intensity, 0)

def effect_glow(img, intensity):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=15)
    glow = cv2.addWeighted(img, 1, blur, intensity * 0.5, 0)
    return glow

def effect_dual_tone(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    color1 = np.full_like(img, (int(255 * intensity), 0, int(255 * (1 - intensity))))
    color2 = np.full_like(img, (0, int(255 * intensity), int(255 * (1 - intensity))))
    mask = th == 255
    result = np.where(mask[:, :, None], color1, color2)
    return result.astype(np.uint8)

def effect_gaussian_blur(img, intensity):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.addWeighted(blurred, intensity, img, 1 - intensity, 0)

def effect_sharpen(img, intensity):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(sharpened, intensity, img, 1 - intensity, 0)

def effect_black_white(img, intensity):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(gray_bgr, intensity, img, 1 - intensity, 0)

def effect_solarize(img, intensity):
    threshold = int(128 * intensity)
    img_float = img.astype(np.float32)
    solarized = np.where(img_float > threshold, 255 - img_float, img_float)
    return solarized.astype(np.uint8)

def effect_posterize(img, intensity):
    levels = int(8 - intensity * 6)
    img = (img // (256 // levels)) * (256 // levels)
    return img.astype(np.uint8)

def effect_soft_focus(img, intensity):
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    return cv2.addWeighted(blur, intensity, img, 1 - intensity, 0)

# ---------------------------
# Effects Registry
# ---------------------------
effects = [
    {"func": effect_pencil_sketch, "name": "Pencil Sketch", "desc": "Converts image to a pencil sketch."},
    {"func": effect_color_pencil_sketch, "name": "Color Sketch", "desc": "Colorful pencil sketch effect."},
    {"func": effect_stylization, "name": "Stylization", "desc": "Applies an artistic stylized look."},
    {"func": effect_cartoon, "name": "Cartoon", "desc": "Creates a cartoon-like effect."},
    {"func": effect_detail_enhance, "name": "Detail Enhance", "desc": "Enhances image details."},
    {"func": effect_edge_sketch, "name": "Edge Sketch", "desc": "Highlights edges in a sketch style."},
    {"func": effect_sepia, "name": "Sepia", "desc": "Applies a vintage sepia tone."},
    {"func": effect_watercolor, "name": "Watercolor", "desc": "Mimics a watercolor painting."},
    {"func": effect_emboss, "name": "Emboss", "desc": "Creates an embossed effect."},
    {"func": effect_oil_painting, "name": "Oil Painting", "desc": "Simulates an oil painting."},
    {"func": effect_vintage, "name": "Vintage", "desc": "Combines sepia and vignette for a retro look."},
    {"func": effect_hdr, "name": "HDR", "desc": "Enhances details for a high dynamic range look."},
    {"func": effect_glitch, "name": "Glitch", "desc": "Applies a digital glitch effect."},
    {"func": effect_tilt_shift, "name": "Tilt Shift", "desc": "Creates a miniature model effect."},
    {"func": effect_duotone, "name": "Duotone", "desc": "Applies a two-tone color effect."},
    {"func": effect_vignette, "name": "Vignette", "desc": "Darkens edges for a focused look."},
    {"func": effect_pop, "name": "Pop Effect", "desc": "Boosts saturation for a vibrant look."},
    {"func": effect_negative, "name": "Negative", "desc": "Inverts the image colors."},
    {"func": effect_thermal, "name": "Thermal", "desc": "Applies a thermal imaging effect."},
    {"func": effect_pixelate, "name": "Pixelate", "desc": "Creates a pixelated, low-res effect."},
    {"func": effect_invert_gray, "name": "Invert Gray", "desc": "Inverts grayscale image."},
    {"func": effect_color_quantization, "name": "Color Quant", "desc": "Reduces color palette."},
    {"func": effect_vortex, "name": "Vortex", "desc": "Applies a swirling distortion."},
    {"func": effect_sketch_colorized, "name": "Sketch Color", "desc": "Colorized sketch effect."},
    {"func": effect_glow, "name": "Glow", "desc": "Adds a glowing effect."},
    {"func": effect_dual_tone, "name": "Dual Tone", "desc": "Two-tone color effect."},
    {"func": effect_gaussian_blur, "name": "Gaussian Blur", "desc": "Softens the image."},
    {"func": effect_sharpen, "name": "Sharpen", "desc": "Enhances image sharpness."},
    {"func": effect_black_white, "name": "Black & White", "desc": "Converts to grayscale."},
    {"func": effect_solarize, "name": "Solarize", "desc": "Inverts colors above a threshold."},
    {"func": effect_posterize, "name": "Posterize", "desc": "Reduces color levels for a poster-like effect."},
    {"func": effect_soft_focus, "name": "Soft Focus", "desc": "Creates a dreamy, soft-focus effect."}
]

effect_names = [effect["name"] for effect in effects]

# ---------------------------
# Image Adjustment Function
# ---------------------------
def adjust_image(img, brightness, contrast, saturation):
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=int((brightness - 1.0) * 50))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------------------------
# UI Components
# ---------------------------
class Slider:
    def __init__(self, label, x, y, w, h, min_val, max_val, value):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        self.handle_rect = pygame.Rect(x + int((value - min_val) / (max_val - min_val) * w) - 5, y - 5, 10, h + 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.dragging = False

    def draw(self, screen, font):
    # Background bar
        pygame.draw.rect(screen, (120, 120, 120), self.rect, border_radius=5)

    # Value overlay (green)
        value_width = int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        value_rect = pygame.Rect(self.rect.x, self.rect.y, value_width, self.rect.height)
        pygame.draw.rect(screen, (100, 255, 100), value_rect, border_radius=5)

    # Handle
        pygame.draw.rect(screen, (200, 200, 80), self.handle_rect, border_radius=5)

    # Label
        txt = font.render(f"{self.label}: {self.value:.2f}", True, (255, 255, 255))
        screen.blit(txt, (self.rect.x, self.rect.y - 25))
    
    def update_handle_position(self):
        self.handle_rect.x = self.rect.x + int((self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width) - 5


    def draw(self, screen, font):
        pygame.draw.rect(screen, (120, 120, 120), self.rect, border_radius=5)
        pygame.draw.rect(screen, (200, 200, 80), self.handle_rect, border_radius=5)
        txt = font.render(f"{self.label}: {self.value:.2f}", True, (255, 255, 255))
        screen.blit(txt, (self.rect.x, self.rect.y - 25))

    def update(self, mouse_pos):
        if self.dragging:
            rel_x = max(0, min(mouse_pos[0] - self.rect.x, self.rect.width))
            self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
            self.update_handle_position()

class ScrollPanel:
    def __init__(self, x, y, width, height, items, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.items = items
        self.font = font
        self.scroll = 0
        self.item_h = 40
        self.hovered = -1

    def draw(self, screen, current):
        pygame.draw.rect(screen, (50, 50, 50), self.rect)
        for idx, item in enumerate(self.items):
            y = self.rect.y + idx * self.item_h - self.scroll
            if self.rect.y <= y < self.rect.y + self.rect.height:
                btn = pygame.Rect(self.rect.x, y, self.rect.width, self.item_h)
                color = (120, 170, 220) if idx == current else (80, 80, 80) if idx == self.hovered else (100, 100, 100)
                pygame.draw.rect(screen, color, btn)
                txt = self.font.render(item["name"], True, (255, 255, 255))
                screen.blit(txt, (self.rect.x + 10, y + 10))

    def handle_event(self, ev):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = -1
        if self.rect.collidepoint(mouse_pos):
            rel_y = mouse_pos[1] - self.rect.y + self.scroll
            idx = rel_y // self.item_h
            if 0 <= idx < len(self.items):
                self.hovered = idx
        if ev.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(ev.pos):
            rel_y = ev.pos[1] - self.rect.y + self.scroll
            idx = rel_y // self.item_h
            if 0 <= idx < len(self.items):
                return idx
        if ev.type == pygame.MOUSEWHEEL and self.rect.collidepoint(mouse_pos):
            self.scroll = max(0, min(self.scroll - ev.y * 20, len(self.items) * self.item_h - self.rect.height))
        return None

class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.hovered = False

    def draw(self, screen, font):
        color = (90, 150, 200) if self.hovered else (70, 130, 180)
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        txt = font.render(self.text, True, (255, 255, 255))
        txt_rect = txt.get_rect(center=self.rect.center)
        screen.blit(txt, txt_rect)

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

# ---------------------------
# Main Application
# ---------------------------
async def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Photo Editor Pro")
    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()
    FPS = 60

    # Panels and UI Elements
    left_panel = ScrollPanel(0, 0, 200, 720, effects, font)
    right_panel_rect = pygame.Rect(1080, 0, 300, 720)
    desc_rect = pygame.Rect(0, 600, 200, 120)
    sliders = [
        Slider("Brightness", 0, 0, 180, 20, 0.0, 2.0, 1.0),
        Slider("Contrast", 0, 0, 180, 20, 0.0, 2.0, 1.0),
        Slider("Saturation", 0, 0, 180, 20, 0.0, 2.0, 1.0),
        Slider("Effect Intensity", 0, 0, 180, 20, 0.0, 1.0, 1.0)
    ]
    buttons = [
        Button(1080, 600, 150, 40, "Load Image"),
        Button(1080, 650, 150, 40, "Save Image"),
        Button(1080, 500, 150, 40, "Undo"),
        Button(1080, 550, 150, 40, "Redo")
    ]

    # Image and History
    img_orig = None
    img_disp = None
    processed = None
    current = 0
    history = []
    redo_stack = []
    max_history = 10
    desc_text = effects[0]["desc"]

    def update_layout(width, height):
        left_panel.rect.height = height - 120
        right_panel_rect.x = width - 200
        right_panel_rect.height = height
        desc_rect.y = height - 120
        for i, s in enumerate(sliders):
            s.rect.x = right_panel_rect.x + 10
            s.handle_rect.x = s.rect.x + (s.value / s.max_val) * s.rect.width 
            s.rect.y = 30 + i * 60
            s.update_handle_position()
            s.handle_rect.y = s.rect.y - 5
        buttons[0].rect.x = right_panel_rect.x + 25
        buttons[1].rect.x = right_panel_rect.x + 25
        buttons[2].rect.x = right_panel_rect.x + 25
        buttons[3].rect.x = right_panel_rect.x + 25

    update_layout(1280, 720)

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.VIDEORESIZE:
                update_layout(ev.w, ev.h)
                screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            idx = left_panel.handle_event(ev)
            if idx is not None:
                current = idx
                desc_text = effects[idx]["desc"]
                redo_stack.clear()
                if img_disp is not None:
                    if len(history) < max_history:
                        history.append(processed.copy())
                    else:
                        history.pop(0)
                        history.append(processed.copy())
                    proc = effects[current]["func"](img_disp, sliders[3].value)
                    processed = adjust_image(proc, sliders[0].value, sliders[1].value, sliders[2].value)
            if ev.type == pygame.MOUSEBUTTONDOWN:
                for s in sliders:
                    if s.rect.collidepoint(ev.pos) or s.handle_rect.collidepoint(ev.pos):
                        s.dragging = True
                for i, btn in enumerate(buttons):
                    if btn.rect.collidepoint(ev.pos):
                        if i == 0:  # Load
                            path = browse_image()
                            if path:
                                img_orig = cv2.imread(path)
                                h, w = img_orig.shape[:2]
                                scale = min((screen.get_width() - 400) / w, screen.get_height() / h)
                                img_disp = cv2.resize(img_orig, (int(w * scale), int(h * scale)))
                                processed = img_disp.copy()
                                history = []
                                redo_stack = []
                        elif i == 1 and processed is not None:  # Save
                            save_image_dialog(processed)
                        elif i == 2 and history:  # Undo
                            redo_stack.append(processed.copy())
                            processed = history.pop()
                        elif i == 3 and redo_stack:  # Redo
                            history.append(processed.copy())
                            processed = redo_stack.pop()
            if ev.type == pygame.MOUSEBUTTONUP:
                for s in sliders:
                    s.dragging = False
            if ev.type == pygame.MOUSEMOTION:
                for s in sliders:
                    if s.dragging:
                        s.update(ev.pos)
                        if img_disp is not None:
                            if len(history) < max_history:
                                history.append(processed.copy())
                            else:
                                history.pop(0)
                                history.append(processed.copy())
                            redo_stack.clear()
                            proc = effects[current]["func"](img_disp, sliders[3].value)
                            processed = adjust_image(proc, sliders[0].value, sliders[1].value, sliders[2].value)

        # Draw
        screen.fill((30, 30, 30))
        left_panel.draw(screen, current)
        if processed is not None:
            surf = pygame.image.frombuffer(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).tobytes(), processed.shape[1::-1], "RGB")
            img_rect = surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(surf, img_rect)
        pygame.draw.rect(screen, (50, 50, 50), right_panel_rect)
        for s in sliders:
            s.draw(screen, font)
        for btn in buttons:
            btn.check_hover(mouse_pos)
            btn.draw(screen, font)
        pygame.draw.rect(screen, (50, 50, 50), desc_rect)
        lines = desc_text.split(' ')
        y = desc_rect.y + 10
        line = ""
        for word in lines:
            test_line = line + word + " "
            if small_font.render(test_line, True, (255, 255, 255)).get_width() < desc_rect.width - 20:
                line = test_line
            else:
                screen.blit(small_font.render(line, True, (255, 255, 255)), (desc_rect.x + 10, y))
                line = word + " "
                y += 20
        if line:
            screen.blit(small_font.render(line, True, (255, 255, 255)), (desc_rect.x + 10, y))

        pygame.display.flip()
        clock.tick(FPS)
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())