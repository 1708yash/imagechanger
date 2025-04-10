import pygame
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ---------------------------
# Helper functions for dialogs
# ---------------------------
def browse_image():
    # Create a hidden Tk window and force it topmost
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
# Basic Effects (Existing)
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

# ---------------------------
# Additional Effects (Inspired by Fotor)
# ---------------------------
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

# ---------------------------
# Adjustment for brightness, contrast, saturation
# ---------------------------
def adjust_image(img, brightness=1.0, contrast=1.0, saturation=1.0):
    adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=(brightness - 1) * 50)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= saturation
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------------------------
# Effects Registry
# ---------------------------
effects = [
    effect_pencil_sketch,
    effect_color_pencil_sketch,
    effect_stylization,
    effect_cartoon,
    effect_detail_enhance,
    effect_edge_sketch,
    effect_sepia,
    effect_watercolor,
    effect_emboss,
    effect_oil_painting,
    effect_vintage,
    effect_hdr,
    effect_glitch,
    effect_tilt_shift,
    effect_duotone,
    effect_vignette,
    effect_pop,
]

effect_names = [
    "Pencil Sketch",
    "Color Sketch",
    "Stylization",
    "Cartoon",
    "Detail Enhance",
    "Edge Sketch",
    "Sepia",
    "Watercolor",
    "Emboss",
    "Oil Painting",
    "Vintage",
    "HDR",
    "Glitch",
    "Tilt Shift",
    "Duotone",
    "Vignette",
    "Pop Effect",
]

# ---------------------------
# Slider Class for UI Controls
# ---------------------------
class Slider:
    def __init__(self, name, x, y, width, height, min_val, max_val, default):
        self.name = name
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_rect = pygame.Rect(x + int((default - min_val) / (max_val - min_val) * width) - 5, y - 5, 10, height + 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.dragging = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, (220, 220, 220), self.rect, border_radius=8)
        pygame.draw.rect(screen, (255, 100, 100), self.handle_rect, border_radius=5)
        text = font.render(f"{self.name}: {self.value:.1f}", True, (255, 255, 255))
        screen.blit(text, (self.rect.x, self.rect.y - 25))
    
    def update(self, mouse_pos):
        if self.dragging:
            new_x = max(self.rect.x, min(mouse_pos[0], self.rect.x + self.rect.width))
            self.handle_rect.x = new_x - 5
            ratio = (new_x - self.rect.x) / self.rect.width
            self.value = self.min_val + ratio * (self.max_val - self.min_val)

# ---------------------------
# Global Layout Updater
# ---------------------------
def update_layout(width, height, sliders):
    global panel_rect, container_rect, convert_button, save_button, prev_button, next_button
    panel_height = 150
    panel_rect = pygame.Rect(0, height - panel_height, width, panel_height)
    container_max_width = int(width * 0.6)
    container_max_height = int((height - panel_height) * 0.7)
    container_rect = pygame.Rect(
        (width - container_max_width) // 2,
        (height - panel_height - container_max_height) // 2,
        container_max_width,
        container_max_height
    )
    padding = 20
    button_width = 170
    button_height = 45
    convert_button = pygame.Rect(padding, height - panel_height + padding, button_width, button_height)
    save_button = pygame.Rect(padding, height - panel_height + 2 * padding + button_height, button_width, button_height)
    nav_button_width = 120
    prev_button = pygame.Rect(convert_button.right + padding, height - panel_height + padding, nav_button_width, button_height)
    next_button = pygame.Rect(prev_button.right + padding, height - panel_height + padding, nav_button_width, button_height)
    slider_width = 220
    slider_height = 20
    start_x = next_button.right + 2 * padding
    # Update positions for 4 sliders: "Effect Intensity", "Brightness", "Contrast", "Saturation"
    sliders[0].rect = pygame.Rect(start_x, height - panel_height + padding, slider_width, slider_height)
    sliders[0].handle_rect.x = start_x + int((sliders[0].value / 100) * slider_width) - 5
    sliders[0].handle_rect.y = height - panel_height + padding - 5

    sliders[1].rect = pygame.Rect(start_x, height - panel_height + padding + 50, slider_width, slider_height)
    sliders[1].handle_rect.x = start_x + int(((sliders[1].value - 0.0) / 2.0) * slider_width) - 5
    sliders[1].handle_rect.y = height - panel_height + padding + 50 - 5

    sliders[2].rect = pygame.Rect(start_x + slider_width + padding, height - panel_height + padding, slider_width, slider_height)
    sliders[2].handle_rect.x = start_x + slider_width + padding + int(((sliders[2].value - 0.0) / 2.0) * slider_width) - 5
    sliders[2].handle_rect.y = height - panel_height + padding - 5

    sliders[3].rect = pygame.Rect(start_x + slider_width + padding, height - panel_height + padding + 50, slider_width, slider_height)
    sliders[3].handle_rect.x = start_x + slider_width + padding + int(((sliders[3].value - 0.0) / 2.0) * slider_width) - 5
    sliders[3].handle_rect.y = height - panel_height + padding + 50 - 5

# ---------------------------
# Main Application (Resizable Window)
# ---------------------------
def main():
    pygame.init()
    # Start with a default window size; enable RESIZABLE to get OS window controls.
    default_width, default_height = 1280, 720
    screen = pygame.display.set_mode((default_width, default_height), pygame.RESIZABLE)
    pygame.display.set_caption("Professional Image Editor")
    window_width, window_height = screen.get_size()

    # Colors and font choices for a modern look.
    bg_color = (35, 35, 35)
    panel_color = (50, 50, 50)
    accent_color = (70, 130, 180)
    font = pygame.font.SysFont("Arial", 22)

    # Create initial sliders for image adjustments.
    sliders = [
        Slider("Effect Intensity", 0, 0, 220, 20, 0, 100, 50),
        Slider("Brightness", 0, 0, 220, 20, 0.0, 2.0, 1.0),
        Slider("Contrast", 0, 0, 220, 20, 0.0, 2.0, 1.0),
        Slider("Saturation", 0, 0, 220, 20, 0.0, 2.0, 1.0)
    ]
    # Update layout using the current window dimensions.
    update_layout(window_width, window_height, sliders)
    
    # Global layout variables to be updated on window resize:
    global panel_rect, container_rect, convert_button, save_button, prev_button, next_button

    # Image variables
    image_original = None  # The loaded full-resolution image (CV2 format)
    image_display = None   # The image resized to fit into the container
    processed_image = None # Processed image after applying effect & adjustments
    current_effect = 0

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Update the window size and re-layout the UI
                window_width, window_height = event.w, event.h
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                update_layout(window_width, window_height, sliders)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for slider in sliders:
                    if slider.handle_rect.collidepoint(mouse_pos) or slider.rect.collidepoint(mouse_pos):
                        slider.dragging = True
                if convert_button.collidepoint(mouse_pos):
                    path = browse_image()
                    if path:
                        image_original = cv2.imread(path)
                        if image_original is None:
                            break
                        # Resize to fit into the container while preserving aspect ratio.
                        h, w = image_original.shape[:2]
                        scale = min(container_rect.width / w, container_rect.height / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        image_display = cv2.resize(image_original, (new_w, new_h))
                        processed = effects[current_effect](image_display, sliders[0].value / 100.0)
                        processed_image = adjust_image(processed, sliders[1].value, sliders[2].value, sliders[3].value)
                if save_button.collidepoint(mouse_pos) and processed_image is not None:
                    save_image_dialog(processed_image)
                if prev_button.collidepoint(mouse_pos):
                    current_effect = (current_effect - 1) % len(effects)
                    if image_display is not None:
                        processed = effects[current_effect](image_display, sliders[0].value / 100.0)
                        processed_image = adjust_image(processed, sliders[1].value, sliders[2].value, sliders[3].value)
                if next_button.collidepoint(mouse_pos):
                    current_effect = (current_effect + 1) % len(effects)
                    if image_display is not None:
                        processed = effects[current_effect](image_display, sliders[0].value / 100.0)
                        processed_image = adjust_image(processed, sliders[1].value, sliders[2].value, sliders[3].value)
            elif event.type == pygame.MOUSEBUTTONUP:
                for slider in sliders:
                    slider.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                for slider in sliders:
                    if slider.dragging:
                        slider.update(event.pos)
                        if image_display is not None:
                            processed = effects[current_effect](image_display, sliders[0].value / 100.0)
                            processed_image = adjust_image(processed, sliders[1].value, sliders[2].value, sliders[3].value)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Draw background
        screen.fill(bg_color)
        
        # Draw the image container (centered with a subtle border)
        pygame.draw.rect(screen, (80, 80, 80), container_rect, border_radius=10)
        if processed_image is not None:
            img_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_mode = img_pil.mode
            img_size = img_pil.size
            image_surface = pygame.image.fromstring(img_pil.tobytes(), img_size, img_mode)
            # Center the image within the container.
            img_x = container_rect.x + (container_rect.width - img_size[0]) // 2
            img_y = container_rect.y + (container_rect.height - img_size[1]) // 2
            screen.blit(image_surface, (img_x, img_y))
        
        # Draw the control panel at the bottom.
        panel_surface = pygame.Surface((window_width, panel_rect.height))
        panel_surface.set_alpha(220)
        panel_surface.fill(panel_color)
        pygame.draw.rect(panel_surface, panel_color, panel_surface.get_rect(), border_radius=20)
        screen.blit(panel_surface, (panel_rect.x, panel_rect.y))
        
        # Draw the buttons with rounded corners.
        pygame.draw.rect(screen, accent_color, convert_button, border_radius=10)
        conv_text = font.render("Convert Image", True, (255, 255, 255))
        screen.blit(conv_text, (convert_button.x + 10, convert_button.y + 10))
        
        pygame.draw.rect(screen, accent_color, save_button, border_radius=10)
        save_text = font.render("Save Image", True, (255, 255, 255))
        screen.blit(save_text, (save_button.x + 10, save_button.y + 10))
        
        pygame.draw.rect(screen, accent_color, prev_button, border_radius=10)
        prev_text = font.render("Prev Effect", True, (255, 255, 255))
        screen.blit(prev_text, (prev_button.x + 5, prev_button.y + 10))
        
        pygame.draw.rect(screen, accent_color, next_button, border_radius=10)
        next_text = font.render("Next Effect", True, (255, 255, 255))
        screen.blit(next_text, (next_button.x + 5, next_button.y + 10))
        
        # Display the current effect name above the navigation buttons.
        effect_text = font.render("Effect: " + effect_names[current_effect], True, (255, 255, 255))
        screen.blit(effect_text, (prev_button.x, prev_button.y - 30))
        
        # Draw the sliders.
        for slider in sliders:
            slider.draw(screen, font)
        
        pygame.display.flip()
        clock.tick(30)
        
    pygame.quit()

if __name__ == '__main__':
    main()
