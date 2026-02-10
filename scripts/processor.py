import cv2
import os
import numpy as np


def apply_anaglyph_full(img, shift=35):
    """Apply a strong red-cyan 3D anaglyph effect to the entire image."""
    b, g, r = cv2.split(img)
    r_shifted = np.roll(r, -shift, axis=1)
    b_shifted = np.roll(b, shift, axis=1)
    anaglyph = cv2.merge((b_shifted, g, r_shifted))
    return anaglyph


def label_image_top(img, text, font_scale=2, thickness=3, padding=10):
    """Add a label above the image with a black background bar."""
    h, w = img.shape[:2]
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = text_size
    label_h = text_h + 2 * padding
    label_img = np.zeros((label_h, w, 3), dtype=np.uint8)
    x_text = (w - text_w) // 2
    y_text = padding + text_h
    cv2.putText(label_img, text, (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    labeled = np.vstack((label_img, img))
    return labeled


def combine_images_horizontally(images, spacing=10, bg_color=(0, 0, 0)):
    """Concatenate images horizontally with spacing and uniform height."""
    max_height = max(img.shape[0] for img in images)
    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        if h < max_height:
            pad = np.zeros((max_height - h, w, 3), dtype=np.uint8)
            img = np.vstack((img, pad))
        padded_images.append(img)
    spaced_images = [padded_images[0]]
    for img in padded_images[1:]:
        spacer = np.full((max_height, spacing, 3), bg_color, dtype=np.uint8)
        spaced_images.append(spacer)
        spaced_images.append(img)
    return np.hstack(spaced_images)


def add_stylish_watermark(img, text="GAGRADUATE AKO"):
    """Add a bold, stylish, centered watermark across the image."""
    wm_img = img.copy()
    h, w = img.shape[:2]

    # Settings
    font_scale = w / 800  # Scale font based on width
    thickness = int(font_scale * 8)
    color = (0, 255, 255)  # Bright yellow
    alpha = 0.7  # Transparency of overlay

    # Create overlay
    overlay = wm_img.copy()
    # Centered position
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    text_w, text_h = text_size
    x = (w - text_w) // 2
    y = h - 50  # 50 px from bottom
    # Draw shadow for bold effect
    cv2.putText(overlay, text, (x + 3, y + 3),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2)
    # Draw main text
    cv2.putText(overlay, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
    # Blend overlay
    cv2.addWeighted(overlay, alpha, wm_img, 1 - alpha, 0, wm_img)
    return wm_img


def process_images():
    input_dir = 'inputs'
    output_dir = 'outputs'

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Dummy image for CI / empty inputs
    if len(os.listdir(input_dir)) == 0:
        dummy = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.putText(dummy, "TEST", (150, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.imwrite(os.path.join(input_dir, "test.jpg"), dummy)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # ---------- 1️⃣ Extreme Reduction ----------
        reduced = img.copy()
        cv2.imwrite(os.path.join(output_dir, f"reduced_{filename}"),
                    reduced, [cv2.IMWRITE_JPEG_QUALITY, 1])
        reduced = label_image_top(reduced, "Reduced", font_scale=2, thickness=3)

        # ---------- 2️⃣ Stylish Watermark ----------
        watermarked = add_stylish_watermark(img, "GAGRADUATE AKO")
        cv2.imwrite(os.path.join(output_dir, f"watermark_{filename}"), watermarked)
        watermarked = label_image_top(watermarked, "Watermark", font_scale=2, thickness=3)

        # ---------- 3️⃣ Stable Fisheye ----------
        map_y, map_x = np.indices((h, w), dtype=np.float32)
        cx, cy = w / 2, h / 2
        nx = (map_x - cx) / cx
        ny = (map_y - cy) / cy
        r = np.sqrt(nx ** 2 + ny ** 2)
        factor = np.where(r != 0, (r ** 1.6) / r, 0)
        fx = (nx * factor * cx + cx).astype(np.float32)
        fy = (ny * factor * cy + cy).astype(np.float32)
        fisheye = cv2.remap(img, fx, fy, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, f"fisheye_{filename}"), fisheye)
        fisheye = label_image_top(fisheye, "Fisheye", font_scale=2, thickness=3)

        # ---------- 4️⃣ FULL IMAGE 3D ANAGLYPH EFFECT ----------
        anaglyph_img = apply_anaglyph_full(img, shift=35)
        anaglyph_img = cv2.convertScaleAbs(anaglyph_img, alpha=1.3, beta=0)
        cv2.imwrite(os.path.join(output_dir, f"3d_anaglyph_effect_{filename}"), anaglyph_img)
        anaglyph_img = label_image_top(anaglyph_img, "3D Anaglyph", font_scale=2, thickness=3)

        # ---------- 5️⃣ PURE GEOMETRIC RECONSTRUCTION ----------
        geometry_img = np.zeros_like(img)
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        pts = np.column_stack(np.where(edges > 0))
        if len(pts) > 800:
            pts = pts[np.random.choice(len(pts), 800, replace=False)]
        for y, x in pts:
            subdiv.insert((int(x), int(y)))
        for c in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            subdiv.insert(c)
        for t in subdiv.getTriangleList():
            tri = [(int(t[i]), int(t[i + 1])) for i in range(0, 6, 2)]
            if not all(0 <= x < w and 0 <= y < h for x, y in tri):
                continue
            cx_t = sum(p[0] for p in tri) // 3
            cy_t = sum(p[1] for p in tri) // 3
            color = img[cy_t, cx_t].tolist()
            cv2.fillConvexPoly(
                geometry_img,
                np.array(tri, dtype=np.int32),
                color
            )
        cv2.imwrite(os.path.join(output_dir, f"geometry_{filename}"), geometry_img)
        geometry_img = label_image_top(geometry_img, "Geometry", font_scale=2, thickness=3)

        # ---------- 6️⃣ SHOW ALL EFFECTS TOGETHER ----------
        combined = combine_images_horizontally(
            [reduced, watermarked, fisheye, anaglyph_img, geometry_img],
            spacing=15
        )

        # Use nearly full screen but not fullscreen to keep window controls visible
        cv2.namedWindow(f"All Effects - {filename}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"All Effects - {filename}", 1600, 900)
        cv2.imshow(f"All Effects - {filename}", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images()
