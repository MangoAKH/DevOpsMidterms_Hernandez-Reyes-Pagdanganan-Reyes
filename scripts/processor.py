import cv2
import os
import numpy as np


# -------------------- 3D ANAGLYPH EFFECT --------------------
def apply_anaglyph_full(img, shift=35):
    """Apply strong red-cyan 3D anaglyph effect to entire image."""
    b, g, r = cv2.split(img)

    r_shifted = np.roll(r, -shift, axis=1)
    b_shifted = np.roll(b, shift, axis=1)

    anaglyph = cv2.merge((b_shifted, g, r_shifted))
    return anaglyph


# -------------------- TOP LABEL --------------------
def label_image_top(img, text, font_scale=1, thickness=1, padding=20):
    """Add large label above image."""
    h, w = img.shape[:2]

    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_w, text_h = text_size

    label_h = text_h + 2 * padding
    label_img = np.zeros((label_h, w, 3), dtype=np.uint8)

    x_text = (w - text_w) // 2
    y_text = padding + text_h

    cv2.putText(
        label_img,
        text,
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return np.vstack((label_img, img))


# -------------------- WATERMARK --------------------
def add_stylish_watermark(img, text="GAGRADUATE AKO"):
    """BIG bold stylish watermark."""
    wm_img = img.copy()
    h, w = img.shape[:2]

    font_scale = w / 300
    thickness = int(font_scale * 5)

    overlay = wm_img.copy()

    text_size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
    )
    text_w, text_h = text_size

    x = (w - text_w) // 2
    y = h - 40

    # Shadow
    cv2.putText(
        overlay,
        text,
        (x + 4, y + 4),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )

    # Main bright text
    cv2.putText(
        overlay,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    cv2.addWeighted(overlay, 0.8, wm_img, 0.2, 0, wm_img)
    return wm_img


# -------------------- PROCESS IMAGES --------------------
def process_images():
    input_dir = "inputs"
    output_dir = "outputs"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Dummy image if empty
    if len(os.listdir(input_dir)) == 0:
        dummy = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.putText(
            dummy, "TEST", (150, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6
        )
        cv2.imwrite(os.path.join(input_dir, "test.jpg"), dummy)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 1️⃣ Reduced
        reduced = img.copy()
        reduced = label_image_top(reduced, "Reduced")
        cv2.imwrite(os.path.join(output_dir, f"reduced_{filename}"), reduced)

        # 2️⃣ Watermark
        watermarked = add_stylish_watermark(img)
        watermarked = label_image_top(watermarked, "Watermark")
        cv2.imwrite(os.path.join(output_dir, f"watermark_{filename}"), watermarked)

        # 3️⃣ Fisheye
        map_y, map_x = np.indices((h, w), dtype=np.float32)
        cx, cy = w / 2, h / 2

        nx = (map_x - cx) / cx
        ny = (map_y - cy) / cy
        r = np.sqrt(nx**2 + ny**2)

        factor = np.where(r != 0, (r ** 1.6) / r, 0)
        fx = (nx * factor * cx + cx).astype(np.float32)
        fy = (ny * factor * cy + cy).astype(np.float32)

        fisheye = cv2.remap(img, fx, fy, cv2.INTER_LINEAR)
        fisheye = label_image_top(fisheye, "Fisheye")
        cv2.imwrite(os.path.join(output_dir, f"fisheye_{filename}"), fisheye)

        # 4️⃣ 3D Anaglyph Effect
        anaglyph = apply_anaglyph_full(img, shift=35)
        anaglyph = cv2.convertScaleAbs(anaglyph, alpha=1.4, beta=0)
        anaglyph = label_image_top(anaglyph, "3D Anaglyph Effect")
        cv2.imwrite(
            os.path.join(output_dir, f"3d_anaglyph_effect_{filename}"),
            anaglyph,
        )

        # 5️⃣ Geometry Reconstruction
        geometry = np.zeros_like(img)
        subdiv = cv2.Subdiv2D((0, 0, w, h))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        pts = np.column_stack(np.where(edges > 0))

        if len(pts) > 1000:
            pts = pts[np.random.choice(len(pts), 1000, replace=False)]

        for y, x in pts:
            subdiv.insert((int(x), int(y)))

        # Insert corners
        for c in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            subdiv.insert(c)

        triangles = subdiv.getTriangleList()

        for t in triangles:
            tri = [(int(t[i]), int(t[i + 1])) for i in range(0, 6, 2)]

            if not all(0 <= x < w and 0 <= y < h for x, y in tri):
                continue

            cx_t = sum(p[0] for p in tri) // 3
            cy_t = sum(p[1] for p in tri) // 3

            color = img[cy_t, cx_t].tolist()

            cv2.fillConvexPoly(
                geometry,
                np.array(tri, dtype=np.int32),
                color,
            )

        geometry = label_image_top(geometry, "Geometry")
        cv2.imwrite(os.path.join(output_dir, f"geometry_{filename}"), geometry)

        # ---------------- OPEN ALL WINDOWS AT ONCE ----------------
        effects = {
            "Reduced": reduced,
            "Watermark": watermarked,
            "Fisheye": fisheye,
            "3D Anaglyph Effect": anaglyph,
            "Geometry": geometry,
        }

        for name, image in effects.items():
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 1000, 750)
            cv2.imshow(name, image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images()
