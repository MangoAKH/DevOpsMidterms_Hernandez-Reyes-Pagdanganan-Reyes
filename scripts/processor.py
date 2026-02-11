import cv2
import os
import numpy as np
from PIL import Image, ImageSequence

# Supported formats
STATIC_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".webp")
ANIMATED_FORMATS = (".gif",)
ALL_FORMATS = STATIC_FORMATS + ANIMATED_FORMATS

# -------------------- 3D ANAGLYPH EFFECT --------------------
def apply_anaglyph_full(img, shift=35):
    b, g, r = cv2.split(img)
    r_shifted = np.roll(r, -shift, axis=1)
    b_shifted = np.roll(b, shift, axis=1)
    return cv2.merge((b_shifted, g, r_shifted))

# -------------------- WATERMARK --------------------
def add_stylish_watermark(img, text="GAGRADUATE AKO"):
    wm_img = img.copy()
    h, w = img.shape[:2]
    font_scale = w / 300
    thickness = int(font_scale * 5)
    overlay = wm_img.copy()
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    text_w, text_h = text_size
    x = (w - text_w) // 2
    y = h - 40
    # Shadow
    cv2.putText(overlay, text, (x + 4, y + 4), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    # Main bright text
    cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, wm_img, 0.2, 0, wm_img)
    return wm_img

# -------------------- STATIC IMAGE PROCESSING --------------------
def process_static_image(img_path, output_dir):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    outputs = []

    # Reduced
    reduced_path = os.path.join(output_dir, f"reduced_{filename}")
    cv2.imwrite(reduced_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    outputs.append(reduced_path)

    # Watermark
    watermarked_path = os.path.join(output_dir, f"watermark_{filename}")
    watermarked = add_stylish_watermark(img)
    cv2.imwrite(watermarked_path, watermarked)
    outputs.append(watermarked_path)

    # Fisheye
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2, h / 2
    nx = (map_x - cx) / cx
    ny = (map_y - cy) / cy
    r = np.sqrt(nx**2 + ny**2)
    factor = np.where(r != 0, (r ** 1.6) / r, 0)
    fx = (nx * factor * cx + cx).astype(np.float32)
    fy = (ny * factor * cy + cy).astype(np.float32)
    fisheye = cv2.remap(img, fx, fy, cv2.INTER_LINEAR)
    fisheye_path = os.path.join(output_dir, f"fisheye_{filename}")
    cv2.imwrite(fisheye_path, fisheye)
    outputs.append(fisheye_path)

    # 3D Anaglyph
    anaglyph = apply_anaglyph_full(img)
    anaglyph = cv2.convertScaleAbs(anaglyph, alpha=1.4, beta=0)
    anaglyph_path = os.path.join(output_dir, f"3d_anaglyph_{filename}")
    cv2.imwrite(anaglyph_path, anaglyph)
    outputs.append(anaglyph_path)

    # Geometry Reconstruction
    geometry = np.zeros_like(img)
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    pts = np.column_stack(np.where(edges > 0))
    if len(pts) > 1000:
        pts = pts[np.random.choice(len(pts), 1000, replace=False)]
    for y, x in pts:
        subdiv.insert((int(x), int(y)))
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
        cv2.fillConvexPoly(geometry, np.array(tri, dtype=np.int32), color)
    geometry_path = os.path.join(output_dir, f"geometry_{filename}")
    cv2.imwrite(geometry_path, geometry)
    outputs.append(geometry_path)

    return outputs

# -------------------- ANIMATED IMAGE PROCESSING --------------------
def process_animated_image(img_path, output_dir):
    filename = os.path.basename(img_path)
    pil_img = Image.open(img_path)
    durations = [frame.info.get("duration", 100) for frame in ImageSequence.Iterator(pil_img)]
    frames_effects = {"reduced": [], "watermark": [], "fisheye": [], "anaglyph": [], "geometry": []}

    for frame in ImageSequence.Iterator(pil_img):
        cv_img = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]

        # Reduced
        _, reduced_enc = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        cv_reduced = cv2.imdecode(reduced_enc, cv2.IMREAD_COLOR)

        # Watermark
        cv_watermark = add_stylish_watermark(cv_img)

        # Fisheye
        map_y, map_x = np.indices((h, w), dtype=np.float32)
        cx, cy = w / 2, h / 2
        nx = (map_x - cx) / cx
        ny = (map_y - cy) / cy
        r = np.sqrt(nx**2 + ny**2)
        factor = np.where(r != 0, (r ** 1.6) / r, 0)
        fx = (nx * factor * cx + cx).astype(np.float32)
        fy = (ny * factor * cy + cy).astype(np.float32)
        cv_fisheye = cv2.remap(cv_img, fx, fy, cv2.INTER_LINEAR)

        # 3D Anaglyph
        cv_anaglyph = cv2.convertScaleAbs(apply_anaglyph_full(cv_img), alpha=1.4, beta=0)

        # Geometry
        geometry = np.zeros_like(cv_img)
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        pts = np.column_stack(np.where(edges > 0))
        if len(pts) > 1000:
            pts = pts[np.random.choice(len(pts), 1000, replace=False)]
        for y, x in pts:
            subdiv.insert((int(x), int(y)))
        for c in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]:
            subdiv.insert(c)
        triangles = subdiv.getTriangleList()
        for t in triangles:
            tri = [(int(t[i]), int(t[i+1])) for i in range(0,6,2)]
            if not all(0<=x<w and 0<=y<h for x,y in tri):
                continue
            cx_t = sum(p[0] for p in tri)//3
            cy_t = sum(p[1] for p in tri)//3
            color = cv_img[cy_t, cx_t].tolist()
            cv2.fillConvexPoly(geometry, np.array(tri, dtype=np.int32), color)

        # Save frames per effect
        frames_effects["reduced"].append(Image.fromarray(cv2.cvtColor(cv_reduced, cv2.COLOR_BGR2RGB)))
        frames_effects["watermark"].append(Image.fromarray(cv2.cvtColor(cv_watermark, cv2.COLOR_BGR2RGB)))
        frames_effects["fisheye"].append(Image.fromarray(cv2.cvtColor(cv_fisheye, cv2.COLOR_BGR2RGB)))
        frames_effects["anaglyph"].append(Image.fromarray(cv2.cvtColor(cv_anaglyph, cv2.COLOR_BGR2RGB)))
        frames_effects["geometry"].append(Image.fromarray(cv2.cvtColor(geometry, cv2.COLOR_BGR2RGB)))

    # Save all effects as GIFs
    saved_paths = []
    for effect, frames in frames_effects.items():
        save_path = os.path.join(output_dir, f"{effect}_{filename}")
        frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=durations)
        saved_paths.append(save_path)

    return saved_paths

# -------------------- MAIN PROCESS --------------------
def process_images(show=False):
    input_dir = "inputs"
    output_dir = "outputs"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(input_dir)) == 0:
        dummy = np.zeros((600, 600, 3), dtype=np.uint8)
        cv2.putText(dummy, "TEST", (150, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.imwrite(os.path.join(input_dir, "test.jpg"), dummy)

    all_output_paths = []

    for filename in os.listdir(input_dir):
        path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in STATIC_FORMATS:
            all_output_paths.extend(process_static_image(path, output_dir))
        elif ext in ANIMATED_FORMATS:
            all_output_paths.extend(process_animated_image(path, output_dir))
        else:
            continue

    HEADLESS = os.environ.get("CI") == "true"
    if show and not HEADLESS:
        for path in all_output_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in STATIC_FORMATS:
                img = cv2.imread(path)
                if img is not None:
                    name = os.path.basename(path)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(name, 1000, 750)
                    cv2.imshow(name, img)
            elif ext in ANIMATED_FORMATS:
                pil_img = Image.open(path)
                for frame in ImageSequence.Iterator(pil_img):
                    img = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
                    name = os.path.basename(path)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(name, 1000, 750)
                    cv2.imshow(name, img)
                    cv2.waitKey(int(frame.info.get("duration", 100)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_images(show=True)
