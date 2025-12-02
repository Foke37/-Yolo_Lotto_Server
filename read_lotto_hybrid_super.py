import argparse
import cv2
import os
import numpy as np
import pytesseract
from ultralytics import YOLO
import time
from multiprocessing import Pool


# Point this to your Tesseract install if needed
default_tess_cmd = "/usr/bin/tesseract" if os.name != "nt" else r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", default_tess_cmd)

def ocr_worker(args):
    gray_crop, resize_scale, invert, threshold_method, adaptive_block, adaptive_c, binary_thresh = args
    text, used_inv = threshold_ocr_with_invert_gray(
        gray_crop,
        resize_scale=resize_scale,
        invert=invert,
        threshold_method=threshold_method,
        adaptive_block=adaptive_block,
        adaptive_c=adaptive_c,
        binary_thresh=binary_thresh,
        save_debug=None
    )
    return text, used_inv


def save_stage_images(debug_base, images):
    """
    Save a dict of stage_name -> image to <debug_base>_<stage>.jpg
    """
    if not debug_base or not images:
        return
    for stage, img in images.items():
        if img is not None:
            cv2.imwrite(f"{debug_base}_{stage}.jpg", img)


def deskew_by_lines_gray(image, thresh_method='adaptive_gaussian', adaptive_block=15,
                         adaptive_c=3, canny_low=50, canny_high=150, hough_thresh=80,
                         min_line_len=50, max_line_gap=10, debug_prefix=None,
                         force_process=True, skip_small_angle_deg=1.0):
    """
    Detect dominant horizontal-ish lines and rotate to level the image (grayscale-friendly).
    Accepts either grayscale (H,W) or BGR (H,W,3) image.
    Returns (rotated_image, median_angle_degrees, did_rotate)
    """
    if image is None:
        return None, 0.0, False

    # Normalize to grayscale for processing
    if image.ndim == 2:
        # (H, W) -> grayscale ตรง ๆ
        gray = image.copy()
        h, w = gray.shape[:2]
        vis_base = None
    elif image.ndim == 3 and image.shape[2] == 1:
        # (H, W, 1) -> squeeze ให้เหลือ 2D
        gray = image[:, :, 0]
        h, w = gray.shape[:2]
        vis_base = None
    else:
        # BGR (H, W, 3)
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_base = image
    # Threshold for calibration
    block = adaptive_block + 1 if adaptive_block % 2 == 0 else adaptive_block
    if thresh_method == 'adaptive_mean':
        calib_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, block, adaptive_c)
    elif thresh_method == 'binary':
        _, calib_thresh = cv2.threshold(gray, adaptive_c, 255, cv2.THRESH_BINARY)
    elif thresh_method == 'otsu':
        _, calib_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # adaptive_gaussian default
        calib_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block, adaptive_c)

    edges = cv2.Canny(calib_thresh, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    if debug_prefix:
        cv2.imwrite(f"{debug_prefix}_calib_thresh.jpg", calib_thresh)
        cv2.imwrite(f"{debug_prefix}_calib_edges.jpg", edges)

    # If no lines found and force_process is True, try alternative strategies
    if (lines is None or len(lines) == 0) and force_process:
        # print(f"    [Calibration] No lines found with initial params, trying alternatives...")

        strategies = [
            {'canny_low': 30, 'canny_high': 100, 'hough_thresh': 50, 'min_line_len': 30},
            {'canny_low': 20, 'canny_high': 80, 'hough_thresh': 40, 'min_line_len': 20},
            {'dilate': True, 'canny_low': 30, 'canny_high': 100, 'hough_thresh': 40, 'min_line_len': 25},
            {'use_otsu': True, 'canny_low': 30, 'canny_high': 100, 'hough_thresh': 50, 'min_line_len': 30},
        ]

        for idx, strategy in enumerate(strategies, 2):
            if strategy.get('use_otsu', False):
                _, calib_thresh_alt = cv2.threshold(gray, 0, 255,
                                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                calib_thresh_alt = calib_thresh.copy()

            edges_alt = cv2.Canny(
                calib_thresh_alt,
                strategy.get('canny_low', canny_low),
                strategy.get('canny_high', canny_high),
                apertureSize=3
            )

            if strategy.get('dilate', False):
                kernel = np.ones((2, 2), np.uint8)
                edges_alt = cv2.dilate(edges_alt, kernel, iterations=1)

            lines = cv2.HoughLinesP(
                edges_alt, 1, np.pi / 180,
                threshold=strategy.get('hough_thresh', hough_thresh),
                minLineLength=strategy.get('min_line_len', min_line_len),
                maxLineGap=strategy.get('max_line_gap', max_line_gap)
            )

            if lines is not None and len(lines) > 0:
                # print(f"    [Calibration] Found {len(lines)} lines with strategy {idx}")
                edges = edges_alt
                break

    # Morphological fallback
    if (lines is None or len(lines) == 0) and force_process:
        # print(f"    [Calibration] Trying morphological operations...")
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        morph = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
        _, morph_thresh = cv2.threshold(morph, 30, 255, cv2.THRESH_BINARY)
        edges_morph = cv2.Canny(morph_thresh, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges_morph, 1, np.pi / 180, threshold=30,
                                minLineLength=20, maxLineGap=15)
        if lines is not None and len(lines) > 0:
            # print(f"    [Calibration] Found {len(lines)} lines with morphological method")
            edges = edges_morph

    # Sobel fallback
    if (lines is None or len(lines) == 0) and force_process:
        # print(f"    [Calibration] Last resort - looking for any horizontal features...")
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h_abs = np.absolute(sobel_h)
        sobel_h_8u = np.uint8(sobel_h_abs)
        _, sobel_thresh = cv2.threshold(sobel_h_8u, 30, 255, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(sobel_thresh, 1, np.pi / 180, threshold=20,
                                minLineLength=15, maxLineGap=20)

    if lines is None or len(lines) == 0:
        # print(f"    [Calibration] WARNING: No lines detected, skipping rotation")
        return image, 0.0, False

    angles = []
    lengths = []
    vis = None
    if debug_prefix:
        if image.ndim == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 60:
            length = np.hypot(x2 - x1, y2 - y1)
            if abs(angle) <= 10:
                length *= 1.5
            angles.append(angle)
            lengths.append(length)
            if vis is not None:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    if debug_prefix and vis is not None:
        cv2.imwrite(f"{debug_prefix}_calib_lines.jpg", vis)

    if not angles:
        # print(f"    [Calibration] No horizontal lines found among {len(lines)} detected lines")
        return image, 0.0, False

    angles_np = np.array(angles)
    lengths_np = np.array(lengths)

    # IQR filter
    q1 = np.percentile(angles_np, 25)
    q3 = np.percentile(angles_np, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (angles_np >= lower) & (angles_np <= upper)
    if np.any(mask):
        angles_np = angles_np[mask]
        lengths_np = lengths_np[mask]

    order = np.argsort(angles_np)
    angles_sorted = angles_np[order]
    weights_sorted = lengths_np[order]
    cumsum = np.cumsum(weights_sorted)
    median_weight = cumsum[-1] / 2.0
    idx = np.searchsorted(cumsum, median_weight)
    median_angle = float(angles_sorted[min(idx, len(angles_sorted) - 1)])

    max_rotation = 15.0
    if abs(median_angle) > max_rotation:
        # print(f"    [Calibration] Angle {median_angle:.2f}° exceeds max {max_rotation}°, clamping")
        median_angle = np.sign(median_angle) * max_rotation

    # If angle is tiny, skip warp for speed
    if abs(median_angle) < skip_small_angle_deg:
        # print(f"    [Calibration] Angle {median_angle:.2f}° < {skip_small_angle_deg}°, skip rotation")
        return image, median_angle, False

    if debug_prefix:
        ref = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ref_y = ref.shape[0] // 2
        cv2.line(ref, (0, ref_y), (ref.shape[1], ref_y), (0, 255, 0), 1)
        cv2.putText(ref, "reference 0 deg", (5, max(ref_y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) <= 60:
                cv2.line(ref, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(ref, f"Median angle: {median_angle:.2f} deg", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{debug_prefix}_calib_ref.jpg", ref)
        with open(f"{debug_prefix}_calib_angle.txt", "w") as f:
            f.write(f"Angle: {median_angle:.3f} degrees\n")
            f.write(f"Lines found: {len(lines)}\n")
            f.write(f"Horizontal lines: {len(angles)}\n")

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    rotated = cv2.warpAffine(gray if image.ndim == 2 else image,
                             rot_mat, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # print(f"    [Calibration] Rotated by {median_angle:.2f}°")
    return rotated, median_angle, True


def check_if_should_invert(gray_image):
    mean_brightness = np.mean(gray_image)
    return mean_brightness < 127


def threshold_ocr_with_invert_gray(gray, resize_scale=None, invert='auto',
                                   threshold_method='otsu', adaptive_block=11,
                                   adaptive_c=2, binary_thresh=70,
                                   save_debug=None):
    """
    Threshold OCR with improved enhancement pipeline:
    - shadow removal
    - adaptive thresholding
    - invert modes (auto, yes, both)
    - OCR enhancement (thinning, clean noise, sharpen, stretch)
    Returns (text, used_invert)
    """

    debug_images = {}

    # ----------------------------------------------------------
    # Resize
    # ----------------------------------------------------------
    if resize_scale and resize_scale > 1:
        gray = cv2.resize(gray, None, fx=resize_scale, fy=resize_scale,
                          interpolation=cv2.INTER_CUBIC)
    debug_images["gray"] = gray

    # ----------------------------------------------------------
    # Shadow removal / lighting correction
    # ----------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bg = cv2.dilate(gray, kernel)
    bg = cv2.medianBlur(bg, 31)

    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_proc = clahe.apply(norm)

    debug_images["gray_shadow_norm"] = norm
    debug_images["gray_proc"] = gray_proc

    gray = gray_proc

    # ----------------------------------------------------------
    # Decide invert
    # ----------------------------------------------------------
    should_invert = False
    if invert == 'auto':
        should_invert = check_if_should_invert(gray)
    elif invert == 'yes':
        should_invert = True
    elif invert == 'both':
        pass

    # ----------------------------------------------------------
    # OCR enhancement function
    # ----------------------------------------------------------
    def enhance_for_ocr(thresh_img):
        """Enhance binary digit image for stronger OCR accuracy (no line removal)."""

        # A) Stroke thinning only
        kernel = np.ones((2, 2), np.uint8)
        thin = cv2.erode(thresh_img, kernel, iterations=1)

        # B) Sharpen edges
        blur = cv2.GaussianBlur(thin, (3,3), 0)
        sharp = cv2.addWeighted(thin, 1.6, blur, -0.6, 0)

        # C) CLAHE contrast boost
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharp)

        # D) Horizontal stretch → OCR accuracy improves
        enhanced = cv2.resize(enhanced, None, fx=1.3, fy=1.0,
                            interpolation=cv2.INTER_CUBIC)

        return enhanced

    # ----------------------------------------------------------
    # Threshold function
    # ----------------------------------------------------------
    def apply_threshold(img, do_invert=False, tag="normal"):
        if do_invert:
            img = 255 - img
            debug_images[f"{tag}_inverted"] = img
        else:
            debug_images.setdefault(f"{tag}_inverted", None)

        # ---- Threshold modes ----
        if threshold_method == 'otsu':
            _, thresh = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif threshold_method == 'adaptive_gaussian':
            block = adaptive_block + 1 if adaptive_block % 2 == 0 else adaptive_block
            thresh = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                adaptive_c
            )
        elif threshold_method == 'adaptive_mean':
            block = adaptive_block + 1 if adaptive_block % 2 == 0 else adaptive_block
            thresh = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block,
                adaptive_c
            )
        elif threshold_method == 'binary':
            _, thresh = cv2.threshold(
                img, binary_thresh, 255,
                cv2.THRESH_BINARY
            )
        else:
            _, thresh = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # ---- Morphological cleaning ----
        kernel_open = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        debug_images[f"{tag}_thresh"] = thresh
        return thresh

    # ==========================================================
    # MODE 1 — invert='both'
    # ==========================================================
    if invert == 'both':
        # ---- NORMAL ----
        thresh_normal = apply_threshold(gray, False, "normal")
        thresh_normal = enhance_for_ocr(thresh_normal)

        text_normal = pytesseract.image_to_string(
            thresh_normal,
            config='--psm 8 -c tessedit_char_whitelist=0123456789'
        ).strip()

        # ---- INVERTED ----
        thresh_inv = apply_threshold(gray, True, "inverted")
        thresh_inv = enhance_for_ocr(thresh_inv)

        text_inv = pytesseract.image_to_string(
            thresh_inv,
            config='--psm 8 -c tessedit_char_whitelist=0123456789'
        ).strip()

        # ---- Decision logic ----
        if len(text_normal) == 6 and len(text_inv) != 6:
            final_text = text_normal
            used_invert = False
            debug_images["final"] = thresh_normal
        elif len(text_inv) == 6 and len(text_normal) != 6:
            final_text = text_inv
            used_invert = True
            debug_images["final"] = thresh_inv
        elif len(text_normal) == len(text_inv):
            final_text = text_normal
            used_invert = False
            debug_images["final"] = thresh_normal
        else:
            if len(text_normal) >= len(text_inv):
                final_text = text_normal
                used_invert = False
                debug_images["final"] = thresh_normal
            else:
                final_text = text_inv
                used_invert = True
                debug_images["final"] = thresh_inv

    # ==========================================================
    # MODE 2 — single invert mode
    # ==========================================================
    else:
        thresh = apply_threshold(gray, should_invert, "single")
        thresh = enhance_for_ocr(thresh)

        final_text = pytesseract.image_to_string(
            thresh,
            config='--psm 8 -c tessedit_char_whitelist=0123456789'
        ).strip()

        used_invert = should_invert
        debug_images["final"] = thresh

    # ----------------------------------------------------------
    # Save debug images if requested
    # ----------------------------------------------------------
    if save_debug:
        save_stage_images(save_debug, debug_images)

    return final_text, used_invert

def read_ticket_hybrid_fast(image_path=None, model_path=None, conf_threshold=0.5,image_array=None,device="cuda",mp_threshold=40,
                            method='threshold', resize_scale=2, padding=5,
                            invert='auto', threshold_method='otsu',
                            adaptive_block=11, adaptive_c=2, binary_thresh=70,
                            save_crops=False, debug_dir="debug",
                            calibrate_lines=False,
                            calibrate_threshold_method='adaptive_gaussian',
                            calibrate_adaptive_block=15,
                            calibrate_adaptive_c=3,
                            calibrate_canny_low=50,
                            calibrate_canny_high=150,
                            calibrate_hough_thresh=80,
                            calibrate_min_line_len=50,
                            calibrate_max_line_gap=10):

    # ------------- Setup -------------
    start = time.time()
    
    model = YOLO(model_path)
    if device == "cpu":
        model.to("cpu")
    else:
        model.to("cuda")

    # image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if image_gray is None:
    #     raise FileNotFoundError(f"Cannot read image: {image_path}")
    if image_array is not None:
        image_gray = image_array.copy()

    elif image_path is not None:
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

    else:
        raise ValueError("You must provide either image_path or image_array.")

    image_for_yolo = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    results = model(image_for_yolo, conf=conf_threshold)[0]

    detections = len(results.boxes) if results.boxes is not None else 0
    yolo_time = time.time() - start

    numbers = []
    invert_used = []

    ocr_start = time.time()

    # No detections
    if results.boxes is None or detections == 0:
        return 0, [], image_for_yolo, {}, []

    # ---------------------------
    # CASE 1: Multiprocessing OCR
    # ---------------------------
    use_mp = detections >= mp_threshold
    if use_mp:

        crop_args = []

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_gray.shape[1], x2 + padding)
            y2 = min(image_gray.shape[0], y2 + padding)

            crop = image_gray[y1:y2, x1:x2]

            # Calibration (fast)
            if calibrate_lines:
                calibrated, angle, did_rotate = deskew_by_lines_gray(
                    crop,
                    thresh_method=calibrate_threshold_method,
                    adaptive_block=calibrate_adaptive_block,
                    adaptive_c=calibrate_adaptive_c,
                    canny_low=calibrate_canny_low,
                    canny_high=calibrate_canny_high,
                    hough_thresh=calibrate_hough_thresh,
                    min_line_len=calibrate_min_line_len,
                    max_line_gap=calibrate_max_line_gap,
                    debug_prefix=None,
                    force_process=True,
                    skip_small_angle_deg=1.0
                )
                crop_for_ocr = calibrated if did_rotate else crop
            else:
                crop_for_ocr = crop
            # ADD THIS (simulate save-crop effect)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encimg = cv2.imencode('.jpg', crop_for_ocr, encode_param)
            crop_for_ocr = cv2.imdecode(encimg, 0)

            crop_args.append((
                crop_for_ocr,
                resize_scale,
                invert,
                threshold_method,
                adaptive_block,
                adaptive_c,
                binary_thresh
            ))

        # Run multiprocessing
        with Pool() as p:
            results_ocr = p.map(ocr_worker, crop_args)

        for text, used_inv in results_ocr:
            numbers.append(text)
            invert_used.append(used_inv)

    # ---------------------------
    # CASE 2: Normal OCR
    # ---------------------------
    else:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_gray.shape[1], x2 + padding)
            y2 = min(image_gray.shape[0], y2 + padding)

            crop = image_gray[y1:y2, x1:x2]

            if calibrate_lines:
                calibrated, angle, did_rotate = deskew_by_lines_gray(
                    crop,
                    thresh_method=calibrate_threshold_method,
                    adaptive_block=calibrate_adaptive_block,
                    adaptive_c=calibrate_adaptive_c,
                    canny_low=calibrate_canny_low,
                    canny_high=calibrate_canny_high,
                    hough_thresh=calibrate_hough_thresh,
                    min_line_len=calibrate_min_line_len,
                    max_line_gap=calibrate_max_line_gap,
                    debug_prefix=None,
                    force_process=True,
                    skip_small_angle_deg=1.0
                )
                crop_for_ocr = calibrated if did_rotate else crop
            else:
                crop_for_ocr = crop

            # ADD THIS (simulate save-crop effect)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encimg = cv2.imencode('.jpg', crop_for_ocr, encode_param)
            crop_for_ocr = cv2.imdecode(encimg, 0)

            text, used_inv = threshold_ocr_with_invert_gray(
                crop_for_ocr,
                resize_scale=resize_scale,
                invert=invert,
                threshold_method=threshold_method,
                adaptive_block=adaptive_block,
                adaptive_c=adaptive_c,
                binary_thresh=binary_thresh,
                save_debug=None
            )

            numbers.append(text)
            invert_used.append(used_inv)

    # ---------------------------
    # Final timing
    # ---------------------------
    ocr_time = time.time() - ocr_start
    total_time = time.time() - start

    times = {
        "total": total_time,
        "yolo": yolo_time,
        "ocr": ocr_time,
        "ocr_per_box": ocr_time / max(detections, 1)
    }

    return detections, numbers, image_for_yolo, times, invert_used

def main():
    parser = argparse.ArgumentParser(
        description="Lottery OCR (grayscale + hybrid fast)")

    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--weights", default="lotto2.pt")

    parser.add_argument("--conf", type=float, default=0.3)

    parser.add_argument("--method",
                        choices=['direct', 'grayscale', 'threshold'],
                        default='threshold')
    parser.add_argument("--resize", type=int, default=3)
    parser.add_argument("--padding", type=int, default=5)

    parser.add_argument("--invert",
                        choices=['auto', 'yes', 'no', 'both'],
                        default='both')

    parser.add_argument("--threshold-method",
                        choices=['otsu', 'adaptive_gaussian',
                                 'adaptive_mean', 'binary'],
                        default='binary')
    parser.add_argument("--adaptive-block", type=int, default=11)
    parser.add_argument("--adaptive-c", type=int, default=3)
    parser.add_argument("--binary-thresh", type=int, default=160)

    parser.add_argument("--calibrate-lines", action="store_true")
    parser.add_argument("--calibrate-threshold-method",
                        choices=['otsu', 'adaptive_gaussian',
                                 'adaptive_mean', 'binary'],
                        default='adaptive_gaussian')
    parser.add_argument("--calibrate-adaptive-block", type=int, default=15)
    parser.add_argument("--calibrate-adaptive-c", type=int, default=3)
    parser.add_argument("--calibrate-canny-low", type=int, default=50)
    parser.add_argument("--calibrate-canny-high", type=int, default=150)
    parser.add_argument("--calibrate-hough-thresh", type=int, default=80)
    parser.add_argument("--calibrate-min-line-len", type=int, default=50)
    parser.add_argument("--calibrate-max-line-gap", type=int, default=10)

    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", default=None)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--debug-dir", default="debug")

    args = parser.parse_args()

    # print(f"Processing (grayscale): {args.image}")
    # print(f"  Weights: {args.weights}")
    # print(f"  Method: {args.method}")
    # print(f"  Resize: {args.resize}x")
    # print(f"  Invert: {args.invert}")
    # if args.method == 'threshold':
    #     print(f"  Threshold: {args.threshold_method}")
    #     if args.threshold_method.startswith('adaptive'):
    #         print(f"  Adaptive params: block={args.adaptive_block}, C={args.adaptive_c}")
    # if args.calibrate_lines:
    #     print("  Line calibration: ENABLED (skip if angle < 1°)")
    # print("=" * 60)

    detections, numbers, annotated, times, invert_used = read_ticket_hybrid_fast(
        args.image,
        args.weights,
        args.conf,
        method=args.method,
        resize_scale=args.resize,
        padding=args.padding,
        invert=args.invert,
        threshold_method=args.threshold_method,
        adaptive_block=args.adaptive_block,
        adaptive_c=args.adaptive_c,
        binary_thresh=args.binary_thresh,
        save_crops=args.save_crops,
        debug_dir=args.debug_dir,
        calibrate_lines=args.calibrate_lines,
        calibrate_threshold_method=args.calibrate_threshold_method,
        calibrate_adaptive_block=args.calibrate_adaptive_block,
        calibrate_adaptive_c=args.calibrate_adaptive_c,
        calibrate_canny_low=args.calibrate_canny_low,
        calibrate_canny_high=args.calibrate_canny_high,
        calibrate_hough_thresh=args.calibrate_hough_thresh,
        calibrate_min_line_len=args.calibrate_min_line_len,
        calibrate_max_line_gap=args.calibrate_max_line_gap
    )

    # Filter only 6-digit numbers for output as requested
    valid_numbers = [(i + 1, n, invert_used[i])
                     for i, n in enumerate(numbers)
                     if len(n) == 6]

    # print(f"\nโมเดลเจอทั้งหมด {detections} กล่อง")
    # print(f"อ่านครบ 6 หลักได้ {len(valid_numbers)} กล่อง (แสดงเฉพาะชุด 6 หลัก)")

    # print("-" * 30)
    for idx, num, inv in valid_numbers:
        
        print(f"{num}")

    # print("-" * 30)
    if detections > 0:
        accuracy = 100 * len(valid_numbers) / detections
        print(f"ความแม่นยำ (นับเฉพาะกล่องที่อ่านครบ 6 หลัก): {accuracy:.1f}%")

    print(f"\n⏱️  Performance:")
    print(f"  Total: {times['total']:.3f}s")
    print(f"  YOLO: {times['yolo']:.3f}s")
    print(f"  OCR: {times['ocr']:.3f}s ({times['ocr_per_box']:.3f}s/box)")

    if annotated is not None and args.show:
        cv2.imshow("Lottery OCR (hybrid fast)", annotated)
        # print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if annotated is not None and args.save:
        cv2.imwrite(args.save, annotated)
        # print(f"Saved annotated image to: {args.save}")


if __name__ == "__main__":
    main()
