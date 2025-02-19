import os
import sys

import cv2
import numpy as np

# Define OpenCV Accelerator function states
OPENCVA_FUNC_DISABLE = 0
OPENCVA_FUNC_ENABLE = 1
OPENCVA_FUNC_NOCHANGE = 2
OPENCVA_FUNCS = 16

KERNEL_0 = np.zeros((0, 0), np.uint8)

OCA_list = np.full((OPENCVA_FUNCS, 1), OPENCVA_FUNC_NOCHANGE, dtype=np.int64)

FILTER2D_KERNEL = np.array([
    [-0.2, -0.2, -0.2],
    [-0.2, 2.6, -0.2],
    [-0.2, -0.2, -0.2]
], dtype=np.float32)

WARP_AFFINE_MATRIX = np.array([
    [0.7071, -0.7071, 649],
    [0.7071, 0.7071, 510.0]
], dtype=np.float32)

WARP_PERSPECTIVE_MATRIX = np.array([
    [0.5, 0.2, 20],
    [-0.1, 0.8, 50],
    [-0.001, 0.001, 1.0]
], dtype=np.float32)


def activate_drp(indexes: tuple, mode: int):
    OCA_list[np.array(indexes), 0] = mode
    return cv2.DRP_Activate(OCA_list.view(np.int32).reshape(2 * OPENCVA_FUNCS, 1))


def measure_cv_time(func, *args):
    """ Measure only the execution time of the OpenCV function itself """
    start_tick = cv2.getTickCount()
    result = func(*args)
    end_tick = cv2.getTickCount()
    execution_time = (end_tick - start_tick) / cv2.getTickFrequency()
    return result, execution_time


def resize(image):
    return cv2.resize(image, (1024, 768))


def resize_wrapper(image):
    return measure_cv_time(resize, image)


def cvt_color(image):
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)


def cvt_color_wrapper(image):
    return measure_cv_time(cvt_color, image)


def cvt_color_two_plane(image1, image2):
    return cv2.cvtColorTwoPlane(image1, image2, cv2.COLOR_YUV2RGB_NV21)


def cvt_color_two_plane_wrapper(image1, image2):
    return measure_cv_time(cvt_color_two_plane, image1, image2)


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0, 0)


def gaussian_blur_wrapper(image):
    return measure_cv_time(gaussian_blur, image)


def dilate(image):
    return cv2.dilate(image, kernel=KERNEL_0, anchor=(-1, -1), iterations=200)


def dilate_wrapper(image):
    return measure_cv_time(dilate, image)


def erode(image):
    return cv2.erode(image, kernel=KERNEL_0, anchor=(-1, -1), iterations=100)


def erode_wrapper(image):
    return measure_cv_time(erode, image)


def morphology_ex(image):
    return cv2.morphologyEx(image, op=cv2.MORPH_OPEN, kernel=KERNEL_0, anchor=(-1, -1), iterations=50)


def morphology_ex_wrapper(image):
    return measure_cv_time(morphology_ex, image)


def filter_2d(image):
    return cv2.filter2D(image, ddepth=-1, kernel=FILTER2D_KERNEL)


def filter_2d_wrapper(image):
    return measure_cv_time(filter_2d, image)


def sobel(image):
    return cv2.Sobel(image, ddepth=-1, dx=1, dy=0)


def sobel_wrapper(image):
    return measure_cv_time(sobel, image)


def apply_adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                 thresholdType=cv2.THRESH_BINARY, blockSize=99, C=0)


def apply_adaptive_threshold_wrapper(image):
    return measure_cv_time(apply_adaptive_threshold, image)


def match_template(image, template, method):
    return cv2.matchTemplate(image, templ=template, method=method)


def match_template_wrapper(image):
    scr_img = image[400:400 + 360, 800:800 + 640]
    scr_img = cv2.UMat(scr_img)
    tpl_image = image[570:570 + 16, 1190:1190 + 16]
    tpl_image = cv2.UMat(tpl_image)

    result, exec_time = measure_cv_time(cv2.matchTemplate, scr_img, tpl_image, cv2.TM_SQDIFF)

    # dst_image = np.array(result, dtype=np.float32)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = (int(min_loc[0] + 800), int(min_loc[1] + 400))
    bottom_right = (int(min_loc[0] + 816), int(min_loc[1] + 416))

    return cv2.rectangle(image, top_left, bottom_right, (128, 0, 128), 3), exec_time


def warp_affine(image):
    return cv2.warpAffine(image, WARP_AFFINE_MATRIX, (1920, 1080))


def warp_affine_wrapper(image):
    return measure_cv_time(warp_affine, image)


def warp_perspective(image):
    return cv2.warpPerspective(image, WARP_PERSPECTIVE_MATRIX, (1920, 1080))


def warp_perspective_wrapper(image):
    return measure_cv_time(warp_perspective, image)


def pyr_down(image):
    return cv2.pyrDown(image, (1920 / 2, 1080 / 2))


def pyr_down_wrapper(image):
    return measure_cv_time(pyr_down, image)


def pyr_up(image):
    return cv2.pyrUp(image, (1920, 1080))


def pyr_up_wrapper(image):
    return measure_cv_time(pyr_up, image)


def load_img(filename: str):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not open or find {filename}")
        sys.exit(1)
    return image


def load_grayscalar_img(filename: str):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not open or find {filename}")
        sys.exit(1)
    return image


def load_mat_npy(filename: str):
    with open(filename, "rb") as f:
        # OpenCV Mat Type to NumPy dtype Mapping
        cv_type_to_dtype = {
            cv2.CV_8U: np.uint8, cv2.CV_8S: np.int8,
            cv2.CV_16U: np.uint16, cv2.CV_16S: np.int16,
            cv2.CV_32S: np.int32, cv2.CV_32F: np.float32,
            cv2.CV_64F: np.float64
        }

        shape = np.frombuffer(f.read(8), dtype=np.int32)  # Read shape (2x int32)
        mat_type = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Read type

        # Check for multi-channel types
        depth = mat_type & 7  # Extracts base type
        channels = (mat_type >> 3) + 1  # Extracts channel count

        if depth not in cv_type_to_dtype:
            raise ValueError(f"Unsupported OpenCV type: {mat_type}")

        dtype = cv_type_to_dtype[depth]

        # Read raw data
        data = np.frombuffer(f.read(), dtype=dtype)

        # Reshape including channels if multi-channel
        if channels > 1:
            data = data.reshape((shape[0], shape[1], channels))
        else:
            data = data.reshape(shape)

    return cv2.UMat(data)  # Convert to UMat


# Define DRP function indices
tests = {
    "[1] resize              FHD(BGR) -> XGA(BGR)": (load_img, ("image.png",), resize_wrapper, (0,)),
    "[2] cvtColor            FHD(YUV) -> FHD(BGR)": (load_mat_npy, ("cvtColor.npy",), cvt_color_wrapper, (2,)),
    "[3] cvtColorTwoPlane    FHD(NV)  -> FHD(BGR)": (
        load_mat_npy, ("cvtColorTwoPlane1.npy", "cvtColorTwoPlane2.npy"), cvt_color_two_plane_wrapper, (2,)),
    "[4] GaussianBlur        FHD(BGR) [7x7]": (load_img, ("image.png",), gaussian_blur_wrapper, (4,)),
    "[5] dilate              FHD(BGR) [iteration=200]": (load_img, ("image.png",), dilate_wrapper, (5,)),
    "[6] erode               FHD(BGR) [iteration=100]": (load_img, ("image.png",), erode_wrapper, (6,)),
    "[7] morphologyEX        FHD(BGR) [iteration= 50]": (
        load_img, ("image.png",), morphology_ex_wrapper, (5, 6)),
    "[8] filter2D            FHD(BGR)": (load_img, ("image.png",), filter_2d_wrapper, (7,)),
    "[9] Sobel               FHD(BGR)": (load_img, ("image.png",), sobel_wrapper, (8,)),
    "[10] adaptiveThreshold  FHD(gray)[kernel= 99x99]": (
        load_grayscalar_img, ("image.png",), apply_adaptive_threshold_wrapper, (9,)),
    "[11] matchTemplate  640x360(BGR) [template 16x16]": (
        load_img, ("image.png",), match_template_wrapper, (10,)),
    "[12] warpAffine         FHD(BGR) [rotate PI/4]": (load_img, ("image.png",), warp_affine_wrapper, (11,)),
    "[13] warpPerspective    FHD(BGR)": (load_img, ("image.png",), warp_perspective_wrapper, (14,)),
    "[14] pyrDown            FHD(BGR) -> QFHD(BGR)": (load_img, ("image.png",), pyr_down_wrapper, (12,)),
    "[15] pyrUp              QFHD(BGR) -> FHD(BGR)": (load_img, ("small.png",), pyr_up_wrapper, (13,)),
}


def main():
    if len(sys.argv) < 3:
        print("Usage: python oca_sample.py <input folder>, <output folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    os.makedirs(output_folder, exist_ok=True)

    print("RZ/V2MA OPENCV SAMPLE")

    for test_name, data_ in tests.items():
        load, input_files, function, index = data_

        args = []
        for e in input_files:
            filename = os.path.join(input_folder, e)
            args.append(load(filename))

        # CPU Test
        activate_drp(index, OPENCVA_FUNC_DISABLE)
        result_cpu, cpu_time = function(*args)

        # DRP Test
        activate_drp(index, OPENCVA_FUNC_ENABLE)
        result_drp, drp_time = function(*args)

        activate_drp(index, OPENCVA_FUNC_NOCHANGE)

        speedup = cpu_time / drp_time if drp_time > 0 else float('inf')
        print(f"{test_name} - CPU Time: {cpu_time * 1e3:.1f} ms")
        print(f"{test_name} - DRP Time: {drp_time * 1e3:.1f} ms")
        print(f"{test_name} - Speedup Factor: {speedup:.2f}x\n")

        file_name = test_name.split("]")[1].strip().split()[0]

        file_path = os.path.join(output_folder, f"{file_name.lower()}_cpu.png")
        cv2.imwrite(file_path, result_cpu)

        file_path = os.path.join(output_folder, f"{file_name.lower()}_drp.png")
        cv2.imwrite(file_path, result_drp)

if __name__ == "__main__":
    main()
