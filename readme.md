# OpenCV Accelerator Sample

## Overview
This repository contains a Python script that demonstrates the usage of OpenCV acceleration on the RZ/V2MA platform. The script benchmarks various OpenCV functions, comparing their execution times when run on the CPU versus the DRP (Dynamic Reconfigurable Processor).

## Features
- Implements a variety of OpenCV image processing functions, such as:
  - Image resizing
  - Color space conversions
  - Filtering and thresholding
  - Morphological operations
  - Template matching
  - Warping transformations
- Measures and compares execution times for CPU and DRP execution modes.
- Outputs processed images for both CPU and DRP modes.
- Provides an easy-to-use interface for running the benchmark tests.

## Prerequisites
- Yocto Weston image for VKRZV2L with OpenCV and enabled DRP OpenCV.
- The image should be based on meta-vkboards ([meta-vkboards repository](https://github.com/Vekatech/meta-vkboards.git)) at commit `b47460e8b4f0ea2315c85e1221b503caa5d621a2` or later.

## Usage

### Running the Script
To run the script, provide the input folder (containing test images) and the output folder (where results will be saved):
### Example
```sh
python oca_sample.py ./resources ./output
```

### Expected Output
- The script will execute multiple image processing functions.
- It will display CPU and DRP execution times along with the speedup factor.
- Processed images will be saved in the specified output folder.

## Performance Metrics
For each operation, the script measures:
- CPU execution time
- DRP execution time
- Speedup factor (CPU time / DRP time)

## Supported OpenCV Functions
The script includes wrappers for measuring execution times of the following OpenCV functions:

- `cv2.resize`
- `cv2.cvtColor`
- `cv2.cvtColorTwoPlane`
- `cv2.GaussianBlur`
- `cv2.dilate`
- `cv2.erode`
- `cv2.morphologyEx`
- `cv2.filter2D`
- `cv2.Sobel`
- `cv2.adaptiveThreshold`
- `cv2.matchTemplate`
- `cv2.warpAffine`
- `cv2.warpPerspective`
- `cv2.pyrDown`
- `cv2.pyrUp`

## File Structure
```
📂 project_root
 ├── oca_sample.py         # Main script
 ├── README.md             # Documentation
 ├── LICENSE               # license file
 ├── resources/            # Input images directory
 ```

## License
This project is released under the MIT License.

## Acknowledgments
This project is designed for benchmarking OpenCV acceleration on the RZ/V2MA platform. Contributions and optimizations are welcome!

