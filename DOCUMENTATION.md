# Documentation

## ArresterBedRadar
This is a program designed map surfaces from above, in this particular case the uneven surface of a sand bed. It uses a linear laser to shine onto the uneven surface and the bended line is recorded on a camera. It's implemented in Python and uses the `cv2` package to mask the brightest points and convert them to x,y,z coordinates. It creates two CSV files in the `measurements/` directory: one contains the points (x,y,z) the oder contains the with, depth and cross section (w,d,A) of each layer.

### Requirements
+ Python with the following packages: `cv2`, `numpy`, `matplotlib`, `time`.

### Calibration 
The program is calibrated for a specific camera. For calibration see [this tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) to obtain the `CAMERA_MATRIX` and `DISTORTION_CF`. To calibrate for precise distances tweak the `Z` parameter. 

### Usage
| Key | Option |
|----:|----------|
| `ESC` | Stops program execution. |
| `c` | Change parameters `D` and `H`. |
| `f` | Change filename. |
| `b` | Change lower bound for masking. |
| `s` | Save current surface. |
| `p` | Shows preview of surface (useful to calibrate). |
| `h` | Short help. |

## Converter

This is a converter program to calculate the width, depth and cross section from only the CSV file containing the coordinates of the points.

**Note:** `ArresterBedRadar.py` automatically generates both CSV files, optimally this is not needed.

### Usage 
Requires filename (automatically searches for `measurements/` directory, no file extension) , creates two new CSV files.
