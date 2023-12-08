# Bolts
Based on ZED Stereo Camera

## Introduction

## System Configuration

![ZED Stereo Vision](./assets/images/ZED.png)

<p align="center">
 Figure 1. ZED Stereo Vision Camera
</p>

![WEGO](./assets/images/WEGO.png)

<p align="center">
 Figure 2. WeGo-ST MINI based on ROS
</p>

## Experiments

### Overview of 3D Point Cloud Data Obtained from the ZED Stereo Vision Camera

[![YouTube](./assets/images/thumbnail.jpg)](https://www.youtube.com/watch?v=R5_7TohjqF8)

## Codes

### Calculate the Distance between Two 3D Points from .SVO File

To run the program, use the following command in your terminal:

```bash
python get_distance.py --input_svo_file <input_svo_file> --p1 x1,y1 --p2 x2,y2
```

Arguments:
* --input_svo_file: Path to an existing .svo file
* --p1: First pixel coordinates in the left 2D image
* --p2: Second pixel coordinates in the left 2D image
