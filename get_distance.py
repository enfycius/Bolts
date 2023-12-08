import os
import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import argparse

def main():
    svo_input_path = opt.input_svo_file

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_input_path)
    init_params.svo_real_time_mode = False 
    init_params.coordinate_units = sl.UNIT.MILLIMETER 

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS: 
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    i = 0

    while i < 50:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, p1 = point_cloud.get_value(871, 985)
            err, p2 = point_cloud.get_value(879, 982)

            if math.isfinite(p1[2]) or math.isfinite(p2[2]):
                distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2) )
                print(f"Distance: {distance}")
            else : 
                print(f"The distance can not be computed at the specific pixel coordinates")    

        i += 1
           
    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    
    opt = parser.parse_args()
    
    if not opt.input_svo_file.endswith(".svo"): 
        print("--input_svo_file parameter should be a .svo file but is not : ", opt.input_svo_file,"Exit program.")
        exit()
    
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ", opt.input_svo_file,"Exit program.")
        exit()
    
    main()