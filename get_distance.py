import os
import pyzed.sl as sl
import math
import numpy as np
import sys
import math
import argparse

def coords(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x, y")

def main():
    svo_input_path = opt.input_svo_file
    frame_rate = opt.frame_rate

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
    sum_of_distance = 0
    count_of_distance = 0

    while i < frame_rate:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            x = round(image.get_width() / 2)
            y = round(image.get_height() / 2)
            err, p1 = point_cloud.get_value(opt.p1[0][0], opt.p1[0][1])
            err, p2 = point_cloud.get_value(opt.p2[0][0], opt.p2[0][1])

            if math.isfinite(p1[2]) or math.isfinite(p2[2]):
                distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2) )

                if not math.isnan(distance):
                    sum_of_distance += distance

                    count_of_distance += 1

                print(f"Distance: {distance}")
            else : 
                print(f"The distance can not be computed at the specific pixel coordinates")    

        i += 1

    if count_of_distance == 0:
        print("No measured value")
    else:
        print("Sum of the distances:", sum_of_distance)
        print("Count of the distances:", count_of_distance)
        print("Average of the distances:", (sum_of_distance / count_of_distance))

    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--frame_rate', type=int, required=True, help='Specify the frame rate')
    parser.add_argument('--p1', help="Coordinate", type=coords, required=True, nargs=1)
    parser.add_argument('--p2', help="Coordinate", type=coords, required=True, nargs=1)

    opt = parser.parse_args()
    
    if not opt.input_svo_file.endswith(".svo"): 
        print("--input_svo_file parameter should be a .svo file but is not : ", opt.input_svo_file,"Exit program.")
        exit()
    
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ", opt.input_svo_file,"Exit program.")
        exit()
    
    main()