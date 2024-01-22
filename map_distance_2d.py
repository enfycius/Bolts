import os
import pyzed.sl as sl
import math
import cv2
import numpy as np
import pandas as pd
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
    coordinates_csv_input_path = opt.input_boundaries_file
    bolts_csv_input_path = opt.input_bolts_file
    frame_rate = opt.frame_rate

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_input_path)
    init_params.svo_real_time_mode = False 
    init_params.coordinate_units = sl.UNIT.MILLIMETER 

    df = pd.read_csv(coordinates_csv_input_path)
    df = df.astype(int)

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS: 
        print("Camera Open : "+repr(status)+". Exit program.")
        exit()

    runtime_parameters = sl.RuntimeParameters()
    
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    i = 0

    sum_of_distances = [0 for i in range(0, len(df))]
    count_of_distances = [0 for i in range(0, len(df))]

    while i < frame_rate:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            for j in range(0, len(df)):
                x1 = df['found_x1'].iloc[j].item()
                y1 = df['found_y1'].iloc[j].item()

                x2 = df['found_x2'].iloc[j].item()
                y2 = df['found_y2'].iloc[j].item()

                err, p1 = point_cloud.get_value(x1, y1)
                err, p2 = point_cloud.get_value(x2, y2)

                if math.isfinite(p1[2]) or math.isfinite(p2[2]):
                    distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)+((p1[2]-p2[2])**2) )

                    if not math.isnan(distance):
                        sum_of_distances[j] += distance
                        count_of_distances[j] += 1

                        print(f"Distance: {distance}")
                else : 
                    print(f"The distance can not be computed at the specific pixel coordinates")

        i += 1

    results = pd.DataFrame(columns = ['sum_of_distances', 'count_of_distances'])

    results['sum_of_distances'] = sum_of_distances
    results['count_of_distances'] = count_of_distances
    results['average_of_distances'] = results['sum_of_distances'] / results['count_of_distances']

    print(results)

    results.to_csv(opt.output_path + '/' + "results.csv")

    ref = pd.read_csv(bolts_csv_input_path, header=None)

    ref.rename(columns={0: 'centerX'}, inplace=True)
    ref.rename(columns={1: 'centerY'}, inplace=True)
    ref.rename(columns={2: 'radius'}, inplace=True)
    
    bolt_holes_info = ref
    
    radius = bolt_holes_info['radius'].mean()

    width = opt.width
    height = opt.height

    img = np.zeros((width, height, 3), np.uint8)

    for i in range(len(bolt_holes_info)):
        centerX = int(bolt_holes_info['centerX'].iloc[i])
        centerY = int(bolt_holes_info['centerY'].iloc[i])
        radius = int(radius)
        
        img = cv2.circle(img, (centerX, centerY), radius, (255, 255, 255), 3)

    for i in range(0, len(df)):
        x1 = df['found_x1'].iloc[i].item()
        y1 = df['found_y1'].iloc[i].item()

        x2 = df['found_x2'].iloc[i].item()
        y2 = df['found_y2'].iloc[i].item()

        img = cv2.putText(img, str(round(results['average_of_distances'].iloc[i].item(), 2)), (int((x1 + x2) / 2 - 15), int((y1 + y2) / 2 - 15)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imwrite(opt.output_path + '/' + "results.png", img)

    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--input_boundaries_file', type=str, required=True, help='Path to the .csv file with info about boundaries of bolts')
    parser.add_argument('--input_bolts_file', type=str, required=True, help='Path to the .csv file with bolts info')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output image')
    parser.add_argument('--width', type=int, required=True, help='Specify the width of the left image')
    parser.add_argument('--height', type=int, required=True, help='Specify the height of the left image')
    parser.add_argument('--frame_rate', type=int, required=True, help='Specify the frame rate')

    opt = parser.parse_args()
    
    if not opt.input_svo_file.endswith(".svo"): 
        print("--input_svo_file parameter should be a .svo file but is not : ", opt.input_svo_file,"Exit program.")
        exit()

    if not opt.input_boundaries_file.endswith(".csv"): 
        print("--input_boundaries_file parameter should be a .csv file but is not : ", opt.input_boundaries_file,"Exit program.")
        exit()

    if not opt.input_bolts_file.endswith(".csv"): 
        print("--input_bolts_file parameter should be a .csv file but is not : ", opt.input_bolts_file,"Exit program.")
        exit()
    
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ", opt.input_svo_file,"Exit program.")
        exit()

    if not os.path.isfile(opt.input_boundaries_file):
        print("--input_boundaries_file parameter should be an existing file but is not : ", opt.input_boundaries_file,"Exit program.")
        exit()

    if not os.path.isfile(opt.input_bolts_file):
        print("--input_bolts_file parameter should be an existing file but is not : ", opt.input_bolts_file,"Exit program.")
        exit()
    
    main()