import cv2
import random
import torch
from computations import compute_histogram
from tracker import Tracker
from add_sup import parse_det_of_images_BYTE, extract_image_part
from byte_tracker.byte_tracker import BYTETracker
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
from os import path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="path to MOT dataset", type=str)
parser.add_argument("-Vout", help="show output", type=str)
parser.add_argument("-Tout", help="show output", type=str)
parser.add_argument("-v", "--verbose", help="-")
parser.add_argument("-cdc", help="show output", type=float)
args = parser.parse_args()

work_dir = args.path
# if args.out == 't':
#     show = True
# else: show = False
show = True

use_color_distance = True
if args.cdc > 1 or args.cdc <= 0.01 or args.cdc is None:
    use_color_distance = False

video_out = args.Vout
tracker_out = args.Tout
dataset_name = args.path.split('\\')[-1]

det_data_BYTE = parse_det_of_images_BYTE(work_dir + '/gt/gt.txt')
img_dir = work_dir + '/img1/'
out_dir = f"{tracker_out}/tracks/HT-train/track/data"

tracker = BYTETracker(track_thresh=0.6, track_buffer=0.1, match_thresh=0.35,
                      use_color_dist=use_color_distance, color_dist_coef=0.11,
                      max_time_lost=15, mot20=None)
# acc = mm.MOTAccumulator(auto_id=True)

hist_list = [0] * 6
cm_len = 0
hist_arr = None

# VIDEO
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(f'{video_out}/output_video.avi', fourcc, 30, (1920, 1080))

#  open out file for track_eval
with open(out_dir + f"/{dataset_name}.txt", 'w') as f_out:
    # another frame
    images_ar = []
    im_it = 0
    color_table = {}
    gt_data_det = det_data_BYTE # parse_det_of_images(work_dir+'train/HT21-01/det/det.txt')
    for ik in range(len(gt_data_det)):
        img = cv2.imread(img_dir + ('0' * (6 - len(str(ik + 1)))) + str(ik + 1) + '.jpg')
        cur_detections = gt_data_det[ik]  # gain detections
        cm_len += len(cur_detections)
        start_time = time.time()
        start_time2 = time.time()
        if use_color_distance:
            hist_arr = []
            for det in cur_detections:
                hist_arr.append(compute_histogram(extract_image_part(img, det), False))
        print("--- %s hist calc seconds ---" % (time.time() - start_time2))

        start_time3 = time.time()
        tmp = cur_detections
        online_targets = tracker.update(torch.Tensor(cur_detections), [1920, 1080], [1920, 1080], hist_arr)
        print("--- %s only tracker seconds ---" % (time.time() - start_time3))
        print("--- %s overall seconds ---" % (time.time() - start_time))

        for track in online_targets:
            if im_it == 0:
                break
            f_out.write(str(im_it) + "," + str(track.track_id) + "," + str(track.tlwh[0]) + "," + str(track.tlwh[1])
                        + "," + str(track.tlwh[2]) + "," + str(track.tlwh[3]) + "," + str(
                float(track.score)) + "," + "-1" + "," + "-1" + "," + "-1" + '\n')
            if track.track_id not in color_table:
                color_table[track.track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = cv2.rectangle(img, (int(track.tlbr[0]), int(track.tlbr[1])),
                                (int(track.tlbr[2]), int(track.tlbr[3])),
                                color=color_table[track.track_id], thickness=2)
            cv2.putText(img, str(track.track_id), (int(track.tlwh[0]),
                                                   int(track.tlwh[1])), 1, 1, color_table[track.track_id], 2, cv2.LINE_AA)

        out.write(img)
        im_it += 1
        print("\n-----------" + str(im_it) + "------------\n")
out.release()

