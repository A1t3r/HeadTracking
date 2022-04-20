import cv2
import random
import numpy as np
from computations import compute_histogram
from add_sup import parse_det_of_images, parse_gt_of_images, extract_image_part, parse_det_of_images_BYTE
import motmetrics as mm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
from os import path
import torch
from byte_tracker.byte_tracker import BYTETracker
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

use_color_distance = True
is_BYTE = False

work_dir = "F:/diploma/HT21/train/HT21-01"
out_dir = "F:/diploma/out/tracks/HT-train/track/data"

#gt_data = parse_gt_of_images(work_dir + '/gt/gt.txt')
#det_data = parse_det_of_images(work_dir + '/det/det.txt')
det_data_BYTE = parse_det_of_images_BYTE(work_dir + '/det/det.txt')
img_dir = work_dir + '/img1/'

if is_BYTE:
    tracker = BYTETracker(track_thresh=0.65, track_buffer=0.1, match_thresh=0.35,
                          use_color_dist=use_color_distance, color_dist_coef=0.11,
                          max_time_lost=5, mot20=None)
else:
    tracker = Tracker(use_color_distance)

hist_list = [0] * 6
cm_len = 0
hist_arr = None

# VIDEO
#fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#out = cv2.VideoWriter('F:\diploma\out\output_video.avi', fourcc, 30, (1920, 1080))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('F:\diploma\out\output_video.mp4', fourcc, 30, (1920, 1080))

#  open out file for track_eval
with open(out_dir + "/HT21-01.txt", mode='w') as f_out:
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
        if is_BYTE:
            online_targets = tracker.update(torch.Tensor(cur_detections), [1920, 1080], [1920, 1080], hist_arr)
        else:
            detections = []
            if use_color_distance:
                for det, hist in zip(cur_detections, hist_arr):
                    detections.append(Detection([det[0], det[1], det[2]-det[0], det[3]-det[1]], det[4], hist))
            else:
                for det in cur_detections:
                    detections.append(Detection([det[0], det[1], det[2]-det[0], det[3]-det[1]], det[4], None))
            tracker.predict()
            tracker.update(detections)
            online_targets = tracker.tracks
        print("--- %s only tracker seconds ---" % (time.time() - start_time3))
        print("--- %s overall seconds ---" % (time.time() - start_time))
        if is_BYTE:
            for track in online_targets:
                f_out.write(str(im_it+1) + "," + str(track.track_id) + "," + str(track.tlwh[0]) + "," + str(track.tlwh[1])
                            + "," + str(track.tlwh[2]) + "," + str(track.tlwh[3]) + "," + str(
                    float(track.score)) + "," + "-1" + "," + "-1" + "," + "-1" + '\n')
                if track.track_id not in color_table:
                    color_table[track.track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = cv2.rectangle(img, (int(track.tlbr[0]), int(track.tlbr[1])),
                                    (int(track.tlbr[2]), int(track.tlbr[3])),
                                    color=color_table[track.track_id], thickness=2)
                cv2.putText(img, str(track.track_id), (int(track.tlwh[0]),
                                                       int(track.tlwh[1])), 1, 1, color_table[track.track_id], 2, cv2.LINE_AA)
        else:
            for track in online_targets:
                f_out.write(str(im_it+1) + "," + str(track.track_id) + "," + str(track.to_tlwh()[0]) + "," + str(track.to_tlwh()[1])
                            + "," + str(track.to_tlwh()[2]) + "," + str(track.to_tlwh()[3]) + "," + str(
                    float(track.conf)) + "," + "-1" + "," + "-1" + "," + "-1" + '\n')
                if track.track_id not in color_table:
                    color_table[track.track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = cv2.rectangle(img, (int(track.to_tlbr()[0]), int(track.to_tlbr()[1])),
                                    (int(track.to_tlbr()[2]), int(track.to_tlbr()[3])),
                                    color=color_table[track.track_id], thickness=2)
                cv2.putText(img, str(track.track_id), (int(track.to_tlwh()[0]),
                                                       int(track.to_tlwh()[1])), 1, 1, color_table[track.track_id], 2, cv2.LINE_AA)

        out.write(img)
       # window_name = 'image'
       # cv2.imshow(window_name, img)
       # cv2.waitKey(0)
       # cv2.destroyAllWindows()

        im_it += 1
        print("\n-----------" + str(im_it) + "------------\n")
out.release()
# print(">90:{} | >85:{} | >75:{} | >70:{} | >65:{} | <65:{}".format(hist_list[0],
#                                                                    hist_list[1], hist_list[2], hist_list[3],
#                                                                    hist_list[4], hist_list[5]))
# print(">90:{} | >85:{} | >75:{} | >70:{} | >65:{} | <65:{}".format(hist_list[0] / cm_len,
#                                                                    hist_list[1] / cm_len, hist_list[2] / cm_len,
#                                                                    hist_list[3] / cm_len,
#                                                                    hist_list[4] / cm_len, hist_list[5] / cm_len))


