import cv2
import random
import numpy as np
from computations import compute_histogram
from add_sup import parse_det_of_images, parse_gt_of_images, extract_image_part, parse_det_of_images_BYTE, parse_det_of_images_CenterNet
import motmetrics as mm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
from os import path
import torch
from byte_tracker.byte_tracker import BYTETracker
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

all_time=[]
all_stamps=[]
use_color_distance = True
is_BYTE = False
for s_it in ['1','2', '3', '4']:
    #work_dir= "F:/diploma/HT21/train/HT21-01"
    work_dir = f"F:/diploma/additional data/new_gt/HT21-0{s_it}"
    out_dir = "F:/diploma/out/tracks/HT-train/track/data"

    #gt_data = parse_gt_of_images(work_dir + '/gt/gt.txt')
    #det_data = parse_det_of_images_BYTE("F:/diploma/HT21/train/HT21-01"+ '/det/det.txt')
    det_data_BYTE = parse_det_of_images_CenterNet(work_dir + '/det/det.txt')
    #mas_stamps = [0]*5
    img_dir = work_dir + '/img1/'
    it_incr_ar = [0, 215, 1873, 2373]
    it_incr = it_incr_ar[int(work_dir[-1])-1]
    #det_data_BYTE = det_data[int(len(det_data)/2):len(det_data)-1]  # COMMENT THIS
    if is_BYTE:
        tracker = BYTETracker(track_thresh=40, track_buffer=0.1, match_thresh=0.1,
                              use_color_dist=use_color_distance, color_dist_coef=0.1,
                              max_time_lost=20, mot20=None)
    else:
        tracker = Tracker(use_color_distance)

    hist_list = [0] * 6
    cm_len = 0
    hist_arr = None

    # VIDEO
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter('F:\diploma\out\output_video.avi', fourcc, 30, (1920, 1080))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'F:\diploma\out\HT21-0{str(work_dir[-1])}_DeepSORTtest.mp4', fourcc, 30, (1920, 1080))
    avg_time = 0
    #  open out file for track_eval
    with open(f'{out_dir}/HT21-0{str(work_dir[-1])}.txt', mode='w') as f_out:
        # another frame
        images_ar = []
        im_it = 0
        color_table = {}
        gt_data_det = det_data_BYTE # parse_det_of_images(work_dir+'train/HT21-01/det/det.txt')
        for ik in range(len(gt_data_det)):
            img = cv2.imread(img_dir + str(ik+it_incr) + '.jpg')
            cur_detections = gt_data_det[ik]  # gain detections
            # for det in cur_detections:
            #     if det[4] < 21:
            #         mas_stamps[0] += 1
            #     if det[4] < 34:
            #         mas_stamps[1] += 1
            #     if det[4] < 64:
            #         mas_stamps[2] += 1
            #     if det[4] < 84:
            #         mas_stamps[3] += 1
            #     else:
            #         mas_stamps[4] += 1
            # all_stamps.append(mas_stamps)
            cm_len += len(cur_detections)
            start_time = time.time()
            start_time2 = time.time()
            if use_color_distance:
                hist_arr = []
                for det in cur_detections:
                    try:
                        hist_arr.append(compute_histogram(extract_image_part(img, det), False))
                    except ValueError:
                        cur_detections.remove(det)
            #print("--- %s hist calc seconds ---" % (time.time() - start_time2))

            start_time3 = time.time()
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
            avg_time += time.time() - start_time
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
    print(avg_time/len(gt_data_det))
    all_time.append(avg_time/len(gt_data_det)+0.017)
    # print(">90:{} | >85:{} | >75:{} | >70:{} | >65:{} | <65:{}".format(hist_list[0],
    #                                                                    hist_list[1], hist_list[2], hist_list[3],
    #                                                                    hist_list[4], hist_list[5]))
    # print(">90:{} | >85:{} | >75:{} | >70:{} | >65:{} | <65:{}".format(hist_list[0] / cm_len,
    #                                                                    hist_list[1] / cm_len, hist_list[2] / cm_len,
    #                                                                    hist_list[3] / cm_len,
    #                                                                    hist_list[4] / cm_len, hist_list[5] / cm_len))
print(all_time)
print(all_stamps)

