import cv2
import random
from computations import compute_histogram
from tracker import Tracker
from add_sup import parse_det_of_images, parse_gt_of_images, extract_image_part
import motmetrics as mm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
from os import path
import argparse

work_dir = "F:/diploma/HT21/train/HT21-02"
out_dir = "F:/diploma/out/tracks/HT-train/track/data"

gt_data = parse_gt_of_images(work_dir + '/gt/gt.txt')
det_data = parse_det_of_images(work_dir + '/det/det.txt')
img_dir = work_dir + '/img1/'

tracker = Tracker(similarity_treshold=0.55, high_con_th=70, low_con_th=60, lifetime=37, alpha=0.7)
# acc = mm.MOTAccumulator(auto_id=True)

hist_list = [0] * 6
cm_len = 0

# VIDEO
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('F:\diploma\out\output_video.avi', fourcc, 30, (1920, 1080))

#  open out file for track_eval
with open(out_dir + "/HT21-01.txt", 'w') as f_out:
    # another frame
    images_ar = []
    im_it = 0
    color_table = {}
    gt_data_det = det_data  # parse_det_of_images(work_dir+'train/HT21-01/det/det.txt')
    for ik in range(len(gt_data_det)):
        #if ik in [1, 31] and work_dir == "F:/diploma/HT21/train/HT21-02":
        #    continue
        img = cv2.imread(img_dir + ('0' * (6 - len(str(ik + 1)))) + str(ik + 1) + '.jpg')
        cur_detections = gt_data_det[ik]  # gain detections
        cm_len += len(cur_detections)
        start_time = time.time()
        start_time2 = time.time()
        for det in cur_detections:

            try:
                # if(im_it==1):
                #     cv2.imshow('test', extract_image_part(img, det))
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                det.hist = compute_histogram(extract_image_part(img, det), False)
            except cv2.error as e:
                cv2.imshow('test', extract_image_part(img, det))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        print("--- %s hist calc seconds ---" % (time.time() - start_time2))

        #   print("1 ", tracker.calculate_IOU(tracker.get_tracks(), cur_detections))
        #    print("2 ", tracker.calc_iou(tracker.get_tracks(), cur_detections))
        start_time3 = time.time()
        tracker.Track(cur_detections)
        print("--- %s only tracker seconds ---" % (time.time() - start_time3))
        print("--- %s overall seconds ---" % (time.time() - start_time))


       #for det in gt_data[ik]:  # for evaluation
        #    det.hist = compute_histogram(extract_image_part(img, det))

        # marked_detections = tracker.update_det(gt_data_det[ik])
        marked_detections = tracker.get_tracks()  # !!!!
        for md in marked_detections:
            f_out.write(str(im_it + 1) + "," + str(md.id) + "," + str(md.coord[0]) + "," + str(md.coord[1])
                        + "," + str(md.l) + "," + str(md.h) + "," + str(
                md.conf) + "," + "-1" + "," + "-1" + "," + "-1" + '\n')
        # det_id = [marked_detections[i].id for i in range(len(marked_detections))]
        # gt_id = [gt_data[ik][i].id for i in range(len(gt_data[ik]))]
        # dist_mat = tracker.calc_iou(gt_data[ik], marked_detections)
        # acc.update(gt_id, det_id, dist_mat)
        #
        for s in tracker.get_tracks():
            if s.id not in color_table:
                color_table[s.id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = cv2.rectangle(img, (int(s.coord[0]), int(s.coord[1])), (int(s.coord[0] + s.l), int(s.coord[1] + s.h)),
                                color=color_table[s.id], thickness=2)
            cv2.putText(img, str(s.id), (int(s.coord[0]), int(s.coord[1])), 1, 1, color_table[s.id], 2, cv2.LINE_AA)
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

# mh = mm.metrics.create()
# summary = mh.compute_many(
#     [acc, acc.events.loc[0:1]],
#     metrics=mm.metrics.motchallenge_metrics,
#     names=['full', 'part'])
#
# strsummary = mm.io.render_summary(
#     summary,
#     formatters=mh.formatters,
#     namemap=mm.io.motchallenge_metric_names
# )
# print(strsummary)
