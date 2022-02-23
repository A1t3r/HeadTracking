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



work_dir = "F:/diploma/HT21/train/HT21-01"

gt_data = parse_gt_of_images(work_dir + '/gt/gt.txt')
det_data = parse_det_of_images(work_dir + '/det/det.txt')
img_dir = work_dir + '/img1/'

tracker = Tracker(similarity_treshold=0.75, high_con_th=85, low_con_th=69, lifetime=27, alpha=0.35)
acc = mm.MOTAccumulator(auto_id=True)

# another frame
images_ar = []
im_it = 0
color_table = {}
gt_data_det = det_data  # parse_det_of_images(work_dir+'train/HT21-01/det/det.txt')
for ik in range(len(gt_data_det) - 1):
    img = cv2.imread(img_dir + ('0' * (6 - len(str(ik + 1)))) + str(ik + 1) + '.jpg')
    cur_detections = gt_data_det[ik]  # gain detections
    for det in cur_detections:
        det.hist = compute_histogram(extract_image_part(img, det))
    for det in gt_data[ik]:  # for evaluation
        det.hist = compute_histogram(extract_image_part(img, det))
    start_time = time.time()
    tracker.Track(cur_detections)
    print("--- %s seconds ---" % (time.time() - start_time))

    # marked_detections = tracker.update_det(gt_data_det[ik])
    marked_detections = tracker.get_tracks()  # !!!!
    det_id = [marked_detections[i].id for i in range(len(marked_detections))]
    gt_id = [gt_data[ik][i].id for i in range(len(gt_data[ik]))]
    dist_mat = tracker.compute_cost_matrix(gt_data[ik], marked_detections)
    acc.update(gt_id, det_id, dist_mat)


    for s in tracker.get_tracks():
        if s.id not in color_table:
            color_table[s.id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.rectangle(img, (int(s.coord[0]), int(s.coord[1])), (int(s.coord[0] + s.l), int(s.coord[1] + s.h)),
                                color=color_table[s.id], thickness=2)
        cv2.putText(img, str(s.id), (int(s.coord[0]), int(s.coord[1])), 1, 1, color_table[s.id], 2, cv2.LINE_AA)

    window_name = 'image'
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im_it += 1
    print("\n-----------" + str(im_it) + "------------\n")

mh = mm.metrics.create()
summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=mm.metrics.motchallenge_metrics,
    names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)