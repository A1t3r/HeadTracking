from PIL import Image, ImageDraw
from tracker import single_detection


def save_gif(path_to, path_from, len):
    images_ar = [Image.open(path_from + str(f) + '.png') for f in range(len)]
    images_ar[0].save(path_to,
                      save_all=True, append_images=images_ar[1:], optimize=False)


def parse_gt_of_images(dir):
    gt_file = open(dir)
    GT_data = []
    gt_data = []
    cur_frame = 1
    for line in gt_file:
        tmp_arr = [float(x) for x in line[:-2].split(',')]
        if int(tmp_arr[0]) != cur_frame:
            cur_frame = int(tmp_arr[0])
            GT_data.append(gt_data)
            gt_data = []
        s = single_detection(int(tmp_arr[1]) - 46,
                             (tmp_arr[2], tmp_arr[3]),
                             tmp_arr[4], tmp_arr[5], 100, 0, 0)
        gt_data.append(s)
    return GT_data


def parse_det_of_images(dir):
    gt_file = open(dir)
    GT_data = []
    gt_data = []
    cur_frame = 1
    for line in gt_file:
        tmp_arr = [float(x) for x in line[:-3].split(',')]
        if int(tmp_arr[0]) != cur_frame:
            cur_frame = int(tmp_arr[0])
            GT_data.append(gt_data)
            gt_data = []
        s = single_detection(None,
                             (tmp_arr[2], tmp_arr[3]),
                             tmp_arr[4], tmp_arr[5], tmp_arr[6], 0, 0)
        gt_data.append(s)
    return GT_data


def extract_image_part(image, detection):
    y = int(detection.coord[0])
    x = int(detection.coord[1])
    return image[x:x + int(detection.h), y:y + int(detection.l)]
