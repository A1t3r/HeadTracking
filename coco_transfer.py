from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from add_sup import parse_det_of_images_BYTE
import cv2

coco = Coco()
coco.add_category(CocoCategory(id=0, name='head'))
dir = 'F:\\diploma\\HT21\\train\\HT21-01\\'
#det_data_BYTE = parse_det_of_images_BYTE(dir + '/gt/gt.txt')
seq_names = ['HT21-01', 'HT21-02', 'HT21-03', 'HT21-04']
out_dir = 'F:/diploma/additional data/coco/annotations/imgs/'
#fout = open(out_dir)
#gt_data_det = det_data_BYTE
global_it = 0
# for name in seq_names:
#     tmp_dir = f'F:\\diploma\\HT21\\train\\{name}\\'
#     gt_data_det = parse_det_of_images_BYTE(tmp_dir + '/gt/gt.txt')
#     for ik in range(int(len(gt_data_det)/2)):
#         img_dir = f'F:\\diploma\\HT21\\train\\{name}\\img1\\'
#       #  img = cv2.imread(img_dir + ('0' * (6 - len(str(ik + 1)))) + str(ik + 1) + '.jpg')
#        # cv2.imwrite(out_dir+str(global_it)+'.jpg', img)
#         cur_detections = gt_data_det[ik]
#         coco_image = CocoImage(file_name=str(global_it) + '.jpg',
#                                height=1080, width=1920)
#         global_it+=1
#         print(global_it)
#         for det in cur_detections:
#             coco_image.add_annotation(
#                 CocoAnnotation(
#                     bbox=[det[0], det[1], det[2]-det[0], det[3]-det[1]],
#                     category_id=0,
#                     category_name='head'
#                 )
#             )
#         coco.add_image(coco_image)
# save_json(data=coco.json, save_path='F:/diploma/additional data/coco/annotations/train_dataset2.json')

coco = Coco()
coco_val = Coco()
coco.add_category(CocoCategory(id=0, name='head'))
coco_val.add_category(CocoCategory(id=0, name='head'))
global_it = 0
out_dir = 'F:/diploma/additional data/coco/annotations/imgs_tst/'
for name in seq_names:
    tmp_dir = f'F:\\diploma\\HT21\\train\\{name}\\'
    gt_data_det = parse_det_of_images_BYTE(tmp_dir + '/gt/gt.txt')
    for ik in range(int(len(gt_data_det)/2), len(gt_data_det)):
        img_dir = f'F:\\diploma\\HT21\\train\\{name}\\img1\\'
       # img = cv2.imread(img_dir + ('0' * (6 - len(str(ik + 1)))) + str(ik + 1) + '.jpg')
      #  cv2.imwrite(out_dir+str(global_it)+'.jpg', img)
        cur_detections = gt_data_det[ik]
        coco_image = CocoImage(file_name=str(global_it) + '.jpg',
                               height=1080, width=1920)
        global_it+=1
        print(global_it)
        for det in cur_detections:
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=[det[0], det[1], det[2]-det[0], det[3]-det[1]],
                    category_id=0,
                    category_name='head'
                )
            )
        if ik<len(gt_data_det)*0.9:
            coco.add_image(coco_image)
           # cv2.imwrite(out_dir + str(global_it) + '.jpg', img)
        else:
            coco_val.add_image(coco_image)
           # cv2.imwrite('F:/diploma/additional data/coco/annotations/imgs_val/' + str(global_it) + '.jpg', img)
save_json(data=coco.json, save_path='F:/diploma/additional data/coco/annotations/test_dataset.json')
save_json(data=coco_val.json, save_path='F:/diploma/additional data/coco/annotations/val_dataset.json')