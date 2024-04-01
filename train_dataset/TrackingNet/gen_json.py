

from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd

dataset_path = "/media/w/719A549756118C56/datasets/TrackingNet/"
train_sets = ['TRAIN_0','TRAIN_1','TRAIN_2','TRAIN_3','TRAIN_4','TRAIN_5','TRAIN_6','TRAIN_7','TRAIN_8','TRAIN_9','TRAIN_10','TRAIN_11']
val_set = ['val']
d_sets = {'videos_val':val_set,'videos_train':train_sets}
d_sets = {'videos_train':train_sets}

def parse_and_sched(dl_dir='.'):
    js = {}
    for d_set in d_sets:
        for dataset in d_sets[d_set]:
            videos = os.listdir(os.path.join(dataset_path,dataset,'frames'))
            for video in videos:
                if video == 'list.txt'or video =='list-original.txt':
                    continue
                # video = dataset+'/frames/'+video
                gt_path = join(dataset_path, dataset,'anno',video)
                video = dataset+'/frames/'+video
                gt_path=gt_path+".txt"
                f = open(gt_path, 'r')
                groundtruth = f.readlines()
                f.close()
                for idx, gt_line in enumerate(groundtruth):
                    gt_image = gt_line.strip().split(',')
                    frame = '%06d' % (int(idx))
                    obj = '%02d' % (int(0))
                    bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                            int(float(gt_image[0])) + int(float(gt_image[2])),
                            int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax

                    if video not in js:
                        js[video] = {}
                    if obj not in js[video]:
                        js[video][obj] = {}
                    js[video][obj][frame] = bbox
        if 'videos_val' == d_set:
            json.dump(js, open('tracker/BAN/train_dataset/TrackingNet/val.json', 'w'), indent=4, sort_keys=True)
        else:
            json.dump(js, open('tracker/BAN/train_dataset/TrackingNet/train.json', 'w'), indent=4, sort_keys=True)
        js = {}

        print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()
