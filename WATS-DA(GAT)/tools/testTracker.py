# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys

sys.path.append('../')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder_gat import ModelBuilder
from toolkit.datasets import DatasetFactory
from pysot.tracker.siamgat_tracker import SiamGATTracker

from ar.pytracking.refine_modules.refine_module import RefineModule
from re_detector.guider.network import Guider

os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser(description='siamgat tracking')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='WATB', #NAT #UAVDark70 #NUT_L
        help='datasets') # OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=False,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='snapshot_WATS-DA/checkpoint_e25_0.03.pth', #48 #37
        help='snapshot of models to eval')
parser.add_argument('--config', type=str, default='/media/w/719A549756118C56/HGG/WATS-DA(GAT)/experiments/siamgat_googlenet/config.yaml',
        help='config file')
parser.add_argument('--THRES', type=float, default=0.1, #NAT #UAVDark70 #NUT_L
        help='datasets')
parser.add_argument('--ar_path', default='/media/w/719A549756118C56/HGG/SAM-DA-main/AR/ltr/checkpoints/SEcmnet_ep0040-c.pth.tar', type=str,  help='path to snapshot')
args = parser.parse_args()

torch.set_num_threads(1)
THRES2 = 0.5
THRES_BIN = 0.4
THRES_AREA = 100
W_MAP = 0.2
def get_ar(img, init_box, ar_path):
    """ set up Alpha-Refine """
    selector_path = 0
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box))
    return RF_module

def main():
    # load config
    cfg.merge_from_file(args.config)
    print(f'Now testing {args.snapshot}')
    # Test dataset
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = '/media/w/719A549756118C56/datasets/WATB/WATB/' #os.path.join(cur_dir, 'test_dataset', args.dataset) #/mnt/sdb/Test_dataset/NAT2021_test/ #/mnt/sdg/Test_dataset/ #/mnt/sdb/Test_dataset/NUT-L_folder

    # set hyper parameters
    params = getattr(cfg.HP_SEARCH, args.dataset)
    cfg.TRACK.LR = params[0]
    cfg.TRACK.PENALTY_K = params[1]
    cfg.TRACK.WINDOW_INFLUENCE = params[2]

    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}


    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamGATTracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[-2]+ '_'+ str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
    model_name = str(args.snapshot.split('/')[-1][:-4])
    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue

        ar_module = None    
        re_module = None
        toc = 0
        pred_bboxes = []
        track_times = []
        scores = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()

            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                # Init ar and re module.
                ar_module = get_ar(img, gt_bbox_, args.ar_path)
                re_module = Guider().cuda().eval()
                re_module.init(img, np.array(gt_bbox_))
                size_1st = [w, h]
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
                scores.append(None)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                # pred_bboxes.append(pred_bbox)
                if outputs['best_score'] < args.THRES:
                    obj_map = re_module.inference(img)  # [x y x y]

                    bk_center_pos = tracker.center_pos  # cy cx
                    bk_size = tracker.size  # h w

                    _, _, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    if obj_map.max() < 0.3:
                        tracker.center_pos = np.array([img.shape[0] / 2, img.shape[1] / 2])
                        tracker.size = np.array(size_1st, np.float32)

                        outputs = tracker.track(img, hp)
                        pred_bbox = outputs['bbox']
                        pred_bbox = ar_module.refine(img, np.array(pred_bbox))
                        # pred_bbox = outputs['bbox']
                    else:
                        # find peak.
                        obj_w, obj_h = np.where(obj_map == obj_map.max())
                        obj_w = obj_w[0]
                        obj_h = obj_h[0]

                        obj_map[obj_map > THRES_BIN] = 1
                        obj_map[obj_map <= THRES_BIN] = 0
                        contours, _ = cv2.findContours(obj_map.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                        if len(contours) != 0 and np.max(cnt_area) > THRES_AREA:
                            contour = contours[np.argmax(cnt_area)]
                            x, y, w, h = cv2.boundingRect(contour)
                            side = np.sqrt(w * h)

                            tracker.center_pos = np.array([y + h / 2.0, x + w / 2.0], dtype=np.float32)  # cy cx
                            tracker.size = np.array(size_1st, dtype=np.float32) * (1 - W_MAP) + np.array([side, side], dtype=np.float32) * W_MAP
                        else:  # empty mask
                            tracker.center_pos = np.array([obj_h, obj_w], dtype=np.float32)
                            tracker.size = np.array(size_1st, np.float32)
                        outputs = tracker.track(img, hp)
                        pred_bbox = outputs['bbox']
                        pred_bbox = ar_module.refine(img, np.array(pred_bbox))
                    if outputs['best_score'] < THRES2:
                        # frame_count+= 1
                        # if frame_count % global_search_interval == 0:
                        #     frame_count=0
                        #     tracker.center_pos = np.array([img.shape[0] / 2, img.shape[1] / 2])
                        #     tracker.size = np.array([img.shape[0] , img.shape[1]], np.float32)
                        #     # tracker.size =np.array(size_1st)*2
                        #     outputs = tracker.track(img, hp)

                        
                        tracker.center_pos = bk_center_pos
                        tracker.size = bk_size

                    # if outputs['best_score'] < 0.2:
                    #     frame_count2+= 1
                    #     if frame_count % global_search_interval == 0:
                    #         frame_count2=0
                    #         tracker.center_pos = np.array([img.shape[0] / 2, img.shape[1] / 2])
                    #         tracker.size = np.array([img.shape[0] , img.shape[1]], np.float32)
                    #         # tracker.size =np.array(size_1st)*2
                    #         outputs = tracker.track(img, hp)

                        
                    #     tracker.center_pos = bk_center_pos
                    #     tracker.size = bk_size


                # pred_bbox = outputs['bbox']
                pred_bbox = ar_module.refine(img, np.array(pred_bbox))

                pred_bbox = pred_bbox.tolist()
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if not any(map(math.isnan,gt_bbox)):
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        if 'GOT-10k' == args.dataset:
            video_path = os.path.join('results_ddam_35', args.dataset, model_name, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            result_path = os.path.join(video_path,
                                       '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))
        else:
            model_path = os.path.join('results_ddam_240309', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
