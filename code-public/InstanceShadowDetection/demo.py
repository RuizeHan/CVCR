# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import skimage.io as io
import numpy as np
import skimage.transform as tf
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from LISA import add_lisa_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        # default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default="./config/LISA_101_FPN_3x_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--config", default="./config/LISA_101_FPN_3x_demo.yaml")
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file.")
    # parser.add_argument("--input", default="/dataset/gyy/shadow/dataset/v20/hor1/*", nargs="+", help="A list of space separated input images")
    # parser.add_argument("--map_output",  default="/dataset/gyy/shadow/dataset/v12/map/")
    # parser.add_argument("--box_output",  default="/dataset/gyy/shadow/dataset/v11/box/")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    cfg = setup_cfg(args)
    # save_path_map = args.map_output
    # save_path_box = args.box_output

    view_ls = ['hor', 'top']
    scene_ls = ['88']
    input_dir = "/dataset/gyy/shadow/dataset/"

    total_time = 0
    total_sample = 0

    for view in view_ls:
        for scene in scene_ls:
            input_path = input_dir + scene + '/' + view + '/'
            save_path_map = input_dir + scene + '/' + view + 'map/'
            save_path_issame = input_dir + scene + '/' + view + 'issame/'

            input_ls = os.listdir(input_path)
            input_ls.sort()

            demo = VisualizationDemo(cfg)
            if not os.path.exists(save_path_map):
                os.mkdir(save_path_map)
            if not os.path.exists(save_path_issame):
                os.mkdir(save_path_issame)

            for path_dir in tqdm.tqdm(input_ls):
                # use PIL, to be consistent with evaluation
                path = os.path.join(input_path, path_dir)
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions[0][0]["instances"]), time.time() - start_time
                    )
                )
                ins,rel = predictions

                shadow_id = np.array(ins[0]['instances'].pred_classes)
                shadow_association = np.array(ins[0]['instances'].pred_associations)
                is_shadow_ls = []
                is_person_ls = []
                for j, s_id in enumerate(list(shadow_id)):
                    if s_id == 1 and shadow_association[j] != 0:
                        is_shadow_ls.append(j)
                        for k, p_id in enumerate(list(shadow_association)):
                            if p_id == shadow_association[j] and j != k:
                                is_person_ls.append(k)
                                break
                            

                shadow_map = []
                for i,mask in enumerate(np.array(ins[0]['instances'].pred_masks)):
                    if i in is_shadow_ls:
                        if shadow_map == []:
                            shadow_map = np.array(mask).astype('uint8')
                        else:
                            shadow_map += np.array(mask).astype('uint8')
                shadow_map = shadow_map * 255
                end_time = time.time()
                total_time += (end_time - start_time) * 1000
                total_sample += 1
                np.save(save_path_map +  os.path.basename(path_dir.split(".")[0]) + '.npy', shadow_map)

                shadow_map = []
                map_ls = []
                score_ls = []
                box_ls = []
                frame_dict = {}
                # idx = 0
                boxes = ins[0]['instances'].pred_boxes
                boxes_ls = []
                for box in boxes:
                    boxes_ls.append(box)
                issame_ls = []
                if view == 'hor':
                    for person, shadow in zip(is_person_ls, is_shadow_ls):
                        issame_tmp = [-10086, -10086]
                        if boxes_ls[person][3] < (boxes_ls[shadow][1] + boxes_ls[shadow][3]) / 2:
                            issame_tmp[1] = -1
                        else:
                            issame_tmp[1] = 1
                        if (boxes_ls[person][0] + boxes_ls[person][2]) / 2 < (boxes_ls[shadow][0] + boxes_ls[shadow][2]) / 2:
                            issame_tmp[0] = 1
                        else:
                            issame_tmp[0] = -1
                        issame_ls.append(issame_tmp)
                if view == 'top':
                    for person, shadow in zip(is_person_ls, is_shadow_ls):
                        issame_tmp = [-10086, -10086]
                        if (boxes_ls[person][1] + boxes_ls[person][3]) / 2 < (boxes_ls[shadow][1] + boxes_ls[shadow][3]) / 2:
                            issame_tmp[1] = -1
                        else:
                            issame_tmp[1] = 1
                        if (boxes_ls[person][0] + boxes_ls[person][2]) / 2 < (boxes_ls[shadow][0] + boxes_ls[shadow][2]) / 2:
                            issame_tmp[0] = 1
                        else:
                            issame_tmp[0] = -1
                        issame_ls.append(issame_tmp)
                if len(issame_ls):
                    issame = np.mean(np.array(issame_ls),axis=0)
                else:
                    issame = np.array([-1, -1])
                    
                        # io.imsave(tmp_path, shadow_map)

                tmp_path = os.path.join(save_path_issame, path_dir.split(".")[0] + '.npy')
                np.save(tmp_path, issame)
    
    print(total_sample, total_time, total_time / total_sample)

                
                
