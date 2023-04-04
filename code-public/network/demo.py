import cv2
import torch
from loader import Loader
import numpy as np
from tqdm import tqdm
# from sklearn.metrics import mean_squared_error
from config import Config
from S2CNet_3 import S2CNet
from S2CNet_t import S2CNet_t
C = Config()
from data_preprocessing import TrainAugmentation, TestTransform
from torch.utils.data import DataLoader
import os
import math
import time

def cal_pitch(pred):
    pred_copy = pred.copy()
    pred_copy *= [w, h]
    pred_copy = pred_copy[0][1]
    h0 = h / 2
    camers_f = (h / 2) / np.tan(np.deg2rad(58.7155 / 2))
    pred_copy = -np.rad2deg(np.arctan((pred_copy - h0) / camers_f))
    return pred_copy

def cal_yaw(pred_hor, pred_top, isSame, h, w, pred_pitch):
    # norm_issame = np.sqrt(np.sum(np.square(isSame)))
    # observe = np.array([0, 1])
    # mean_angle = np.rad2deg(np.arccos(np.dot(isSame, observe.T) / (norm_issame)))
    # mean_angle = mean_angle  if isSame[0] < 0 else -mean_angle
    pred_hor *= [w, h]
    pred_hor = pred_hor[0][0]
    pred_top = pred_top[0] * [w, h]
    w0 = w / 2
    camers_f = (h / 2) / np.tan(np.deg2rad(58.7155 / 2))
    pred_hor = - np.rad2deg(np.arctan((pred_hor - w0) / camers_f)) * np.cos(np.deg2rad(pred_pitch))
    # pred_hor = pred_hor if abs(pred_hor - mean_angle) % 360 < 90  else pred_hor + 180
    if -pred_top[1] > 0:
        top_shadow_pred = np.rad2deg(np.arctan(-pred_top[0]/pred_top[1]))
    else:
        top_shadow_pred = np.rad2deg(np.arctan(-pred_top[0]/pred_top[1])) - 180
    res = (top_shadow_pred + pred_hor + 360) % 360
    return res


print('loading...')

class_names = ['0', '1']
num_classes = len(class_names)

checkpoint_path = '/home/realgump/documents/network_v0/model14/models'
checkpoint_path_t = '/home/realgump/documents/network_v0/model14/model_t1'

checkpoint = torch.load(checkpoint_path)
checkpoint_t = torch.load(checkpoint_path_t)

model = S2CNet().cuda()
model_t = S2CNet_t().cuda()
model.load_state_dict(checkpoint['model'])
model_t.load_state_dict(checkpoint_t['model'])
model.eval()
model_t.eval()

input_path_boot = '/home/realgump/documents/network_v0/case/'

if not os.path.exists(input_path_boot + 'hormask/'):
    os.makedirs(input_path_boot + 'hormask/')
if not os.path.exists(input_path_boot + 'topmask/'):
    os.makedirs(input_path_boot + 'topmask/')
if not os.path.exists(input_path_boot + 'horout/'):
    os.makedirs(input_path_boot + 'horout/')
if not os.path.exists(input_path_boot + 'topout/'):
    os.makedirs(input_path_boot + 'topout/')


input_path_hor = input_path_boot + 'hormap/'
input_path_top = input_path_boot + 'topmap/'

hor_ls = os.listdir(input_path_hor)
top_ls = os.listdir(input_path_top)

hor_ls.sort()

total_time = 0
total_sample = 0
total_time_rel = 0

for hor in hor_ls:
    img_name = hor.split('.')[0]
    hor_p = input_path_hor + hor
    top_p = input_path_top + hor
    if not os.path.exists(hor_p):
        continue
    if not os.path.exists(top_p):
        continue
    hor = np.load(hor_p)
    top = np.load(top_p)
    if hor.shape == (0,):
        continue
    cv2.imwrite(input_path_boot + 'hormask/' + img_name + '.png', hor)
    cv2.imwrite(input_path_boot + 'topmask/' + img_name + '.png', top)
    hor = cv2.resize(hor, (300, 300))
    hor = np.expand_dims(hor, axis=2)
    hor = np.expand_dims(hor, axis=0)
    hor = torch.tensor(hor)
    top = cv2.resize(top, (300, 300))
    top = np.expand_dims(top, axis=2)
    top = np.expand_dims(top, axis=0)
    top = torch.tensor(top)

    start_time = time.time()
    out_hor = model(hor)
    out_top = model_t(top)
    end_time = time.time()
    total_time += (end_time - start_time) * 1000
    total_sample += 1

    print(hor_p)

    if os.path.exists(input_path_boot + 'hor/' + img_name + '.png'):
        img = cv2.imread(input_path_boot + 'hor/' + img_name + '.png')
    elif os.path.exists(input_path_boot + 'hor/' + img_name + '.jpeg'):
        img = cv2.imread(input_path_boot + 'hor/' + img_name + '.jpeg')
    else:
        img = cv2.imread(input_path_boot + 'hor/' + img_name + '.jpg')
    h, w, _ = img.shape
    cv2.circle(img, (out_hor[0][0] * w, out_hor[0][1] * h), 25, (0, 0, 255), -1)
    cv2.imwrite(input_path_boot + 'horout/' + img_name + '.png', img)

    out_hor = out_hor.detach().cpu().numpy()
    out_top = out_top.detach().cpu().numpy()

    start_time = time.time()
    pitch_pred = cal_pitch(out_hor)
    same = np.load(input_path_boot + 'horissame/' + img_name + '.npy')
    # print(same)
    yaw_pred = cal_yaw(out_hor, out_top, same, h, w, pitch_pred)
    end_time = time.time()
    total_time_rel += (end_time - start_time) * 1000

    if os.path.exists(input_path_boot + 'top/' + img_name + '.png'):
        img = cv2.imread(input_path_boot + 'top/' + img_name + '.png')
    elif os.path.exists(input_path_boot + 'top/' + img_name + '.jpeg'):
        img = cv2.imread(input_path_boot + 'top/' + img_name + '.jpeg')
    else:
        img = cv2.imread(input_path_boot + 'top/' + img_name + '.jpg')
    out_top *= [w, h]
    n = math.sqrt(out_top[0][0] ** 2 + out_top[0][1] ** 2)
    norm_pred = np.sqrt(np.sum(np.square(out_top)))
    x1 = int(w / 2)
    y1 = int(h / 2)
    x2 = x1 + int(out_top[0][0] / n * 300)
    y2 = y1 + int(out_top[0][1] / n * 300)

    # cv2.line(img, (x1, y1), (int(x2), int(y2)), (0, 0, 255), 2)
    # y1 = h - y1
    # y2 = h - y2
    # yaw_pred = math.radians(yaw_pred)
    # yaw_pred = math.radians(90)
    # x2 = ((x2-x1)*math.cos(yaw_pred) - (y2-y1)*math.sin(yaw_pred)) + x1
    # y2 = ((x2-x1)*math.sin(yaw_pred) + (y2-y1)*math.cos(yaw_pred)) + y1
    # y2 = h - y2

    # cv2.circle(img, (x1, y1), 5, (0, 0, 255), -1)
    # cv2.line(img, (x1, y1), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.line(img, (x1, y1), (int(x1 - 300 * math.sin(math.radians(yaw_pred))), int(y1 + 300 * math.cos(math.radians(yaw_pred)))), (0, 0, 255), 2)

    cv2.imwrite(input_path_boot + 'topout/' + img_name + '.png', img)
    # cv2.imwrite(img_name + '.png', img)
    print(img_name, out_top, yaw_pred)


print('vanishing point detection', total_sample, total_time, total_time / total_sample)
print('relation', total_sample, total_time_rel, total_time_rel / total_sample)

