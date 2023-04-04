from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import config
import os
import numpy as np
import cv2
import copy
import torch
C = config.Config()

class Loader(Dataset):
    def __init__(self, mode='train', view='hor', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.image_h = C.IMAGE_H
        self.image_w = C.IMAGE_W
        self.human_height = C.HUMAN_HEIGHT
        self.light_angle = C.SUNSHINE_ANGLE
        self.mode = mode
        self.view = view


        self.image_dir = C.IMAGE_HOR_DIR if view == 'hor' else C.IMAGE_TOP_DIR


        # bndbox_ls, shadow_line_ls, image_path_ls, vanishing_point_ls = self.readAnno()
        # train_num = int(0.8 * len(image_path_ls))
        # if mode == 'train':
        #     self.bndbox_ls = bndbox_ls[:train_num]
        #     self.shadow_line_ls = shadow_line_ls[:train_num]
        #     self.image_path_ls = image_path_ls[:train_num]
        #     self.vanishing_point_ls = vanishing_point_ls[:train_num]
        # if mode == 'test':
        #     self.bndbox_ls = bndbox_ls[train_num:]
        #     self.shadow_line_ls = shadow_line_ls[train_num:]
        #     self.image_path_ls = image_path_ls[train_num:]
        #     self.vanishing_point_ls = vanishing_point_ls[train_num:]
        # self.loader_len = len(self.image_path_ls)
        self.bndbox_ls = []
        self.shadow_line_ls = []
        self.image_path_ls = []
        self.vanishing_point_ls = []
        self.light_ls = []
        self.gt_ls = []
        self.issame_ls = []

        self.IMAGE_TOP_DIR = C.IMAGE_TOP_DIR if mode == 'test' else C.IMAGE_TOP_DIR_GT
        self.IMAGE_HOR_DIR = C.IMAGE_HOR_DIR if mode == 'test' else C.IMAGE_HOR_DIR_GT

        self.IMAGE_TOP_DIR = C.IMAGE_TOP_DIR
        self.IMAGE_HOR_DIR = C.IMAGE_HOR_DIR
        # self.IMAGE_TOP_DIR = C.IMAGE_TOP_DIR_GT
        # self.IMAGE_HOR_DIR = C.IMAGE_HOR_DIR_GT

        for self.scene, self.camera_anno_dir, self.person_anno_dir, self.image_hor_dir, self.image_top_dir, self.light_anno_dir, self.issame_anno_dir\
                in zip(C.SCENE_LIST, C.CAMERA_ANNO_DIR, C.PERSON_ANNO_DIR, self.IMAGE_HOR_DIR, self.IMAGE_TOP_DIR, C.LIGHT_ANNO_DIR, C.ISSAME_DIR):
            # self.boxnpy = np.load('box/box' + self.scene[1:] + view + '.npy', allow_pickle=True).item()
            self.boxnpy = np.load('box/box' + '19' + view + '.npy', allow_pickle=True).item()
            self.image_dir = self.image_hor_dir if view == 'hor' else self.image_top_dir
            bndbox_ls, shadow_line_ls, image_path_ls, vanishing_point_ls, gt_ls, issame_ls = self.readAnno()
            self.bndbox_ls += bndbox_ls
            self.shadow_line_ls += shadow_line_ls
            self.image_path_ls += image_path_ls
            self.vanishing_point_ls += vanishing_point_ls
            self.gt_ls += gt_ls
            self.issame_ls += issame_ls
        self.loader_len = len(self.image_path_ls)


    def humanPosition2shadowPosition(self, human_position, light_rotation):
        shadow_len = self.human_height / np.tan(np.deg2rad(light_rotation))
        shadow_position = [human_position[0], human_position[1], human_position[2] + shadow_len]
        return shadow_position

    def world2screen(self, world_position, camera_position, camera_rotation):
        world_position = np.array([world_position]).T
        camera_position = np.array([camera_position]).T
        theta_x = np.deg2rad(camera_rotation[0])
        theta_y = np.deg2rad(camera_rotation[1])
        r_x = np.array([[1, 0, 0], [0, np.cos(theta_x), np.sin(theta_x)], [0, - np.sin(theta_x), np.cos(theta_x)]])
        r_y = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)], [0, 1, 0], [np.sin(theta_y), 0, np.cos(theta_y)]])
        r = np.dot(r_x, r_y)
        p_c = np.dot(r, world_position - camera_position)
        p_c = np.vstack((p_c, [0]))
        f = C.CAMERA_F
        k1 = np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0]])
        p_i = np.dot(k1, p_c)
        p_i /= p_i[-1] + 1e-8
        p_i = [int(p_i[0] + self.image_w/ 2), int(- p_i[1] + self.image_h / 2)]
        return p_i

    def angle2cor(self, camera_rotation):
        if self.view == 'hor':
            # 影子朝向相机
            pitch = camera_rotation[0]
            yaw = camera_rotation[1]
            if yaw >= 90 and yaw <= 270:
                cam_yaw = 180 - yaw
            if yaw < 90 or yaw > 270:
                cam_yaw = 360 - yaw
            if pitch >= 180:
                cam_pitch = pitch - 360
            if pitch < 180:
                cam_pitch = pitch
            x = np.tan(np.deg2rad(cam_yaw)) / np.cos(np.deg2rad(cam_pitch)) * C.CAMERA_F + self.image_w/ 2
            y = -np.tan(np.deg2rad(cam_pitch)) * C.CAMERA_F + self.image_h / 2
            return [x, y]
        if self.view == 'top':
            yaw = -camera_rotation[1]
            if yaw < -180:
                yaw = yaw + 360
            x = C.IMAGE_W / 2 + 100 * np.sin(np.deg2rad(yaw))
            y = C.IMAGE_H / 2 - 100 * np.cos(np.deg2rad(yaw))
            return [x, y]


    def readAnno(self):

        camera_anno_ls = os.listdir(self.camera_anno_dir)
        person_anno_ls = os.listdir(self.person_anno_dir)
        frame_name_ls = os.listdir(self.image_dir)
        issame_anno_ls = os.listdir(self.issame_anno_dir)
        light_anno_ls = os.listdir(self.light_anno_dir)
        camera_anno_ls.sort(key=lambda x: int(x[:-4]))
        person_anno_ls.sort(key=lambda x: int(x[:-4]))
        light_anno_ls.sort(key=lambda x: int(x[:-4]))
        issame_anno_ls.sort(key=lambda x: int(x[:-4]))

        frame_name_ls.sort(key=lambda x: int(x[:-4]))
        camera_anno_ls = camera_anno_ls[1:]
        person_anno_ls = person_anno_ls[1:]
        light_anno_ls = light_anno_ls[1:]
        frame_name_ls = frame_name_ls[1:]
        issame_anno_ls = issame_anno_ls[1:]

        if C.DEBUG:
            debug = C.DEBUG
            camera_anno_ls = camera_anno_ls[:debug]
            person_anno_ls = person_anno_ls[:debug]
            light_anno_ls = light_anno_ls[:debug]
            frame_name_ls = frame_name_ls[:debug]

        train_num = int(0.8 * len(frame_name_ls))
        if self.mode == 'train':
            camera_anno_ls = camera_anno_ls[:train_num]
            person_anno_ls = person_anno_ls[:train_num]
            light_anno_ls = light_anno_ls[:train_num]
            frame_name_ls = frame_name_ls[:train_num]
            issame_anno_ls = issame_anno_ls[:train_num]
        if self.mode == 'test':
            camera_anno_ls = camera_anno_ls[train_num:]
            person_anno_ls = person_anno_ls[train_num:]
            light_anno_ls = light_anno_ls[train_num:]
            frame_name_ls = frame_name_ls[train_num:]
            issame_anno_ls = issame_anno_ls[train_num:]

        num = len(camera_anno_ls)
        assert num == len(person_anno_ls) and num == len(light_anno_ls) and num == len(issame_anno_ls) and num == len(frame_name_ls)
        bndbox_ls = []
        shadow_line_ls = []
        image_path_ls = []
        vanishing_point_ls = []
        gt_ls = []
        issame_ls = []

        for camera_anno, person_anno, light_anno, issame_anno, frame_name in zip(camera_anno_ls, person_anno_ls, light_anno_ls, issame_anno_ls, frame_name_ls):
            camera_anno_path = os.path.join(self.camera_anno_dir, camera_anno)
            person_anno_path = os.path.join(self.person_anno_dir, person_anno)
            light_anno_path = os.path.join(self.light_anno_dir, light_anno)
            issame_anno_path = os.path.join(self.issame_anno_dir, issame_anno)
            frame_name_path = os.path.join(self.image_dir, frame_name)
            # person_bbox_frame = []
            shadow_line_frame = []
            with open(camera_anno_path, 'r') as camera_anno_file:
                camera_rotation_line = camera_anno_file.readlines()[0] if self.view == 'hor' else camera_anno_file.readlines()[1]
                camera_line = [float(i) for i in (camera_rotation_line.strip('\n').split(','))]
                camera_rotation = camera_line[3:5]
                camera_position = camera_line[0:3]
                vanishing_point_ls.append(self.angle2cor(camera_rotation))
                gt_ls.append(camera_rotation)

            with open(light_anno_path, 'r') as light_anno_file:
                light_rotation_line = light_anno_file.readlines()[0]
                light_line = [float(i) for i in (light_rotation_line.strip('\n').split(','))]
                light_rotation = light_line[0]


            with open(person_anno_path, 'r') as person_anno_file:
                for person_position_i in person_anno_file.readlines():
                    shadow_feet_position = [float(i) for i in (person_position_i.strip('\n').split(','))]
                    shadow_head_position = self.humanPosition2shadowPosition(shadow_feet_position, light_rotation)
                    shadow_line_frame.append(self.world2screen(shadow_feet_position, camera_position, camera_rotation) + self.world2screen(shadow_head_position, camera_position, camera_rotation))

            person_bbox_frame = self.boxnpy[frame_name[:-4]][:-1]
            # person_bbox_frame = self.boxnpy[str(int(frame_name[:-4]) - 1)][:-1]
            # frame_person_name_ls = os.listdir(frame_name_path)
            # frame_person_name_ls.sort(key=lambda x: int(x[:-4]))
            # image0_path = os.path.join(frame_name_path, frame_person_name_ls[0])
            image_path_ls.append(frame_name_path)
            # for frame_person_name in frame_person_name_ls[1:]:
            #     frame_person_path = os.path.join(frame_name_path, frame_person_name)
            #     person_i_map = self.image2shadow(image0_path, frame_person_path)
            #     person_i_bbox = self.map2box(person_i_map)
            #     person_bbox_frame.append(person_i_bbox)
            box_num = len(person_bbox_frame)
            isIn = lambda x: (x[0] > 0 and x[0] < self.image_w and x[1] > 0 and x[1] < self.image_h and x[2] > 0 and x[
                2] < self.image_w and x[3] > 0 and x[3] < self.image_h)
            for i in range(box_num - 1, -1, -1):
                if ([self.image_w, self.image_h, 0, 0] == person_bbox_frame[i]) or not isIn(shadow_line_frame[i]):
                    del person_bbox_frame[i]
                    del shadow_line_frame[i]
            # while [self.image_w, self.image_h, 0, 0] in person_bbox_frame:
            #     person_bbox_frame.remove([self.image_w, self.image_h, 0, 0])
            # image = cv2.imread(image0_path)
            # ctr = (800, 450)
            # vns = (int(self.angle2cor(camera_rotation)[0]), int(self.angle2cor(camera_rotation)[1]))
            # cv2.line(image, ctr, vns, (0,0,255), 2)
            # cv2.imwrite('test.png', image)
            shadow_line_ls.append(shadow_line_frame)
            bndbox_ls.append(person_bbox_frame)
            issame = np.load(issame_anno_path,  allow_pickle=True)
            issame_ls.append(issame)
        return bndbox_ls, shadow_line_ls, image_path_ls, vanishing_point_ls, gt_ls, issame_ls

    def __getitem__(self, item):
        # print(item)
        # image = cv2.imread(self.image_path_ls[item])
        image = np.load(self.image_path_ls[item])
        if image.shape == (0,):
            return self.__getitem__(item - 1)
        pts = np.array(self.vanishing_point_ls[item])
        bndbox = np.array(self.bndbox_ls[item], dtype=np.float32)
        line = np.array(self.shadow_line_ls[item], dtype=np.float32)
        if len(line) == 0:
            return self.__getitem__(item - 1)
        # try:
        #     a = line[0]
        # except:
        #     a = 1
        isSameg = 1 if line[0][3] < line[0][1] else 0
        isSame = self.issame_ls[item]
        if isSame.dtype != 'float64':
            return self.__getitem__(item - 1)
        if image.shape[0] == 1:
            return self.__getitem__(item - 1)


        labels = np.array([1] * len(bndbox))

        # if self.transform:
        #     image, bndbox, labels = self.transform(image, bndbox, labels)
        # self.draw(image, bndbox, item)
        #
        # if self.target_transform:
        #     bndbox, labels = self.target_transform(bndbox, labels)
        # return image, bndbox, labels
        pts = np.expand_dims(pts, 0)
        ctr = np.array([[C.IMAGE_W / 2, C.IMAGE_H / 2]])
        # if C.IFDRAW:
        #     self.draw(image, bndbox, line, item, pts)
        if bndbox.shape ==(0,):
            return self.__getitem__(item - 1)
            a = 1 # for debug
        if self.transform and self.view == 'hor':
            image, bndbox, labels, line, pts = self.transform(image, bndbox, labels, line, pts)
            # image, bndbox, labels, line, pts = self.transform(image, None, None, None, pts)
        if self.transform and self.view == 'top':
            image, bndbox, labels, line, pts = self.transform(image, bndbox, labels, line, pts)
            ## ctr is no use, just a placeholder

        image = np.expand_dims(image, axis=2)
        if C.IFDRAW:
            print(self.image_path_ls[item])
            self.draw(image, bndbox, line, item, pts)

        #
        if self.target_transform:
            bndbox, labels, line = self.target_transform(bndbox, labels, line)
        #
        pts = np.squeeze(pts, axis=0)
        # # test = np.array(image).astype(np.uint8)
        # # test = np.transpose(test, [1, 2, 0])
        # # cv2.imwrite('test.png', test)
        if self.mode == 'train':
            return image, pts, isSame
        else:
            return image, pts, isSame, item
        # return image,line, pts

    def draw(self, image, box_ls, point_ls, item, pts_ls):
        image = np.array(image).astype(np.uint8)
        # image = np.transpose(image, [1, 2, 0])
        point_size = 1
        color = (255, 255, 255)  # BGR
        thickness = 2
        pts = pts_ls[0]
        vns = (int(pts[0] * 300), int(pts[1] * 300))
        ctr = (150, 150)
        cv2.circle(image, vns, point_size, (255, 255, 255), 5)
        cv2.line(image, ctr, vns, color, 2)
        # for point, box in zip(point_ls, box_ls):
        #     pts1 = (int(point[0] * 300), int(point[1] * 300))
        #     pts2 = (int(point[2] * 300), int(point[3] * 300))
        #
        #     cv2.circle(image, pts1, point_size, color, thickness)
        #     cv2.circle(image, pts2, point_size, color, thickness)
        #
        #     cv2.rectangle(image, (int(box[0] * 300), int(box[1] * 300)), (int(box[2] * 300), int(box[3] * 300)), color, point_size)

        cv2.imwrite(C.GT_IMAGE_DIR + str(item) + '.png', image)
    # def draw(self, image, box_ls, item):
    #     image = np.array(image).astype(np.uint8)
    #     image = np.transpose(image, [1, 2, 0])
    #     point_size = 1
    #     color = (0, 0, 255)  # BGR
    #     thickness = 2
    #     for box in box_ls:
    #         cv2.rectangle(image, (int(box[0] * 300), int(box[1] * 300)), (int(box[2] * 300), int(box[3] * 300)), color,
    #                       point_size)
    #     cv2.imwrite(C.GT_IMAGE_DIR + str(item) + '.png', image)

    def __len__(self):
        return self.loader_len
        # return 24

if __name__ == '__main__':
    dataloader = Loader(view='top')
    train_loader = DataLoader(dataloader, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    for i, j in enumerate(train_loader):
        a = 1