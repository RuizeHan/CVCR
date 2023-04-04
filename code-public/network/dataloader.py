from torch.utils.data import Dataset
import config
import os
import numpy as np

C = config.Config()

class DataLoader(Dataset):
    def __init__(self):
        self.x0 = C.IMAGE_W / 2
        self.y0 = C.IMAGE_H / 2
        self.label_ls = self.getLabel()
        self.data_ls = self.getData()


    def getLabel(self):
        def angle2cor(angle):
            # 影子朝向相机
            if angle > 90 and angle < 270:
                image_angle = 180 - angle
            if angle < 90 or angle > 270:
                image_angle = 360 - angle
            x = np.tan(np.deg2rad(image_angle)) / np.tan(C.CAMERA_VIEW_W) * self.x0 + self.x0
            y = 0
            return [x, y]

        label_ls = []
        anno_file_ls = sorted(os.listdir(C.ANNO_PATH))
        for anno_i in anno_file_ls:
            full_path = os.path.join(C.ANNO_PATH, anno_i)
            with open(full_path, 'r') as anno_file:
                line = anno_file.readlines()[0]
                angle = float(line.strip('\n').split(' ')[4])
                label = angle2cor(angle)
                label_ls.append(label)
        return label_ls[1:]

    def getDataset(self):
        data_ls = []
        data_file_ls = sorted(os.listdir(C.DATASET_PATH))
        for data_i in data_file_ls:
            full_path = os.path.join(C.DATASET_PATH, data_i)
            data_ls.append(full_path)
        return data_ls[1:]

    def __getitem__(self, item):
        map = np.load(self.data_ls[item])
        label = np.array(self.label_ls[item])
        return map, label

if __name__ == '__main__':
    data = DataLoader()
