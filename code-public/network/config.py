import numpy as np

class Config:
    def __init__(self):
        self.BASE_NET = 'model/mobilenet_v1_with_relu_69_5.pth'
        self.NET = "mb1-ssd"
        self.INPUT_TYPE = 'map'
        self.ROOT_DIR = '/dataset/gyy/shadow/dataset/'
        self.SCENE_LIST = ['v13', 'v14', 'v15', 'v16']
        self.SCENE_LIST = ['v21', 'v22', 'v23', 'v24']
        self.SCENE_LIST = ['v31', 'v32', 'v33', 'v34']
        # self.SCENE_LIST = ['v32', 'v33', 'v34']
        # self.SCENE_LIST = ['v33']
        self.CAMERA_ANNO_DIR = []
        self.PERSON_ANNO_DIR = []
        self.IMAGE_HOR_DIR_GT = []
        self.IMAGE_TOP_DIR_GT = []
        self.IMAGE_HOR_DIR = []
        self.IMAGE_TOP_DIR = []
        self.LIGHT_ANNO_DIR = []
        self.ISSAME_DIR = []
        for scene in self.SCENE_LIST:
            self.CAMERA_ANNO_DIR.append(self.ROOT_DIR + scene + '/CameraData/')
            self.PERSON_ANNO_DIR.append(self.ROOT_DIR + scene + '/PositionData/')
            self.ISSAME_DIR.append(self.ROOT_DIR + scene + '/hor1issame/')
            self.IMAGE_HOR_DIR_GT.append(self.ROOT_DIR + scene + '/hor1gtmap/')
            self.IMAGE_TOP_DIR_GT.append(self.ROOT_DIR + scene + '/topgtmap/')
            self.IMAGE_HOR_DIR.append(self.ROOT_DIR + scene + '/hor1map/')
            self.IMAGE_TOP_DIR.append(self.ROOT_DIR + scene + '/topmap1/')
            # self.IMAGE_HOR_DIR.append('/dataset/wydata/shadow/'+ scene + '/hor1mapzh/')
            # self.IMAGE_TOP_DIR.append('/dataset/wydata/shadow/' + scene + '/topmapzh/')
            self.LIGHT_ANNO_DIR.append(self.ROOT_DIR + scene + '/LightData/')
        self.DEMO_OURPUT_DIR = 'demo/'
        self.GT_IMAGE_DIR = 'gt/'
        self.MODEL_SAVING_PATH = 'train_model/'
        self.MODEL_SAVING_PATH = '/dataset/gyy/shadow/model/'
        # self.INPUT_TYPE = 'map' if self.DATASET_PATH[-2] == 'p' else 'rgb'
        # self.INPUT_TYPE = 'map' if self.DATASET_PATH[-2] == 'p' else 'rgb'


        ## dataset
        self.IMAGE_H = 545
        self.IMAGE_W = 968

        # self.IMAGE_H = 900
        # self.IMAGE_W = 1600

        self.HUMAN_HEIGHT = 1.8
        self.SUNSHINE_ANGLE = 35
        FielfOfView = 58.7155
        self.VIEW_ANGLE_H = np.tan(np.deg2rad(FielfOfView / 2))
        self.VIEW_ANGLE_W = np.tan(np.deg2rad(90 / 2))

        self.RESIZE_SIZE = 300
        self.IMAGE_MEAN = np.array([167.66493, 182.60864, 196.42943])  # RGB layout
        # self.IMAGE_MEAN = np.array([0,0,0])  # RGB layout
        self.IMAGE_STD = 2
        self.IMAGE_PADDING = 5

        # FielfOfView = 90


        self.CAMERA_VIEW_H = np.deg2rad(FielfOfView / 2)
        # self.CAMERA_VIEW_W = np.deg2rad(53.1)
        self.CAMERA_F = (self.IMAGE_H / 2) / np.tan(self.CAMERA_VIEW_H)

        self.LEARNING_RATE = 0.001
        self.MAX_EPOCH = 1000
        self.BATCH_SIZE = 32

        self.SCHEDULER = 'cosine'
        self.TMAX = 200

        self.IOU_THRESHOLD = 0.5
        self.CENTER_VARIANCE = 0.1
        self.SIZE_VARIANCE = 0.2



        self.GPUS = '1'
        self.USING_TB = 1

        self.DEBUG = 0
        self.IFDRAW = 0



