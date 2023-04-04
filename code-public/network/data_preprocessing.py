from transforms import *


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # PhotometricDistort(),
            # Expand(self.mean),
            # RandomSampleCrop(),
            # RandomMirror(),
            # UpdownMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None, points=None, vns=None: (img / std, boxes, labels, points, vns),
            # lambda img, boxes=None, labels=None, points=None, vns=None: (img, boxes, labels, points, vns),
            # ToTensor(),
        ])

    def __call__(self, img, boxes, labels, points, vns):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels, points, vns)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None, points=None, vns=None :(img / std, boxes, labels, points, vns),
            # ToTensor(),
        ])

    def __call__(self, image, boxes, labels, points, vns):
        return self.transform(image, boxes, labels, points, vns)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None, points=None, vns=None :(img / std, boxes, labels, points, vns),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _, _, _ = self.transform(image)
        return image