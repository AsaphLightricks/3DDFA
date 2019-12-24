
import torch
import mobilenet_v1
import dlib
import torchvision.transforms as transforms
import numpy as np
import cv2
import imageio as iio
import os
import os.path as osp

from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.inference import parse_roi_box_from_bbox, parse_roi_box_from_landmark, crop_img, predict_68pts

STD_SIZE = 120


class FaceAligner(object):

    def __init__(self):
        # 1. load pre-tained model
        checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = self.model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]

        self.model.load_state_dict(model_dict)

        # if args.mode == 'gpu':
        #     cudnn.benchmark = True
        #     model = model.cuda()
        self.model.eval()

        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        self.face_regressor = dlib.shape_predictor(dlib_landmark_model)
        self.face_detector = dlib.get_frontal_face_detector()

        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    def dlib_detection(self, im):
        rect = self.face_detector(im)[0]
        landmarks = self.face_regressor(im, rect)
        pts = np.array([[pt.x, pt.y] for pt in landmarks.parts()]).T

        return pts

    def get_roi_box(self, im, bbox=None):
        if bbox:
            bbox = [bbox.left(), bbox.top(), bbox.right(), bbox.bottom()]
            return parse_roi_box_from_bbox(bbox)
        else:
            return parse_roi_box_from_landmark(self.dlib_detection(im))

    def get_landmarks(self, im, bbox=None, two_pass=False):
        roi_box = self.get_roi_box(im, bbox)
        img = crop_img(im[:, :, ::-1], roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            param = self.model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        pts = predict_68pts(param, roi_box)

        if not two_pass:
            return pts
        else:
            roi_box = parse_roi_box_from_landmark(pts)
            img = crop_img(im[:, :, ::-1], roi_box)
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                param = self.model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        return pts, predict_68pts(param, roi_box)


if __name__ == '__main__':
    vid_dir = '/Users/ashamir/PycharmProjects/snap/data/input_vids/full_length'
    landmarks_dir = '/Users/ashamir/PycharmProjects/snap/data/vids_3ddfa_landmarks'

    fa = FaceAligner()

    for vid_f in sorted([f for f in os.listdir(vid_dir) if not f.startswith('.')]):
        print(vid_f)
        reader = iio.get_reader(osp.join(vid_dir, vid_f))
        landmarks = []

        for im in reader:
            try:
                l = fa.get_landmarks(im, two_pass=False)
                landmarks.append(l[:2, :].T)
            except Exception as e:
                landmarks.append(np.full((68, 2), -1))

        np.save(osp.join(landmarks_dir, '{}.landmarks.npy'.format(vid_f)), np.array(landmarks))
