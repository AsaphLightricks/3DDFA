#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
import os.path as osp
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
from imageio import get_reader, get_writer
from utils.lighting import RenderPipeline

STD_SIZE = 120
IMAGE_SUFFS = ['.png', '.JPG', '.jpeg', '.jpg', '.bmp', '.tiff']
VID_SUFFS = ['.mov', '.mp4', '.avi', '.MOV']

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (0, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # for rendering video:
    app = RenderPipeline(**cfg)
    triangles = sio.loadmat('tri_refine.mat')['tri']  # mx3
    triangles = _to_ctype(triangles).astype(np.int32)

    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    for in_file in args.files:
        print(in_file)
        filename = osp.basename(osp.splitext(in_file)[0])
        if osp.splitext(in_file)[1] in IMAGE_SUFFS:
            ims = [cv2.imread(in_file)]

        elif osp.splitext(in_file)[1] in VID_SUFFS:
            with get_reader(in_file) as reader:
                fps = reader.get_meta_data()['fps']
                ims = [im for im in reader]

            writer = get_writer(osp.join(args.out_dir, osp.basename(in_file)), fps=fps)
        else:
            raise IOError

        for i, img_ori in enumerate(ims):
            img_ori_bgr = img_ori[:, :, ::-1]
            i += 1
            if args.dlib_bbox:
                rects = face_detector(img_ori_bgr, 1)
            else:
                rects = []

            if len(rects) == 0:
                writer.append_data(img_ori / 255)
                continue
                # rects = dlib.rectangles()
                # rect_fp = in_file + '.bbox'
                # lines = open(rect_fp).read().strip().split('\n')[1:]
                # for l in lines:
                #     l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                #     rect = dlib.rectangle(l, r, t, b)
                #     rects.append(rect)

            pts_res = []
            Ps = []  # Camera matrix collection
            poses = []  # pose collection, [todo: validate it]
            vertices_lst = []  # store multiple face vertices
            ind = 0
            suffix = get_suffix(in_file)
            for rect in rects:
                # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
                if args.dlib_landmark:
                    # - use landmark for cropping
                    pts = face_regressor(img_ori_bgr, rect).parts()
                    pts = np.array([[pt.x, pt.y] for pt in pts]).T
                    roi_box = parse_roi_box_from_landmark(pts)
                else:
                    # - use detected face bbox
                    bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                    roi_box = parse_roi_box_from_bbox(bbox)

                img = crop_img(img_ori_bgr, roi_box)

                # forward: one step
                img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                # 68 pts
                pts68 = predict_68pts(param, roi_box)

                # two-step for more accurate bbox to crop face
                if args.bbox_init == 'two':
                    roi_box = parse_roi_box_from_landmark(pts68)
                    img_step2 = crop_img(img_ori, roi_box)
                    img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                    input = transform(img_step2).unsqueeze(0)
                    with torch.no_grad():
                        if args.mode == 'gpu':
                            input = input.cuda()
                        param = model(input)
                        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                    pts68 = predict_68pts(param, roi_box)

                pts_res.append(pts68)
                P, pose = parse_pose(param)
                Ps.append(P)
                poses.append(pose)
                vertices = np.copy(predict_dense(param, roi_box).T, order='C')
                triangles = _to_ctype((tri - 1).T.astype(np.int32))
                out_im = app(vertices, triangles, (img_ori / 255).astype(np.float32))
                writer.append_data(out_im)

                # dense face 3d vertices
                if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
                    vertices_lst.append(vertices)
                if args.dump_ply:
                    dump_to_ply(vertices, tri, '{}/{}_{}.ply'.format(args.out_dir, filename, i))
                if args.dump_vertex:
                    dump_vertex(vertices, '{}/{}_{}.mat'.format(args.out_dir, filename, i))
                if args.dump_pts:
                    wfp = '{}/{}_{}.txt'.format(args.out_dir, filename, i)
                    np.savetxt(wfp, pts68, fmt='%.3f')
                    print('Save 68 3d landmarks to {}'.format(wfp))
                if args.dump_roi_box:
                    wfp = '{}/{}_{}.roibox'.format(args.out_dir, filename, i)
                    np.savetxt(wfp, roi_box, fmt='%.3f')
                    print('Save roi box to {}'.format(wfp))
                if args.dump_paf:
                    wfp_paf = '{}/{}_{}_paf.jpg'.format(args.out_dir, filename, i)
                    wfp_crop = '{}/{}_{}_crop.jpg'.format(args.out_dir, filename, i)
                    paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

                    cv2.imwrite(wfp_paf, paf_feature)
                    cv2.imwrite(wfp_crop, img)
                    print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
                if args.dump_obj:
                    wfp = '{}/{}_{}.obj'.format(args.out_dir, filename, i)
                    colors = get_colors(img_ori, vertices)
                    write_obj_with_colors(wfp, vertices, tri, colors)
                    print('Dump obj with sampled texture to {}'.format(wfp))
                ind += 1

            if args.dump_pose:
                # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
                img_pose = plot_pose_box(img_ori, Ps, pts_res)
                wfp = in_file.replace(suffix, '_pose.jpg')
                cv2.imwrite(wfp, img_pose)
                print('Dump to {}'.format(wfp))
            if args.dump_depth:
                wfp = in_file.replace(suffix, '_depth.png')
                # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
                depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
                cv2.imwrite(wfp, depths_img)
                print('Dump to {}'.format(wfp))
            if args.dump_pncc:
                wfp = in_file.replace(suffix, '_pncc.png')
                pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
                cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
                print('Dump to {}'.format(wfp))
            if args.dump_res:
                draw_landmarks(img_ori, pts_res, wfp=in_file.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-o', '--out_dir', default='output')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', action='store_true', help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', action='store_true', help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', action='store_true',
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', action='store_true')
    parser.add_argument('--dump_pts', action='store_true')
    parser.add_argument('--dump_roi_box', action='store_true')
    parser.add_argument('--dump_pose', action='store_true')
    parser.add_argument('--dump_depth', action='store_true')
    parser.add_argument('--dump_pncc', action='store_true')
    parser.add_argument('--dump_paf', action='store_true')
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', action='store_true')
    parser.add_argument('--dlib_bbox', action='store_true', default=True, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', action='store_true',
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)
