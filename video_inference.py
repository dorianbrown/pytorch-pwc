#!/usr/bin/env python

import torch
import argparse
import cv2
import flowiz
import numpy as np

from Network import Network
from utils import image2tensor

torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


def parse_args():
    parser = argparse.ArgumentParser(description='Run optical-flow algorithm on designated video file')
    parser.add_argument('-i', '--input', required=True, help='input video')
    parser.add_argument('-m', '--model', required=True, help="path of pytorch-pwc .pth file")
    parser.add_argument('-o', '--output', default="flow.mp4", help='input video')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    vidcap = cv2.VideoCapture(args.input)
    FPS = int(vidcap.get(cv2.CAP_PROP_FPS))
    VID_WIDTH = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VID_HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    success = True

    while success:
        success, image = vidcap.read()
        if success:
            frames.append(image2tensor(image))

    print("Converted images to frames")

    vidout = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (VID_WIDTH, VID_HEIGHT))
    moduleNetwork = Network().cuda().eval()
    moduleNetwork.load_state_dict(torch.load(args.model))

    for i in range(len(frames) - 1):
        tensorOutput = moduleNetwork.estimate(frames[i], frames[i+1])
        flow = np.array(tensorOutput.numpy().transpose(1, 2, 0), np.float32)
        flow_img = flowiz.convert_from_flow(flow)
        vidout.write(flow_img)

    vidout.release()
    print(f"Done! \nFlowviz written to {args.output}")
