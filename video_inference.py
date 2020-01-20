#!/usr/bin/env python

import argparse
import cv2
import flowiz
import numpy as np
from torch.utils.data import DataLoader


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
    FPS = vidcap.get(cv2.CAP_PROP_FPS)
    VID_WIDTH = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    VID_HEIGHT = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = []
    success = True
    count = 0

    while success:
        success, image = vidcap.read()
        frames.append(image)

        if not success:
            break
        print(count)
        count += 1
        cv2.imwrite(f"temp/frame{count}.jpg", image)

    print("Done")

    vidout = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MP4'), FPS, (VID_WIDTH, VID_HEIGHT))

    for i in range(len(frames)):
        flow = do_pwc_torch(frame[i], frame[i+1])
        flow_img = flowiz.convert_from_flow(flow)
        vidout.write(flow_img)

    vidout.release()
    print(f"Done! \nFlowviz written to {args.out}")