#!/usr/bin/env python

import torch
import numpy
import PIL.Image
import argparse
from utils import image2tensor
from Network import Network


torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance


def parse_args():
    parser = argparse.ArgumentParser(description='Run optical-flow algorithm on images')
    parser.add_argument('-f', '--first', required=True, help='first input frame')
    parser.add_argument('-s', '--second', required=True, help="second input frame")
    parser.add_argument('-m', '--model', required=True, help="path of pytorch-pwc .pth file")
    parser.add_argument('-o', '--output', default="output.flo", help='output location for flo file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    tensorFirst = image2tensor(PIL.Image.open(args.first))
    tensorSecond = image2tensor(PIL.Image.open(args.second))

    moduleNetwork = Network().cuda().eval()
    moduleNetwork.load_state_dict(torch.load(args.model))

    tensorOutput = moduleNetwork.estimate(tensorFirst, tensorSecond)

    objectOutput = open(args.output, 'wb')

    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objectOutput)
    numpy.array([tensorOutput.size(2), tensorOutput.size(1)], numpy.int32).tofile(objectOutput)
    numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

    objectOutput.close()
