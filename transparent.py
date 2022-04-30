#!/usr/bin/env python3
#coding: UTF-8

from argparse import ArgumentParser
import pandas as pd
import cv2
import numpy as np
import scipy

def main():
    args = getargs()

    img = cv2.imread(args.input, -1)
    median = np.median(np.median(img, axis = 0), axis=0)
    mask_weak = np.logical_and(
            np.logical_and(
                np.logical_and(img[:,:,0] >= median[0] - args.threshold_weak[0], img[:,:,0] <= median[0] + args.threshold_weak[0]),
                np.logical_and(img[:,:,1] >= median[1] - args.threshold_weak[1], img[:,:,1] <= median[1] + args.threshold_weak[1])),
            np.logical_and(img[:,:,2] >= median[2] - args.threshold_weak[2], img[:,:,2] <= median[2] + args.threshold_weak[2])
    )

    mask_strong = np.logical_and(
            np.logical_and(
                np.logical_and(img[:,:,0] >= median[0] - args.threshold_strong[0], img[:,:,0] <= median[0] + args.threshold_strong[0]),
                np.logical_and(img[:,:,1] >= median[1] - args.threshold_strong[1], img[:,:,1] <= median[1] + args.threshold_strong[1])),
            np.logical_and(img[:,:,2] >= median[2] - args.threshold_strong[2], img[:,:,2] <= median[2] + args.threshold_strong[2])
    )
    mask_weak = (255 - 255*mask_weak).astype(np.uint8)
    mask_strong = (255 - 255*mask_strong).astype(np.uint8)

    kernel = np.ones((args.erosion, args.erosion),np.uint8)
    mask_weak_eroded = cv2.erode(mask_weak, kernel, iterations = 3)
    mask_or = np.logical_or(mask_strong > 128, mask_weak_eroded > 128)

    mask_or = 255*mask_or

    img_masked = np.concatenate([img, mask_or[:,:,None]], axis=2)
    cv2.imwrite(args.output, img_masked)

    img_weak = np.concatenate([img, mask_weak[:,:,None]], axis=2)
    cv2.imwrite('tmp1.png', img_weak)

    img_strong = np.concatenate([img, mask_strong[:,:,None]], axis=2)
    cv2.imwrite('tmp2.png', img_strong)


def getargs():
    argparser = ArgumentParser()
    argparser.add_argument('input', type=str, default="")
    argparser.add_argument('output',type=str, default="")
    argparser.add_argument('--range', type=int, default=10, help='in pix')
    argparser.add_argument('--threshold_weak', type=str2intlist, default="150,40,150")
    argparser.add_argument('--threshold_strong', type=str2intlist, default="180,80,180")
    argparser.add_argument('--erosion', type=int, default=3)
    args = argparser.parse_args()
    return args

def str2intlist(inputstr):
    return [int(one) for one in inputstr.split(",")]

if __name__ == '__main__':
    main()