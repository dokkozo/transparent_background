#!/usr/bin/env python3
#coding: UTF-8

from argparse import ArgumentParser
import pandas as pd
import cv2
import numpy as np
import scipy

def main():
    args = getargs()

    # load img
    img = cv2.imread(args.input, -1)

    # get background reference
    median = np.median(np.median(img, axis = 0), axis=0)

    # calculate mask
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

    kernel = np.ones((args.weak_mask_erosion, args.weak_mask_erosion),np.uint8)
    mask_weak_eroded = cv2.erode(mask_weak, kernel, iterations = 3)
    mask_or = np.logical_or(mask_strong > 128, mask_weak_eroded > 128)
    mask_or = (255*mask_or).astype(np.uint8)

    # final erosion
    if args.final_erosion > 0:
        cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (args.final_erosion, args.final_erosion))
        mask_final = cv2.erode(mask_or, cross_kernel, iterations = 1)
    else:
        mask_final = mask_or

    # apply mask as alpha channel
    img_masked = np.concatenate([img, mask_final[:,:,None]], axis=2)
    cv2.imwrite(args.output, img_masked)

def getargs():
    argparser = ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output',type=str, required=True)
    argparser.add_argument('--range', type=int, default=10, help='Background color reference would be a square region of 0<=x<args.range and 0<=y<args.range. Specify in pix')
    argparser.add_argument('--threshold_weak', type=str2intlist, default="150,40,150", help='Specify threshold to be background. 0,0,0 would be the weakest and 255,255,255 is the strongest.')
    argparser.add_argument('--threshold_strong', type=str2intlist, default="180,80,180", help='Strong thredshold to cut the edge of the front object strictly.')
    argparser.add_argument('--weak_mask_erosion', type=int, default=3)
    argparser.add_argument('--final_erosion', type=int, default=3, help='Erosion kernel size for final mask. Specify 0 to pass erosion.')
    args = argparser.parse_args()
    return args

def str2intlist(inputstr):
    return [int(one) for one in inputstr.split(",")]

if __name__ == '__main__':
    main()