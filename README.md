# transparent_background
A script to make the background of a .png file transparent.

## requirement 
- python3
- cv2, scipy, pandas, numpy, argparse

## options
- --input: Required. Input png file path.
- --output: Required. Output png file path.
- --range: Background color reference would be a square region of 0<=x<args.range and 0<=y<args.range. Specify in pix
- --threshold_weak: Specify threshold to be background. 0,0,0 would be the weakest and 255,255,255 is the strongest. Final mask will be "or" calculation of eroded weak mask and strong mask.
- --threshold_strong: Strong thredshold to cut the edge of the front object strictly.
- --weak_mask_erosion: Erosion kernel size of weak mask
- --final_erosion: Erosion kernel size for final mask. Specify 0 to pass erosion.
