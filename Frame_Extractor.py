import cv2
from os import listdir
from tqdm import tqdm
import sys


dir = sys.argv[1]



files_in_dir = listdir(dir)
save_dir = sys.argv[2]

def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        #print(hasFrames)
        if hasFrames:
            cv2.imwrite(save_dir+video+"-Frame-{}".format(str(sec*cv2.CAP_PROP_FPS))+".jpg", image)     # save frame as JPG file
        return hasFrames

pbar= tqdm(total=len(files_in_dir))
for video in files_in_dir: 
    print(video)
    pbar.update(1)
    vidcap = cv2.VideoCapture(dir+video)
    #cv2.imshow(video)
    sec = 0
    frameRate = 10
    count=1
    success = getFrame(sec)
    while success:
      count = count + 1
      sec = sec + frameRate
      sec = round(sec, 2)
      success = getFrame(sec)


