import os
import cv2
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
# from tqdm import tqdm

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

class CBVIR():
    def __init__(self, VideoFilePath, ExtractedFramesPath, DetectedPath):
        self.VideoFilePath = VideoFilePath
        self.ExtractedFramesPath = ExtractedFramesPath
        self.DetectedPath = DetectedPath

    def _TimeStampOfFrame(self, Number):
	    return str(datetime.timedelta(seconds = Number * 10)).replace(":", "_")
    
    def _GetFrame(self , VideoName, vidcap, sec, count):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(self.ExtractedFramesPath + VideoName + "-FrameAt_{}".format(self._TimeStampOfFrame(count)) + ".png", image)
        return hasFrames
    
    def FrameExtractor(self):
        for video in os.listdir(self.VideoFilePath):
            vidcap = cv2.VideoCapture(os.path.join(self.VideoFilePath, video))
            sec = 10
            GetFrameAfter = 10
            count = 1
            success = self._GetFrame(video, vidcap, sec, count)
            while success:
                count = count + 1
                sec = sec + GetFrameAfter
                sec = round(sec, 2)
                success = self._GetFrame(video, vidcap, sec, count)
        print("Frames have been extracted!")

    def ExtractBoundedBoxesData(self, DetectionModelPath, LabelMapPath):
        # Number of classes the object detector can identify ()
        NUM_CLASSES = 2

        label_map = label_map_util.load_labelmap(LabelMapPath)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(DetectionModelPath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for Image in os.listdir(self.ExtractedFramesPath):
            print("Currently Extracting Bounded Boxes from : ", Image)
            IMAGE_NAME = Image

            PATH_TO_IMAGE = os.path.join(self.ExtractedFramesPath,IMAGE_NAME)
            image = cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.60)

            ImageHeight, ImageWidth, BlaBlaBlah = image.shape

            ExtractedBoxes = boxes[np.where(scores > 0.6)]
            ExtractedClass = classes[np.where(scores > 0.6)]

            i = 0

            IMAGE_NAME = IMAGE_NAME[:str(IMAGE_NAME).rfind(".")]

            for EachBoxExtracted in ExtractedBoxes:
                y1 = int(EachBoxExtracted[0] * ImageHeight)
                x1 = int(EachBoxExtracted[1] * ImageWidth)
                y2 = int(EachBoxExtracted[2] * ImageHeight)
                x2 = int(EachBoxExtracted[3] * ImageWidth)

                OriginalImage = cv2.imread(PATH_TO_IMAGE)

                CroppedPart = OriginalImage[y1 : y2, x1 : x2]

                if ExtractedClass[i]==1.0:
                    cv2.imwrite(self.DetectedPath + "/UrduLines/" + IMAGE_NAME + "_" + str(i) + ".PNG", CroppedPart)
                elif ExtractedClass[i]==2.0:
                    cv2.imwrite(self.DetectedPath + "/EnglishLines/" + IMAGE_NAME + "_" + str(i) + ".PNG", CroppedPart)

                i = i + 1
        print("Detection Completed!")

    def _Binarize(self, ImagePath):
        raw_image = cv2.imread(os.path.join(self.DetectedPath, "UrduLines", ImagePath), 0)
        grayScale = raw_image

        m, n = grayScale.shape

        high_thresh, thresh_im = cv2.threshold(grayScale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lowThresh = 0.5 * high_thresh

        canny_Image = cv2.Canny(grayScale,350,700)

        kernel = np.ones((3,3),np.uint8)
        closing = self._ImFill(canny_Image)
        inds = np.where(closing==255)
        inds2 = np.where(closing==0)

        pix_val = grayScale[inds]
        pix_val_back = grayScale[inds2]
        
        
        median_text=np.median(pix_val,axis=0)
        median_back=np.median(pix_val_back,axis=0)
        if median_text>median_back:
            pre_binarize = cv2.bitwise_not(grayScale)
        else:
            pre_binarize=grayScale
    
        sharpen_kernel = np.array([[0,-0.9,0], [-1,4.9,-1], [0,-1,-0]])
        new_img = cv2.filter2D(pre_binarize, -1, sharpen_kernel)

        img = self._ApplyThreshold(new_img, self._WolfThreshold(new_img))

        img = cv2.medianBlur(img,3)
    
        name = os.path.basename(ImagePath)
        cv2.imwrite(os.path.join(self.DetectedPath, "BinarizedUrduLines", name), img)

    def _WolfThreshold(self, img, w_size=15,k=0.5):
        rows, cols = img.shape
        i_rows, i_cols = rows + 1, cols + 1

        integ = np.zeros((i_rows, i_cols), np.float)
        sqr_integral = np.zeros((i_rows, i_cols), np.float)

        integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
        sqr_img = np.square(img.astype(np.float))
        sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

        x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

        hw_size = w_size // 2
        x1 = (x - hw_size).clip(1, cols)
        x2 = (x + hw_size).clip(1, cols)
        y1 = (y - hw_size).clip(1, rows)
        y2 = (y + hw_size).clip(1, rows)

        l_size = (y2 - y1 + 1) * (x2 - x1 + 1)
        sums = (integ[y2, x2] - integ[y2, x1 - 1] - integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
        sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] - sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

        means = sums / l_size
        stds = np.sqrt(sqr_sums / l_size - np.square(means))
        max_std = np.max(stds)
        min_v = np.min(img)

        thresholds = ((1.0 - k) * means + k * min_v + k * stds / max_std * (means - min_v))
        return thresholds
        
        
    def _ImFill(self, im_in):
        
        th, im_th = cv2.threshold(im_in, 0, 128, cv2.THRESH_BINARY)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        im_out = im_th | im_floodfill_inv

        return im_out    

    def _ApplyThreshold(self, img, threshold=128, wp_val=255):
        return ((img >= threshold) * wp_val).astype(np.uint8)

    def WolfBinarization(self):
        for File in os.listdir(os.path.join(self.DetectedPath, "UrduLines")):
            print("Currently Binarizing : ", File)
            self._Binarize(File)
        print("Binarization Complete!")
        
        with open(PathToTxtFile, "wb+") as TxtTranscriptions:
            TxtTranscriptions.writelines(transcriptions)
        
        print("Recognition Completed!")

    def __str__(self):
        print("----- CBVIR -----")
        print("-----------------")

if __name__ == "__main__":
    ObjectOfCBVIR = CBVIR("C:/Users/DELl/Desktop/ImplementationTools/Videos/", "C:/Users/DELl/Desktop/ImplementationTools/Frames/", "C:/Users/DELl/Desktop/ImplementationTools/Detected/")
    ObjectOfCBVIR.FrameExtractor()
    ObjectOfCBVIR.ExtractBoundedBoxesData("C:/Users/DELl/Desktop/Classified_Implementation/Resnet/frozen_inference_graph.pb", "C:/Users/DELl/Desktop/Classified_Implementation/LabelMap/labelmap.pbtxt")
    ObjectOfCBVIR.WolfBinarization()