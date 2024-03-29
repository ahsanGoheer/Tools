import os
import cv2
import numpy as np
import tensorflow as tf
import sys


sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util
path="E:\\TextDetection\\TextDetection2019\\models-master\\research\\object_detection\\Frames"
i=0
flag = tf.app.flags
flag.DEFINE_string('model','','Name of the folder in which the inference graph exists')
flag.DEFINE_string('savedir','HiRes','Directory in which you wish to save the images')
FLAGS = flag.FLAGS
if(FLAGS.savedir ==''):
    detected_image_path="E:\\TextDetection\\TextDetection2019\\models-master\\research\\object_detection\\Detected Images Text\\"
else:
    detected_image_path="E:\\TextDetection\\TextDetection2019\\models-master\\research\\object_detection\\"+FLAGS.savedir+"\\"


MODEL_NAME = FLAGS.model 

TEST_IMAGE_FOLDER= 'Frames'
            
CWD_PATH = os.getcwd()

            
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

            
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','label_map.pbtxt')

            
         
# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
for r,d,f in os.walk(path):
    for file in f:
        if ('.jpg' or '.png') in file:
            IMAGE_NAME = file            
            # Name of the directory containing the object detection module we're using
           
            PATH_TO_IMAGE = os.path.join(CWD_PATH,TEST_IMAGE_FOLDER,IMAGE_NAME)
            print(PATH_TO_IMAGE)
            image = cv2.imread(PATH_TO_IMAGE)
            #temp_img=image
            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.60)

            # All the results have been drawn on image. Now display the image.
            cv2.imshow('UrduEnglish', image)
            cv2.imwrite(detected_image_path+IMAGE_NAME,image)
            print('\n')
            print(IMAGE_NAME+' SAVED!\n ')
            print('\n')
            cv2.waitKey(300)
            height,width,x = image.shape
            # print(width)
            # print("\n"+str(height))
        
            #boxes = boxes[scores>0.6]
            box = []
            _class = []
            for j in scores:
                if j>0.6:
                    box.append(boxes[j])
                    _class.append(classes[j])

            for box in boxes:
                ymin, xmin, ymax, xmax = box
                
                ymin =int(ymin*height)
                ymax = int(ymax*height)
                xmin = int(xmin*width)
                xmax = int(xmax*width)
                print(xmin, " ", xmax, " ", ymin, " ", ymax)
              
                temp_img = cv2.imread(PATH_TO_IMAGE)
                
                line = temp_img[ymin:ymax,xmin:xmax]
                i=i+1
                if _class==1.0:
                    line_path="E:\\TextDetection\\TextDetection2019\\models-master\\research\\object_detection\\Detected Images Text\\Urdu_Lines\\"            
                    cv2.imwrite(line_path+IMAGE_NAME+str(i),line)
                    cv2.imshow("Urdu",line)
                elif _class==2.0:
                    line_path="E:\\TextDetection\\TextDetection2019\\models-master\\research\\object_detection\\Detected Images Text\\English_Lines\\"
                    cv2.imwrite(line_path+IMAGE_NAME+str(i),line)
                    cv2.imshow("English",line)
                    
                cv2.waitKey(200)        
            cv2.destroyAllWindows()

print('Process Successfully Completed! \n'+'Images are located in the directory : '+detected_image_path)

