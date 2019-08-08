# -*- coding: utf-8 -*-
# @Time    : 2017/10/26 14:09
# @Author  : zhoujun

import tensorflow as tf
#from scipy.misc import imread
import time
import os
import sys
import Binarize
import glob
import cv2

# img_file_path = sys.argv[1]
# model_path = sys.argv[2]
#y = sys.argv[2]
#model=0
list=[]
img_file_path="C:/Users/ahsan/Pictures/Grayscale_/*"
model_path ="C:/Users/ahsan/Documents/Code that I need to change/Recognition Binarization Implementation/New recognition Model/1555142663"
class PredictionModel:
    
    def __init__(self, model_dir, session=None):
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        start = time.time()
        self.model = tf.saved_model.loader.load(self.session, ['serve'], model_dir)

#        print('load_model_time:', time.time() - start)

        self._input_dict, self._output_dict = _signature_def_to_tensors(self.model.signature_def['predictions'])

    def predict(self, image):
        output = self._output_dict
        # ??predict  op
        start = time.time()
        result = self.session.run(output, feed_dict={self._input_dict['images']: image})
#        print('predict_time:',time.time()-start)
        return result


def _signature_def_to_tensors(signature_def):  # from SeguinBe
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}


def predict(model_dir, image,gpu_id = 0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tf.Session() as sess:
        start = time.time()
        #model = PredictionModel(model_dir,session=sess)
        predictions = model.predict(image)
        transcription = predictions['words']
        score = predictions['score']
        return [transcription[0].decode(), score, time.time() - start]

 
if __name__ == '__main__':

    #model_dir = 'E:/Atif/Software/Python_tensor_flow/Train/export_model_full_year_256_complex_mod_hidden_units/export/1539245406'
    model_dir = model_path
    save_dir= "C:/Users/ahsan/Pictures/Improved Binarization/"
    # in_txt_file = open(img_txt_file_path, 'r')
    # image_arr = in_txt_file.readlines()
    # in_txt_file.close()
    
    Binarize.wolf_multi(img_file_path,save_dir)
    Binarized_Images = glob.glob(save_dir+"*")
    filenames = []
    transcriptions = []
    outF=open("new.txt","w+")
    #outF = open('../../../CBVIR.BLL/model/recognition/recognizedLabels.txt', "w+")
    # outF = open('./CBVIR.BLL/model/recognition/recognizedLabels.txt', "w+")
    with tf.Session() as sess:
        model = PredictionModel(model_dir,session=sess)
        for img  in Binarized_Images:
            # print(cur_image)
            # Binarize image using Wolf's Algorithm 
            # binarizedImage = binarization.wolf(cur_image.rstrip("\n\r"))
            #image = imread(img.rstrip("\n\r"), mode='L')[:, :, None]
            # image = imread(binarizedImage, mode='L')[:, :, None]
            image = cv2.imread(img,0)[:,:,None]
            predictions = model.predict(image)
            transcription = predictions['words']
            transcriptions.append(os.path.basename(img))
            transcriptions.append(transcription[0].decode()+'\n')
    # for item in image_arr:
    #     outF.write(item)
    outF.writelines(transcriptions)
    outF.close()
    print('successful')
    # print(img_txt_file_path)
