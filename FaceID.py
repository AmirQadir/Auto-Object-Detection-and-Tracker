# -*- coding: utf-8 -*-
"""
@author: Admin
"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.platform import gfile
import detect_face
import pickle, os


EMB_FILE = './evaluation.pkl'
thresh = 0.97

class faceID:

    def __init__(self):
        
        print("\nFaceID Constructor Called" );
        if os.path.exists(EMB_FILE):
            with open(EMB_FILE, 'rb') as f:
                self.db = pickle.load(f,allow_pickle=True)
        else:
            self.db = {}

        self.minsize = 20
        self.factor = 0.709
        self.threshold = [0.6, 0.7, 0.7]
        self.embed = None
        gpu_memory_fraction=1.0
        self.sess = None
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, 'models/')

        self.sess = tf.Session()
        print("Loading ....")
        with gfile.FastGFile('models/20180408-102900.pb','rb') as f:
            graph = tf.GraphDef()
            graph.ParseFromString(f.read())
            tf.import_graph_def(graph, name='')

        self.my_graph = tf.get_default_graph()
        self.image_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0") 
        print(self.embeddings.get_shape())
        
    def get_face(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = detect_face.detect_face(img, self.minsize, \
                    self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        
        return bounding_boxes

    def add_image(self, emb, id):
        if id not in self.db.keys():
            self.db[id] = [emb]
        else:
            self.db[id].append(emb)

        with open(EMB_FILE, 'wb') as f:
            pickle.dump(self.db, f)

    def load_dataset(self):
        if os.path.exists(EMB_FILE):
            with open(EMB_FILE, 'rb') as f:
                self.db = pickle.load(f)
        else:
            self.db = {}
            
    def search_img(self, emb):

        print("Compare");
        print();
        for key in self.db.keys():
            for face_emd in self.db[key]:
                print(np.linalg.norm(face_emd - emb));
                if np.linalg.norm(face_emd - emb) < 0.8:
                    print();
                    return key
        print();
        return None
    
    def search_img_fast(self, emb):
        print("Compare");
        print();
        
        bestMatchDiff = 0.8+0.15
        
      
        for key in self.db.keys():
            countWorst = 0
            print("Class: ", key);
            for face_emd in self.db[key]:
                
                
                diff = np.linalg.norm(face_emd - emb);
                print(diff);
                if diff > 1:
                    countWorst += 1;
                elif countWorst>0:
                    countWorst -= 1;
                
                if(countWorst>2):
                    break;
                if diff < bestMatchDiff:
                    return key
                
                    
        print("END");
       
        print();
        return None
    
    def search_img_thorough(self, emb):
        print("Compare123");
        print();
        bestMatchDiff = thresh
        bestMatchKey = None
        for key in self.db.keys():
            for face_emd in self.db[key]:
                
                diff = np.linalg.norm(face_emd - emb);
                print(diff);
                if diff < bestMatchDiff:
                    bestMatchDiff = diff
                    print(key)
                    bestMatchKey = key
                
                    
        print("END");
        print(bestMatchKey);
        print();
        return bestMatchKey

    def search_img_thorough2(self, emb, data):
        #print("Compare123");
        #print();
        
        bestMatchDiff = thresh
        bestMatchKey = None
        for key in data.keys():
            for face_emd in data[key]:
                
                diff = np.linalg.norm(face_emd - emb);
                #print(diff);
                if diff < bestMatchDiff:
                    bestMatchDiff = diff
                    #print(key)
                    bestMatchKey = key
                
        
        #print("END");
        #print(bestMatchKey);
        #print(bestMatchDiff);
        #print();
        return bestMatchKey


    def get_crops(self, faces, img):

        if faces.shape[0] < 1:
            return

        images = []

        for f in faces:
            f = f.astype(int)
            f[0] = np.maximum(f[0] - 20, 0)
            f[1] = np.maximum(f[1] - 20, 0)
            f[2] = np.minimum(f[2] + 20, 480)
            f[3] = np.minimum(f[3] + 20, 640)

            # print(f)
            crop = img[f[1]:f[3], f[0]:f[2]]
            # print('crops =     ',crop.shape)
            if not 0 in crop.shape:
                images.append(self.prewhiten(cv2.resize(crop, (160, 160))))
                
#        for img in images:
#             cv2.imshow('frame', img);
#             cv2.waitKey(0)

        
                
        
        ans = np.array(images)
        if ans.shape[0] > 0:
            return ans
        else:
            return None
        
        
    def getEmbed(self, img):

        feed_dict = {self.image_placeholder: img, self.phase_train_placeholder: False}
        embed = self.sess.run(self.embeddings, feed_dict = feed_dict)
        return embed


    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    def getBoxes(self, image):
        
        box, _ = detect_face.detect_face(image, self.minsize, \
                self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        return box
    
    def to_rgb(img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    
    def get_crops3(self, face, img):

        if face.shape[0] < 1:
            return

        images = [];

        
        face = face.astype(int)
        face[0] = np.maximum(face[0] - 20, 0)
        face[1] = np.maximum(face[1] - 20, 0)
        face[2] = np.minimum(face[2] + 20, 480)
        face[3] = np.minimum(face[3] + 20, 640)

            # print(f)
        crop = img[face[1]:face[3], face[0]:face[2]]
            # print('crops =     ',crop.shape)
        if not 0 in crop.shape:
            images.append(self.prewhiten(cv2.resize(crop, (160, 160))))


        #print("Images:")
        #print(images)
        
        #for imgg in images:
             #cv2.imshow('frame', imgg);
             #cv2.waitKey(0)
             #print ("Image: ")
             #print (imgg)

        #print("Len")
        #print(len(images))
        
        if len(images) > 0:
            return images
        else:
            return None
    
    def get_crops2(self, face, img):

        if face.shape[0] < 1:
            return

        images = [];

        
        face = face.astype(int)
        face[0] = np.maximum(face[0] - 20, 0)
        face[1] = np.maximum(face[1] - 20, 0)
        face[2] = np.minimum(face[2] + 20, 480)
        face[3] = np.minimum(face[3] + 20, 640)

            # print(f)
        crop = img[face[1]:face[3], face[0]:face[2]]
            # print('crops =     ',crop.shape)
        if not 0 in crop.shape:
            images.append(cv2.resize(crop, (160, 160)))


        #print("Images:")
        #print(images)
        
        #for imgg in images:
             #cv2.imshow('frame', imgg);
             #cv2.waitKey(0)
             #print ("Image: ")
             #print (imgg)

        #print("Len")
        #print(len(images))
        
        if len(images) > 0:
            return images
        else:
            return None
        
        
    def prewhiten2(self, image):
        
        images = []
        images.append(self.prewhiten(image))
        
        ans = np.array(images);
        
        
        if ans.shape[0] > 0:
            return ans
        else:
            return None