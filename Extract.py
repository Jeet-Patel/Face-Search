# extracting faces from the custom dataset

# Importing required librabies
#import numpy as np
from os import listdir
import cv2
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
#from tensorflow import keras
from models import VGGFace, OpenFace, FbDeepFace, Facenet
import copy


face_size = {'facenet':(160,160),
             'deepface':(152,152),
             'openface':(96,96),
             'vggface':(224,224)}

detector = MTCNN()

facenet = Facenet.loadModel()
deep_face = FbDeepFace.loadModel()
open_face = OpenFace.loadModel()
vgg_face = VGGFace.loadModel()
models = {'facenet': facenet,
          'openface': open_face,
          'deepface': deep_face,
          'vggface': vgg_face}

class Extract:
    def __init__(self,directory, face_size, detector, models, predict = 0):
        self.directory = directory
        self.face_size = face_size
        self.detector = detector
        self.models = models
        self.predict = predict
        self.face_dict = {}
        self.facenet_embed = []
        self.deepface_embed = []
        self.vggface_embed = []
        self.openface_embed = []
        self.model_embed = {'facenet':self.facenet_embed,
                            'deepface':self.deepface_embed,
                            'vggface':self.vggface_embed,
                            'openface':self.openface_embed}
        
    
    def lst_images(self,directory):
        return listdir(directory)

    
    def _load_(self, directory, face_size, models):
        self.face_dict = {}
        self.facenet_embed = []
        self.deepface_embed = []
        self.vggface_embed = []
        self.openface_embed = []
        self.face_identity = []
        img_fldr = listdir(self.directory)
        #print(img_fldr)
        for fldr in img_fldr:
            path = self.directory + '/' + fldr
            list_images = listdir(path)
            for i,filename in enumerate(list_images):
                # setting the path of the image
                img_path = path+'/'+filename
                # using opencv lib to load the image
                print(img_path)
                image = cv2.imread(img_path)
                faces = detector.detect_faces(image)
                # using the bounding box from result to extract face
                #print(result)
                x1,y1,width,height = faces[0]['box']
                # the top-left and bottom-right co-ordinates
                x1,y1 = abs(x1),abs(y1)
                x2,y2 = x1+width,y1+height
                #face
                face = image[y1:y2,x1:x2]
                #print(str(rank+i) + ' ' + str(faces) + str(len(faces)))
                #print(filename+str([x1,y1,width,height]))
                print(fldr + '_' + filename)
                self.face_dict[filename] = {'name':fldr , 'box':[x1,y1,width,height]}
                self.face_identity.append(filename)
                for m in list(models.keys()):
                    #resize face to required size
                    face_copy = copy.deepcopy(face)
                    required_size = face_size[m]    
                    face_1 = cv2.resize(face_copy,required_size)
                    embed = self.get_embedding(models[m], face_1)
                    self.model_embed[m].append(embed)
        return self.face_dict, self.face_identity, self.model_embed
        
        
    def _loadpredict_(self, directory, face_size,models):
        self.face_dict = {}
        self.facenet_embed = []
        self.deepface_embed = []
        self.vggface_embed = []
        self.openface_embed = []
        # using opencv lib to load the image
        image = cv2.imread(directory)
        result = detector.detect_faces(image)
        # using the bounding box from result to extract face
        
        for faces in result:
            x1,y1,width,height = faces['box']
            # the top-left and bottom-right co-ordinates
            x1,y1 = abs(x1),abs(y1)
            x2,y2 = x1+width,y1+height
            #face
            face = image[y1:y2,x1:x2]
            #resize face to required size
            for m in list(models.keys()):
                #resize face to required size
                face_copy = copy.deepcopy(face)
                required_size = face_size[m]    
                face_1 = cv2.resize(face_copy,required_size)
                embed = self.get_embedding(models[m], face_1)
                self.model_embed[m].append(embed)
        return self.model_embed
        
        
    # get the face embedding for one face
    def get_embedding(self, model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        
        embed_vec = model.predict(samples)
        return embed_vec[0]
    
        
    
    # Extracting face ffrom a photograph
    def get_face_data(self, directory, face_size,models):
        if self.predict ==0:
            return self._load_(directory, face_size, models)
        else:
            return self._loadpredict_(directory, face_size, models)


