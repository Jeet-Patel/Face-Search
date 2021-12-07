
# loading the Face_embeddings_dict pickle file
import pickle
import numpy as np
from annoy import AnnoyIndex
#import json

with open("data/face_embeddings_dict.pkl", "rb") as data:
    face_embeddings_dict = pickle.load(data)
data.close()

facenet_embeddings = np.asarray(face_embeddings_dict['facenet'])
deepface_embeddings = np.asarray(face_embeddings_dict['deepface'])
vggface_embeddings = np.asarray(face_embeddings_dict['vggface'])
openface_embeddings = np.asarray(face_embeddings_dict['openface'])


# Make Annoy Index 
facenet_tree =  AnnoyIndex(128, 'euclidean')
deepface_tree =  AnnoyIndex(4096, 'euclidean')
vggface_tree =  AnnoyIndex(2622, 'euclidean')
openface_tree =  AnnoyIndex(128, 'euclidean')


# Fill Annoy
for i, v in enumerate(facenet_embeddings):
    facenet_tree.add_item(i,v)
for i, v in enumerate(deepface_embeddings):
    deepface_tree.add_item(i,v)
for i, v in enumerate(vggface_embeddings):
    vggface_tree.add_item(i,v)
for i, v in enumerate(openface_embeddings):
    openface_tree.add_item(i,v)


# Building the trees(instead you know of planting them)
facenet_tree.build(1000)
deepface_tree.build(1000)
vggface_tree.build(1000)
openface_tree.build(1000)


# Saving the trees
facenet_tree.save('Annoy Trees/facenet_annoytree.ann')
deepface_tree.save('Annoy Trees/deepface_annoytree.ann')
vggface_tree.save('Annoy Trees/vggface_annoytree.ann')
openface_tree.save('Annoy Trees/openface_annoytree.ann')
