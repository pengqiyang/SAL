import os
import numpy as np
import scipy.io as sio
from collections import Counter
import pdb
#id:  id of identity
#camid:  id of camera
def import_pa100k(dataset_dir):
    peta_dir = os.path.join(dataset_dir,'PA100K')
    if not os.path.exists(peta_dir):
        print('Please Download PETA Dataset and check if the sub-folder name exists')

    file_list = os.listdir(peta_dir)

    for name in file_list:
        id = name.split('.')[0]
        globals()[id] = name
   
    
    
    #cc = sio.loadmat(os.path.join('dataset/PETA/', 'PETA.mat'))
    #cc['peta'][0][0][0][0][4:]
    f = sio.loadmat(os.path.join(dataset_dir, 'annotation.mat'))
    attr_dict = {}
    #pdb.set_trace()
    train_nrow, train_ncol = f['train_images_name'].shape
    gallery_nrow, gallery_ncol = f['test_images_name'].shape
  
    globals()['train'] = {}
    globals()['query'] = {}
    globals()['gallery'] = {}
    globals()['train']['data'] = []
    globals()['train']['ids'] = []
    globals()['query']['data'] = []
    globals()['query']['ids'] = []
    globals()['gallery']['data'] = []
    globals()['gallery']['ids'] = []
    train_attribute = {}
    test_attribute = {}
    
    
    for id in range(train_nrow):
        index = f['train_images_name'][id][0][0].split('.')[0]
        camid = np.int64(1)
        name = globals()[index]
        images = os.path.join(peta_dir,name)
        globals()['train']['ids'].append(id)
        globals()['train']['data'].append([images, id, id, camid, name])
        train_attribute[id] = f['train_label'][id]
    
    
    for id in range(gallery_nrow):
        index = f['test_images_name'][id][0][0].split('.')[0]
        name = globals()[index]
        images = os.path.join(peta_dir,name)
        camid = np.int64(1)
        globals()['query']['ids'].append(id)
        globals()['gallery']['ids'].append(id)
        globals()['query']['data'].append([images, id, id, camid, name])
        globals()['gallery']['data'].append([images, id, id, camid, name])
        test_attribute[id] = f['test_label'][id]


    attributes = []  # gives the names of attributes, label
    for i in range(f['attributes'].shape[0]):
        attributes.append(f['attributes'][i][0])
    

    #pdb.set_trace()
    return train, query, gallery, train_attribute, test_attribute, attributes

