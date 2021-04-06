import sys
import re
import string
import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import pandas as pd
import subprocess
import itertools
import base64

#TO make the heading about the location(places)_JS
def make_heading_loc(file_set,loc_to_idx,loc_list):
    list = {}
    loc_heading = 0
    viewpoint_loc = []
    loc_info = {}
    heading_idx=[]

    for i in range(0,len(file_set)):
        append_loc = []
        with open(file_set[i]) as json_file:
            data = json.load(json_file)
        viewpoint_id = data['viewpoint_id']
        # Make the heading list based on skybox num
        # panorama images: skybox1,skybox2, skybox3, skybox4

        if data['skybox']=='skybox1':
            heading_idx = [1,2,3]
        elif data['skybox']=='skybox2':
            heading_idx = [4,5,6]
        elif data['skybox']=='skybox3':
            heading_idx = [7,8,9]
        else: #data['skybox']=='skybox4'
            heading_idx = [10,11,12]
        #print('data', data)

        for i in range(0, 1):
            if float(data['location_detection'][i]['probability']) >= 0.01:
                loc_name = data['location_detection'][i]['location']
                loc_prob = data['location_detection'][i]['probability']
                if loc_name not in loc_list:
                    loc_name = 'None'
                    loc_prob = 0.0
                    #loc_heading = 0
                for heading in heading_idx:
                    loc = (loc_to_idx[loc_name], heading, loc_prob)
                    append_loc.append(loc)# THIS SILHUM PILYO
                list[data['skybox']] = append_loc
            else:
                for heading in heading_idx:
                    loc = (loc_to_idx['None'], heading, 0.0)
                    append_loc.append(loc)
                list[data['skybox']] = append_loc
    viewpoint_loc = list['skybox1'] + list['skybox2'] + list['skybox3'] + list['skybox4']
    loc_info[viewpoint_id] = viewpoint_loc
    #print ('loc_info',loc_info[viewpoint_id])
    #arr = np.array(loc_info)
    arr = np.array(viewpoint_loc)#  numpy.ndarray
    return arr


# To Make the heading about the objects_JS
def make_heading_obj(file_set,obj_to_idx,obj_list):
    heading_idx = []
    list = {}
    obj_heading = 0
    viewpoint_obj =[] #for add each skybox info(1+2+3+4)
    obj_info = {} #It will contain all 4(skybox 1,2,3,4) objects

    for i in range(0,len(file_set)):#len(file_set)
        append_obj = []

        # Open the data(Input: i-th json file path)
        with open(file_set[i]) as json_file:
            data = json.load(json_file)
        viewpoint_id = data['viewpoint_id']
        # Make the heading list based on skybox number
        # Panorama images : skybox1, skybox2, skybox3, skybox4

        if data['skybox']=='skybox1':
            heading_idx = [1,2,3]
        elif data['skybox']=='skybox2':
            heading_idx = [4,5,6]
        elif data['skybox']=='skybox3':
            heading_idx = [7,8,9]
        else: #data['skybox']=='skybox4'
            heading_idx = [10,11,12]

        for i in range(0,len(data['object_list'])):
            if data['object_list'][i]['x'] > 0 and data['object_list'][i]['x'] <= 341:
                obj_heading = heading_idx[0]
            elif data['object_list'][i]['x'] > 342 and data['object_list'][i]['x'] <= 682:
                obj_heading = heading_idx[1]
            elif data['object_list'][i]['x'] > 683 and data['object_list'][i]['x'] <= 1024:
                obj_heading = heading_idx[2]

            obj_name=data['object_list'][i]['object_name']
            obj_prob=data['object_list'][i]['probability']
            if obj_name not in obj_list:
                if i > 0 and (0,0,0.0) in list[data['skybox']]:
                    continue
                else:
                    obj_name = 'None'
                    obj_heading = 0
                    obj_prob = 0.0
            obj = (obj_to_idx[obj_name], obj_heading, obj_prob)
            append_obj.append(obj)
            list[data['skybox']]=append_obj

            #print('object_id :', data['object_list'][i]['object_id'])
            #print('object_name:', data['object_list'][i]['object_name'])
            #print('object_probability:', data['object_list'][i]['probability'])
            #print('object_x:', data['object_list'][i]['x'])#image size : 1024 * 4 = 4096 [0:341][342:682][683:1024]
            # obj_info.append(zip(data['object_list'][i]['object_id'],obj_loc_idx))
            if i > 0 and (0, 0) in list[data['skybox']]:
                continue
            else:
                obj = (obj_to_idx['None'], 0, 0.0)
            append_obj.append(obj)
            list[data['skybox']]=append_obj

    viewpoint_obj = list['skybox1']+list['skybox2']+list['skybox3']+list['skybox4']
    obj_info[viewpoint_id] = viewpoint_obj
    #arr = np.array(obj_info)
    arr = np.array(viewpoint_obj)
    return arr

def search_file(path,viewpointId):
    result=[]
    nonfile_list =[]#skybox0, skybox5
    nonused_skybox = ['skybox1','skybox2','skybox3','skybox4']
    for root, dirs, files in os.walk(path):
        rootpath = os.path.join(os.path.abspath(path), root)
        for file in files:
            if os.path.splitext(file)[1].lower() == '.json':
                filepath = os.path.join(rootpath, file)
                for skybox in nonused_skybox:#1,2,3,4 for
                    if viewpointId in filepath:
                        if skybox in filepath:
                            result.append(filepath)
    #print('result',result)
    return result

def make_obj_idx():
    obj_list_file = open('/media/ai8503/f4837b59-5cdd-4faa-955b-befb9a99e668/JS/jisu_selfmonitoring-agent/selfmonitoring-agent/tasks/R2R-pano/objects.txt', 'r')
    obj_list = []
    obj_to_idx = {}

    while True:
        line = obj_list_file.readline()
        line = line[:-1]
        if not line: break
        obj_list.append(line)
    obj_list_file.close()

    # Make the list of objects and matching these into num_JS # THIS SHOULD BE IN ENV.py
    for i, obj_name in enumerate(obj_list):
        obj_to_idx[obj_name] = i  # None: 0 , pottedplant : 1, ...
    #print('obj_to_idx', obj_to_idx)

    return (obj_to_idx,obj_list)

def make_loc_idx():
    loc_list_file = open('/media/ai8503/f4837b59-5cdd-4faa-955b-befb9a99e668/JS/jisu_selfmonitoring-agent/selfmonitoring-agent/tasks/R2R-pano/places.txt', 'r')
    loc_list = []
    loc_to_idx = {}
    while True:
        line = loc_list_file.readline()
        line = line[:-1]
        if not line: break
        loc_list.append(line)
    loc_list_file.close()

    # Make the list of locations and matching these into num_JS # THIS SHOULD BE IN ENV.py
    for i, loc_name in enumerate(loc_list):
        loc_to_idx[loc_name] = i# None: 0 ,
    #print('loc_to_idx',loc_to_idx)

    return (loc_to_idx,loc_list)


#------- Encoding feature -------#
# Encoded feature = (object_heading,object_label)
def encode_object(obj_info,obj_to_idx):
    encoded_obj = np.zeros((12,17))#12: Heading, 17: Object labels New Object label : 81


    for idx in range(0,len(obj_info)):
        object_label = int(obj_info[idx][0])
        object_heading =  int(obj_info[idx][1])
        object_probability = obj_info[idx][2]
        #print('object_label',object_label,object_heading,object_probability)

        if object_label == 0:
            continue
        else:
            if encoded_obj[object_heading-1][object_label-1] == 0:# if isn't zero : Already Masking
                encoded_obj[object_heading-1][object_label-1] = object_probability

    return encoded_obj#object_feature


def encode_loc(loc_info,loc_to_idx):
    encoded_loc = np.zeros((12, 10)) # 12: Heading, 10: 10
    place_feature = np.zeros((36, 10))

    for idx in range(0, len(loc_info)):
        place_label = int(loc_info[idx][0])
        place_heading = int(loc_info[idx][1])
        place_probability = loc_info[idx][2]
        if place_label == 0:
            continue
        else:
            if encoded_loc[place_heading-1][place_label - 1] == 0:  # if isn't zero : Already Masking
                encoded_loc[place_heading-1][place_label - 1] = place_probability


    return encoded_loc#place_feature


def json_read_main(scanId,viewpointId):

    (obj_to_idx, obj_list) = make_obj_idx()
    (loc_to_idx, loc_list) = make_loc_idx()

    scanId= scanId
    viewpointId=viewpointId

    obj_data_path=("/media/ai8503/f4837b59-5cdd-4faa-955b-befb9a99e668/JS/jisu_selfmonitoring-agent/selfmonitoring-agent/tasks/R2R-pano/data/objects/%s/"%(scanId))
    loc_data_path = ("/media/ai8503/f4837b59-5cdd-4faa-955b-befb9a99e668/JS/jisu_selfmonitoring-agent/selfmonitoring-agent/tasks/R2R-pano/data/location/%s/"%(scanId))
    obj_file_set = search_file(obj_data_path,viewpointId)
    loc_file_set = search_file(loc_data_path,viewpointId)

    obj_info = make_heading_obj(obj_file_set,obj_to_idx,obj_list)
    loc_info = make_heading_loc(loc_file_set,loc_to_idx,loc_list)

    #----object
    obj_encoding = encode_object(obj_info,obj_to_idx)#nparray, (12,17)
    #----location
    loc_encoding = encode_loc(loc_info,loc_to_idx)#(12,10)
    return (obj_encoding, loc_encoding)#return as tuple
