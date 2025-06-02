# -*- coding: utf-8 -*-
# !pip install tensorflow==2.10.1

from google.colab import drive
drive.mount('/content/drive')

import io
import imageio
import ipywidgets
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
# import wandb
# from wandb.keras import WandbCallback


import PIL
from IPython import display
from tqdm.notebook import tqdm_notebook
import random
from tensorflow.keras import layers, Model, Input
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
# from keras_preprocessing.image import img_to_array, load_img

# Setting seed for reproducibility
SEED = 42

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ['PYTHONHASHSEED']='0'
tf.keras.utils.set_random_seed(SEED)
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.callbacks import EarlyStopping

# import tensorflow_addons as tfa
from tensorflow.keras.layers import Concatenate, Conv2D, Conv3D, Layer, Dense, Attention, GlobalAveragePooling2D, Lambda,\
 Flatten, Reshape, AveragePooling2D, Add, LSTM, Multiply, Softmax, Dropout, LeakyReLU, GRU, TimeDistributed

from keras.losses import binary_crossentropy
import keras.backend as K
from tensorflow.keras.layers import Layer, Input, Concatenate, Conv3D, Add, Dense, Lambda, Activation, Multiply, RepeatVector, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
import tensorflow. keras.backend as K

"""# PIE/JAAD data and some processing functions"""

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/MyDrive/obj4'
import sys
# from sf_gru_o import SFGRU
sys.path.insert(0,'/content/drive/MyDrive/datasets/PIE')
from pie_data2 import PIE

data_opts ={'fstride': 1,
            'subset': 'default',
            'data_split_type': 'random',  # kfold, random, default
            'seq_type': 'crossing',
            'min_track_size': 61} ## for obs length of 15 frames + 60 frames tte. This should be adjusted for different setup
imdb = PIE(data_path='/content/drive/MyDrive/datasets/PIE') # change with the path to the dataset

beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

def convert_normalize_bboxes(all_bboxes, normalize, bbox_type):
    '''input box type is x1y1x2y2 in original resolution'''
    for i in range(len(all_bboxes)):
        if len(all_bboxes[i]) == 0:
            continue
        bbox = np.array(all_bboxes[i])
        # NOTE ltrb to cxcywh
        if bbox_type == 'cxcywh':
            bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
            bbox[..., [0, 1]] += bbox[..., [2, 3]]/2
        # NOTE Normalize bbox
        if normalize == 'zero-one':
            # W, H  = all_resolutions[i][0]
            _min = np.array(self.args.min_bbox)[None, :]
            _max = np.array(self.args.max_bbox)[None, :]
            bbox = (bbox - _min) / (_max - _min)
        elif normalize == 'plus-minus-one':
            # W, H  = all_resolutions[i][0]
            _min = np.array(self.args.min_bbox)[None, :]
            _max = np.array(self.args.max_bbox)[None, :]
            bbox = (2 * (bbox - _min) / (_max - _min)) - 1
        elif normalize == 'none':
            pass
        else:
            raise ValueError(normalize)
        all_bboxes[i] = bbox
    return all_bboxes

def get_traj_tracks(dataset, data_types, observe_length, predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:# KeyError:
                raise KeyError('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        d['age'] = dataset['age']
        d['gen'] = dataset['gen']
        d['signalized'] = dataset['signalized']
        d['Num_lanes'] = dataset['Num_lanes']
        d['activities'] = dataset['activities']
        # d['resolution'] = dataset['resolution']
        # d['flow'] = []
        num_trks = len(d['image'])
        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                for i in range(0, len(track) - seq_length + 1, overlap_stride):
                    tracks.append(track[i:i + seq_length])
            d[k] = tracks
        #  Normalize tracks using FOL paper method,
        d['bbox'] = convert_normalize_bboxes(d['bbox'],'none', 'ltrb')
        return d

def get_traj_data(data, **model_opts):
    """
    Main data generation function for training/testing
    :param data: The raw data
    :param model_opts: Control parameters for data generation characteristics (see below for default values)
    :return: A dictionary containing training and testing data
    """

    opts = {
        'normalize_bbox': True,
        'track_overlap': 0.5,
        'observe_length': 15,
        'predict_length': 45,
        'enc_input_type': ['bbox'],
        'dec_input_type': [],
        'prediction_type': ['bbox']
    }
    for key, value in model_opts.items():
        assert key in opts.keys(), 'wrong data parameter %s' % key
        opts[key] = value

    observe_length = opts['observe_length']
    data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
    data_tracks = get_traj_tracks(data, data_types, observe_length,
                                  opts['predict_length'], opts['track_overlap'],
                                  opts['normalize_bbox'])
    obs_slices = {}
    pred_slices = {}
    #  Generate observation/prediction sequences from the tracks
    for k in data_tracks.keys():
        obs_slices[k] = []
        pred_slices[k] = []
        # NOTE: Add downsample function
        down = 1
        obs_slices[k].extend([d[down-1:observe_length:down] for d in data_tracks[k]])
        pred_slices[k].extend([d[observe_length+down-1::down] for d in data_tracks[k]])

    ret =  {'obs_image': obs_slices['image'],
            'obs_pid': obs_slices['pid'],

            'pred_image': pred_slices['image'],
            'pred_pid': pred_slices['pid'],


            'obs_bbox': np.array(obs_slices['bbox']), #enc_input,

            'pred_bbox': np.array(pred_slices['bbox']), #pred_target,
            'age': np.array(obs_slices['age']),
            'gen': np.array(obs_slices['gen']),
            'signalized': np.array(obs_slices['signalized']),
            'lanes': np.array(obs_slices['Num_lanes']),
            'activities': np.array(obs_slices['activities'])
            }

    return ret

import numpy as np
traj_model_opts = {'normalize_bbox': True,
              'track_overlap': 0.5,
              'observe_length': 15,
              'predict_length': 45,
              'enc_input_type': ['bbox'],
              'dec_input_type': [],
              'prediction_type': ['bbox']
              }

traj_model_opts['enc_input_type'].extend(['obd_speed', 'heading_angle'])
traj_model_opts['prediction_type'].extend(['obd_speed', 'heading_angle'])

train_data = get_traj_data(beh_seq_train, **traj_model_opts)
# val_data = get_traj_data(beh_seq_val, **traj_model_opts)
# test_data = get_traj_data(beh_seq_test, **traj_model_opts)

train_data.keys()

import tensorflow as tf

def squarify(bbox, squarify_ratio, img_width):
    """
    Changes the ratio of bounding boxes to a fixed ratio
    :param bbox: Bounding box tensor [x_min, y_min, x_max, y_max]
    :param squarify_ratio: Ratio to be changed to
    :param img_width: Image width tensor
    :return: Squarified bounding box tensor [x_min, y_min, x_max, y_max]
    """
    width = tf.abs(bbox[0] - bbox[2])
    height = tf.abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox_x_min = bbox[0] - width_change / 2
    bbox_x_max = bbox[2] + width_change / 2

    # Cast img_width to float64
    img_width = tf.cast(img_width, tf.float64)

    # Clamp bounding box to image borders
    bbox_x_min = tf.maximum(bbox_x_min, 0)
    bbox_x_max = tf.minimum(bbox_x_max, img_width)

    # Concatenate x_min, y_min, x_max, y_max into a tensor
    squarified_bbox = tf.stack([bbox_x_min, bbox[1], bbox_x_max, bbox[3]])

    return squarified_bbox

def bbox_sanity_check(img_size, bbox):
	"""
	Confirms that the bounding boxes are within image boundaries.
	If this is not the case, modifications is applied.
	:param img_size: The size of the image
	:param bbox: The bounding box coordinates
	:return: The modified/original bbox
	"""
	img_width, img_heigth = img_size
	if bbox[0] < 0:
		bbox[0] = 0.0
	if bbox[1] < 0:
		bbox[1] = 0.0
	if bbox[2] >= img_width:
		bbox[2] = img_width - 1
	if bbox[3] >= img_heigth:
		bbox[3] = img_heigth - 1
	return bbox

def jitter_bbox(img_path, bbox, mode, ratio):
	"""
	This method jitters the position or dimensions of the bounding box.
	:param img_path: The to the image
	:param bbox: The bounding box to be jittered
	:param mode: The mode of jittere:
	'same' returns the bounding box unchanged
		  'enlarge' increases the size of bounding box based on the given ratio.
		  'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
		  'move' moves the center of the bounding box in each direction based on the given ratio
		  'random_move' moves the center of the bounding box in each direction by randomly
						sampling a value in [-ratio,ratio)
	:param ratio: The ratio of change relative to the size of the bounding box.
		   For modes 'enlarge' and 'random_enlarge'
		   the absolute value is considered.
	:return: Jittered bounding box
	"""

	assert(mode in ['same','enlarge','move','random_enlarge','random_move']), \
			'mode %s is invalid.' % mode

	if mode == 'same':
		return bbox

	img = load_img(img_path)

	if mode in ['random_enlarge', 'enlarge']:
		jitter_ratio  = abs(ratio)
	else:
		jitter_ratio  = ratio

	if mode == 'random_enlarge':
		jitter_ratio = np.random.random_sample()*jitter_ratio
	elif mode == 'random_move':
		# for ratio between (-jitter_ratio, jitter_ratio)
		# for sampling the formula is [a,b), b > a,
		# random_sample * (b-a) + a
		jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

	jit_boxes = []
	for b in bbox:
		bbox_width = b[2] - b[0]
		bbox_height = b[3] - b[1]

		width_change = bbox_width * jitter_ratio
		height_change = bbox_height * jitter_ratio

		if width_change < height_change:
			height_change = width_change
		else:
			width_change = height_change

		if mode in ['enlarge','random_enlarge']:
			b[0] = b[0] - width_change //2
			b[1] = b[1] - height_change //2
		else:
			b[0] = b[0] + width_change //2
			b[1] = b[1] + height_change //2

		b[2] = b[2] + width_change //2
		b[3] = b[3] + height_change //2

		# Checks to make sure the bbox is not exiting the image boundaries
		b = bbox_sanity_check(img.size, b)
		jit_boxes.append(b)
	# elif crop_opts['mode'] == 'border_only':
	return jit_boxes

def img_pad(img, mode = 'warp', size = 224):
	"""
	Pads a image given the boundries of the box needed
	:param img: The image to be coropped and/or padded
	:param mode: The type of padding or resizing:
			warp: crops the bounding box and resize to the output size
			same: only crops the image
			pad_same: maintains the original size of the cropped box  and pads with zeros
			pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
						the desired output size in that direction while maintaining the aspect ratio. The rest
						of the image is	padded with zeros
			pad_fit: maintains the original size of the cropped box unless the image is bigger than the size
					in which case it scales the image down, and then pads it
	:param size: Target size of image
	:return:
	"""
	assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
	image = img.copy()
	if mode == 'warp':
		warped_image = image.resize((size,size), PIL.Image.NEAREST)
		return warped_image
	elif mode == 'same':
		return image
	elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
		img_size = image.size  # size is in (width, height)
		ratio = float(size)/max(img_size)
		if mode == 'pad_resize' or	\
			(mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
			img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
			image = image.resize(img_size, PIL.Image.NEAREST)
		padded_image = PIL.Image.new("RGB", (size, size))
		padded_image.paste(image, ((size-img_size [0])//2,
					(size-img_size [1])//2))
		return padded_image

"""#Custom Data Generator"""

#Custom Data Generator
save_path1 = '/content/drive/MyDrive/obj4/features'
save_path2 = '/content/drive/MyDrive/obj4/seg_features'
# convnet = tf.keras.applications.efficientnet.EfficientNetB4(input_shape=(224, 224, 3),
#                               include_top=False, weights='imagenet')


import tensorflow as tf
import PIL

import random
import numpy as np
class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self,x,y,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True, train=1):


        self.x=x
        self.y =y

        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.train= train
        self.n = len(self.x[0])

    def on_epoch_end(self):
        if self.shuffle:
            ind = np.random.randint(0, len(self.x[0]), len(self.x[0]))
            self.x[0] = self.x[0][ind]
            self.x[1] = self.x[1][ind]
            self.x[2] = self.x[2][ind]


            self.y = self.y[ind]




    def __get_input(self, path_batch, bbox, target_size, seg=0):


      if seg==0:
        img_seq=[]
        for path,b in zip(path_batch,bbox):

           img_seq.append(path)



      elif seg==1:

        img_seq=[]
        for path,b in zip(path_batch,bbox):
           seg_path= path.replace("images", "seg_images")
           img_seq.append(seg_path)

      elif seg==4:
         img_seq=[]

         for c in context:
              img_seq.append(c)





      else:
          box_seq = []
          img_seq=[]
            # Calculate minimum and maximum values

          x_max = 1920
          y_max = 1080


            # Normalize the bounding box sequence
          for b in bbox:
            img_seq.append(b)

              # x_bleft, y_bleft, x_tright, y_tright = b
              # xbl_normalized = (x_bleft) / (x_max)
              # ybl_normalized = (y_bleft) / (y_max)
              # xtr_normalized = (x_tright) / (x_max)
              # ytr_normalized = (y_tright) / (y_max)



              # normalized_b = (
              #     xbl_normalized,
              #     ybl_normalized,
              #     xtr_normalized,
              #     ytr_normalized
              # )


          #     box_seq.append(normalized_b)





      return img_seq


    def __get_data(self, batches_b, batches_p, batches_i):
        # # Generates data containing batch_size samples


        bbox_batch = batches_b
        pred_batch = batches_p
        image_batch = batches_i
        x_im_bat=[]
        x_seg_bat=[]
        x_bbox_bat=[]
        x_pred_bat=[]


        for i,j in zip(image_batch, bbox_batch):


          x_bbox_bat.append(self.__get_input(i, j, self.input_size, seg=2))
        X_bbox_batch =np.asarray(x_bbox_bat)



        for i,j in zip(image_batch, pred_batch):


          x_pred_bat.append(self.__get_input(i, j, self.input_size, seg=2))
        X_pred_batch =np.asarray(x_pred_bat)

        for i,j in zip(image_batch, bbox_batch):

          inp = self.__get_input(i, j, self.input_size, seg=0)
          x_im_bat.append(inp)

        X_batch =np.asarray(x_im_bat)

        for i,j in zip(image_batch, bbox_batch):
          inp = self.__get_input(i, j, self.input_size, seg=1)
          x_seg_bat.append(inp)
        X_seg_batch =np.asarray(x_seg_bat)



        return tf.squeeze(X_bbox_batch, axis=0), tf.squeeze(X_batch, axis=0), tf.squeeze(X_seg_batch, axis=0)


    def __getitem__(self, index):


        batches_obs = self.x[0][index * self.batch_size:(index + 1) * self.batch_size]
        batches_pred =  self.x[1][index * self.batch_size:(index + 1) * self.batch_size]
        batches_im = self.x[2][index * self.batch_size:(index + 1) * self.batch_size]

        batches_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        X_obs_box, X, X_seg = self.__get_data(batches_obs, batches_pred, batches_im)

        # X_obs_box, X_pred_box, X = self.__get_data(batches_obs, batches_pred, batches_im)
        return tf.convert_to_tensor(X), tf.convert_to_tensor(X_seg), tf.convert_to_tensor(X_obs_box), np.unique(batches_y)
        # return tf.convert_to_tensor(X_obs_box), tf.convert_to_tensor(X_pred_box), tf.convert_to_tensor(X)
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__()-1:
                self.on_epoch_end()
    def __len__(self):
        return self.n // self.batch_size

BATCH_SIZE=8
x_train = [train_data['obs_bbox'], train_data['pred_bbox'], train_data['obs_image']]
y_train = train_data['activities']
# x_val = [val_data['obs_bbox'], val_data['pred_bbox'], val_data['obs_image']]
# x_test = [test_data['obs_bbox'], test_data['pred_bbox'], test_data['obs_image']]
traingen = CustomDataGen(x_train, y_train, batch_size=1)
# valgen = CustomDataGen([x_val[0], x_val[1]],batch_size=BATCH_SIZE)
# testgen = CustomDataGen([x_test[0], x_test[1]], batch_size=BATCH_SIZE)

crossing_ind=[]
for i in range(len(traingen)):

  print(i)
  if traingen[i][3]==1:

     crossing_ind.append(i)

len(crossing_ind)

traingen[crossing_ind[0]][3]

ot = tf.string, tf.string, tf.float64, tf.float64

os = (None)
ds = tf.data.Dataset.from_generator(traingen,
                                    output_types = ot,
                                    output_shapes = os)
# ds = ds.prefetch(tf.data.AUTOTUNE).cache().batch(8, num_parallel_calls=tf.data.AUTOTUNE)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D


def preprocess_and_compute_features_img(a,g):
    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    all_tensors_1 = tf.TensorArray(dtype=tf.float32, size=len(a), dynamic_size=True, clear_after_read=False)


    for i in range(len(a)):
        flip_condition = tf.strings.regex_full_match(a[i], ".*flip.*")

        image = tf.io.read_file(a[i])  # Read image file
        image = tf.image.decode_png(image, channels=3)  # Decode PNG image
        image = tf.cond(flip_condition, lambda: tf.image.flip_left_right(image), lambda: image)
        try:
              box = g[i]
              # box = squarify(box, 1, tf.shape(image)[0])
              box= tf.cast(box, tf.int32)
              offset_height = box[1]
              offset_width = box[0]
              # target_height = tf.abs(box[3]-box[1])
              # target_width = tf.abs(box[2]-box[0])
              target_height = tf.maximum(tf.abs(box[3] - box[1]), 1)
              target_width = tf.maximum(tf.abs(box[2] - box[0]), 1)

              # Check if target_width is less than or equal to 0

              cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
        except Exception as e:
              cropped_image = image
        # # Resize or pad the cropped image to 224x224
        cropped_image = tf.image.resize_with_pad(cropped_image, 224, 224)

        # Preprocess image for EfficientNet
        image = tf.cast(cropped_image, tf.float32)  # Cast image to float32
        image /= 255.0
        image = tf.keras.applications.efficientnet.preprocess_input(image)  # Preprocess image for EfficientNet
        features = base_model(tf.expand_dims(image, 0))  # Compute EfficientNet features
        features = tf.squeeze(GlobalAveragePooling2D()(features))
        all_tensors_1=all_tensors_1.write(i, features)
    stacked_tensor_1 = all_tensors_1.stack()
    return stacked_tensor_1

def preprocess_and_compute_features_seg(a):
    # a,b,c,d,e,f,g =x

    base_model = EfficientNetB0(include_top=False, weights='imagenet')
    all_tensors_2 = tf.TensorArray(dtype=tf.float32, size=len(a), dynamic_size=True, clear_after_read=False)


    for i in range(len(a)):
        flip_condition = tf.strings.regex_full_match(a[i], ".*flip.*")

        image = tf.io.read_file(a[i])  # Read image file
        image = tf.image.decode_png(image, channels=3)  # Decode PNG image
        image = tf.cond(flip_condition, lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.image.resize(image, (224, 224))  # Resize image to fit EfficientNet input size
        image = tf.cast(image, tf.float32)  # Cast image to float32
        image = tf.keras.applications.efficientnet.preprocess_input(image)  # Preprocess image for EfficientNet
        features = base_model(tf.expand_dims(image, 0))  # Compute EfficientNet features
        features = tf.squeeze(GlobalAveragePooling2D()(features))
        all_tensors_2=all_tensors_2.write(i, features)
    stacked_tensor_2 = all_tensors_2.stack()
    return stacked_tensor_2

def preprocess_bbox(g):
    img_height=1080
    img_width=1920

    all_tensors_3 = tf.TensorArray(dtype=tf.float64, size=len(g), dynamic_size=True, clear_after_read=False)

    for k in range(len(g)):
        bbox = g[k]
        x_1, y_1, x_2, y_2 = tf.split(bbox, 4, axis=-1)

        # Normalize bounding box coordinates
        y_1 = y_1 / img_height
        x_1 = x_1 / img_width
        y_2 = y_2 / img_height
        x_2 = x_2 / img_width

        normalized_bbox = tf.concat([x_1, y_1, x_2, y_2], axis=-1)
        all_tensors_3 = all_tensors_3.write(k, normalized_bbox)

    stacked_tensor_3 = all_tensors_3.stack()
    return stacked_tensor_3

def encode(data, categories):
  one_hot_encoded = tf.squeeze(tf.one_hot(tf.cast(data, tf.int32), depth=categories))
  return one_hot_encoded


dataset=ds
# Map preprocessing and feature computation function to the dataset for the first and fourth elements
dataset = dataset.map(lambda a,d,g,y: (preprocess_and_compute_features_img(a,g),preprocess_and_compute_features_seg(d),g, y), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(lambda a,d,g,y: (a,d,preprocess_bbox(g),y), num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.filter(lambda a,d,g,y: tf.shape(a[0])[0] > 0)

def prepare_dataset(a, d, g, y):
    x = (a, d, g)  # Combine inputs into a tuple
    threshold = 0.5
    binary_labels = tf.cast(y >= threshold, tf.int32)
    return x, tf.squeeze(binary_labels,axis=-1)

# Map the dataset to prepare the data
dataset = dataset.map(prepare_dataset, num_parallel_calls=tf.data.AUTOTUNE)
# Prefetch the dataset
dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(8, num_parallel_calls=tf.data.AUTOTUNE).cache()
# dataset = dataset.prefetch(tf.data.AUTOTUNE).cache()
# Filter out examples with empty images (assuming image is the first element in the dataset tuple)


# Example usage of the dataset
# for x,y in dataset.take(1):
#     a,b,c,d,e,f,g=x

# for x,y in dataset.take(1):
#     a,b,c=x
# print(c.shape)

"""# Model architecture modules"""

from tensorflow.keras import layers
class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim =embed_dim
        self.patch_size = patch_size
        self.projection =tf.keras.layers.Conv1D(
    self.embed_dim,
    3,
    strides=1,
    padding='same',
    data_format=None,
    dilation_rate=1,
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
        self.projection2 = layers.GRU(self.embed_dim, return_sequences=True, return_state=True)
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos, vid):
        if vid==0:
          projected_patches = self.projection(videos)

        elif vid==1:
          projected_patches,_ = self.projection2(videos)


        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
        })
        return config

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim

        })
        return config

# DATA
BATCH_SIZE = 2
INPUT_SHAPE = (15, 1796)
INPUT_SHAPE2 = (15, 4)
INPUT_SHAPE4 =(15,)
NUM_CLASSES = 2

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 50

# TUBELET EMBEDDING
PATCH_SIZE = (2,8,8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
# PROJECTION_DIM2= 4
NUM_HEADS = 4

import tensorflow as tf
from tensorflow.keras import layers

class DenoisingUNet(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(DenoisingUNet, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.encoder1 = self.conv_block(embed_dim // 4)
        self.encoder2 = self.conv_block(embed_dim // 2)
        self.middle = self.conv_block(embed_dim)
        self.decoder2 = self.deconv_block(embed_dim // 2)
        self.decoder1 = self.deconv_block(embed_dim // 4)
        self.final_conv = layers.Conv1D(embed_dim, kernel_size=1, padding='same')

    def conv_block(self, filters):
        block = tf.keras.Sequential([
            layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu'),
            layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')
        ])
        return block

    def deconv_block(self, filters):
        block = tf.keras.Sequential([
            layers.Conv1DTranspose(filters, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')
        ])
        return block

    def pad_or_crop(self, tensor, target_shape):
        current_shape = tf.shape(tensor)[1]
        diff = target_shape - current_shape
        pad_left = tf.maximum(diff // 2, 0)
        pad_right = tf.maximum(diff - pad_left, 0)
        crop_left = tf.maximum(-diff // 2, 0)
        crop_right = tf.maximum(-diff - crop_left, 0)
        tensor = tensor[:, crop_left:current_shape-crop_right, :]
        tensor = tf.pad(tensor, [[0, 0], [pad_left, pad_right], [0, 0]], mode='CONSTANT')
        return tensor

    def call(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(layers.MaxPooling1D(pool_size=2)(e1))
        m = self.middle(layers.MaxPooling1D(pool_size=2)(e2))
        d2 = self.decoder2(layers.UpSampling1D(size=2)(m))
        d2 = self.pad_or_crop(d2, tf.shape(e2)[1])
        d2 = layers.Concatenate()([d2, e2])
        d1 = self.decoder1(layers.UpSampling1D(size=2)(d2))
        d1 = self.pad_or_crop(d1, tf.shape(e1)[1])
        d1 = layers.Concatenate()([d1, e1])
        return self.final_conv(d1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config
class PDAM(tf.keras.layers.Layer):
    def __init__(self, embed_dim=128, max_iters=5, tolerance=1e-3, noise_std=0.1, **kwargs):
        super(PDAM, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.noise_std = noise_std
        self.query = layers.Dense(units=embed_dim)
        self.key = layers.Dense(units=embed_dim)
        self.value = layers.Dense(units=embed_dim)
        self.denoise_net = DenoisingUNet(embed_dim)

    def add_noise(self, tensor):
        noise = tf.random.normal(shape=tf.shape(tensor), mean=0.0, stddev=self.noise_std, dtype=tf.float32)
        return tensor + noise

    def self_attention(self, Q, K, V):
        d_k = tf.cast(self.embed_dim, dtype=tf.float32)
        score = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)
        score = tf.nn.softmax(score, axis=-1)
        Z = tf.matmul(score, V)
        return Z

    def call(self, x1, x2):
        Q = self.query(x1)
        K = self.key(x1)
        V = self.value(x2)
        Q = self.add_noise(Q)
        K = self.add_noise(K)
        V = self.add_noise(V)
        Z = self.self_attention(Q, K, V)
        prev_Z = tf.zeros_like(Z)

        def loop_cond(i, Z, prev_Z, Q, K, V):
            return tf.logical_and(tf.less(i, self.max_iters), tf.greater(tf.reduce_mean(tf.abs(Z - prev_Z)), self.tolerance))

        def loop_body(i, Z, prev_Z, Q, K, V):
            prev_Z = Z
            Z = self.self_attention(Q, K, V)
            Z = tf.nn.softmax(self.denoise_net(Z), axis=-1)
            Q, K, V = Z, Z, Z
            Q = self.add_noise(Q)
            K = self.add_noise(K)
            V = self.add_noise(V)
            return tf.add(i, 1), Z, prev_Z, Q, K, V

        i = tf.constant(0)
        i, Z, prev_Z, Q, K, V = tf.while_loop(loop_cond, loop_body, [i, Z, prev_Z, Q, K, V], shape_invariants=[i.get_shape(), tf.TensorShape([None, None, self.embed_dim]), tf.TensorShape([None, None, self.embed_dim]), Q.get_shape(), K.get_shape(), V.get_shape()])
        return Z

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "max_iters": self.max_iters,
            "tolerance": self.tolerance,
            "noise_std": self.noise_std
        })
        return config

tubelet_embedder=TubeletEmbedding(embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE)
positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM)
input_shape=INPUT_SHAPE
num_heads=NUM_HEADS
embed_dim=PROJECTION_DIM
layer_norm_eps=LAYER_NORM_EPS
num_classes=NUM_CLASSES

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Hyperparameters (adjust as needed)
attention_dim = 256
timesteps =15

# Pedestrian Attributes Branch
rgb_input = layers.Input(shape=(timesteps, 1280))
ped_attr = rgb_input


# Scene Attributes Branch
seg_input = layers.Input(shape=(timesteps, 1280))
scene_attr = seg_input

traj_input = layers.Input(shape=(timesteps, 4))

Feed_MLP = keras.Sequential(
      [
          layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu, kernel_initializer=keras.initializers.HeNormal(),kernel_regularizer= keras.regularizers.L2(1e-6) ),
          layers.Dropout(0.5),
          layers.Dense(units=embed_dim, activation=tf.nn.gelu, kernel_initializer=keras.initializers.HeNormal(),kernel_regularizer= keras.regularizers.L2(1e-6) ),

      ])


patches_0 = tubelet_embedder(ped_attr, 0)
# Encode patches.
encoded_patches_0 = positional_encoder(patches_0)
x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_0)

#Get the second input
encoded_patches_1 = tubelet_embedder(traj_input, 1)
traj_encoded = layers.GlobalAvgPool1D()(encoded_patches_1)
# Encode patches.
x2 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_1)


attention_output = PDAM()(x1,x2)

x3 = layers.Add()([attention_output, encoded_patches_0])

x4 = layers.LayerNormalization(epsilon=1e-6)(x2)
x4 = Feed_MLP(x3)

# Skip connection
encoded_patches = layers.Add(name='encoded_R')([x4, x3])
representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
representation = layers.GlobalAvgPool1D()(representation)
representation =layers.Dropout(0.5)(representation)


# Create patches.
patches_0 = tubelet_embedder(scene_attr, 0)
# Encode patches.
encoded_patches_0 = positional_encoder(patches_0)
x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_0)

x2 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_1)

attention_output = PDAM()(x1,x2)

x3 = layers.Add()([attention_output, encoded_patches_0])

x4 = layers.LayerNormalization(epsilon=1e-6)(x2)
x4 = Feed_MLP(x3)

# Skip connection
encoded_patches = layers.Add(name='encoded_S')([x4, x3])
representation2 = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
representation2 = layers.GlobalAvgPool1D()(representation2)
representation2 =layers.Dropout(0.5)(representation2)

# Define trainable weights for adaptive addition
weight_representation = tf.Variable(1.0, trainable=True, name='weight_representation')
weight_representation2 = tf.Variable(1.0, trainable=True, name='weight_representation2')
weight_traj_encoded = tf.Variable(1.0, trainable=True, name='weight_traj_encoded')

# Stack weights and apply softmax to get normalized weights
weights = tf.stack([weight_representation, weight_representation2, weight_traj_encoded])
normalized_weights = tf.nn.softmax(weights)

# Compute weighted sum of input tensors
final_rep = (normalized_weights[0] * representation +
             normalized_weights[1] * representation2 +
             normalized_weights[2] * traj_encoded)

# Classify outputs.
output_r = layers.Dense(units=1, activation='sigmoid', name='final', kernel_initializer=keras.initializers.HeNormal())(final_rep)

model = keras.Model(inputs=[rgb_input,seg_input,traj_input], outputs=[output_r])

model.summary()

"""# Model Training- Phase-I"""

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=tf.keras.metrics.BinaryAccuracy())
steps_per_epoch = len(traingen)/8
# steps_per_epoch =60
model.fit(dataset,epochs=1, steps_per_epoch=steps_per_epoch)

"""Model Training- Phase-II"""

# Get the existing inputs from the model
inputs = model.input

# Define the counterfactual RGB input (all zeros)
# rgb_input_counterfactual = layers.Input(shape=(timesteps, 1280), name='rgb_input_counterfactual')
rgb_input_counterfactual = tf.zeros_like(rgb_input)
# Replace the original RGB input with the counterfactual one for the new model
counterfactual_inputs = [rgb_input_counterfactual if inp.name == 'rgb_input' else inp for inp in inputs]

# Get the outputs from the original model
original_output = model.output

# Create a new model with the counterfactual input
counterfactual_model = keras.Model(inputs=counterfactual_inputs, outputs=original_output)

# Compile the model with a custom loss function
counterfactual_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=tf.keras.metrics.BinaryAccuracy())

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

# Define the loss function
def counterfactual_loss(y_true, y_pred_normal, y_pred_counterfactual):
    prediction_loss = binary_crossentropy(y_true, y_pred_normal)
    counterfactual_diff_loss = tf.reduce_mean(tf.square(y_pred_normal - y_pred_counterfactual))
    return prediction_loss + counterfactual_diff_loss

# Training parameters
batch_size = 32
epochs = 10

counterfactual_model.fit(dataset,epochs=1, steps_per_epoch=steps_per_epoch)

"""Model Training- Phase-III"""

# Get the existing inputs from the model
inputs = counterfactual_model.input

# Define the counterfactual RGB input (all zeros)
seg_input_counterfactual = tf.zeros_like(seg_input)
# Replace the original RGB input with the counterfactual one for the new model
counterfactual_inputs = [seg_input_counterfactual if inp.name == 'seg_input' else inp for inp in inputs]
counterfactual_inputs = [rgb_input if inp.name == 'counterfactual_rgb_input' else inp for inp in inputs]
# Get the outputs from the original model
original_output = counterfactual_model.output

# Create a new model with the counterfactual input
counterfactual_model2 = keras.Model(inputs=counterfactual_inputs, outputs=original_output)

# Compile the model with a custom loss function
counterfactual_model2.compile(optimizer='adam', loss='binary_crossentropy',metrics=tf.keras.metrics.BinaryAccuracy())

# Training parameters
batch_size = 32
epochs = 10

counterfactual_model2.fit(dataset,epochs=1, steps_per_epoch=steps_per_epoch)

"""Representation model"""

# Get the existing inputs from the model
inputs = counterfactual_model2.input

counterfactual_inputs = [seg_input if inp.name == 'counterfactual_seg_input' else inp for inp in inputs]
counterfactual_inputs = [rgb_input if inp.name == 'counterfactual_rgb_input' else inp for inp in inputs]

Rep_model = keras.Model(inputs=inputs, outputs=[counterfactual_model2.get_layer('encoded_R').output, counterfactual_model2.get_layer('encoded_S').output])

# Verify the new model's architecture
Rep_model.summary()

"""Saving the model"""

# Save the entire model
Rep_model.save('/content/drive/MyDrive/obj-5/intent_model.tf')

intent_model = load_model('/content/drive/MyDrive/obj-5/intent_model.h5', custom_objects={
    'TubeletEmbedding': TubeletEmbedding,
    'PositionalEncoder': PositionalEncoder,
    'PDAM': PDAM,
    'DenoisingUNet': DenoisingUNet
})

