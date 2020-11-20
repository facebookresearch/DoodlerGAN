# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import json
import copy
import numpy as np
import cairocffi as cairo


def vector_to_raster(vector_images, part_label=False, nodetail=False, side=64, line_diameter=16, padding=16, bg_color=(1,1,1), fg_color=(0,0,0)):
	"""
	padding and line_diameter are relative to the original 512x512 image.
	"""
	original_side = 512.
	surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
	ctx = cairo.Context(surface)
	ctx.set_antialias(cairo.ANTIALIAS_BEST)
	ctx.set_line_cap(cairo.LINE_CAP_ROUND)
	ctx.set_line_join(cairo.LINE_JOIN_ROUND)
	ctx.set_line_width(line_diameter)
	# scale to match the new size
	# add padding at the edges for the line_diameter
	# and add additional padding to account for antialiasing
	total_padding = padding * 2. + line_diameter
	new_scale = float(side) / float(original_side + total_padding)
	ctx.scale(new_scale, new_scale)
	ctx.translate(total_padding / 2., total_padding / 2.)
	raster_images = []
	for i, vector_data in enumerate(vector_images):
		# clear background
		ctx.set_source_rgb(*bg_color)
		ctx.paint()
		vector_image = []
		x_max = y_max = 0
		for step in vector_data['output']['all_strokes']:
			vector_image.append([]) # for each step
			for stroke in step:
				if len(stroke) == 0: # skip the empty stroke like the one in step 3
					vector_image[-1].append([])
					continue
				vector_image[-1].append(np.array([stroke[0][:2]]+[point[2:4] for point in stroke])) # add each stroke N x 2
				x_max_stroke, y_max_stroke = np.max(vector_image[-1][-1], 0)
				x_max = x_max_stroke if x_max_stroke>x_max else x_max
				y_max = y_max_stroke if y_max_stroke>y_max else y_max
		offset = ((original_side, original_side) - np.array([x_max, y_max])) / 2.
		offset = offset.reshape(1,2)
		for j in range(len(vector_image)):
			for k in range(len(vector_image[j])):
				vector_image[j][k] = vector_image[j][k]+offset  if len(vector_image[j][k]) > 0 else vector_image[j][k]
		# draw strokes, this is the most cpu-intensive part
		ctx.set_source_rgb(*fg_color)
		for j, step in enumerate(vector_image): 
			if part_label:
				if j == 0:
					ctx.set_source_rgb(*COLORS['initial'])
				elif j == 1:
					ctx.set_source_rgb(*COLORS['eye'])
				elif j == len(vector_image)-1 and vector_data['output']['partsUsed'][j-2] == 'none': # last one and details
					ctx.set_source_rgb(*COLORS['details'])
				elif vector_data['output']['partsUsed'][j-2] != 'none':
					ctx.set_source_rgb(*COLORS[vector_data['output']['partsUsed'][j-2]])
			if nodetail and j == len(vector_image)-1 and vector_data['output']['partsUsed'][j-2] == 'none':
				continue
			for stroke in step:
				if len(stroke) == 0:
					continue
				ctx.move_to(stroke[0][0], stroke[0][1])
				for x, y in stroke:
					ctx.line_to(x, y)
				ctx.stroke()
		surface_data = surface.get_data()
		if part_label:
			raster_image = np.copy(np.asarray(surface_data)).reshape(side, side, 4)[:, :, :3]
		else:
			raster_image = np.copy(np.asarray(surface_data))[::4].reshape(side, side)
		raster_images.append(raster_image)
	return raster_images


def vector_image_to_vector_part(vector_images, target_part, side=64, line_diameter=16, padding=16, data_name='bird'):
	"""
	save processed vector image for target_parts: input partial images, input parts and target images with target parts
	"""
	original_side = 512.
	# scale to match the new size
	# add padding at the edges for the line_diameter
	# and add additional padding to account for antialiasing
	total_padding = padding * 2. + line_diameter
	new_scale = float(side) / float(original_side + total_padding)
	processed_vector_input_parts = []
	processed_vector_parts = []
	# each item in processed_vector_images is a list that corresponds to all target parts that appear in that sketch
	for i, vector_data in enumerate(vector_images):
		# check if target part is drawn
		processed_vector_input_parts.append([])
		processed_vector_parts.append([])
		# store the strokes for each part
		if data_name == 'bird':
			strokes_input_parts = {'initial':[], 'eye':[], 'beak':[], 'body':[], 'head':[], 'legs':[], 'mouth':[], 'tail':[], 'wings':[]}
		elif data_name == 'creature':
			strokes_input_parts = {'initial':[], 'eye':[], 'arms':[], 'beak':[], 'mouth':[], 'body':[], 'ears':[], 'feet':[],  'fin':[], 'hair':[], 'hands':[], 
									'head':[], 'horns':[], 'legs':[],  'nose':[], 'paws':[], 'tail':[], 'wings':[]}
		if target_part not in vector_data['output']['partsUsed'] and target_part != 'eye':
			continue
		vector_image = []
		x_max = y_max = 0
		for step in vector_data['output']['all_strokes']:
			vector_image.append([]) # for each step
			for stroke in step:
				if len(stroke) == 0: # skip the empty stroke like the one in step 3
					vector_image[-1].append([])
					continue
				vector_image[-1].append(np.array([stroke[0][:2]]+[point[2:4] for point in stroke])) # add each stroke N x 2
				x_max_stroke, y_max_stroke = np.max(vector_image[-1][-1], 0)
				x_max = x_max_stroke if x_max_stroke>x_max else x_max
				y_max = y_max_stroke if y_max_stroke>y_max else y_max
		offset = ((original_side, original_side) - np.array([x_max, y_max])) / 2.
		offset = offset.reshape(1,2)
		for j in range(len(vector_image)):
			for k in range(len(vector_image[j])):
				vector_image[j][k] = vector_image[j][k]+offset  if len(vector_image[j][k]) > 0 else vector_image[j][k]
		# save strokes
		for j, step in enumerate(vector_image): 
			if j>2 and vector_data['output']['partsUsed'][j-2] == target_part or j==1 and target_part=='eye': # find one part
				processed_vector_input_parts[-1].append(copy.deepcopy(strokes_input_parts))
			if j == 0:
				strokes_input_parts['initial'] += step
			elif j == 1:
				strokes_input_parts['eye'] += step
			elif j != len(vector_image)-1 and vector_data['output']['partsUsed'][j-2] != 'none': # last one and details
				strokes_input_parts[vector_data['output']['partsUsed'][j-2]] += step
			else:
				continue
			if j>2 and vector_data['output']['partsUsed'][j-2] == target_part or j==1 and target_part=='eye':
				# record the input + part
				processed_vector_parts[-1].append(step)
		# record all the parts
		processed_vector_input_parts[-1].append(copy.deepcopy(strokes_input_parts))
	return processed_vector_input_parts, processed_vector_parts



########################################################################################################################
########################################################################################################################
# basic setups, load data
data_name = 'bird' # or 'creature'
side=64 # size of the rendered image

## data format: ['assignment_id', 'hit_id', 'worker_id', 'output', 'submit_time']
## 'output' --> ['all_strokes', 'prompts', 'comment', 'description', 'partsUsed']
if data_name == 'bird':
	COLORS = {'initial':np.array([45, 169, 145])/255., 'eye':np.array([243, 156, 18])/255., 'none':np.array([149, 165, 166])/255., 
			'beak':np.array([211, 84, 0])/255., 'body':np.array([41, 128, 185])/255., 'details':np.array([171, 190, 191])/255.,
			'head':np.array([192, 57, 43])/255., 'legs':np.array([142, 68, 173])/255., 'mouth':np.array([39, 174, 96])/255., 
			'tail':np.array([69, 85, 101])/255., 'wings':np.array([127, 140, 141])/255.}
	part_to_id = {'initial': 0, 'eye': 1, 'head': 4, 'body': 3, 'beak': 2, 'legs': 5, 'wings': 8, 'mouth': 6, 'tail': 7}
	target_parts = ['eye', 'beak', 'body', 'head', 'legs', 'mouth', 'tail', 'wings', 'none']
	data = [json.loads(line) for fn in ['doodle_short_birds_%d'%i for i in range(1, 11)]+['doodle_real_individual_pilot7_birds_short_initial_strokes'] for line in open('raw_data/%s.txt'%fn)]
	wid_rej = [line.rstrip() for line in open('raw_data/reject_bird_short_workids_all.txt')]
elif data_name == 'creature':
	COLORS = {'initial':np.array([45, 169, 145])/255., 'eye':np.array([243, 156, 18])/255., 'none':np.array([149, 165, 166])/255., 
			'arms':np.array([211, 84, 0])/255., 'beak':np.array([41, 128, 185])/255., 'mouth':np.array([54, 153, 219])/255.,
			'body':np.array([192, 57, 43])/255., 'ears':np.array([142, 68, 173])/255., 'feet':np.array([39, 174, 96])/255., 
			'fin':np.array([69, 85, 101])/255., 'hair':np.array([127, 140, 141])/255., 'hands':np.array([45, 63, 81])/255.,
			'head':np.array([241, 197, 17])/255., 'horns':np.array([51, 205, 117])/255., 'legs':np.array([232, 135, 50])/255., 
			'nose':np.array([233, 90, 75])/255., 'paws':np.array([160, 98, 186])/255., 'tail':np.array([58, 78, 99])/255., 
			'wings':np.array([198, 203, 207])/255., 'details':np.array([171, 190, 191])/255.}
	part_to_id = {'initial': 0, 'eye': 1, 'arms': 2, 'beak': 3, 'mouth': 4, 'body': 5, 'ears': 6, 'feet': 7, 'fin': 8, 
                            'hair': 9, 'hands': 10, 'head': 11, 'horns': 12, 'legs': 13, 'nose': 14, 'paws': 15, 'tail': 16, 'wings':17}
	target_parts = ['arms', 'beak', 'mouth', 'body', 'eye', 'ears', 'feet',  'fin', 'hair', 'hands', 
		'head', 'horns', 'legs',  'nose', 'paws', 'tail',  'wings', 'none']
	data = [json.loads(line) for j in range(1, 12) for line in open('raw_data/doodle_generic_%d.txt'%j)]
	wid_rej = [line.rstrip() for line in open('raw_data/reject_generic_workids_all.txt')]


########################################################################################################################
# visualize all the sketches by rendering raster images
raster_images_gs = vector_to_raster(data, part_label=False, nodetail=True, side=side, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1))
raster_images_rgb = vector_to_raster(data, part_label=True, nodetail=True, side=side, line_diameter=16, padding=16, bg_color=(1,1,1), fg_color=(0,0,0))

for i, raster_image, raster_image_rgb in enumerate(zip(raster_images_gs, raster_images_rgb)):
	wid = data[i]['worker_id']
	aid = data[i]['assignment_id']
	if wid in wid_rej:
		continue
	outpath = os.path.join('data/%s_short_full_nodetail_%d'%(data_name, side))
	outpath_rgb = os.path.join('data/%s_short_full_nodetail_rgb_%d'%(data_name, side))
	if not os.path.exists(outpath):
		os.mkdir(outpath)
		os.mkdir(outpath_rgb)
	cv2.imwrite(os.path.join(outpath, "%s.png"%(aid)), raster_image)
	cv2.imwrite(os.path.join(outpath_rgb, "%s.png"%(aid)), raster_image_rgb)


########################################################################################################################
## process vectors images for doodlerGAN
for target_part in target_parts:
	print('rendering %s...'%target_part)
	vector_input_parts, vector_parts = vector_image_to_vector_part(data, target_part=target_part, side=side, line_diameter=16, padding=16, data_name=data_name)
	outpath_train = 'data/%s_short_%s_json_%d_train'%(data_name, target_part, side)
	outpath_test = 'data/%s_short_%s_json_%d_test'%(data_name, target_part, side)
	if not os.path.exists(outpath_test):
		os.mkdir(outpath_test)
		os.mkdir(outpath_train)
	for i in range(len(data)-500):
		wid = data[i]['worker_id']
		aid = data[i]['assignment_id']
		if wid in wid_rej:
			continue
		if len(vector_input_parts[i]) == 0:
			continue
		for j in range(len(vector_input_parts[i])-1):
			if data_name == 'bird':
				json_data = {'input_parts':{'initial': [], 'eye': [], 'head': [], 'body': [], 'beak': [], 'legs': [], 'wings': [], 'mouth': [], 'tail': []}, 'target_part':[]}
			elif data_name == 'creature':
				json_data = {'input_parts':{'initial':[], 'eye':[], 'arms':[], 'beak':[], 'mouth':[], 'body':[], 'ears':[], 'feet':[],  'fin':[], 'hair':[], 'hands':[], 
					'head':[], 'horns':[], 'legs':[],  'nose':[], 'paws':[], 'tail':[], 'wings':[]}, 'target_part':[]}
			if target_part != 'none':
				json_data['target_part'] = [item.tolist() for item in vector_parts[i][j] if len(item) > 0]
			for key in vector_input_parts[i][j].keys():
				json_data['input_parts'][key] = [item.tolist() for item in vector_input_parts[i][j][key] if len(item) > 0]
			with open(outpath_train+"/%s_%d.json"%(data[i]['assignment_id'], j), 'w') as fw:
				json.dump(json_data, fw)
	for i in range(len(data)-500, len(data)):
		wid = data[i]['worker_id']
		aid = data[i]['assignment_id']
		if wid in wid_rej:
			continue
		if len(vector_input_parts[i]) == 0:
			continue
		for j in range(len(vector_input_parts[i])-1):
			if data_name == 'bird':
				json_data = {'input_parts':{'initial': [], 'eye': [], 'head': [], 'body': [], 'beak': [], 'legs': [], 'wings': [], 'mouth': [], 'tail': []}, 'target_part':[]}
			elif data_name == 'creature':
				json_data = {'input_parts':{'initial':[], 'eye':[], 'arms':[], 'beak':[], 'mouth':[], 'body':[], 'ears':[], 'feet':[],  'fin':[], 'hair':[], 'hands':[], 
					'head':[], 'horns':[], 'legs':[],  'nose':[], 'paws':[], 'tail':[], 'wings':[]}, 'target_part':[]}
			if target_part != 'none':
				json_data['target_part'] = [item.tolist() for item in vector_parts[i][j] if len(item) > 0]
			for key in vector_input_parts[i][j].keys():
				json_data['input_parts'][key] = [item.tolist() for item in vector_input_parts[i][j][key] if len(item) > 0]
			with open(outpath_test+"/%s_%d.json"%(data[i]['assignment_id'], j), 'w') as fw:
				json.dump(json_data, fw)