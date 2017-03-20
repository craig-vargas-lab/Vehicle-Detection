import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label

from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

import car_cv

TOP_DIR = '/Users/cvar/selfdrivingcar/term_one/projectfive'
MAIN_DIR = TOP_DIR + '/CarND-Vehicle-Detection'
TRAIN_DIR = TOP_DIR + '/training_data'
EXAMPLES_DIR = MAIN_DIR + '/examples'
TEST_IMGS_DIR = MAIN_DIR + '/test_images'

with open('svc.p', 'rb') as svc_file:
	svc = pickle.load(svc_file)
with open('scaler.p', 'rb') as scaler_file:
	X_scaler = pickle.load(scaler_file)

VISUALIZE = True
USING_MEMORY = False
heatmaps = []



def train_svc():
	car_glob_path = TRAIN_DIR + '/vehicles/*/*.png'
	not_cars_glob_path = TRAIN_DIR + '/non-vehicles/*/*.png'
	car_files, notcar_files = car_cv.get_file_names(car_glob_path, not_cars_glob_path)

	num_cars = len(car_files)
	num_notcars = len(notcar_files)

	# # Explore data
	# print('Number of car data points, num_cars)
	# print('Number of non car data points, num_notcars)
	# car_cv.visualize(car_files, notcar_files)

	# Train SVC
	t = time.time()
	n_samples = None
	cspace = 'YCrCb' # 'YUV' # 'HLS' # 'LUV' # 'HSV' # 'YCrCb'
	spatial_dim = 16
	hist_bins = 16
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	hog_channel = 'ALL'
	spatial_feat = True
	hist_feat = True
	hog_feat = True

	n_ops = 1
	acc_list = []
	for op in range(n_ops):
		print()
		print(":::::: Training iteration: {0} of {1} ::::::".format(op + 1, n_ops))
		print()

		svc, acc, scaler, data_size = car_cv.train_svc(car_files, notcar_files, n_samples=n_samples, cspace=cspace, spatial_dim=spatial_dim, 
								hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
								hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

		acc_list.append(acc)

	ovarall_acc = np.mean(acc_list)
	print("STATS")
	print("=====")
	print("=====")
	print()
	print("Overall accuracy:", round(ovarall_acc, 4))
	print("Data set size:", data_size)
	print()
	print("With parameters")
	print("===============")
	print('n_samples =', n_samples)
	print('cspace =', cspace)
	print('spatial_dim =', spatial_dim)
	print('hist_bins =', hist_bins)
	print('orient =', orient)
	print('pix_per_cell =', pix_per_cell)
	print('cell_per_block =', cell_per_block)
	print('hog_channel =', hog_channel)
	print('spatial_feat =', spatial_feat)
	print('hist_feat =', hist_feat)
	print('hog_feat =', hog_feat)
	print()
	print('Total time', round((time.time() - t)/60, 3), 'mins')
	print()

	print("Saving SVC")
	print("=====================")
	t=time.time()
	with open('svc.p', 'wb') as svc_file:
		pickle.dump(svc, svc_file)
	t2=time.time()
	print("SVC saved in {0:.3f} secs".format(t2-t))
	print()

	print("Saving Scaler")
	print("=====================")
	t=time.time()
	with open('scaler.p', 'wb') as scaler_file:
		pickle.dump(scaler, scaler_file)
	t2=time.time()
	print("Scaler saved in {0:.3f} secs".format(t2-t))
	print()


def find_cars_test():

	test_files = glob.glob('/Users/cvar/selfdrivingcar/term_one/projectfive/CarND-Vehicle-Detection/test_images/test6.jpg')
	for file in test_files:
		img = mpimg.imread(file)
		found_img = find_cars(img)


def find_cars(img):
	max_memory=30

	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	spatial_size = (16, 16)
	hist_bins = 16
	conv = 'RGB2YCrCb'
	threshold = 0

	# XSmall
	scale_xsm = 0.75
	step_xsm = int(64*scale_xsm*0.25)
	ystart_xsm = 400
	ystop_xsm = ystart_xsm + (step_xsm*6)
	# Small
	scale_sm = 1
	step_sm = int(64*scale_sm*0.25)
	ystart_sm = 400
	ystop_sm = ystart_sm + (step_sm*6)
	# Medium
	scale_md = 1.5
	step_md = int(64*scale_md*0.25)
	ystart_md = 400
	ystop_md = ystart_md + (step_md*6) 
	# Large
	scale_lg = 2
	step_lg = int(64*scale_lg*0.25)
	ystart_lg = 400
	ystop_lg = ystart_lg + (step_lg*8) 

	start_find = time.time()

	sizes_found = []

	# Find extra small cars
	# =====================
	t1 = time.time()
	sizes_found.append(car_cv.find_cars(img, ystart_xsm, ystop_xsm, scale_xsm, svc, X_scaler, orient, pix_per_cell,
								cell_per_block, spatial_size, hist_bins, conv=conv, threshold=threshold))
	t2 = time.time()
	if(VISUALIZE):
		print("Found extra small cars in {0:.3f} secs".format(t2-t1))

	# # Find small cars
	# t1 = time.time()
	# sizes_found.append(car_cv.find_cars(img, ystart_sm, ystop_sm, scale_sm, svc, X_scaler, orient, pix_per_cell,
	# 							cell_per_block, spatial_size, hist_bins, conv=conv, threshold=threshold))
	# t2 = time.time()
	# print("Found small cars in {0:.3f} secs".format(t2-t1))

	# Find medium cars
	# ================
	t1 = time.time()
	sizes_found.append(car_cv.find_cars(img, ystart_md, ystop_md, scale_md, svc, X_scaler, orient, pix_per_cell,
								cell_per_block, spatial_size, hist_bins, conv=conv, threshold=threshold))
	t2 = time.time()
	if(VISUALIZE):
		print("Found medium cars in {0:.3f} secs".format(t2-t1))

	# Find large cars
	# ================
	t1 = time.time()
	sizes_found.append(car_cv.find_cars(img, ystart_lg, ystop_lg, scale_lg, svc, X_scaler, orient, pix_per_cell,
								cell_per_block, spatial_size, hist_bins, conv=conv, threshold=threshold))
	t2 = time.time()
	if VISUALIZE:
		print("Found large cars in {0:.3f} secs".format(t2-t1))	

	if VISUALIZE:
		print()
		print(":::::: Total find time = {0:.3f} secs ::::::".format(time.time()-start_find))

	rows = 4
	cols = 2
	charts_per_frame = 2
	titles = ['Boxes 48', 'Heat 48', 'Boxes 96', 'Heat 96', 'Boxes 128', 'Heat 128', 'Final Boxes', 'Final Heat']
	# Plot the examples
	fig = plt.figure()
	collective_heat = np.zeros_like(img[:,:,0])
	if VISUALIZE:
		for index, size in enumerate(sizes_found):
			plt.subplot(rows, cols, index*charts_per_frame + 1)
			plt.imshow(size['draw'])
			plt.title(titles[index*charts_per_frame])
			plt.subplot(rows, cols, index*charts_per_frame + 2)
			plt.imshow(size['heat'], cmap='gist_heat')
			plt.title(titles[index*charts_per_frame + 1])
			collective_heat += size['heat']
	else:
		for size in sizes_found:
			collective_heat += size['heat']

	if USING_MEMORY:
		final_heat = np.zeros_like(img[:,:,0])
		global heatmaps
		if len(heatmaps) == max_memory:
			heatmaps.pop(0)
		heatmaps.append(collective_heat)

		for heatmap in heatmaps:
			final_heat += heatmap

		final_threshold = 1.0*len(heatmaps)
		final_heat[final_heat <= final_threshold] = 0
	else:
		final_threshold = 2 
		collective_heat[collective_heat <= final_threshold] = 0
		final_heat = collective_heat

	labels = label(final_heat)
	final_img = car_cv.draw_labeled_bboxes(img, labels)

	if VISUALIZE:			
		plt.subplot(rows, cols, (index+1)*charts_per_frame + 1)
		plt.imshow(final_img)
		plt.title('Final Boxes')
		plt.subplot(rows, cols, (index+1)*charts_per_frame + 2)
		plt.imshow(final_heat, cmap='gist_heat')
		plt.title('Final Heat')
		plt.savefig('carboxes.pdf', bbox_inches='tight')
		plt.show()

	return final_img


def process_test_video():
	global VISUALIZE
	VISUALIZE = False
	global USING_MEMORY
	USING_MEMORY = True
	out_vid = 'test.mp4'
	clip = VideoFileClip('test_video_cv_end.mp4')
	test_clip = clip.fl_image(find_cars)
	test_clip.write_videofile(out_vid, audio=False)


def process_project_video():
	global VISUALIZE
	VISUALIZE = False
	global USING_MEMORY
	USING_MEMORY = True
	out_vid = 'track_cars.mp4'
	clip = VideoFileClip('project_video.mp4')
	test_clip = clip.fl_image(find_cars)
	test_clip.write_videofile(out_vid, audio=False)


def image_pipeline():
	# Train params
	t = time.time()
	n_samples = None
	cspace = 'YCrCb' # 'YUV' # 'HLS' # 'LUV' # 'HSV' # 'YCrCb'
	spatial_dim = 16
	hist_bins = 16
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	save = True
	# Get file names
	car_glob_path = TRAIN_DIR + '/vehicles/*/*.png'
	not_cars_glob_path = TRAIN_DIR + '/non-vehicles/*/*.png'
	car_files, notcar_files = car_cv.get_file_names(car_glob_path, not_cars_glob_path)
	# Start pipeline
	car_cv.find_cars_img_pipline(car_files, notcar_files, n_samples=n_samples, cspace=cspace, spatial_dim=spatial_dim, 
                            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, save=save)


def main():
	# experiment()
	find_cars_test()
	# process_test_video()
	# process_project_video()
	# image_pipeline()


if __name__ == '__main__':
	main()
