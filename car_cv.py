import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from skimage.feature import hog
from scipy.ndimage.measurements import label

from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


def get_file_names(car_glob_path, not_car_glob_path):
    # Get file list
    cars = glob.glob(car_glob_path)
    notcars = glob.glob(not_car_glob_path)

    return cars, notcars


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2HSV' or conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == 'RGB2LUV' or conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'RGB2HLS' or conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == 'RGB2YUV' or conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif conv == 'RGB2YCrCb' or conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def bin_spatial(img, size=(32, 32)):
    feature_image = np.copy(img)
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features
        

def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict

    
def visualize(cars, notcars):

    data_info = data_look(cars, notcars)

    print('Your function returned a count of', 
          data_info["n_cars"], ' cars and', 
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', 
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images   
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))
        
    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    car_cs = convert_color(car_image, conv='COLOR_RGB2YCrCb')
    notcar_cs = convert_color(notcar_image, conv='COLOR_RGB2YCrCb')

    car_hog_feats, car_hog = get_hog_features(car_cs[:,:,0], vis=True)
    notcar_hog_feats, notcar_hog = get_hog_features(notcar_cs[:,:,0], vis=True)

    rows = 2
    cols = 2
    # Plot the examples
    fig = plt.figure()
    plt.subplot(rows, cols, 1)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(rows, cols, 2)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    plt.subplot(rows, cols, 3)
    plt.imshow(car_hog)
    plt.title('Car HOG Image')
    plt.subplot(rows, cols, 4)
    plt.imshow(notcar_hog)
    plt.title('Not-car HOG Image')
    plt.show()


def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features



def extract_features(img_files, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, 
                        hist_range=(0,256), spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    t = time.time()
    n_images = len(img_files)
    for index, file in enumerate(img_files):
        if index % 500 == 0:
            print("Processing image: {0} of {1}.  Elapsed time: {2:.3f} secs".format(index, n_images, time.time() - t))
        # Read in each one by one
        image = mpimg.imread(file)
        # Append feature vector to feature list
        features.append(single_img_features(image, color_space=cspace, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat))

    print()

    return features



def single_img_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def train_svc(car_files, notcar_files, n_samples=None, cspace='YCrCb', spatial_dim=32, 
    hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', 
    spatial_feat=True, hist_feat=True, hog_feat=True):

    print()
    print("Begining training")
    print("=================")
    print()

    func_start = time.time()

    if n_samples is not None:
        file_limit = min(len(car_files), len(notcar_files))
        car_files, notcar_files = shuffle(car_files[:file_limit], notcar_files[:file_limit], n_samples=n_samples)

    # Time operation
    t = time.time()

    print("Processing car images....")
    print("=========================")
    car_features = extract_features(car_files, cspace=cspace, spatial_size=(spatial_dim, spatial_dim),
                        hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    print("Processing non car images....")
    print("=============================")
    notcar_features = extract_features(notcar_files, cspace=cspace, spatial_size=(spatial_dim, spatial_dim),
                        hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    # Record end time
    t2 = time.time()
    print(round(t2-t, 3), 'Seconds to extract features')
    print()

    print("Organizing data")
    print('===============')
    t=time.time()
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    t2=time.time()
    print('Data organization took {0:.3f} secs'.format(t2-t))
    print('Using spatial binning of:',spatial_dim,
        'and', hist_bins,'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    print()

    print('Training SVC')
    print('============')
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    test_acc = svc.score(X_test, y_test)
    print('Test Accuracy of SVC = ', round(test_acc, 4))
    print()
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    print()

    func_end = time.time()
    print("Entire training operation took:", round(func_end - func_start, 4), "secs")
    print()

    return svc, test_acc, X_scaler, len(X)



def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, conv='RGB2YCrCb', threshold=1):
    
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0], dtype=np.uint8)
    img_boxes = []
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=conv)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                img_boxes.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))) 
                heatmap[ytop_draw+ystart:ytop_draw+ystart+win_draw, xbox_left:xbox_left+win_draw] += 1

    if threshold > 0:
        heatmap[heatmap <= threshold] = 0
    # # print("Heatmap:", np.min(heatmap), np.max(heatmap))
    # heatmax = np.max(heatmap)
    # print(heatmax)
    # print(np.max(heatmap*255/heatmax))
    # shape = heatmap.shape
    # color_heatmap = np.zeros((shape[0], shape[1], 3))            
    # color_heatmap[:, :, 0] = heatmap #*255/heatmax #.reshape((shape[0], shape[1], 1))
    # print(np.max(color_heatmap))

    return {'draw':draw_img, 'heat':heatmap, 'boxes':img_boxes} #draw_img, heatmap, img_boxes


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def find_cars_img_pipline(car_files, notcar_files, n_samples=None, cspace='YCrCb', spatial_dim=32, 
                            hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, save=False):

    print()
    print("Running image pipeline")
    print("======================")
    print()

    func_start = time.time()

    if n_samples is not None:
        file_limit = min(len(car_files), len(notcar_files))
        car_files, notcar_files = shuffle(car_files[:file_limit], notcar_files[:file_limit], n_samples=n_samples)

    # Just for fun choose random car / not-car indices and plot example images   
    car_ind = np.random.randint(0, len(car_files))
    notcar_ind = np.random.randint(0, len(notcar_files))
        
    # Read in car / not-car images
    car_image = mpimg.imread(car_files[car_ind])
    notcar_image = mpimg.imread(notcar_files[notcar_ind])

    car_cs = convert_color(car_image, conv=cspace)
    notcar_cs = convert_color(notcar_image, conv=cspace)

    img_dicts = {'car':{'img':car_cs}, 'notcar':{'img':notcar_cs}}
    cols = 4
    for _, img_dict in img_dicts.items():
        for channel in range(3):
            # Get data
            img_channel = img_dict['img'][:,:,channel]
            _, hog = get_hog_features(img_channel, orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, vis=True)
            spatial = cv2.resize(img_channel, (spatial_dim, spatial_dim))
            hist = np.histogram(img_channel, bins=hist_bins)
            # Load values into dict
            img_dict['img_ch' + str(channel+1)] = {'data':img_channel, 'index':cols*channel + 1}
            img_dict['hog_ch' + str(channel+1)] = {'data':hog, 'index':cols*channel + 2}
            img_dict['spatial_ch' + str(channel+1)] = {'data':spatial, 'index':cols*channel + 3}
            img_dict['hist_ch' + str(channel+1)] = {'data':hist, 'index':cols*channel + 4}

    # Plot the examples
    rows = 3
    cols = 4
    for key, img_dict in img_dicts.items():
        fig = plt.figure()
        index = 0
        for inner_key, feat_data in img_dict.items():
            if inner_key == 'img':
                continue
            plt.subplot(rows, cols, feat_data['index'])
            if 'hist' in inner_key:
                bin_edges = feat_data['data'][1]
                bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
                plt.bar(bin_centers, feat_data['data'][0])
                # plt.xlim(0, hist_bins)
            else:
                plt.imshow(feat_data['data'], cmap='gray')
            plt.title(inner_key)
            index+=1
        if save:
            plt.savefig(key+'plot.pdf', bbox_inches='tight')
        else:
            plt.show()

    # rows = 2
    # cols = 2
    # # Plot the examples
    # fig = plt.figure()
    # plt.subplot(rows, cols, 1)
    # plt.imshow(car_image)
    # plt.title('Example Car Image')
    # plt.subplot(rows, cols, 2)
    # plt.imshow(notcar_image)
    # plt.title('Example Not-car Image')
    # plt.subplot(rows, cols, 3)
    # plt.imshow(car_hog)
    # plt.title('Car HOG Image')
    # plt.subplot(rows, cols, 4)
    # plt.imshow(notcar_hog)
    # plt.title('Not-car HOG Image')
    # plt.show()

    # # Time operation
    # t = time.time()

    # print("Processing car images....")
    # print("=========================")
    # car_features = extract_features(car_files, cspace=cspace, spatial_size=(spatial_dim, spatial_dim),
    #                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
    #                     hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    # print("Processing non car images....")
    # print("=============================")
    # notcar_features = extract_features(notcar_files, cspace=cspace, spatial_size=(spatial_dim, spatial_dim),
    #                     hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
    #                     hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
    # # Record end time
    # t2 = time.time()
    # print(round(t2-t, 3), 'Seconds to extract features')
    # print()
