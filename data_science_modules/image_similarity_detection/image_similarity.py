#################################################
# Imports and function definitions
#################################################
# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For saving 'feature vectors' into a txt file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os
import os.path

# from tkinter import Tk for Python 3.x
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# json for storing data in json file
import json

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial

# visualisation libraries
import cv2
from matplotlib import pyplot as plt

#################################################
# This function:
# Loads the JPEG image at the given path
# Decodes the JPEG image to a uint8 W X H X 3 tensor
# Resizes the image to 224 x 224 x 3 tensor
# Returns the pre processed image as 224 x 224 x 3 tensor
#################################################
def load_img(path):

  # Reads the image file and returns data type of string
  img = tf.io.read_file(path)

  # Decodes the image to W x H x 3 shape tensor with type of uint8
  img = tf.io.decode_jpeg(img, channels=3)

  # Resize the image to 224 x 244 x 3 shape tensor
  img = tf.image.resize_with_pad(img, 224, 224)

  # Converts the data type of uint8 to float32 by adding a new axis
  # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
  # This is required for the mobilenet model we are using
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

  return img

#################################################
# This function:
# Loads the mobilenet model in TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_image_feature_vectors():

  i = 0

  start_time = time.time()

  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" %time.ctime())
  print("---------------------------------")

  # Definition of module with using tfhub.dev handle
  module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4" 
  
  # Load the module
  module = hub.load(module_handle)

  print("---------------------------------")
  print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors -  Started at %s" %time.ctime())

  # function to select input
  def select_input():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(initialdir = "/home/anm/image_similarity_detection/fashion_web_scraping/images_scraped/test/",title = "Select file", filetypes = (("jpeg files","*.jpg"),("all files","*.*"))) # show an "Open" dialog box and return the path to file
    return filename

  filename = select_input()

  print("-----------------------------------------------------------------------------------------")
  #print("Image count                     :%s" %i)
  print("Image in process is             :%s" %filename)

  # Loads and pre-process the image
  img = load_img(filename)

  # Calculate the image feature vector of the img
  features = module(img)   

  # Remove single-dimensional entries from the 'features' array
  feature_set = np.squeeze(features)  

  # Saves the image feature vectors into a file for later use

  outfile_name = "temp" + ".npz"
  #outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
  out_path = os.path.join('/home/anm/image_similarity_detection/fashion_web_scraping/images_scraped/feature-vectors', outfile_name)

  # Saves the 'feature_set' to a text file
  np.savetxt(out_path, feature_set, delimiter=',')

  print("Image feature vector saved to   :%s" %out_path)
  
  print("---------------------------------")
  print ("Step.2 of 2 - Generating Feature Vectors - Completed at %s" %time.ctime())
  print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
  #print("--- %s images processed ---------" %i)
  return filename # return query image for visualisation

filename = get_image_feature_vectors() # return query image for visualisation

#################################################
# This function reads from 'image_data.json' file
# Looks for a specific 'filename' value
# Returns the product id when product image names are matched 
# So it is used to find product id based on the product image name
#################################################
def match_id(filename):
  with open('/home/anm/image_similarity_detection/fashion_web_scraping/jsonFiles/image_data.json') as json_file:
    
    for file in json_file:
        seen = json.loads(file)

        for line in seen:
          
          if filename==line['imageName']:
            print(line)
            return line['productId']
            break

#################################################
# This function; 
# Reads all image feature vectores stored in /feature-vectors/*.npz
# Adds them all in Annoy Index
# Builds ANNOY index
# Calculates the nearest neighbors and image similarity metrics
# Stores image similarity scores with productID in a json file
#################################################
def cluster():

  start_time = time.time()
  
  print("---------------------------------")
  print ("Step.1 - ANNOY index generation - Started at %s" %time.ctime())
  print("---------------------------------")

  # Defining data structures as empty dict
  file_index_to_file_name = {}
  file_index_to_file_vector = {}
  file_index_to_product_id = {}

  # Configuring annoy parameters
  dims = 1792
  n_nearest_neighbors = 20
  trees = 10000

  # Reads all file names which stores feature vectors 
  allfiles = glob.glob('/home/anm/image_similarity_detection/fashion_web_scraping/images_scraped/feature-vectors/*.npz')

  t = AnnoyIndex(dims, metric='angular')

  for file_index, i in enumerate(allfiles):
    
    # Reads feature vectors and assigns them into the file_vector 
    file_vector = np.loadtxt(i)

    # Assigns file_name, feature_vectors and corresponding product_id
    file_name = os.path.basename(i).split('.')[0]
    file_index_to_file_name[file_index] = file_name
    file_index_to_file_vector[file_index] = file_vector
    file_index_to_product_id[file_index] = match_id(file_name)

    # Adds image feature vectors into annoy index   
    t.add_item(file_index, file_vector)

    print("---------------------------------")
    print("Annoy index     : %s" %file_index)
    print("Image file name : %s" %file_name)
    print("Product id      : %s" %file_index_to_product_id[file_index])
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))


  # Builds annoy index
  t.build(trees)

  print ("Step.1 - ANNOY index generation - Finished")
  print ("Step.2 - Similarity score calculation - Started ") 
  
  named_nearest_neighbors = []
  image_names = []

  # Loops through all indexed items
  for i in file_index_to_file_name.keys():

    # Only includes similarity score calculation for 'temp'
    if(file_index_to_file_name[i] == 'temp'):

      # Assigns master file_name, image feature vectors and product id values
      master_file_name = file_index_to_file_name[i]
      master_vector = file_index_to_file_vector[i]
      master_product_id = file_index_to_product_id[i]

      # Calculates the nearest neighbors of the master item
      nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

      # Loops through the nearest neighbors of the master item
      for j in nearest_neighbors:

        print(j)

        # Assigns file_name, image feature vectors and product id values of the similar item
        neighbor_file_name = file_index_to_file_name[j]
        image_names.append(file_index_to_file_name[j])
        neighbor_file_vector = file_index_to_file_vector[j]
        neighbor_product_id = file_index_to_product_id[j]

        # Calculates the similarity score of the similar item
        similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0

        # Appends master product id with the similarity score 
        # and the product id of the similar items
        named_nearest_neighbors.append({
          'similarity': rounded_similarity,
          'master_pi': master_product_id,
          'similar_pi': neighbor_product_id})

      print("---------------------------------") 
      print("Similarity index       : %s" %i)
      print("Master Image file name : %s" %file_index_to_file_name[i]) 
      print("Nearest Neighbors.     : %s" %nearest_neighbors) 
      print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

    
      print ("Step.2 - Similarity score calculation - Finished ") 

      # Writes the 'named_nearest_neighbors' to a json file
      with open('nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)

      print ("Step.3 - Data stored in 'nearest_neighbors.json' file ") 
      print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time)/60))

      return image_names

#################################################
# This function:
# Stores the image names from cluster()
# Reads the JPEG images and stores them in a list
# Plots the query image with prediction images
#################################################
def plot_pred_neighbors():
  image_names = cluster()

  pred_neighbors_list = []
  for name in image_names:
    pred_neighbor = name
    pred_neighbors_list.append(pred_neighbor)

  # create figure
  fig = plt.figure(figsize=(10, 10))
  
  # setting values to rows and column variables
  rows = 4
  columns = 5
  
  # reading images

  # Changing the CWD
  os.chdir('/home/anm/image_similarity_detection/fashion_web_scraping/images_scraped/test/')
  query_image = cv2.imread('{}'.format(filename))

  # Changing the CWD
  os.chdir('/home/anm/image_similarity_detection/fashion_web_scraping/images_scraped/full/')
  pred_image1 = cv2.imread('{}.jpg'.format(pred_neighbors_list[1]))
  pred_image2 = cv2.imread('{}.jpg'.format(pred_neighbors_list[2]))
  pred_image3 = cv2.imread('{}.jpg'.format(pred_neighbors_list[3]))
  pred_image4 = cv2.imread('{}.jpg'.format(pred_neighbors_list[4]))
  pred_image5 = cv2.imread('{}.jpg'.format(pred_neighbors_list[5]))
  pred_image6 = cv2.imread('{}.jpg'.format(pred_neighbors_list[6]))
  pred_image7 = cv2.imread('{}.jpg'.format(pred_neighbors_list[7]))
  pred_image8 = cv2.imread('{}.jpg'.format(pred_neighbors_list[8]))
  pred_image9 = cv2.imread('{}.jpg'.format(pred_neighbors_list[9]))
  pred_image10 = cv2.imread('{}.jpg'.format(pred_neighbors_list[10]))
  pred_image11 = cv2.imread('{}.jpg'.format(pred_neighbors_list[11]))
  pred_image12 = cv2.imread('{}.jpg'.format(pred_neighbors_list[12]))
  pred_image13 = cv2.imread('{}.jpg'.format(pred_neighbors_list[13]))
  pred_image14 = cv2.imread('{}.jpg'.format(pred_neighbors_list[14]))
  pred_image15 = cv2.imread('{}.jpg'.format(pred_neighbors_list[15]))
  pred_image16 = cv2.imread('{}.jpg'.format(pred_neighbors_list[16]))
  pred_image17 = cv2.imread('{}.jpg'.format(pred_neighbors_list[17]))
  pred_image18 = cv2.imread('{}.jpg'.format(pred_neighbors_list[18]))
  pred_image19 = cv2.imread('{}.jpg'.format(pred_neighbors_list[19]))
  pred_images = [pred_image1, pred_image2, pred_image3, pred_image4, pred_image5, pred_image6,
                 pred_image7, pred_image8, pred_image9, pred_image10, pred_image11, pred_image12,
                 pred_image13, pred_image14, pred_image15, pred_image16, pred_image17, pred_image18,
                 pred_image19]

  # Adds a subplot at the 1st position
  fig.add_subplot(rows, columns, 1)
  
  # showing query image
  plt.imshow(query_image)
  plt.title("query_image")
  plt.axis('off')
  plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))

  i = 1
  for image in pred_images:
    i = i + 1
    # Adds a subplot at the nth position
    fig.add_subplot(rows, columns, i)
  
    # showing prediction images
    plt.imshow(image)
    plt.title("pred_image" + str(i-1))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  
  plt.show()

plot_pred_neighbors()
