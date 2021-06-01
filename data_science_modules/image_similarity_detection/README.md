#########################################################################################
# HOW TO RUN THE APPLICATION
#########################################################################################

# open linux terminal and run the following commands.
$ virtualenv --system-site-packages -p python3 ./TFvenv
$ source ./TFvenv/bin/activate

# only need to install these when running for the first time.
$ pip install tensorflow
$ pip install tensorflow-hub
$ pip install annoy

# there is no need to run the following as it is included in image_similarity.py.
# however get_image_feature_vectors.py will need to be run if you use a 
# new image dataset as their feature vectors will need to be calculated first.
$ python get_image_feature_vectors.py
$ python cluster_image_feature_vectors.py

# full pipeline that takes an input image amd clusters the nearest neighbors of the 
# input image only and stores their similarity and product ids in nearest_neighbors.json.
$ python image_similarity.py

#########################################################################################
# INSTRUCTIONS FOR USING A NEW DATASET
#########################################################################################

# if using a new image dataset, a new image_data.json will need to be create to reflect
# this. use image_data_prep.ipynb in 'image_similarity_detection/fashion_web_scraping'
# to do this. 

# if there are no products ids available, then find a way to append
# a list of consecutive positive integers that match the length of the dataset 
# in place of them. 

# once complete, remove every '.jpg' from image_data.json and
# append one additional object as follows: {"imageName": "temp", "productId": "temp"}.
# the reason for this is that it allows for the input image to be compared to find the
# nearest neighbors. 

# make sure this file is placed in 'image_similarity_detection/
# fashion_web_scraping/jsonFiles'.
