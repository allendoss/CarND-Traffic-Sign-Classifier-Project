#Traffic Sign Classifier
# Load pickled data
import pickle
import matplotlib.pyplot as plt
import random
import cv2
from scipy import ndimage
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'D:/SDCND/Project_2/CarND-Traffic-Sign-Classifier-Project/train.p'
validation_file = 'D:/SDCND/Project_2/CarND-Traffic-Sign-Classifier-Project/valid.p'
testing_file = 'D:/SDCND/Project_2/CarND-Traffic-Sign-Classifier-Project/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
# Each X_train corresponds to one image
# eg: X_train[0] is one image
# X_train has a length of 34799, this means 34799 images
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('X_valid shape:', X_valid.shape)
print('y_valid shape:', y_valid.shape)

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#import matplotlib.pyplot as plt
#import random

# Visualizations will be shown in the notebook.
#fig, axisImage = plt.subplots(2,4)
#axisImage = axisImage.ravel() #returns flattened array of image objects from a multidimensional array
#for i in range(8):
#    index = random.randint(0, len(X_train))
#    image = X_train[index]
#    axisImage[i].axis('off')
#    axisImage[i].imshow(image)

plt.hist(y_train, n_classes, histtype = 'stepfilled')
plt.show()

#==============================================================================
# # Conversion to grayscale
# def grayscale(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return image
# 
# # Normalize the data
# def normalize(data):
#     return data / 255 * 0.8 + 0.1
#     #return cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# 
# # Iterate through all the images
# def preprocess(data):
#     grayImages=[]
#     for image in data:
#         gray = grayscale(image)
#         grayNormalize = normalize(gray)
#         grayImages.append(grayNormalize)
#     return np.array(grayImages) # converts the list into an array
# 
# # Preprocessing the images
# from numpy import newaxis
# 
# print('Preprocessing training data...')
# 
# # Iterate through grayscale
# X_train = preprocess(X_train)
# X_train = X_train[..., newaxis]
# 
# # Normalize
# X_train = normalize(X_train) 
# 
# print('Finished preprocessing training data.')
# 
# # Double-check that the image is changed to depth of 1
# image_shape2 = X_train.shape
# print("Processed training data shape =", image_shape2)
# 
# print('Preprocessing testing data...')
# 
# # Iterate through grayscale
# X_test = preprocess(X_test)
# X_test = X_test[..., newaxis]
# 
# # Normalize
# X_test = normalize(X_test) 
# 
# print('Finished preprocessing testing data.')
# 
# # Double-check that the image is changed to depth of 1
# image_shape3 = X_test.shape
# print("Processed testing data shape =", image_shape3)
# 
# print('All data preprocessing complete.')
#==============================================================================


# Visualizations will be shown in the notebook.
fig, axisImage = plt.subplots(2,4) #axxisImage is an object used to hold/contain/display each image
axisImage = axisImage.ravel() #returns flattened array of image objects from a multidimensional array
for i in range(8):
    index = random.randint(0, len(X_train))
    axisImage[i].axis('off')
    axisImage[i].imshow(X_train[index].squeeze(), cmap="gray") # To remove single-dimensional entries from the shape of an array

# Data Augmentation

# Avoid transformation after normalizing as pixel definition will differ
def translate(image):
    #image = cv2.imread(image) doesn't work cause it converts image to numpy like array
    #but you have already converted the image passed here in an array format before
    #print(image.shape) #32x32x1
    rows, cols, ch = image.shape
    #plt.imshow(image.squeeze(), cmap='gray')
    dx,dy = np.random.randint(-5,5,2) 
    M = np.float32([[1,0,dx],[0,1,dy]]) # Translation matrix, M=[1 0 dx;0 1 dy]
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def affineTransform(image):
    rows, cols, ch = image.shape
    #fix one point and change the other two
    rndx = np.random.rand(3)
    rndx *= cols * 0.06
    rndy = np.random.rand(3)
    rndy *= rows * 0.06
    x1 = 0.25*cols
    x2 = 0.75*cols
    y1 = 0.25*rows
    y2 = 0.75*rows
    pts1 = np.float32([[y1,x1],[y2,x1],[y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]], [y2+rndy[1],x1+rndx[1]], [y1+rndy[2],x2+rndx[2]]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(image,M,(cols,rows))   
    return dst
    
def plot(sampleImage, sampletransform):
    fig, ax = plt.subplots(1,2, figsize=(5,2))
    ax[0].axis('off')
    ax[0].imshow(sampleImage.squeeze())
    ax[0].set_title('Original image')
    ax[1].axis('off')
    ax[1].imshow(sampletransform.squeeze())
    ax[1].set_title('Transformed image')

def transform(image):
    img1=translate(image)
    img2=affineTransform(img1)
    return img2

sampleImage = X_train[111]
sampletransform = transform(sampleImage)
plot(sampleImage, sampletransform)
# Data additions
imagesInEachClass = np.bincount(y_train)
meanImages = int(np.mean(imagesInEachClass))
for i in range(len(imagesInEachClass)):
    if imagesInEachClass[i] < meanImages:
        extra = meanImages - imagesInEachClass[i]
        location = np.where(y_train == i)
        addX=[]
        addy=[]
        for j in range(extra):
            newImage = transform(X_train[location[0][0]]) #??
            addX.append(newImage)
            addy.append(i)
        X_train = np.append(X_train, np.array(addX), axis=0)
        y_train = np.append(y_train, np.array(addy), axis=0)
    
plt.hist(y_train, n_classes, histtype = 'stepfilled')
print("The updated number of training examples are ", len(X_train))

# Normalization and grayscale

# Conversion to grayscale
def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Normalize the data
def normalize(data):
    return data / 255 * 0.8 + 0.1
    #return cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Iterate through all the images
def preprocess(data):
    grayImages=[]
    for image in data:
        gray = grayscale(image)
        grayNormalize = normalize(gray)
        grayImages.append(grayNormalize)
    return np.array(grayImages) # converts the list into an array

# Preprocessing the images
from numpy import newaxis

print('Preprocessing training data...')

# Iterate through grayscale
X_train = preprocess(X_train)
X_train = X_train[..., newaxis]
print('Finished preprocessing training data.')
image_shape1 = X_train.shape
print("Processed training data shape =", image_shape1)
print('Preprocessing testing data...')

X_valid = preprocess(X_valid)
X_valid = X_valid[..., newaxis]
print('Finished preprocessing validation data.')
image_shape2 = X_valid.shape
print("Processed validation data shape =", image_shape2)
print('Finished preprocessing validation data.')

# Iterate through grayscale
X_test = preprocess(X_test)
X_test = X_test[..., newaxis]
print('Finished preprocessing testing data.')
image_shape3 = X_test.shape
print("Processed testing data shape =", image_shape3)
print('All data preprocessing complete.')

fig, axisImage = plt.subplots(2,4) #axxisImage is an object used to hold/contain/display each image
axisImage = axisImage.ravel() #returns flattened array of image objects from a multidimensional array
for i in range(8):
    index = random.randint(0, len(X_train))
    axisImage[i].axis('off')
    axisImage[i].imshow(X_train[index].squeeze(), cmap="gray") # To remove single-dimensional entries from the shape of an array

X_train, y_train = shuffle(X_train, y_train) 

# Model
tf.reset_default_graph()
epochs = 1
batchSize=100
x = tf.placeholder(tf.float32, (None,32,32,1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
def leNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Weight and bias
    # If not using grayscale, the third number in shape would be 3
    c1_weight = tf.Variable(tf.truncated_normal(shape = (5, 5, 1, 6), mean = mu, stddev = sigma))
    c1_bias = tf.Variable(tf.zeros(6))
    # Apply convolution
    conv_layer1 = tf.nn.conv2d(x, c1_weight, strides=[1, 1, 1, 1], padding='VALID') + c1_bias
    
    # Activation for layer 1
    conv_layer1 = tf.nn.relu(conv_layer1)
    
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv_layer1 = tf.nn.avg_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    # Note: The second layer is implemented the exact same as layer one, with layer 1 as input instead of x
    # And then of course changing the numbers to fit the desired ouput of 10x10x16
    # Weight and bias
    c2_weight = tf.Variable(tf.truncated_normal(shape = (5, 5, 6, 16), mean = mu, stddev = sigma))
    c2_bias = tf.Variable(tf.zeros(16))
    # Apply convolution for layer 2
    conv_layer2 = tf.nn.conv2d(conv_layer1, c2_weight, strides=[1, 1, 1, 1], padding='VALID') + c2_bias
    
    # Activation for layer 2
    conv_layer2 = tf.nn.relu(conv_layer2)
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv_layer2 = tf.nn.avg_pool(conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten to get to fully connected layers. Input = 5x5x16. Output = 400.
    flat = tf.contrib.layers.flatten(conv_layer2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    # Although this is fully connected, the weights and biases still are implemented similarly
    # There is no filter this time, so shape only takes input and output
    # Weight and bias
    fc1_weight = tf.Variable(tf.truncated_normal(shape = (400, 200), mean = mu, stddev = sigma))
    fc1_bias = tf.Variable(tf.zeros(200))
    # Here is the main change versus a convolutional layer - matrix multiplication instead of 2D convolution
    fc1 = tf.matmul(flat, fc1_weight) + fc1_bias
    
    # Activation for the first fully connected layer.
    # Same thing as before
    fc1 = tf.nn.relu(fc1)
    
    # Dropout, to prevent overfitting
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    # Same as the fc1 layer, just with updated output numbers
    fc2_weight = tf.Variable(tf.truncated_normal(shape = (200, 100), mean = mu, stddev = sigma))
    fc2_bias = tf.Variable(tf.zeros(100))
    # Again, matrix multiplication
    fc2 = tf.matmul(fc1, fc2_weight) + fc2_bias
    
    # Activation.
    fc2 = tf.nn.relu(fc2)
    
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # Layer 5 Fully Connected. Input = 84. Output = 43.
    # Since this is the final layer, output needs to match up with the number of classes
    fc3_weight = tf.Variable(tf.truncated_normal(shape = (100, 43), mean = mu, stddev = sigma))
    fc3_bias = tf.Variable(tf.zeros(43))
    # Again, matrix multiplication
    logits = tf.matmul(fc2, fc3_weight) + fc3_bias
    
    return logits

rate = 0.005
logits =leNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batchSize):
        batch_x, batch_y = X_data[offset:offset+batchSize], y_data[offset:offset+batchSize]
        accuracy =  sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

save_file = 'D:/SDCND/Project_2/CarND-Traffic-Sign-Classifier-Project/train_model.ckpt'
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batchSize):
            end = offset + batchSize
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            loss = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')