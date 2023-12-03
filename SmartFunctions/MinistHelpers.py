import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def MNIST_GetDataSet():
    X, y = fetch_openml('mnist_784', as_frame=False, return_X_y=True)
    return X, y

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")


#Used chat gpt to create a helper function as to not plot all the data but only partially.
#It isn't needed as you could plot the data as a whole but this makes it much nicer to look at
def MNIST_PlotDigitArray(data_array):
    # Determine the number of rows and columns for subplots based on the array length
    num_digits = len(data_array)
    num_rows = int(num_digits ** 0.5)
    num_cols = (num_digits + num_rows - 1) // num_rows

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Iterate through the data_array and plot each digit
    for i, data in enumerate(data_array):
        ax = axes[i // num_cols, i % num_cols] if num_digits > 1 else axes
        image = data.reshape(28, 28)
        ax.imshow(image, cmap='binary')
        ax.axis('off')

    # Remove any unused subplots if num_digits is not a perfect square
    for i in range(num_digits, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])
    #End of helper function
    plt.show()

def ReshapeData(X,y):
   if X.ndim==3:
    print("reshaping X..")
    assert y.ndim==1
    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
    assert X.ndim==2

    print("done reshaping X..")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def show_some_digits(images, targets, sample_size=24, title_text='Digit {}' ):
    '''
    Visualize random digits in a grid plot
    images - array of flatten gidigs [:,784]
    targets - final labels
    '''
    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0],nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))


    img = plt.figure(1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
        plt.axis('off')
        #each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))
    plt.show()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots confusion matrix, 
    
    cm - confusion matrix
    """
    plt.figure(1, figsize=(15, 12), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()  
    


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_param_space_heatmap(scores, C_range, gamma_range):
    """
    Draw heatmap of the validation accuracy as a function of gamma and C
    
    
    Parameters
    ----------
    scores - 2D numpy array with accuracies
    
    """
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.jet,
               norm=MidpointNormalize(vmin=0.5, midpoint=0.9))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    

def plot_param_space_bubble(scores, x_range, y_range):
    """
    Plot scatter plot of the validation accuracy as a function of gamma and C
        
    Parameters
    ----------
    scores - 2D numpy array with accuracies
    
    """
    
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    
    # Change color with c and alpha. I map the color to the X axis value.
    plt.scatter(x_range, y_range, s=scores*2000, c=scores, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)


    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.colorbar()
    # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    # plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()
    

    
   
   

