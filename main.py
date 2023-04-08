import cv2 as cv
import os
import numpy as np
import pydicom
import imageio
import matplotlib.pyplot as plt

# Create save directories
if not os.path.exists('./GIF'):
    os.makedirs('./GIF')


if not os.path.exists('./ChemicalRemoved'):
    os.makedirs('./ChemicalRemoved')


if not os.path.exists('./SpeckleRemoved'):
    os.makedirs('./SpeckleRemoved')


def GifShower(gif):
    # Create a figure and axis for displaying the frames
    fig, ax = plt.subplots()

    # Loop through each frame in the GIF and display it
    for frame in gif:
        ax.imshow(frame)
        plt.pause(0.1)
        ax.cla()

    # Close the figure when done
    plt.close(fig)


# -----------------Task1-----------------

img = cv.imread('./noisy/chemical/inchi5.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply bilateral filter
filtered_img = cv.bilateralFilter(gray, d=15, sigmaColor=110, sigmaSpace=110)

# Create a structuring element
kernel = np.ones((2, 2), np.uint8)

# Invert the image
inverted_img = cv.bitwise_not(filtered_img)

# Perform dilation
dilated_img = cv.dilate(inverted_img, kernel, iterations=1)

# Invert the result back and apply median blur
result_img = cv.bitwise_not(dilated_img)
result_img = cv.medianBlur(result_img, 3)

cv.imwrite('./ChemicalRemoved/DenoisedChemical.png', result_img)

# Trying morphological method

# Create a structuring element for dilation
dilation_kernel = np.array([[0, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0]], dtype=np.uint8)

# Create a structuring element for erosion
erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

# Threshold the image to invert it
_, thresholded = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)

# Apply dilation
dilated = cv.dilate(thresholded, dilation_kernel)

# Apply erosion
eroded = cv.erode(dilated, erosion_kernel)

eroded = cv.erode(eroded, erosion_kernel)

# Show the original image and the result
cv.imwrite('./ChemicalRemoved/Morph.png', eroded)

# Working on my own image

my_img = cv.imread('./MyImage/IMG_3581.jpg')

# Add Gaussian noise to the image
mean = 0
var = 100
sigma = var ** 0.05
gauss = np.random.normal(mean, sigma, my_img.shape).astype(np.uint8)
noisy_img = cv.add(my_img, gauss)

# Apply a Gaussian filter to remove the noise
filtered_img = cv.GaussianBlur(noisy_img, (15, 15), 0)

cv.imwrite('./MyImage/Noised.jpg', noisy_img)
cv.imwrite('./MyImage/Denoised.jpg', filtered_img)

# -----------------Task2----------------

img = cv.imread('./noisy/speckle/5.png', cv.IMREAD_GRAYSCALE)


def crimmins(data):
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])

    # Dark pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i - 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol - 1):
            if data[i, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i - 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol - 1):
            if data[i - 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i - 1, j] > data[i, j]) and (data[i, j] <= data[i + 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j + 1] > data[i, j]) and (data[i, j] <= data[i, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j - 1] > data[i, j]) and (data[i, j] <= data[i + 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j + 1] > data[i, j]) and (data[i, j] <= data[i + 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] > data[i, j]) and (data[i, j] <= data[i - 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j - 1] > data[i, j]) and (data[i, j] <= data[i, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j + 1] > data[i, j]) and (data[i, j] <= data[i - 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j - 1] > data[i, j]) and (data[i, j] <= data[i - 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] >= (data[i, j] + 2)):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol):
            if (data[i, j - 1] >= (data[i, j] + 2)):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            if (data[i + 1, j + 1] >= (data[i, j] + 2)):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1):
        for j in range(1, ncol):
            if (data[i + 1, j - 1] >= (data[i, j] + 2)):
                new_image[i, j] += 1
    data = new_image

    # Light pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if (data[i - 1, j] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol - 1):
            if (data[i, j + 1] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if (data[i - 1, j - 1] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol - 1):
            if (data[i - 1, j + 1] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i - 1, j] < data[i, j]) and (data[i, j] >= data[i + 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j + 1] < data[i, j]) and (data[i, j] >= data[i, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j - 1] < data[i, j]) and (data[i, j] >= data[i + 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j + 1] < data[i, j]) and (data[i, j] >= data[i + 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] < data[i, j]) and (data[i, j] >= data[i - 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j - 1] < data[i, j]) and (data[i, j] >= data[i, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j + 1] < data[i, j]) and (data[i, j] >= data[i - 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j - 1] < data[i, j]) and (data[i, j] >= data[i - 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol):
            if (data[i, j - 1] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            if (data[i + 1, j + 1] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1):
        for j in range(1, ncol):
            if (data[i + 1, j - 1] <= (data[i, j] - 2)):
                new_image[i, j] -= 1
    data = new_image
    return new_image.copy()


speckle_reduced_img = cv.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
speckle_reduced_wf = crimmins(img)

cv.imwrite('./SpeckleRemoved/FastNLMethod.png', speckle_reduced_img)
cv.imwrite('./SpeckleRemoved/WithCustomFunc.png', speckle_reduced_wf)

# --------------Task3-----------------

# Visualizing the image layers

flair_dir = './test/00037/FLAIR'
flair_files = [os.path.join(flair_dir, f) for f in os.listdir(flair_dir) if f.endswith('.dcm')]

flair_images = []
for filename in flair_files:
    ds = pydicom.dcmread(filename)
    image = ds.pixel_array
    # Convert the image data to uint8
    image = (image / image.max() * 128).astype('uint8')
    flair_images.append(image)

gif_path = './GIF/flair.gif'
imageio.mimsave(gif_path, flair_images)

#GifShower(flair_images)

t1w_dir = './test/00037/T1w'
t1w_files = [os.path.join(t1w_dir, f) for f in os.listdir(t1w_dir) if f.endswith('.dcm')]

t1w_images = []
for filename in t1w_files:
    ds = pydicom.dcmread(filename)
    image = ds.pixel_array
    # Convert the image data to uint8
    image = (image / image.max() * 128).astype('uint8')
    t1w_images.append(image)

gif_path = './GIF/t1w.gif'
imageio.mimsave(gif_path, t1w_images)

#GifShower(t1w_images)

t1wce_dir = './test/00037/T1wCE'
t1wce_files = [os.path.join(t1wce_dir, f) for f in os.listdir(t1wce_dir) if f.endswith('.dcm')]

t1wce_images = []
for filename in t1wce_files:
    ds = pydicom.dcmread(filename)
    image = ds.pixel_array
    # Convert the image data to uint8
    image = (image / image.max() * 128).astype('uint8')
    t1wce_images.append(image)

gif_path = './GIF/t1wce.gif'
imageio.mimsave(gif_path, t1wce_images)

#GifShower(t1wce_images)

t2w_dir = './test/00037/T2w'
t2w_files = [os.path.join(t2w_dir, f) for f in os.listdir(t2w_dir) if f.endswith('.dcm')]

t2w_images = []
for filename in t2w_files:
    ds = pydicom.dcmread(filename)
    image = ds.pixel_array
    # Convert the image data to uint8
    image = (image / image.max() * 128).astype('uint8')
    t2w_images.append(image)

gif_path = './GIF/t2w.gif'
imageio.mimsave(gif_path, t2w_images)

#GifShower(t2w_images)

# Describing the contents and metadata

ds = pydicom.dcmread('./test/00037/FLAIR/Image-147.dcm')

print(ds)
