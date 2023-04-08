
# Assignment 2 - Digital Image Preprocessing

This assignment is about denoising and reconstruction of images. Apart from that, it was required to work with test dataset of RSNA MRI data. 
Overall, there was 3 tasks: (Tasks are conducted on images number 5 for first two tasks and on test/00037 for the third task)
  * Task 1
    - Denoising and reconstructing chemical formula image
    - Adding noise to my image from previous assignment and removing it
  * Task 2
    - Removing speckle noise from the provided image
  * Task 3
    - Showing metadata and contents of files from RSNA dataset
    - Visualizing images inside the dataset
  
These images are saved in the **"noisy/chemical"**, **"MyImage"**, **"noisey/speckle"**, and **"test/00037"** folders respectively.  
### MyImage Folder 
**MyImage** folder contains my custom image and results of the noisy and denoised images.  
### ChemicalRemoved Folder 
**ChemicalRemoved** folder contains the results of the denoised chemical image. Outputs of the all tried methods (spatial, morphological) are saved here.  
### SpeckleRemoved Folder 
**SpeckleRemoved** folder contains the results of the images where speckle noise is removed. It is done by two methods - custom function and OpenCV function
and both results are saved here.
### GIF Folder 
**GIF** folder contains the gifs that are generated to visualize the images from the test RSNA dataset.


## main.py
All the code is contained inside the **main.py** file. The code is well commented. Displaying the gifs are done with a custom function named **GifShowe**.
Also, there is a custom **crimmins** function to remove the speckle noise.


None of the results are being shown to the user with imshow since all of the results are being saved in their corresponding folders. Apart from that,
these folders are generated on the root location of the **main.py** through the code itself.  
