import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
import math
import tarfile#Разархивировать файлы
import imageio.v2 as imageio #for reading image
# ============================================
# Write number of images
total_number_of_image=368
# ============================================
class FaceDetector():

    def __init__(self,faceCascadePath):
        self.faceCascade=cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.3,
               minNeighbors=4,
               minSize=(30,30)):
        
        #function return rectangle coordinates of faces for given image
        rects=self.faceCascade.detectMultiScale(image,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
        return rects


#Frontal face of haar cascade
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
frontal_cascade_path = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
#Detector object created
fd=FaceDetector(frontal_cascade_path)

def show_image(image):
    plt.figure(figsize=(18,15))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([]) 
    plt.savefig("Detected_faces.png")
    plt.show()

class FetchLFW:
    def __init__(self, path):
        self.path=path
        
    def _initialize(self,dim):
        self.dim_of_photo_gallery=dim
        self.number_of_images=self.dim_of_photo_gallery*self.dim_of_photo_gallery
        
        
        self.random_face_indexes=np.arange(total_number_of_image)
        self.n_random_face_indexes=self.random_face_indexes[:np.random.shuffle(self.random_face_indexes)]
        
        
    def get_lfw_images(self,dim=5):
        
        self._initialize(dim)
        
        
        self.lfw_images=self._get_images()
        
        return self.lfw_images
        
    
    def _get_images(self):
        image_list=[]
        tar = tarfile.open(path, "r:tar")
        counter=0
        counter_zero_size_jpg=0
        for tarinfo in tar:
            
            tar.extract(tarinfo.name)#extract one file from directory
            if tarinfo.name[-4:]==".jpg":
                if counter in self.n_random_face_indexes:
                    image=cv2.imread(tarinfo.name, cv2.IMREAD_COLOR)
                    # image=cv2.resize(image,(250,250),fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
                    # image_list.append(np.array(image))
                    height, width = image.shape[:2]
                    if (height<40) or (width<40):
                        print('{} size is too low'.format(tarinfo.name)) 
                        counter_zero_size_jpg+=1
                    else:
                        image=cv2.resize(image,(250,250),fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
                        image_list.append(np.array(image)) 
                    
                counter+=1

                        
            # if tarinfo.isdir():#Checking the existence of a directory
            #     pass
            # else:
            #     os.remove(tarinfo.name)
        tar.close()
        print('Count_bad_images == {}'.format(counter_zero_size_jpg))
        return np.array(image_list),counter_zero_size_jpg
    
path=os.path.join('data', 'with_my_photo.tar')
fetchLFW=FetchLFW(path)
# ============================================================
#Write size of matrix. size <= sqrt(total_number_of_image)
dimension=19# That is, there will be n*n scanned images
# ======================================================



images,counter_zero_size_jpg=fetchLFW.get_lfw_images(dim=dimension)

def get_photo_gallery():
    counter=0
    himages=[]
    vimages=[]

    dim_matrix=(dimension**2-counter_zero_size_jpg)**(0.5)
    if dim_matrix %1 ==0:
        x=dim_matrix
        y=dim_matrix
    else:
        x=math.floor(dim_matrix)
        y=x+1


    for i in range(x):
        for j in range(y):
                himages.append(images[counter])
                counter+=1
        vimages.append(np.hstack((himages)))
        himages=[]
    image_matrix=np.vstack((vimages))
    return image_matrix


face_counter=0
for image_org in images:
    image_gray=cv2.cvtColor(image_org,cv2.COLOR_BGR2GRAY)
    faceRect=fd.detect(image_gray,
                       scaleFactor=1.05,
                       minNeighbors=5,
                       minSize=(30,30))
    first_detection=False
    for (x,y,w,h) in faceRect:
        if first_detection==False:
            face_counter+=1
            cv2.rectangle(image_org,(x,y),(x+w,y+h),(127,255,0),2)
            first_detection=True
        else:
            # print("Second detection ignored in a image")
            pass

print("{} images have been scaned".format(dimension*dimension))
print("{} faces have been detected".format(face_counter))

photo_gallery=get_photo_gallery()
show_image(photo_gallery)
