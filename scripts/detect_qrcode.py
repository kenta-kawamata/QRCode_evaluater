#! /usr/bin/env python3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, exposure
from pyzbar.pyzbar import decode, ZBarSymbol

def output_hog_featurers(image):

    fd, hog_image = hog(image, 
                        orientations=8, 
                        pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), 
                        visualize=True, 
                        channel_axis=2,
                        feature_vector=True)

    # HOG in histgram
    plt.hist(fd,
                bins = 9,
                range = (0,1),
                color ='Blue')
    plt.show()

    #HOG in image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def detect_qr(img):

    qcd = cv2.QRCodeDetector()

    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)

    img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 3)

    for s, p in zip(decoded_info, points):
        img = cv2.putText(img, s, p[0].astype(int),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    print(points[0])
    # When the image that read from Pillow
    # the image's coordinate X Y is reverse.
    x = points[0][:,0]
    y = points[0][:,1]

    margin = 1
    
    x1 = int(np.amin(x)-margin)
    x2 = int(np.amax(x)+margin)
    y1 = int(np.amin(y)-margin)
    y2 = int(np.amax(y)+margin)

    trimed_img = img[y1:y2,x1:x2]

    return trimed_img


def detect_qr_using_pyzbar(img):

    decoded_list = decode(img)
   
    print(decoded_list[0])


def pil2cv(image):

    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)

    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー

        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    elif new_image.shape[2] == 4:  # 透過

        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)

    return new_image

def main():

    #img = cv2.imread('./../qr_code.png')
    img = Image.open('./../qr_code.png')
    img = pil2cv(img)

    img = cv2.resize(img,None,fx=1,fy=1)

    detect_qr_using_pyzbar(img)

    trimed_img = detect_qr(img)
    output_hog_featurers(trimed_img)

    #cv2.imshow("qr code",trimed_img)
    #cv2.waitKey(0)

if __name__ == "__main__":

    main()
