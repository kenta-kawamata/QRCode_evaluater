#! /usr/bin/env python3
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, exposure

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

    #HOG in image
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    #ax1.axis('off')
    #ax1.imshow(image, cmap=plt.cm.gray)
    #ax1.set_title('Input image')

    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    #ax2.axis('off')
    #ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    #ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def detect_qr(img):

    qcd = cv2.QRCodeDetector()

    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)

    img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 3)

    for s, p in zip(decoded_info, points):
        img = cv2.putText(img, s, p[0].astype(int),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    #print(points[0][2][1])
    y1 = int(points[0][0][0])
    y2 = int(points[0][2][0])
    x1 = int(points[0][0][1])
    x2 = int(points[0][2][1])

    trimed_img = img[y1:y2,x1:x2]

    return trimed_img


def main():

    img = cv2.imread('./../qr_code.png')

    trimed_img = detect_qr(img)
    output_hog_featurers(trimed_img)

    #cv2.imshow("qr code",trimed_img)
    #cv2.waitKey(0)

if __name__ == "__main__":

    main()
