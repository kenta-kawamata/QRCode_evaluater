#! /usr/bin/env python3

import cv2

def detect_qr():

    img = cv2.imread('./../qr_code.png')

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

    cv2.imshow("qr code",trimed_img)
    cv2.waitKey(0)


def main():

    detect_qr()

if __name__ == "__main__":

    main()
