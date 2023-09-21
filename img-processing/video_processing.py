#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys

import cv2  # type: ignore
import numpy as np  # type: ignore

"""
Created on Wed Mar  2 13:04:53 2022

@author: Roberto Hernandez Ruiz
COMPUTER VISION ASSIGNMENT 1
Master in Artificial Intelligence - KU Leuven
"""

"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""

# python cv_assignment1.py -i PATH/input_video.mp4 -o PATH/output_video.mp4


def between(cap, lower: int, upper: int) -> bool:
    """ Helper function to change what you do based on video seconds. """
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Font
        font = cv2.FONT_HERSHEY_DUPLEX
        font_color = (255, 255, 255)
        text = ''
              
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            
            # ############# 1. BASIC IMAGE PROCESSING ##############
            
            # Between 0-4s, switch full img from gray to rgb and viceversa
            if between(cap, 1000, 4000):
                text = "Switching into GRAY and RGB colorspaces..."
                if between(cap, 1000, 2000):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif between(cap, 2500, 3500):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
            # Between 4-8s, Gaussian filter
            if between(cap, 4000, 5500):
                text = "GAUSSIAN FILTER"
                frame = cv2.GaussianBlur(frame, (5, 5), 20)
            if between(cap, 5500, 8000):
                text = "GAUSSIAN FILTER (+ Kernel size = + smoothing)"
                if between(cap, 5500, 7000):
                    kernel_size = (15, 15)
                elif between(cap, 7000, 8000):
                    text = "GAUSSIAN FILTER (+++ Kernel size = +++ smoothing)"
                    kernel_size = (25, 25)  # more kernel size, more smoothing
                frame = cv2.GaussianBlur(frame, kernel_size, 20)
                
            # Between 8-12s, bilateral filter (difference: sharp edges are still preserved)
            if between(cap, 8000, 10000):
                text = "BILATERAL FILTER (sharp edges are preserved)"
                frame = cv2.bilateralFilter(frame, 5, 50, 50)
            if between(cap, 10000, 12000):
                text = "BILATERAL FILTER (sharp edges are preserved)"
                frame = cv2.bilateralFilter(frame, 5, 400, 400)
            
            # Between 12-16s, grab yellow balloon in RGB and HSV
            if between(cap, 12000, 14000):
                text = "Grab yellow object in RGB"  # search bgr range for yellow
                lower_bound = np.array([0, 180, 180], dtype="uint8")
                upper_bound = np.array([125, 255, 255], dtype="uint8")
                frame = cv2.inRange(frame, lower_bound, upper_bound)
                frame = np.stack([frame, frame, frame], axis=2)
            if between(cap, 14000, 16000):
                text = "Grab yellow object in HSV (with noise)"
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # easier to work in hsv to detect colors
                lower_bound = np.array([23, 93, 0], dtype="uint8")
                upper_bound = np.array([48, 255, 255], dtype="uint8")
                frame = cv2.inRange(frame, lower_bound, upper_bound)
                frame = np.stack([frame, frame, frame], axis=2)
                           
            # Between 16-20s, grab object and apply binary morphological operations
            if between(cap, 16000, 20000):
                text = "Improve grabbing with opening (erosion + dilation)"
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_bound = np.array([23, 93, 0], dtype="uint8")
                upper_bound = np.array([48, 255, 255], dtype="uint8")
                mask = cv2.inRange(frame, lower_bound, upper_bound)
                kernel = np.ones((19, 19), np.uint8)
                frame = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # opening is the one with the better results
                frame = np.stack([frame, frame, frame], axis=2)
                        
            # ############# 2. OBJECT DETECTION ##############
            # Between 20-25s, Sobel filter to detect and visualize horizontal and vertical edges
            if between(cap, 20000, 22000):
                text = "SOBEL: Vertical edges"
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = cv2.Sobel(frame, cv2.CV_8U, 1, 0, ksize=3)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # to color edges
                
            if between(cap, 22000, 24000):
                text = "SOBEL: Horizontal edges"
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = cv2.Sobel(frame, cv2.CV_8U, 0, 1, ksize=3)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # to color edges
                    
            if between(cap, 24000, 25000):
                text = "Now: ++ k_size = ++ blurry edges"  # all increased bc of kernel size
                font_color = (0, 255, 0)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = cv2.Sobel(frame, cv2.CV_8U, 1, 0, ksize=5)
                frame = cv2.Sobel(frame, cv2.CV_8U, 0, 1, ksize=5)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # to color edges
                
            # Between 25-35s, detect circular shapes (Hough transform)
            if between(cap, 25000, 35000):
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if between(cap, 25000, 28000):
                    text = "HOUGH: detect circles (many False Positives: so tune parameters!!!)"
                    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=45, minRadius=20, maxRadius=80)
        
                if between(cap, 28000, 33000):
                    # by tunning radius range and acc threshold (param2 of method)
                    text = "HOUGH: detect circular shapes better (changing radius range, accThreshold/param2)"
                    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=60, param2=40, minRadius=20, maxRadius=40)
        
                if between(cap, 33000, 35000):
                    # by tunning radius range and setting minDistance the lowest
                    text = "HOUGH: detect small circular shapes"
                    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1.2, 1, param1=50, param2=26, minRadius=5, maxRadius=10)
        
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                   
                    for (x, y, radius) in circles:
                        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)  # draw center
                
            # Between 35-38s, draw a flashy rectangle around the object in certain position
            if between(cap, 35500, 38000):
                text = "Detect object (rice) in certain position"
                                  
                template = cv2.imread("rice.png", 0)  # template in gray
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame in gray
                
                w, h = template.shape[::-1]

                # Template matching with squares differences
                result = cv2.matchTemplate(frame_gray, template, cv2.TM_SQDIFF)
                
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
                (startX, startY) = minLoc  # because we are using sqdiff
                
                endX = startX + template.shape[1]
                endY = startY + template.shape[0]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                
            # Between 38-40s, grayscale with values of likelihood of object being at that location
            if between(cap, 38000, 40000):
                text = "Grayscale likelihood map of the object (rice)"
                
                template = cv2.imread("rice.png", 0)  # template in gray
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # frame in gray
                w, h = template.shape[::-1]
                result = cv2.matchTemplate(frame_gray, template, cv2.TM_SQDIFF)
            
                inv_probs = cv2.normalize(result, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                probs = cv2.bitwise_not(inv_probs)  # invert probabilities
                probs = cv2.resize(probs, (frame.shape[1], frame.shape[0]))  # resize matrix to original frame dims
                frame = cv2.cvtColor(probs, cv2.COLOR_GRAY2BGR)
                
            # ############# 3. CARTE BLANCHE ##############
            
            if between(cap, 40500, 55000):
                # Now we are going to detect the object by thresholding, not by matching template
                         
                # Eye detection with pretrained model (attached files)
                if between(cap, 40500, 44000):
                    text = "Tracking of green balloon (thresholding) and eye detection (pretrained model, in blue)"
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces_detection = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
                    for (x, y, w, h) in faces_detection:
                        roi_gray = frame_gray[y:y + h, x:x + w]
                        roi_color = frame[y:y + h, x:x + w]
                    eyes_detection = eye_cascade.detectMultiScale(roi_gray)  # only detect eyes within face range
                    for (eye_x, eye_y, eye_w, eye_h) in eyes_detection:
                        cv2.rectangle(roi_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 1)
                
                # Object detection invariant to rotation(thresholding)
                if between(cap, 45000, 49000):
                    text = "Tracking of green balloon (invariant to rotation)"
                    frame = cv2.flip(frame, 0)
                    
                # Object detection invariant to changes in illumination
                if between(cap, 49000, 55000):  # in these seconds video turns brighter
                    text = "Tracking of green balloon (invariant to illumination changes and going out of scene)"
            
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower = np.array([36, 0, 0], dtype="uint8")
                upper = np.array([86, 255, 255], dtype="uint8")
                mask = cv2.inRange(frame_hsv, lower, upper)
                cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)  # to draw rectangle
                    
                    if (w > 75) and (h > 75):  # knowing its dims, such as in the template matching case
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        obj = "Green balloon"
                        font_color_obj = (0, 255, 0)
                        cv2.putText(frame, obj, (x + w + 5, y + h + 5), font, 0.5, font_color_obj, 1)
                 
            # Another way for small circles detection
            if between(cap, 55000, 60000):
                text = "Another way for small circles detection (SimpleBlobDetector)"
                p = cv2.SimpleBlobDetector_Params()
                # Set Area, Circularity and Convexity filtering parameters
                p.filterByArea = True
                p.minArea = 60
                p.maxArea = 500
                p.filterByCircularity = True
                p.minCircularity = 0.5
                p.filterByConvexity = True
                p.minConvexity = 0.2
               
                detector = cv2.SimpleBlobDetector_create(p)
                                    
                keypoints = detector.detect(frame)
                                
                blank = np.zeros((1, 1))
                # blobs as blue circles
                frame = cv2.drawKeypoints(frame, keypoints, blank, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
             
            # Change color of whole object for a few seconds
            if between(cap, 60000, 65000):
                text = "Change a color (colouring wall)"
        
                frame = cv2.blur(frame, (5, 5))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                ddepth = cv2.CV_16S
                grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)  # equally
                ret, thresh = cv2.threshold(grad, 10, 255, cv2.THRESH_BINARY_INV)
                c, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                areas = [cv2.contourArea(c1) for c1 in c]
                maxAreaIndex = areas.index(max(areas))
                cv2.drawContours(frame, c, maxAreaIndex, (0, 255, 0), -1)
                                       
            # Write performed operations
            textsize = cv2.getTextSize(text, font, 0.6, 2)[0]
            textX = int((frame_width - textsize[0]) / 2)
            textY = int((frame_height + textsize[1]) - 30)
            cv2.putText(frame, text, (textX, textY), font, 0.6, font_color, 1)

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Computer Vision: Assignment 1', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
