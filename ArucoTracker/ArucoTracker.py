'''
Aruco tracker for Physics demonstrations
Written by Shahar Seifer, Weizmann Institute of Science 2025
License:  CC-BY-SA 4.0. (https://creativecommons.org/licenses/by/4.0/legalcode)
Use either live internal camera, or remote webcam camera (install IP Webcam app on your phone, start by clicking: start server),or use pre- recorded mp4 file.
The program calculates position, rotation, two-axis size, and acceleration of Aruco tags (up to two tags on first come basis),and exports data to Excel file.
To generate Aruco images: go to https://chev.me/arucogen/ and choose the 5X5 100 series with any valid ID.
'''

import cv2 as cv
import numpy as np
import requests
import imutils
import time
import xlsxwriter
import os
import win32gui  #install pywin32
import win32con
from pynput import keyboard
from pynput.keyboard import Controller

#Cam_width=1920 #720 #960
#Cam_height=1080 # 480 #540

print("<<<<<<<  Aruco PhysicsTracker  ver 1.0  >>>>>>>")
print("BY Shahar Seifer, Weizmann Institute of Science")
print("The program tracks ArUco tags of 5x5 family.")
print("```````````````````````````````````````````````")
print("")

kbd=Controller()
try:
    Fileobj=open("ip_address.txt","r")
    ipAddress=Fileobj.readline()
    Fileobj.close()
    print("Previously used IP camera address: "+ipAddress)
    ipAddress2 = input('Presss Enter to accept, or insert a new IP address, or 0 for local camera, or -1 to load movie file: ')
    if len(ipAddress2) > 0:
        ipAddress = ipAddress2
        Fileobj2 = open("ip_address.txt", "w")
        Fileobj2.write(ipAddress)
        Fileobj2.close()
        if ipAddress=="-1":
            FileName=input('Enter full filename:  ')
            FrameRate=input('Enter frame rate of recorded file (images per second): ')
            if FrameRate<=0:
                FrameRate=20
except:
    ipAddress=input("Enter ip address of web camera (example: 10.0.0.5) or 0= local camera, -1 = recorded file:   ")
    if ipAddress=="-1":
        FileName = input('Enter full filename:  ')
    Fileobj2 = open("ip_address.txt", "w")
    Fileobj2.write(ipAddress)
    Fileobj2.close()
while True:
    url = "http://"+ipAddress+":8080/shot.jpg"
    try:
        if ipAddress=="0":
            localcamera=cv.VideoCapture(0)
            if not localcamera.isOpened():
                print("Error: Could not open local camera.")
            ret,img_resp = localcamera.read()
        elif ipAddress=="-1":
            movieFile = cv.VideoCapture(FileName)
        else:
            img_resp = requests.get(url)
        break
    except:
        print("Web camera has not started or the address/filename is wrong.")
        ipAddress2 = input("Enter a new ip address/ filename or press Enter to accept "+ipAddress+" :")
        if len(ipAddress2)>0:
            ipAddress=ipAddress2
            if ipAddress == "-1":
                FileName = input('Enter full filename:  ')
            Fileobj2 = open("ip_address.txt", "w")
            Fileobj2.write(ipAddress)
            Fileobj2.close()
try:
    workbook = xlsxwriter.Workbook('sample.xlsx')
except:
    print("Close the excel file and run program again")
    exit(1)

print("Running ... (Press space to exit, s to speed up acquisition.)")

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
parameters =  cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)


ac_mode=0  #live stream (0- live stream and processing, 1- fast acquisition with delayed processing)
store_size=2000
store_count=0
end_store_count=0
#store_im_arr=np.zeros((store_size,Cam_height,Cam_width,3), dtype=np.uint8)
storetime_arr=np.zeros(store_size, dtype=float)


cv.destroyAllWindows()

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
parameters = cv.aruco.DetectorParameters()
arr_result=np.zeros((2,100000,10),dtype=float)

timestart= time.time()
tick_index = 0
index1=0
index2=0

for tick in range(0,100000):
    if tick>0:
        k = cv.waitKey(3)
    else:
        k=-1
    if k == 32: #space bar: to end
        print('Going to end session')
        if ac_mode>=2:
            ac_mode=3
        elif ac_mode==1:
            print('processing stored images')
            ac_mode=3 #first do processing, then exit
            end_store_count=store_count
            store_count=0
        else:
            cv.destroyAllWindows()
            max_tick = tick-1
            break
    elif k == 115:  #letter s: to toggle between normal mode and fast acquisition mode
        if ac_mode==0:
            print('Starting fast acquisition mode')
            ac_mode=1
            store_count=0
            end_store_count = 0
        elif ac_mode==1:
            print('processing stored images')
            ac_mode=2 #first do processing, then continue acquisition
            end_store_count=store_count
            store_count=0
    elif k == -1:

        if ac_mode<=1:
            if ipAddress == "0":
                ret,img_resp = localcamera.read()
                img = np.array(img_resp, dtype=np.uint8)
            elif ipAddress == "-1":
                ret, img_resp = movieFile.read()
                img = np.array(img_resp, dtype=np.uint8)
            else:
                img_resp = requests.get(url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv.imdecode(img_arr, -1)
            im_out = img #imutils.resize(img, width=Cam_width, height=Cam_height)
            if ipAddress == "-1":
                timepass=float(tick)/float(FrameRate)
            else:
                timepass= time.time()-timestart
            if ac_mode==1:
                if store_count<store_size:
                    store_im_arr[store_count,:,:,:]=im_out
                    storetime_arr[store_count]=timepass
                    store_count=store_count+1
                else:
                    print('Storage space ended')
                    print('processing stored images')
                    ac_mode=2 #first do processing, then continue acquisition
                    end_store_count=store_count
                    store_count=0
                    continue
        elif ac_mode>=2:
            if store_count<end_store_count:
                im_out = store_im_arr[store_count,:,:,:]
                timepass = storetime_arr[store_count]
                store_count=store_count+1
            else:
                if ac_mode==3:
                    cv.destroyAllWindows()
                    max_tick = tick - 1
                    break
                elif ac_mode==2:
                    ac_mode=0 #end fast mode, but continue to acquire
                    continue
        if ac_mode!=1:
            markerCorners, markerIds, _ = detector.detectMarkers(im_out)
            detections_img = cv.aruco.drawDetectedMarkers(im_out, markerCorners, markerIds)

            id_index=0
            if len(markerCorners) > 0:
                tick_index = tick_index + 1
                im_out_float=im_out.astype(float)
                brightness_val =np.mean(im_out_float)
                markerIds = markerIds.flatten()

                for (marker_corner, marker_id) in zip(markerCorners, markerIds):
                    # Extract the marker corners
                    if marker_id==index1:
                        id_index=1
                    elif marker_id==index2:
                        id_index=2
                    elif index1==0:
                        index1=marker_id
                        id_index=1
                    elif index2==0:
                        index2=marker_id
                        id_index=2
                    else:
                        id_index=3
                    corners = marker_corner.reshape((4, 2))
                    (top_left, top_right, bottom_right, bottom_left) = corners
                    center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                    center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                    size_iconxtag=0.5*(np.sqrt(pow((top_right[1]-top_left[1]),2)+pow((top_right[0]-top_left[0]),2))+np.sqrt(pow((bottom_right[0]-bottom_left[0]),2)+pow((bottom_right[1]-bottom_left[1]),2)))
                    size_iconytag=0.5*(np.sqrt(pow((top_right[1]-bottom_right[1]),2)+pow((top_right[0]-bottom_right[0]),2))+np.sqrt(pow((top_left[0]-bottom_left[0]),2)+pow((top_left[1]-bottom_left[1]),2)))
                    alpharad1=-np.arcsin((top_right[1]-top_left[1])/(np.sqrt(pow((top_right[1]-top_left[1]),2)+pow((top_right[0]-top_left[0]),2))))
                    if (top_right[0]-top_left[0])<0:
                        alpharad1=np.sign(alpharad1)*np.pi-alpharad1
                    alpharad2=np.arcsin((bottom_left[1]-bottom_right[1])/(np.sqrt(pow((bottom_left[1]-bottom_right[1]),2)+pow((bottom_left[0]-bottom_right[0]),2))))
                    if (bottom_right[0]-bottom_left[0])<0:
                        alpharad2=np.sign(alpharad2)*np.pi-alpharad2
                    alpharad3=np.arcsin((bottom_right[0]-top_right[0])/(np.sqrt(pow((bottom_right[0]-top_right[0]),2)+pow((bottom_right[1]-top_right[1]),2))))
                    if (bottom_right[1]-top_right[1])<0:
                        alpharad3=np.sign(alpharad3)*np.pi-alpharad3
                    alpharad4=np.arcsin((bottom_left[0]-top_left[0])/(np.sqrt(pow((bottom_left[0]-top_left[0]),2)+pow((bottom_left[1]-top_left[1]),2))))
                    if (bottom_left[1]-top_left[1])<0:
                        alpharad4=np.sign(alpharad4)*np.pi-alpharad4
                    alphadeg=(180/np.pi)*(alpharad1+alpharad2+alpharad3+alpharad4)/4.0


                    print ("id=",marker_id,"  x=",center_x,"  y=",center_y,"  rotation=",int(alphadeg))

                    if id_index<=2:
                        arr_result[(id_index-1, tick_index-1, 1-1)] = timepass
                        arr_result[(id_index-1, tick_index-1, 2-1)] = marker_id
                        arr_result[(id_index-1, tick_index-1, 3-1)] = center_x
                        arr_result[(id_index-1, tick_index-1, 4-1)] = center_y
                        arr_result[(id_index - 1, tick_index - 1, 5 - 1)] = alphadeg
                        arr_result[(id_index - 1, tick_index - 1, 6 - 1)] = 0
                        arr_result[(id_index - 1, tick_index - 1, 7 - 1)] = 0
                        arr_result[(id_index - 1, tick_index - 1, 8 - 1)] = size_iconxtag
                        arr_result[(id_index - 1, tick_index - 1, 9 - 1)] = size_iconytag
                        arr_result[(id_index - 1, tick_index - 1, 10 - 1)] = brightness_val
                    if tick_index > 2:
                        if arr_result[(id_index-1, tick_index-1-1, 1-1)]>0 and arr_result[(id_index-1, tick_index-1-2, 1-1)]>0:
                            dtB = (arr_result[(id_index-1, tick_index-1, 1-1)]-arr_result[(id_index-1, tick_index-1-2, 1-1)])/2.0
                            dt_cur = (arr_result[(id_index-1, tick_index-1, 1-1)]-arr_result[(id_index-1, tick_index-1-1, 1-1)])
                            dt_prev = (arr_result[(id_index-1, tick_index-1-1, 1-1)]-arr_result[(id_index-1, tick_index-1-2, 1-1)])
                            #print(dtB,dt_cur,dt_prev)
                            if id_index <= 2:
                                a_x = ((arr_result[(id_index-1, tick_index-1, 3-1)]-arr_result[(id_index-1, tick_index-1-1, 3-1)])/dt_cur - (arr_result[(id_index-1, tick_index-1-1, 3-1)]-arr_result[(id_index-1, tick_index-1-2, 3-1)])/dt_prev)/dtB
                                a_y = ((arr_result[(id_index-1, tick_index-1, 4-1)]-arr_result[(id_index-1, tick_index-1-1, 4-1)])/dt_cur - (arr_result[(id_index-1, tick_index-1-1, 4-1)]-arr_result[(id_index-1, tick_index-1-2, 4-1)])/dt_prev)/dtB
                                arr_result[(id_index-1, tick_index-2, 6-1)] = a_x
                                arr_result[(id_index-1, tick_index-2, 7-1)] = a_y

            for mark in markerCorners:

                mask = np.zeros([im_out.shape[0], im_out.shape[1]], dtype=np.uint8)
                warped_image = im_out.astype(float)
                mask3 = np.zeros_like(warped_image)
                ch=3
                for i in range(0, ch):
                    mask3[:, :, i] = mask / 255
                im_out = cv.multiply(im_out.astype(float), 1-mask3)

            newFrame = im_out.astype(np.uint8)
            newFrame2 = newFrame #imutils.resize(newFrame, width=Cam_width, height=Cam_height)
            cv.imshow('Press space to exit', newFrame2)
            if tick==0:
                Cam_width=newFrame.shape[1]
                Cam_height=newFrame.shape[0]
                store_im_arr = np.zeros((store_size, Cam_height, Cam_width, 3), dtype=np.uint8)
                # Get the window handle
                hwnd = win32gui.FindWindow(None, "Press space to exit")
                # Set the window to always be on top and to be in focus so key press is accepted
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOSIZE) #win32con.SWP_NOMOVE |
                #Get keyboard access to image screen
                kbd.press(keyboard.Key.alt)
                try:
                    win32gui.SetForegroundWindow(hwnd)
                finally:
                    kbd.release(keyboard.Key.alt)

    else:
        continue

if tick_index>0:
    worksheet = workbook.add_worksheet('First ID')
    worksheet.write('A1', 'first id')
    worksheet.write('B2', '[s]')
    worksheet.write('B1', 'time')
    worksheet.write('F1', 'ax')
    worksheet.write('F2', '[pix/s^2]')
    worksheet.write('G1', 'ay')
    worksheet.write('G2', '[pix/s^2]')
    worksheet.write('H1', 'sizex\'')
    worksheet.write('H2', '[pix]')
    worksheet.write('I1', 'sizey\'')
    worksheet.write('I2', '[pix]')
    worksheet.write('J1', 'brightness')
    worksheet.write('C2', '[deg]')
    worksheet.write('C1', 'rotation')
    worksheet.write('D1', 'x')
    worksheet.write('D2', '[pix]')
    worksheet.write('E1', 'y')
    worksheet.write('E2', '[pix]')
    for row in range(1,max_tick+1):
        worksheet.write(row+1, 1, arr_result[(1-1, row+1-1, 1-1)])
        worksheet.write(row+1, 5, arr_result[(1-1, row+1-1, 6-1)])
        worksheet.write(row+1, 6, arr_result[(1-1, row+1-1, 7-1)])
        worksheet.write(row+1, 7, arr_result[(1-1, row+1-1, 8-1)])
        worksheet.write(row+1, 8, arr_result[(1-1, row+1-1, 9-1)])
        worksheet.write(row+1, 9, arr_result[(1-1, row+1-1, 10-1)])
        worksheet.write(row+1, 2, arr_result[(1-1, row+1-1, 5-1)])
        worksheet.write(row + 1, 3, arr_result[(1 - 1, row + 1 - 1, 3 - 1)])
        worksheet.write(row + 1, 4, arr_result[(1 - 1, row + 1 - 1, 4  - 1)])
    worksheet = workbook.add_worksheet('Second ID')
    worksheet.write('A1', 'second id')
    worksheet.write('B2', '[s]')
    worksheet.write('B1', 'time')
    worksheet.write('F1', 'ax')
    worksheet.write('F2', '[pix/s^2]')
    worksheet.write('G1', 'ay')
    worksheet.write('G2', '[pix/s^2]')
    worksheet.write('C2', '[deg]')
    worksheet.write('C1', 'rotation')
    worksheet.write('D1', 'x')
    worksheet.write('D2', '[pix]')
    worksheet.write('E1', 'y')
    worksheet.write('E2', '[pix]')
    worksheet.write('H1', 'sizex\'')
    worksheet.write('H2', '[pix]')
    worksheet.write('I1', 'sizey\'')
    worksheet.write('I2', '[pix]')

    for row in range(1,max_tick+1):
        worksheet.write(row+1, 1, arr_result[(2-1, row+1-1, 1-1)])
        worksheet.write(row+1, 5, arr_result[(2-1, row+1-1, 6-1)])
        worksheet.write(row+1, 6, arr_result[(2-1, row+1-1, 7-1)])
        worksheet.write(row+1, 7, arr_result[(1-1, row+1-1, 8-1)])
        worksheet.write(row+1, 8, arr_result[(1-1, row+1-1, 9-1)])
        worksheet.write(row+1, 2, arr_result[(2-1, row+1-1, 5-1)])
        worksheet.write(row + 1, 3, arr_result[(2 - 1, row + 1 - 1, 3 - 1)])
        worksheet.write(row + 1, 4, arr_result[(2 - 1, row + 1 - 1, 4  - 1)])

    workbook.close()
    print('Data saved to Excel file in folder: '+os.getcwd())
else:
    print("program ended")
    if ipAddress == "0":
        localcamera.release()
    elif ipAddress == "-1":
        movieFile.release()
    exit(0)