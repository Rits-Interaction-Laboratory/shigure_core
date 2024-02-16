# coding: utf-8

# # カメラキャプチャ＆基本的なフィルタ処理サンプル

import sys
import cv2
import numpy as np
import random
import time



def increase_param():
    global gaussian_scale, median_scale, bilateral_scale
    gaussian_scale += 6
    if gaussian_scale > 149:
        gaussian_scale = 149
    median_scale += 6
    if median_scale > 199:
        median_scale = 199
    bilateral_scale += 5
    if bilateral_scale > 30:
        bilateral_scale = 30

def decrease_param():
    global gaussian_scale, median_scale, bilateral_scale
    gaussian_scale -= 6
    if gaussian_scale < 1:
        gaussian_scale = 1
    median_scale -= 6
    if median_scale < 1:
        median_scale = 1
    bilateral_scale -= 6
    if bilateral_scale < 1:
        bilateral_scale = 1

def reset_param():
    global gaussian_scale, median_scale, bilateral_scale
    gaussian_scale = 1


def NDI_setup():
    try:
        global ndi
        import NDIlib as ndi
    except ModuleNotFoundError as e:
        print(e)
        print("NDI module not found, so this function unavailable.")
        return False, None, None

    if not ndi.initialize():
        print("NDI module found but cannot be initialized.")
        return False, None, None
    else:
        print("NDI successfully initialized.")

    send_settings = ndi.SendCreate()
    send_settings.ndi_name = "ndi-python"
    ndi_send = ndi.send_create(send_settings)
    ndi_frame = ndi.VideoFrameV2()

    return True, ndi_send, ndi_frame


def NDI_finish(ndi_send):
    if ndi_send != None:
        ndi.send_destroy(ndi_send)
        ndi.destroy()
        return True
    else:
        return False

#
# main
#

#if( len(sys.argv) == 1 ):
	#cam = 0
#else :
	#cam = int(sys.argv[1])
	
#print("camera device num: %d"%cam)

cap = cv2.VideoCapture('/home/azuma/ros2_ws/src/dir/data/movie/sample2.mp4')

print(type(cap))
print(cap.isOpened())

cap.set(cv2.CAP_PROP_FPS,10)

print("FPS:%f"%cap.get(cv2.CAP_PROP_FPS))
print("Image Height:%d"%int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Image Width:%d"%int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("FOURCC:%d"%int(cap.get(cv2.CAP_PROP_FOURCC)))

proc = "color"
ndi_send = None

count = 0
ret,img = cap.read()
if ret == False:
    print("cam not captured")
    sys.exit(1)

height = img.shape[0]
print(height )
width = img.shape[1]
print(width)
img = cv2.resize(img,((int)(width/2),(int)(height/2)))

img2 = img[65:225, 100:200]

prvs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
obj2 = prvs[65:225, 100:200]
#
## for HSV color space transform
#
hsv = np.zeros_like(img2)
hsv[...,1] = 255
colors_hsv = [[0,0,0]]
for i in range(1, 256):
    colors_hsv.append(np.array([random.randint(0,180), random.randint(120,255), 255]))
colors_hsv = np.array(colors_hsv).astype(np.uint8)

#
## for facedetection by OpenCV Haar-like feature based face detector
#
# haarcascade_frontalface_default.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_frontalface_default.xml')
# haarcascade_eye.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
eye_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_eye.xml')
eye_tree = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
# haarcascade_upperbody.xml from https://github.com/opencv/opencv/tree/master/data/haarcascades
body_cascade = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_upperbody.xml')


profileface = cv2.CascadeClassifier('/home/azuma/ros2_ws/src/dir/data/haarcascades/haarcascade_profileface.xml')
profile_face = True

#
## for LK tracking
#
#ShiTomasiコーナー検出器のためのパラメータ
lk_fnum = 1000
feature_params = dict( maxCorners = lk_fnum,
                       qualityLevel = 0.001,
                       minDistance = 7,
                       blockSize = 7 )
# Lucas-Kanade法によるオプティカル・フローのためのパラメータ
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# random color map
lk_color = np.random.randint(0,255,(lk_fnum,3))

#
## for Bluring preprocess of input image
#
blur_proc = "gaussian"
gaussian_scale = 1
median_scale = 5
bilateral_scale = 20

#
## for histogram equalization
#
clahe = None


#
## main loop
#

t_prev = time.perf_counter()
while(True):
    #print("FPS",cap.get(cv2.CAP_PROP_FPS))
    #print("Image Height",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print("Image Width ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #print("FOURCC",int(cap.get(cv2.CAP_PROP_FOURCC)))

    t_now = time.perf_counter()
    duration = t_now-t_prev
    t_prev = t_now
    
    # image capture
    ret,img = cap.read()
    
    #count = count + 1
    #print(count)
    #if count % 2 == 0:
       #while True:
          #next_key = cv2.waitKey(80)
          #if next_key != -1 : 
             #key = next_key
          #else:
             #break
             
       #print('cintunue')
       #continue
    if ret == False:
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      ret,img = cap.read()  
      count = 0

    # resize for processing rate
    img = cv2.resize(img,((int)(width/2),(int)(height/2)))


    # the newest key entry accepted, others flushed
    key = -1
    while True:
        next_key = cv2.waitKey(80)
        if next_key != -1 : 
           key = next_key
        else:
           break

    key = key&0xFF 

    if(ret==True):
        if key == ord('q') :
            break;
        elif key == 0 :        # Up arrow
            increase_param()
        elif key == 1 :        # Down arrow
            decrease_param()
        elif key == 27 :       # ESC
            reset_param()
        elif key == ord('G') : # Gaussian blur
            blur_proc = "gaussian"
            reset_param()
        elif key == ord('m') : # Median blur
            blur_proc = "median"
            reset_param()
        elif key == ord('B') : # Bilateral blur
            blur_proc = "bilateral"
            reset_param()
        elif key == ord('g') : # Monocolor
            proc = "gray"
        elif key == ord('c') : # Color
            proc = "color"
        elif key == ord('x') : # Sobel x-gradient
            proc = "sobelx"
        elif key == ord('y') : # Sobel y-gradient
            proc = "sobely"
        elif key == ord('a') : # Sobel angle
            proc = "angle"
        elif key == ord('s') : # Sobel grad
            proc = "sobel"
        elif key == ord('h') : # Hue image
            proc = "hue"
        elif key == ord('b') : # binarize (otsu)
            proc = "binary"
        elif key == ord(':') : # SIFT features
            proc = "sift"
            sift = cv2.SIFT_create()
        elif key == ord('f') : # Fanneback Denseflow
            proc = "denseflow"
        elif key == ord('l') : # Labeling
            proc = "labeling"
        elif key == ord(';') : # Labeling
            proc = "labeling2"
        elif key == ord('t') : # Lukas-Kanade feature tracking
            proc = "lktracking"
            lk_reset = True
        elif key == ord('F') : # Face detection
            proc = "facedetect"
        elif key == ord('U') : # Upper body detection
            proc = "bodydetect"
        elif key == ord('p') : # Profile Face detection
            if proc == "facedetect":
                profile_face = not profile_face
                if profile_face:
                    print("profile face mode")
                else:
                    print("frontal face mode")

        elif key == ord('n') : # NDI camera sending
            if ndi_send == None:
                ret,ndi_send,ndi_frame = NDI_setup()
            else:
                NDI_finish(ndi_send)
                ndi_send = None
        elif key == ord('d') : # DoG (Difference of Gaussian) filter
            proc = "dog"
            blur_proc == "gaussian"
        elif key == ord('e') : # Histogram equalization
            proc = "hist_eq"
            blur_proc == "gaussian"
        elif key == ord('E') : # Constrast Limited Histogram equalization
            proc = "hist_eq_CLAHE"
            blur_proc == "gaussian"

        # blurring preprocessing 
        if blur_proc == "gaussian" :
            img = cv2.GaussianBlur(img.astype(np.float32),(gaussian_scale,gaussian_scale),0).astype(np.uint8)
            #img = cv2.GaussianBlur(img.astype(np.float32),(0,0),gaussian_scale).astype(np.uint8)
        elif blur_proc == "median" :
            img = cv2.medianBlur(img,median_scale)
        elif blur_proc == "bilateral" :
            img = cv2.bilateralFilter(img.astype(np.float32),bilateral_scale,75,75).astype(np.uint8)


        # image processing
        if proc == "color" :
            result = img
        elif proc == "gray" : 
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif proc == "sobelx" :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
            result = np.abs(dx*10)
            result = np.where(result>255,255,result).astype(np.uint8)
        elif proc == "sobely" :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
            result = np.abs(dy*10)
            result = np.where(result>255,255,result).astype(np.uint8)
        elif proc == "sobel" :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
            dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
            result = np.sqrt((dx*dx+dy*dy)/2)*10
            result = np.where(result>255,255,result).astype(np.uint8)
        elif proc == "angle" :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
            dy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
            #result = np.arctan2(dy,dx)
            h = gray
            h[:] = np.arctan2(dy[:],dx[:])/np.pi*128
            h = np.where(h<0, h+128, h)
            h = np.where(h>255, 255, h).astype(np.uint8)
            s = np.full(h.shape,255,np.uint8)
            #v = np.full(h.shape,255,np.uint8)
            v = np.sqrt((dx*dx+dy*dy)/2)*10
            v = np.where(v>255,255,v).astype(np.uint8)
            result = img
            result[:,:,0] = h
            result[:,:,1] = s
            result[:,:,2] = v
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
        elif proc == "hue" :
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = 255
            hsv[:,:,2] = 255
            result = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        elif proc =="binary" :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, result = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif proc == "sift" :
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kp = sift.detect(gray,None)
            #img=cv2.drawKeypoints(gray,kp,img)
            result = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        elif proc == "denseflow" :

            next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            obj = next[65:225, 100:200]
            flow = cv2.calcOpticalFlowFarneback(obj2,obj, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # colored denseflow
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            result = cv2.addWeighted(img2,0.5,rgb,0.5,0)
            # sparse arrow
            flowx = mag*np.cos(ang)
            flowy = mag*np.sin(ang)
            #result = (np.minimum(np.abs(flowx)*100,255)).astype(np.uint8)
            for i in range(0,result.shape[0],15):
                for j in range(0,result.shape[1],15):
                    cv2.line(result,(j,i),(int(j+flowx[i,j]*3),int(i+flowy[i,j]*3)),
                             (255,255,255),1)
            prvs = next
        elif proc == "labeling" :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            labelnum, label = cv2.connectedComponents(bin)
            #print(label.shape)
            #print(label.dtype)
            result = np.zeros_like(img)
            result = colors_hsv[label%256]
            #for i in range(label.shape[0]):
            #    for j in range(label.shape[1]):
            #        result[i,j] = colors_hsv[label[i,j]%256]
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
        elif proc == "labeling2" :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            num, label, stats, centroids = cv2.connectedComponentsWithStats(bin)
            result = np.zeros_like(img)
            result = colors_hsv[label%256]
            result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)
            for i in range(1,num):
                if stats[i][4] < 100:
                    continue
                x0 = stats[i][0]
                y0 = stats[i][1]
                x1 = stats[i][0]+stats[i][2]
                y1 = stats[i][1]+stats[i][3]
                cv2.rectangle(result,(x0,y0),(x1,y1),(0,0,255))
                cv2.putText(result,"ID: "+str(i+1),(x0,y1+15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
                cv2.putText(result,"S: "+str(stats[i][4]),(x0,y1+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
                cv2.putText(result,"X: "+str(int(centroids[i][0])),(x1-10,y1+15),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
                cv2.putText(result,"Y: "+str(int(centroids[i][1])),(x1-10,y1+30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
        elif proc == "facedetect" :
            result = img.copy()
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                result = cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = result[y:y+h, x:x+w]
                #eyes = eye_cascade.detectMultiScale(roi_gray)
                eyes = eye_tree.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            if profile_face:
                flipped_gray = cv2.flip(gray,1) # flip horizontally
                profiles = profileface.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in profiles:
                    result = cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),2)
                profiles = profileface.detectMultiScale(flipped_gray, 1.3, 5)
                for (x,y,w,h) in profiles:
                    H,W = flipped_gray.shape
                    result = cv2.rectangle(result,(W-1-x-w,y),(W-1-x,y+h),(0,0,255),2)
        elif proc == "bodydetect" :
            result = img.copy()
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in bodies:
                result = cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)

        elif proc == "lktracking" :
            next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if( lk_reset == True ):
                prvs = next
                track_list = []
                obj = next[65:225, 15:200]
                p0 = cv2.goodFeaturesToTrack(obj, mask = None, **feature_params)
                lk_reset = False
            else :
                p0 = good_new.reshape(-1,1,2)

            # オプティカル・フローを計算
            p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next, p0, None, **lk_params)

            # 良い特徴点を選択
            good_new = p1[st==1]
            good_old = p0[st==1]

            # 物体追跡を描画
            track = np.zeros_like(img)
            result = img.copy()
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                cv2.line(track, (int(a),int(b)),(int(c),int(d)), lk_color[i].tolist(), 2)
                result = cv2.circle(result,(int(a),int(b)),3,lk_color[i].tolist(),-1)

            track_list.append(track)
            if( len(track_list)>20 ):
                track_list.pop(0)

            for t in track_list :
                result = np.where(t!=0,t,result)

            prvs = next

        elif proc == "dog" :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blurred_gray = cv2.GaussianBlur(gray,(gaussian_scale*2+1,gaussian_scale*2+1),0)
            #blurred_gray = cv2.GaussianBlur(gray,(0,0),gaussian_scale*2+1)
            result = np.clip(((gray - blurred_gray)*3.0+128.0), a_min=0, a_max=255).astype(np.uint8)

        elif proc == "hist_eq" :
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            result = equ

        elif proc == "hist_eq_CLAHE" :
            if clahe is None:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = clahe.apply(gray)
            
        result = cv2.resize(result,(width,height))
        cv2.putText(result,
                    text="FPS:%f"%(1./duration),org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,color=(0,255,0),thickness=2,lineType=cv2.LINE_4)
        cv2.namedWindow('OpenCV Capture', cv2.WINDOW_NORMAL)
        cv2.imshow("OpenCV Capture", result)

        # send image to NDI camera
        if ndi_send != None:
            ndi_img = cv2.cvtColor(result,cv2.COLOR_BGR2BGRA)
            ndi_frame.data = ndi_img
            ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            ndi.send_send_video_v2(ndi_send, ndi_frame)
        
if ndi_send != None:
    NDI_finish(ndi_send)       

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
