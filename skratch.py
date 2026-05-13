import cv2 as cv 
import numpy as np 
import os 

Colour_Range = [
    [(0, 0, 0), (255, 255, 20), "BLACK", 0, (0, 0, 0)],
    [(0, 90, 10), (15, 250, 100), "BROWN", 1, (0, 51, 102)],
    [(0, 30, 80), (10, 255, 200), "RED", 2, (0, 0, 255)],
    [(5, 150, 150), (15, 235, 250), "ORANGE", 3, (0, 128, 255)],  # ok
    [(50, 100, 100), (70, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(45, 100, 50), (75, 255, 255), "GREEN", 5, (0, 255, 0)],  # ok
    [(100, 150, 0), (140, 255, 255), "BLUE", 6, (255, 0, 0)],  # ok
    [(120, 40, 100), (140, 250, 220), "VIOLET", 7, (255, 0, 127)],
    [(0, 0, 50), (179, 50, 80), "GRAY", 8, (128, 128, 128)],
    [(0, 0, 90), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
]


Red_top_low = (160, 30, 80)
Red_top_high = (179, 255, 200)


def Best_k_filling (binary_image):

    Dtransform=cv.distanceTransform(binary_image, distanceType=cv.DIST_L2, maskSize=5)

    _, maxValue, _, _ = cv.minMaxLoc(Dtransform)
    
    if int(maxValue) % 2 ==0 :
        maxValue +=1 
    if  int(maxValue) < 3  :
        maxValue = 3 


    k_fill = cv.getStructuringElement(cv.MORPH_RECT, (int(maxValue*0.9), int(maxValue*0.9)) )  

    return k_fill

def find_color_band(image):
    image_hight, image_width = image.shape[:2] # for valid contours

    image_biltrate= cv.bilateralFilter(image, 40, 90, 90) # the values was taken from websites 
    image_gray    = cv.cvtColor(image_biltrate, cv.COLOR_BGR2GRAY)
    image_hsv     = cv.cvtColor(image_biltrate, cv.COLOR_BGR2HSV)
    thr= cv.adaptiveThreshold(image_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 79, 3) # the values was taken from websites 
    thr_inv= cv.bitwise_not(thr)
    cv.imshow("test", thr_inv)

    all_bands=[]

    for  i,clr in enumerate(Colour_Range):
        mask = cv.inRange(image_hsv,clr[0], clr[1])
        if clr[2] == "RED":
            red_mask= cv.inRange(image_hsv, Red_top_low, Red_top_high)
            mask= cv.bitwise_or(mask, red_mask)
        mask=cv.bitwise_and(mask, thr_inv, mask)
        cv.imshow("mask", mask)

        contour_s, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        
        for cont in contour_s :
            if validContours(cont, image_hight, image_width):
                cv.drawContours(image, [cont], -1, (0, 255, 0), 2)

                band=[]
                band.append(clr[2])
                band.append(clr[3])
    

        
                x, y, _, _ = cv.boundingRect(cont) # I need to make a function to return a valid countor 

                band.append(( x, y ))
                all_bands.append(band)
                
    
    
    rearrange_bands= sorted(all_bands, key=lambda all_bands: all_bands[2][0], reverse=False)  #final step rearrange the bands to ready to extract values 
    print(rearrange_bands)
    cv.imshow("the final result", image)
    

    
    cv.waitKey(0)
    cv.destroyAllWindows()


def validContours(countors, img_h, img_w):
    area = cv.contourArea(countors)
    x, y, w, h = cv.boundingRect(countors)
    
    min_area = (img_h * img_w) * 0.003 
    
    aspect_ratio = float(w) / h
    
    extent = float(area) / (w * h)

    if area > min_area and aspect_ratio < 0.85 and extent > 0.5:
        return True
    
    return False
    


    





image= cv.imread(r"C:\Users\hp\Downloads\python\git_hup_resistor.jpg")


imag_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY )

# filtration 
filter= cv.bilateralFilter(imag_gray, 9, sigmaColor=10, sigmaSpace=10 )
cv.imshow("filter", filter)

ret1, th1 =cv.threshold(filter, None, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU) 
# cv.imshow("th1", th1)


#--------------------filling   ---------------------

k_fill = Best_k_filling(th1)
filling_image = cv.morphologyEx(th1, cv.MORPH_CLOSE, k_fill)
# cv.imshow("filling_image", filling_image)



#--------------------distance transform  ---------------------
Dtransform=cv.distanceTransform(filling_image, distanceType=cv.DIST_L2, maskSize=5)

minValue, maxValue, minLoc, maxLoc = cv.minMaxLoc(Dtransform)


N_image= cv.normalize(Dtransform, None, 0, 255, cv.NORM_MINMAX) #ignor  for the dista_th
D_image= np.uint8(N_image)
# cv.imshow("D_image", D_image)

# distance threshold
ret2, th_distance =cv.threshold(Dtransform, maxValue*0.35, 255, cv.THRESH_BINARY ) # it maybe deleted 

N_image= cv.normalize(th_distance, None, 0, 255, cv.NORM_MINMAX) #ignor  for the dista_th
th_distance_image= np.uint8(N_image)
# cv.imshow("th_distance_image",th_distance_image )


# Find a contor 
contor, hierarchy =cv.findContours(th_distance_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if len (contor) >0 :

    c_max= max(contor, key=cv.contourArea)

    rectangle= cv.minAreaRect(c_max)
    center, (w_rect, h_rect), angle =rectangle




#------------------ check if the image is horizantal-----------------------------    
    if w_rect < h_rect :
        angle= angle+90
        w_rect, h_rect= h_rect, w_rect

    
    h_img, w_img =image.shape[:2]

    M=cv.getRotationMatrix2D(center, angle,1 )

    rotating_image= cv.warpAffine(image, M, ( w_img,h_img) )


#--------------------croping a image ---------------------
crop_image = cv.getRectSubPix(rotating_image, (int(w_rect  ), int(h_rect*1.3 )), center )

cv.imshow("crop_image",crop_image)


find_color_band(crop_image)




cv.waitKey(0)
cv.destroyAllWindows()