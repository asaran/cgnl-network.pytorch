import cv2
import os

img_names = os.listdir('data/ut-lfd/pouring/VD_images/')


for i,name in enumerate(img_names):
	if 'KT2_4_' in name:
		# print(i)
		img = cv2.imread('data/ut-lfd/pouring/VD_images/'+name)
		crop_img = img[1:1080, 400:1920-400]
		# print(img.shape)
		img = cv2.resize(crop_img,None,fx=0.5,fy=1.0)
		
		
		# cv2.imshow("cropped", crop_img)
		# cv2.waitKey(0)
		cv2.imwrite('data/ut-lfd/pouring/VD_images_subsampled_small/'+name,img)