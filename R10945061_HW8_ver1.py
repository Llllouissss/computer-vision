import sys
import numpy as np
import cv2
import os
import math

def dilation(img):
	dil = np.zeros(img.shape, np.int)
	kernel = [[-2,-1],[-2,0],[-2,1],
			[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
			[0,-2],[0,-1],[0,0],[0,1],[0,2],
			[1,-2],[1,-1],[1,0],[1,1],[1,2],
			[2,-1],[2,0],[2,1]]
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] > 0:
				max_value = 0
				for item in kernel:
					row, column = item
					if y+row < 0:
						continue
					elif y+row > (img.shape[0]-1):
						continue
					elif x+column < 0:
						continue
					elif x+column > (img.shape[1]-1):
						continue
					else:
						if img[y+row][x+column] > max_value:
							max_value = img[y+row][x+column]
				#change the max_value to all elements
				for item in kernel:
					row, column = item
					if y+row < 0:
						continue
					elif y+row > (img.shape[0]-1):
						continue
					elif x+column < 0:
						continue
					elif x+column > (img.shape[1]-1):
						continue
					else:
						dil[y+row][x+column] = max_value

	return dil
def erosion(img):
	ero = np.zeros(img.shape, np.int)
	kernel = [[-2,-1],[-2,0],[-2,1],
			[-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
			[0,-2],[0,-1],[0,0],[0,1],[0,2],
			[1,-2],[1,-1],[1,0],[1,1],[1,2],
			[2,-1],[2,0],[2,1]]
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] > 0:
				white = True
				min_value = np.inf
				for item in kernel:
					row, column = item
					if y+row >= 0 and y+row < (img.shape[0]) and x+column >= 0 and x+column < (img.shape[1]):
						if img[y+row][x+column] == 0:
							white = False
							break
						if img[y+row][x+column] < min_value:
							min_value = img[y+row][x+column]

				if white:
					ero[y][x] = min_value

	return ero

def opening(img):
	img_ero = erosion(img)
	img_open = dilation(img_ero)

	return img_open

def closing(img):
	img_dil = dilation(img)
	img_close = erosion(img_dil)

	return img_close

def gaussian(img):
	mu = 0
	sigma = 1
	amp10 = img + 10*np.random.normal(mu, sigma, img.shape)
	amp30 = img + 30*np.random.normal(mu, sigma, img.shape)

	return amp10, amp30

def saltpepper(img):
	random1 = np.random.uniform(0, 1, img.shape)
	random2 = np.random.uniform(0, 1, img.shape)
	img_sp005 = np.zeros(img.shape, np.int)
	img_sp01 = np.zeros(img.shape, np.int)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if random1[y][x] < 0.05:
				img_sp005[y][x] = 0
			elif random1[y][x] > 0.95:
				img_sp005[y][x] = 255
			else:
				img_sp005[y][x] = img[y][x]

	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if random2[y][x] < 0.1:
				img_sp01[y][x] = 0
			elif random2[y][x] > 0.9:
				img_sp01[y][x] = 255
			else:
				img_sp01[y][x] = img[y][x]

	return img_sp01, img_sp005

def box_filter(img, size):
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			boxlist = []
			localOrigin = (y-(size//2),x-(size//2))
			for box_y in range(size):
				for box_x in range(size):
					count_y = localOrigin[0]+box_y
					count_x = localOrigin[1]+box_x
					if count_y >= 0 and count_y < img.shape[0] and count_x >= 0 and count_x < img.shape[1]:
						boxlist.append(img[count_y][count_x])

			img[y][x] = sum(boxlist)/len(boxlist)

	return img

def median_filter(img, size):
	def median(boxlist):
		half = len(boxlist) // 2
		boxlist.sort()
		if len(boxlist) % 2 == 0:
			return (boxlist[half-1] + boxlist[half]) / 2
		else:
			return boxlist[half]

	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			boxlist = []
			localOrigin = (y-(size//2),x-(size//2))
			for box_y in range(size):
				for box_x in range(size):
					count_y = localOrigin[0]+box_y
					count_x = localOrigin[1]+box_x
					if count_y >= 0 and count_y < img.shape[0] and count_x >= 0 and count_x < img.shape[1]:
						boxlist.append(img[count_y][count_x])

			img[y][x] = median(boxlist)

	return img

def open_close(img):
	img_op = opening(img)
	img_result = closing(img_op)

	return img_result

def close_open(img):
	img_cl = closing(img)
	img_result = opening(img_cl)

	return img_result

def SNR(original, noise):
	us = 0
	vs = 0
	un = 0
	vn = 0
	size = original.shape[0]*original.shape[1]

	for y in range(original.shape[0]):
		for x in range(original.shape[1]):
			us += original[y][x]
	us /= size

	for y in range(original.shape[0]):
		for x in range(original.shape[1]):
			vs += math.pow(original[y][x] - us, 2)
	vs /= size

	for y in range(original.shape[0]):
		for x in range(original.shape[1]):
			un += (noise[y][x] - original[y][x])
	un /= size

	for y in range(original.shape[0]):
		for x in range(original.shape[1]):
			vn += math.pow(noise[y][x] - original[y][x] - un, 2)
	vn /= size

	return 20 * math.log(math.sqrt(vs) / math.sqrt(vn), 10)

def main():
	assert len(sys.argv) == 2

	img = cv2.imread('lena.bmp', 0)

	if not os.path.exists('Gaussian_Noise'):
		os.makedirs('Gaussian_Noise')

	if not os.path.exists('SaltandPepper'):
		os.makedirs('SaltandPepper')

	if not os.path.exists('box_filter'):
		os.makedirs('box_filter')

	if not os.path.exists('median_filter'):
		os.makedirs('median_filter')

	if not os.path.exists('open_and_close'):
		os.makedirs('open_and_close')

	gaussian_10, gaussian_30 = gaussian(img)
	cv2.imwrite('Gaussian_Noise/gaussian_10.bmp', gaussian_10)
	cv2.imwrite('Gaussian_Noise/gaussian_30.bmp', gaussian_30)

	sp_01, sp_005 = saltpepper(img)
	cv2.imwrite('SaltandPepper/saltpepper_01.bmp', sp_01)
	cv2.imwrite('SaltandPepper/saltpepper_005.bmp', sp_005)

	if sys.argv[1] == 'box':
	#box filter
		cv2.imwrite('box_filter/gaussian_10_box3.bmp', box_filter(gaussian_10, 3))
		cv2.imwrite('box_filter/gaussian_30_box3.bmp', box_filter(gaussian_30, 3))
		cv2.imwrite('box_filter/gaussian_10_box5.bmp', box_filter(gaussian_10, 5))
		cv2.imwrite('box_filter/gaussian_30_box5.bmp', box_filter(gaussian_30, 5))
		cv2.imwrite('box_filter/saltpepper_01_box3.bmp', box_filter(sp_01, 3))
		cv2.imwrite('box_filter/saltpepper_005_box3.bmp', box_filter(sp_005, 3))
		cv2.imwrite('box_filter/saltpepper_01_box5.bmp', box_filter(sp_01, 5))
		cv2.imwrite('box_filter/saltpepper_005_box5.bmp', box_filter(sp_005, 5))

		file = open("SNR.txt", "w")
		file.write('box3x3_gaussian_10: ' + str(SNR(img, box_filter(gaussian_10, 3))) + '\n' )
		file.write('box3x3_gaussian_30: ' + str(SNR(img, box_filter(gaussian_30, 3))) + '\n' )
		file.write('box5x5_gaussian_10: ' + str(SNR(img, box_filter(gaussian_10, 5))) + '\n' )
		file.write('box5x5_gaussian_30: ' + str(SNR(img, box_filter(gaussian_30, 5))) + '\n' )
		file.write('box3x3_saltpepper_01: ' + str(SNR(img, box_filter(sp_01, 3))) + '\n' )
		file.write('box3x3_saltpepper_005: ' + str(SNR(img, box_filter(sp_005, 3))) + '\n' )
		file.write('box5x5_saltpepper_01: ' + str(SNR(img, box_filter(sp_01, 5))) + '\n' )
		file.write('box5x5_saltpepper_005: ' + str(SNR(img, box_filter(sp_005, 5))) + '\n' )

	elif sys.argv[1] == 'median':
	#median filter
		cv2.imwrite('median_filter/gaussian_10_median3.bmp', median_filter(gaussian_10, 3))
		cv2.imwrite('median_filter/gaussian_30_median3.bmp', median_filter(gaussian_30, 3))
		cv2.imwrite('median_filter/gaussian_10_median5.bmp', median_filter(gaussian_10, 5))
		cv2.imwrite('median_filter/gaussian_30_median5.bmp', median_filter(gaussian_30, 5))
		cv2.imwrite('median_filter/saltpepper_01_median3.bmp', median_filter(sp_01, 3))
		cv2.imwrite('median_filter/saltpepper_005_median3.bmp', median_filter(sp_005, 3))
		cv2.imwrite('median_filter/saltpepper_01_median5.bmp', median_filter(sp_01, 5))
		cv2.imwrite('median_filter/saltpepper_005_median5.bmp', median_filter(sp_005, 5))

		file = open("SNR.txt", "w")
		file.write('median3x3_gaussian_10: ' + str(SNR(img, median_filter(gaussian_10, 3))) + '\n' )
		file.write('median3x3_gaussian_30: ' + str(SNR(img, median_filter(gaussian_30, 3))) + '\n' )
		file.write('median5x5_gaussian_10: ' + str(SNR(img, median_filter(gaussian_10, 5))) + '\n' )
		file.write('median5x5_gaussian_30: ' + str(SNR(img, median_filter(gaussian_30, 5))) + '\n' )
		file.write('median3x3_saltpepper_01: ' + str(SNR(img, median_filter(sp_01, 3))) + '\n' )
		file.write('median3x3_saltpepper_005: ' + str(SNR(img, median_filter(sp_005, 3))) + '\n' )
		file.write('median5x5_saltpepper_01: ' + str(SNR(img, median_filter(sp_01, 5))) + '\n' )
		file.write('median5x5_saltpepper_005: ' + str(SNR(img, median_filter(sp_005, 5))) + '\n' )

	elif sys.argv[1] == 'opcl':
	#open then close
		cv2.imwrite('open_and_close/gaussian_10_open_close.bmp', open_close(gaussian_10))
		cv2.imwrite('open_and_close/gaussian_30_open_close.bmp', open_close(gaussian_30))
		cv2.imwrite('open_and_close/saltpepper_01_open_close.bmp', open_close(sp_01))
		cv2.imwrite('open_and_close/saltpepper_005_open_close.bmp', open_close(sp_005))
	
	#close then open
		cv2.imwrite('open_and_close/gaussian_10_close_open.bmp', close_open(gaussian_10))
		cv2.imwrite('open_and_close/gaussian_30_close_open.bmp', close_open(gaussian_30))
		cv2.imwrite('open_and_close/saltpepper_01_close_open.bmp', close_open(sp_01))
		cv2.imwrite('open_and_close/saltpepper_005_close_open.bmp', close_open(sp_005))

		file = open("SNR.txt", "w")
		file.write('closingthenopening_gaussian_10: ' + str(SNR(img, close_open(gaussian_10))) + '\n' )
		file.write('closingthenopening_gaussian_30: ' + str(SNR(img, close_open(gaussian_30))) + '\n' )
		file.write('openingthenclosing_gaussian_10: ' + str(SNR(img, open_close(gaussian_10))) + '\n' )
		file.write('openingthenclosing_gaussian_30: ' + str(SNR(img, open_close(gaussian_30))) + '\n' )
		file.write('closingthenopening_saltpepper_01: ' + str(SNR(img, close_open(sp_01))) + '\n' )
		file.write('closingthenopening_saltpepper_005: ' + str(SNR(img, close_open(sp_005))) + '\n' )
		file.write('openingthenclosing_saltpepper_01: ' + str(SNR(img, open_close(sp_01))) + '\n' )
		file.write('openingthenclosing_saltpepper_005: ' + str(SNR(img, open_close(sp_005))) + '\n' )

if __name__ == '__main__':
    main()