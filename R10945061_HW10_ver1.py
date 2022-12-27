import sys
import numpy as np
import cv2
import math

def Magnitude(pixel, mask, alpha):
	sizeY = len(mask)
	sizeX = len(mask[0])
	result = 0

	for y in range(sizeY):
		for x in range(sizeX):
			result += pixel[y][x] * mask[y][x]

	result *= alpha

	return result

def Laplace_Mask1(img, threshold):
	alpha = 1
	img_label = np.full(img.shape, 255, np.int)
	mask = [[0,1,0],[1,-4,1],[0,1,0]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask, alpha)

			if G > threshold:
				img_label[y][x] = 1
			elif G < -threshold:
				img_label[y][x] = -1
			else:
				img_label[y][x] = 0

	return img_label

def Laplace_Mask2(img, threshold):
	alpha = 1/3.0
	img_label = np.full(img.shape, 255, np.int)
	mask = [[1,1,1],[1,-8,1],[1,1,1]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask, alpha)

			if G > threshold:
				img_label[y][x] = 1
			elif G < -threshold:
				img_label[y][x] = -1
			else:
				img_label[y][x] = 0

	return img_label

def Minimum_Variance_Laplacian(img, threshold):
	alpha = 1/3.0
	img_label = np.full(img.shape, 255, np.int)
	mask = [[2,-1,2],[-1,-4,-1],[2,-1,2]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask, alpha)

			if G > threshold:
				img_label[y][x] = 1
			elif G < -threshold:
				img_label[y][x] = -1
			else:
				img_label[y][x] = 0

	return img_label

def Laplace_Gaussian(img, threshold):
	alpha = 1
	img_label = np.full(img.shape, 255, np.int)
	mask = [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
		[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
		[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
		[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
		[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
		[-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
		[-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
		[-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
		[0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
		[0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
		[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]]
	for y in range(5,img.shape[0]-5):
		for x in range(5,img.shape[1]-5):
			neighbors = []
			neighbors.append([img[y-5][x-5],img[y-5][x-4],img[y-5][x-3],img[y-5][x-2],img[y-5][x-1],img[y-5][x],img[y-5][x+1],img[y-5][x+2],img[y-5][x+3],img[y-5][x+4],img[y-5][x+5]])
			neighbors.append([img[y-4][x-5],img[y-4][x-4],img[y-4][x-3],img[y-4][x-2],img[y-4][x-1],img[y-4][x],img[y-4][x+1],img[y-4][x+2],img[y-4][x+3],img[y-4][x+4],img[y-4][x+5]])
			neighbors.append([img[y-3][x-5],img[y-3][x-4],img[y-3][x-3],img[y-3][x-2],img[y-3][x-1],img[y-3][x],img[y-3][x+1],img[y-3][x+2],img[y-3][x+3],img[y-3][x+4],img[y-3][x+5]])
			neighbors.append([img[y-2][x-5],img[y-2][x-4],img[y-2][x-3],img[y-2][x-2],img[y-2][x-1],img[y-2][x],img[y-2][x+1],img[y-2][x+2],img[y-2][x+3],img[y-2][x+4],img[y-2][x+5]])
			neighbors.append([img[y-1][x-5],img[y-1][x-4],img[y-1][x-3],img[y-1][x-2],img[y-1][x-1],img[y-1][x],img[y-1][x+1],img[y-1][x+2],img[y-1][x+3],img[y-1][x+4],img[y-1][x+5]])
			neighbors.append([img[y][x-5],img[y][x-4],img[y][x-3],img[y][x-2],img[y][x-1],img[y][x],img[y][x+1],img[y][x+2],img[y][x+3],img[y][x+4],img[y][x+5]])
			neighbors.append([img[y+1][x-5],img[y+1][x-4],img[y+1][x-3],img[y+1][x-2],img[y+1][x-1],img[y+1][x],img[y+1][x+1],img[y+1][x+2],img[y+1][x+3],img[y+1][x+4],img[y+1][x+5]])
			neighbors.append([img[y+2][x-5],img[y+2][x-4],img[y+2][x-3],img[y+2][x-2],img[y+2][x-1],img[y+2][x],img[y+2][x+1],img[y+2][x+2],img[y+2][x+3],img[y+2][x+4],img[y+2][x+5]])
			neighbors.append([img[y+3][x-5],img[y+3][x-4],img[y+3][x-3],img[y+3][x-2],img[y+3][x-1],img[y+3][x],img[y+3][x+1],img[y+3][x+2],img[y+3][x+3],img[y+3][x+4],img[y+3][x+5]])
			neighbors.append([img[y+4][x-5],img[y+4][x-4],img[y+4][x-3],img[y+4][x-2],img[y+4][x-1],img[y+4][x],img[y+4][x+1],img[y+4][x+2],img[y+4][x+3],img[y+4][x+4],img[y+4][x+5]])
			neighbors.append([img[y+5][x-5],img[y+5][x-4],img[y+5][x-3],img[y+5][x-2],img[y+5][x-1],img[y+5][x],img[y+5][x+1],img[y+5][x+2],img[y+5][x+3],img[y+5][x+4],img[y+5][x+5]])
			G = Magnitude(neighbors, mask, alpha)

			if G > threshold:
				img_label[y][x] = 1
			elif G < -threshold:
				img_label[y][x] = -1
			else:
				img_label[y][x] = 0

	return img_label

def Difference_Gaussian(img, threshold):
	alpha = 1
	img_label = np.full(img.shape, 255, np.int)
	mask =  [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
		[-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
		[-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
		[-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
		[-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
		[-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
		[-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
		[-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
		[-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
		[-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
		[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]
	for y in range(5,img.shape[0]-5):
		for x in range(5,img.shape[1]-5):
			neighbors = []
			neighbors.append([img[y-5][x-5],img[y-5][x-4],img[y-5][x-3],img[y-5][x-2],img[y-5][x-1],img[y-5][x],img[y-5][x+1],img[y-5][x+2],img[y-5][x+3],img[y-5][x+4],img[y-5][x+5]])
			neighbors.append([img[y-4][x-5],img[y-4][x-4],img[y-4][x-3],img[y-4][x-2],img[y-4][x-1],img[y-4][x],img[y-4][x+1],img[y-4][x+2],img[y-4][x+3],img[y-4][x+4],img[y-4][x+5]])
			neighbors.append([img[y-3][x-5],img[y-3][x-4],img[y-3][x-3],img[y-3][x-2],img[y-3][x-1],img[y-3][x],img[y-3][x+1],img[y-3][x+2],img[y-3][x+3],img[y-3][x+4],img[y-3][x+5]])
			neighbors.append([img[y-2][x-5],img[y-2][x-4],img[y-2][x-3],img[y-2][x-2],img[y-2][x-1],img[y-2][x],img[y-2][x+1],img[y-2][x+2],img[y-2][x+3],img[y-2][x+4],img[y-2][x+5]])
			neighbors.append([img[y-1][x-5],img[y-1][x-4],img[y-1][x-3],img[y-1][x-2],img[y-1][x-1],img[y-1][x],img[y-1][x+1],img[y-1][x+2],img[y-1][x+3],img[y-1][x+4],img[y-1][x+5]])
			neighbors.append([img[y][x-5],img[y][x-4],img[y][x-3],img[y][x-2],img[y][x-1],img[y][x],img[y][x+1],img[y][x+2],img[y][x+3],img[y][x+4],img[y][x+5]])
			neighbors.append([img[y+1][x-5],img[y+1][x-4],img[y+1][x-3],img[y+1][x-2],img[y+1][x-1],img[y+1][x],img[y+1][x+1],img[y+1][x+2],img[y+1][x+3],img[y+1][x+4],img[y+1][x+5]])
			neighbors.append([img[y+2][x-5],img[y+2][x-4],img[y+2][x-3],img[y+2][x-2],img[y+2][x-1],img[y+2][x],img[y+2][x+1],img[y+2][x+2],img[y+2][x+3],img[y+2][x+4],img[y+2][x+5]])
			neighbors.append([img[y+3][x-5],img[y+3][x-4],img[y+3][x-3],img[y+3][x-2],img[y+3][x-1],img[y+3][x],img[y+3][x+1],img[y+3][x+2],img[y+3][x+3],img[y+3][x+4],img[y+3][x+5]])
			neighbors.append([img[y+4][x-5],img[y+4][x-4],img[y+4][x-3],img[y+4][x-2],img[y+4][x-1],img[y+4][x],img[y+4][x+1],img[y+4][x+2],img[y+4][x+3],img[y+4][x+4],img[y+4][x+5]])
			neighbors.append([img[y+5][x-5],img[y+5][x-4],img[y+5][x-3],img[y+5][x-2],img[y+5][x-1],img[y+5][x],img[y+5][x+1],img[y+5][x+2],img[y+5][x+3],img[y+5][x+4],img[y+5][x+5]])
			G = Magnitude(neighbors, mask, alpha)

			if G > threshold:
				img_label[y][x] = 1
			elif G < -threshold:
				img_label[y][x] = -1
			else:
				img_label[y][x] = 0

	return img_label

def CheckNeighbors(label, size):
	img_new = np.full(label.shape, 255, np.int)
	rawSize = label.shape[0]
	half = size[0]//2

	for y in range(label.shape[0]):
		for x in range(label.shape[1]):
			img_new[y][x] = 255
			#check neighbors
			if label[y][x] == 1:
				for row in range(-half,half+1):
					for col in range(-half,half+1):
						if y+row >= 0 and y+row <= rawSize-1 and x+col >= 0 and x+col <= rawSize-1:
							if label[y+row][x+col] == -1:
								img_new[y][x] = 0

	return img_new

def main():
	assert len(sys.argv) == 2

	img = cv2.imread('lena.bmp', 0)

	if sys.argv[1] == 'Laplace1':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		label = Laplace_Mask1(img, threshold)
		answer = CheckNeighbors(label,[3,3])
		cv2.imwrite('LaplaceMask1_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Laplace2':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		label = Laplace_Mask2(img, threshold)
		answer = CheckNeighbors(label,[3,3])
		cv2.imwrite('LaplaceMask2_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Minimum':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		label = Minimum_Variance_Laplacian(img, threshold)
		answer = CheckNeighbors(label,[3,3])
		cv2.imwrite('MinimumVarianceLaplacian_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Gaussian':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		label = Laplace_Gaussian(img, threshold)
		answer = CheckNeighbors(label,[11,11])
		cv2.imwrite('LaplaceGaussian_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'DoG':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		label = Difference_Gaussian(img, threshold)
		answer = CheckNeighbors(label,[11,11])
		cv2.imwrite('DoG_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
    main()