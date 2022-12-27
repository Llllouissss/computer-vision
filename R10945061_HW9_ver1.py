import sys
import numpy as np
import cv2
import math

def MaxMagnitude(pixel, mask):
	num = len(mask)
	sizeY = len(mask[0])
	sizeX = len(mask[0][0])
	magnitude = []

	for i in range(num):
		r = 0
		for y in range(sizeY):
			for x in range(sizeX):
				r += pixel[y][x] * mask[i][y][x]

		magnitude.append(r)

	return max(magnitude)	

def Magnitude(pixel, mask):
	num = len(mask)
	sizeY = len(mask[0])
	sizeX = len(mask[0][0])
	magnitude = []

	for i in range(num):
		r = 0
		for y in range(sizeY):
			for x in range(sizeX):
				r += pixel[y][x] * mask[i][y][x]

		magnitude.append(r**2)

	return math.sqrt(sum(magnitude))

def Robert(img, threshold):
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-1,0],[0,1]],[[0,-1],[1,0]]]
	for y in range(img.shape[0]-1):
		for x in range(img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new

def Prewitt(img, threshold):
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-1,-1,-1],[0,0,0],[1,1,1]],[[-1,0,1],[-1,0,1],[-1,0,1]]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new

def Sobel(img, threshold):
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-1,-2,-1],[0,0,0],[1,2,1]],[[-1,0,1],[-2,0,2],[-1,0,1]]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new	

def Frei_and_Chen(img, threshold):
	value = math.sqrt(2)
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-1,-value,-1],[0,0,0],[1,value,1]],[[-1,0,1],[-value,0,value],[-1,0,1]]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = Magnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new	

def Kirsch(img, threshold):
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
					[[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
					[[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
					[[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
					[[5, -3, -3], [5, 0, -3], [5, -3, -3]],
					[[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
					[[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
					[[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = MaxMagnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new	

def Robinson(img, threshold):
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
					[[0, -1, -2], [1, 0, -1], [2, 1, 0]],
					[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
					[[2, 1, 0], [1, 0, -1], [0, -1, -2]],
					[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
					[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
					[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
					[[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]
	for y in range(1,img.shape[0]-1):
		for x in range(1,img.shape[1]-1):
			neighbors = []
			neighbors.append([img[y-1][x-1],img[y-1][x],img[y-1][x+1]])
			neighbors.append([img[y][x-1],img[y][x],img[y][x+1]])
			neighbors.append([img[y+1][x-1],img[y+1][x],img[y+1][x+1]])
			G = MaxMagnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new	

def NevatiaBabu(img, threshold):
	img_new = np.full(img.shape, 255, np.int)
	mask = [[[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100]],
	[[100, 100, 100, 32, -100], [100, 100, 92, -78, -100], [100, 100, 0, -100, -100], [100, 78, -92, -100, -100], [100, -32, -100, -100, -100]],		
	[[100, 100, 100, 100, 100], [100, 100, 100, 78, -32], [100, 92, 0, -92, -100], [32, -78, -100, -100, -100], [-100, -100, -100, -100, -100]],
	[[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]],
	[[100, 100, 100, 100, 100], [-32, 78, 100, 100, 100], [-100, -92, 0, 92, 100], [-100, -100, -100, -78, 32], [-100, -100, -100, -100, -100]],
	[[-100, 32, 100, 100, 100], [-100, -78, 92, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, -92, 78, 100], [-100, -100, -100, -32, 100]]]
	for y in range(2,img.shape[0]-2):
		for x in range(2,img.shape[1]-2):
			neighbors = []
			neighbors.append([img[y-2][x-2],img[y-2][x-1],img[y-2][x],img[y-2][x+1],img[y-2][x+2]])
			neighbors.append([img[y-1][x-2],img[y-1][x-1],img[y-1][x],img[y-1][x+1],img[y-1][x+2]])
			neighbors.append([img[y][x-2],img[y][x-1],img[y][x],img[y][x+1],img[y][x+2]])
			neighbors.append([img[y+1][x-2],img[y+1][x-1],img[y+1][x],img[y+1][x+1],img[y+1][x+2]])
			neighbors.append([img[y+2][x-2],img[y+2][x-1],img[y+2][x],img[y+2][x+1],img[y+2][x+2]])
			G = MaxMagnitude(neighbors, mask)

			if G > threshold:
				img_new[y][x] = 0
			else:
				img_new[y][x] = 255

	return img_new

def main():
	assert len(sys.argv) == 2

	img = cv2.imread('lena.bmp', 0)

	if sys.argv[1] == 'Robert':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = Robert(img, threshold)
		cv2.imwrite('Robert_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Prewitt':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = Prewitt(img, threshold)
		cv2.imwrite('Prewitt_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Sobel':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = Sobel(img, threshold)
		cv2.imwrite('Sobel_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'FAC':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = Frei_and_Chen(img, threshold)
		cv2.imwrite('FreiandChen_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Kirsch':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = Kirsch(img, threshold)
		cv2.imwrite('Kirsch_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'Robinson':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = Robinson(img, threshold)
		cv2.imwrite('Robinson_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if sys.argv[1] == 'NevatiaBabu':
		threshold = input('Enter Threshold: ')
		threshold = int(threshold)
		answer = NevatiaBabu(img, threshold)
		cv2.imwrite('Nevatia_Babu_%s_Lena.bmp' %(threshold), answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
    main()