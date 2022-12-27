import sys
import numpy as np
import cv2
import copy

def Thin(img):
	img_bin = np.zeros(img.shape, np.int)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] >= 128:
				img_bin[y][x] = 255

	img_ds = np.zeros((64,64), np.int)
	for y in range(img_ds.shape[0]):
		for x in range(img_ds.shape[1]):
			img_ds[y][x] = img_bin[8*y][8*x]

	img_thin = img_ds

	while True:
		img_thinned_original = copy.deepcopy(img_thin)

		#Step1: mark the interior/border pixels
		#input: original symbolic image
		#output: interior/border image 
		img_ib = interior_border(img_thin)

		#Step2: pair relationship operator
		#input: interior/border image
		#output: marked image
		img_marked = pair_relationship_operator(img_ib)

		#Step3: marked-pixel connected shrink operator
		#input: original symbolic image + marked image
		#output: thinned image
		img_yokoi = yokoi(img_thin)
		for y in range(img_thin.shape[0]):
			for x in range(img_thin.shape[1]):
				if img_yokoi[y][x] == 1 and img_marked[y][x] == 1:
					img_thin[y][x] = 0

		#Use thinned image as next original symbolic image
		#Repeat Step1,2,3 until the last output never changed
		if np.sum(img_thin == img_thinned_original) == img_thin.shape[0] * img_thin.shape[1]:
			break

	return img_thin

def interior_border(img):
	def h(c, d):
		if c == d:
			return c
		else:
			return 'b'

	img_ib = np.zeros(img.shape, np.int)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] > 0:
				x1, x2, x3, x4 = 0, 0, 0, 0
				if y == 0:
					if x == 0:
						x1, x4 = img[y][x+1], img[y+1][x]
					elif x == img.shape[1]-1:
						x3, x4 = img[y][x-1], img[y+1][x]
					else:
						x1, x3, x4 = img[y][x+1], img[y][x-1], img[y+1][x]
				elif y == img.shape[0]-1:
					if x == 0:
						x1, x2 = img[y][x+1], img[y-1][x]
					elif x == img.shape[1]-1:
						x2 ,x3 = img[y-1][x], img[y][x-1]
					else:
						x1, x2, x3 = img[y][x+1], img[y-1][x], img[y][x-1]
				else:
					if x == 0:
						x1, x2, x4 = img[y][x+1], img[y-1][x], img[y+1][x]
					elif x == img.shape[1]-1:
						x2, x3, x4 = img[y-1][x], img[y][x-1], img[y+1][x]
					else:
						x1, x2, x3, x4 = img[y][x+1], img[y-1][x], img[y][x-1], img[y+1][x]

				x1 /= 255
				x2 /= 255
				x3 /= 255
				x4 /= 255
				a1 = h(1, x1)
				a2 = h(a1, x2)
				a3 = h(a2, x3)
				a4 = h(a3, x4)
				if a4 == 'b':
					img_ib[y][x] = 2 # mark border pixel as 2 
				else:
					img_ib[y][x] = 1 # mark interior pixel as 1

	return img_ib

def pair_relationship_operator(img):
	def h(a, i):
		if a == i:
			return 1
		else:
			return 0

	img_marked = np.zeros(img.shape, np.int)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] > 0:
				x1, x2, x3, x4 = 0, 0, 0, 0
				if y == 0:
					if x == 0:
						x1, x4 = img[y][x+1], img[y+1][x]
					elif x == img.shape[1]-1:
						x3, x4 = img[y][x-1], img[y+1][x]
					else:
						x1, x3, x4 = img[y][x+1], img[y][x-1], img[y+1][x]
				elif y == img.shape[0]-1:
					if x == 0:
						x1, x2 = img[y][x+1], img[y-1][x]
					elif x == img.shape[1]-1:
						x2 ,x3 = img[y-1][x], img[y][x-1]
					else:
						x1, x2, x3 = img[y][x+1], img[y-1][x], img[y][x-1]
				else:
					if x == 0:
						x1, x2, x4 = img[y][x+1], img[y-1][x], img[y+1][x]
					elif x == img.shape[1]-1:
						x2, x3, x4 = img[y-1][x], img[y][x-1], img[y+1][x]
					else:
						x1, x2, x3, x4 = img[y][x+1], img[y-1][x], img[y][x-1], img[y+1][x]

				if h(x1, 1) + h(x2, 1) + h(x3, 1) + h(x4, 1) >= 1 and img[y][x] == 2:
					img_marked[y][x] = 1 # can be deleted
				else:
					img_marked[y][x] = 2 # cannot be deleted

	return img_marked

def yokoi(img):
	def yokoi_function(b,c,d,e):
		if b == c and (d != b or e != b):
			return 'q'
		elif b == c and (d == b and e == b):
			return 'r'
		elif b != c:
			return 's'

	ans = [[' ' for x in range(64)] for y in range(64)]

	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] > 0:
				if y == 0:
					if x == 0:
						x7, x2, x6 = 0, 0, 0
						x3, x0, x1 = 0, img[y][x], img[y][x+1]
						x8, x4, x5 = 0, img[y+1][x], img[y+1][x+1]
					elif x == img.shape[1]-1:
						x7, x2, x6 = 0, 0, 0
						x3, x0, x1 = img[y][x-1], img[y][x], 0
						x8, x4, x5 = img[y+1][x-1], img[y+1][x], 0
					else:
						x7, x2, x6 = 0, 0, 0
						x3, x0, x1 = img[y][x-1], img[y][x], img[y][x+1]
						x8, x4, x5 = img[y+1][x-1], img[y+1][x], img[y+1][x+1]
				elif y == img.shape[0]-1:
					if x == 0:
						x7, x2, x6 = 0, img[y-1][x], img[y-1][x+1]
						x3, x0, x1 = 0, img[y][x], img[y][x+1]
						x8, x4, x5 = 0, 0, 0
					elif x == img.shape[1]-1:
						x7, x2, x6 = img[y-1][x-1], img[y-1][x], 0
						x3, x0, x1 = img[y][x-1], img[y][x], 0
						x8, x4, x5 = 0, 0, 0
					else:
						x7, x2, x6 = img[y-1][x-1], img[y-1][x], img[y-1][x+1]
						x3, x0, x1 = img[y][x-1], img[y][x], img[y][x+1]
						x8, x4, x5 = 0, 0, 0
				else:
					if x == 0:
						x7, x2, x6 = 0, img[y-1][x], img[y-1][x+1]
						x3, x0, x1 = 0, img[y][x], img[y][x+1]
						x8, x4, x5 = 0, img[y+1][x], img[y+1][x+1]
					elif x == img.shape[1]-1:
						x7, x2, x6 = img[y-1][x-1], img[y-1][x], 0
						x3, x0, x1 = img[y][x-1], img[y][x], 0
						x8, x4, x5 = img[y+1][x-1], img[y+1][x], 0
					else:
						x7, x2, x6 = img[y-1][x-1], img[y-1][x], img[y-1][x+1]
						x3, x0, x1 = img[y][x-1], img[y][x], img[y][x+1]
						x8, x4, x5 = img[y+1][x-1], img[y+1][x], img[y+1][x+1]

				a1 = yokoi_function(x0, x1, x6, x2)
				a2 = yokoi_function(x0, x2, x7, x3)
				a3 = yokoi_function(x0, x3, x8, x4)
				a4 = yokoi_function(x0, x4, x5, x1)

				if a1 == a2 == a3 == a4 == 'r':
					ans[y][x] = 5
				else:
					ans[y][x] = 0
					for a in [a1, a2, a3, a4]:
						if a == 'q':
							ans[y][x] += 1


	return ans				

def main():
	assert len(sys.argv) == 2

	img = cv2.imread('lena.bmp', 0)

	if sys.argv[1] == 'Thin':
		answer = Thin(img)
		cv2.imwrite('Thinned_Lena.bmp', answer)
		cv2.imshow('image', answer)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print("wrong command")

if __name__ == '__main__':
    main()