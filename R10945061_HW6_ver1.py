import sys
import numpy as np
import cv2

def yokoi_function(b,c,d,e):
	if b == c and (d != b or e != b):
		return 'q'
	elif b == c and (d == b and e == b):
		return 'r'
	elif b != c:
		return 's'

def yokoi(img):
	ans = [[' ' for x in range(64)] for y in range(64)]
	img_bin = np.zeros(img.shape, np.int)
	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y][x] >= 128:
				img_bin[y][x] = 255

	img_ds = np.zeros((64,64), np.int)
	for y in range(img_ds.shape[0]):
		for x in range(img_ds.shape[1]):
			img_ds[y][x] = img_bin[8*y][8*x]

	for y in range(img_ds.shape[0]):
		for x in range(img_ds.shape[1]):
			if img_ds[y][x] == 255:
				if y == 0:
					if x == 0:
						x7, x2, x6 = 0, 0, 0
						x3, x0, x1 = 0, img_ds[y][x], img_ds[y][x+1]
						x8, x4, x5 = 0, img_ds[y+1][x], img_ds[y+1][x+1]
					elif x == img_ds.shape[1]-1:
						x7, x2, x6 = 0, 0, 0
						x3, x0, x1 = img_ds[y][x-1], img_ds[y][x], 0
						x8, x4, x5 = img_ds[y+1][x-1], img_ds[y+1][x], 0
					else:
						x7, x2, x6 = 0, 0, 0
						x3, x0, x1 = img_ds[y][x-1], img_ds[y][x], img_ds[y][x+1]
						x8, x4, x5 = img_ds[y+1][x-1], img_ds[y+1][x], img_ds[y+1][x+1]
				elif y == img_ds.shape[0]-1:
					if x == 0:
						x7, x2, x6 = 0, img_ds[y-1][x], img_ds[y-1][x+1]
						x3, x0, x1 = 0, img_ds[y][x], img_ds[y][x+1]
						x8, x4, x5 = 0, 0, 0
					elif x == img_ds.shape[1]-1:
						x7, x2, x6 = img_ds[y-1][x-1], img_ds[y-1][x], 0
						x3, x0, x1 = img_ds[y][x-1], img_ds[y][x], 0
						x8, x4, x5 = 0, 0, 0
					else:
						x7, x2, x6 = img_ds[y-1][x-1], img_ds[y-1][x], img_ds[y-1][x+1]
						x3, x0, x1 = img_ds[y][x-1], img_ds[y][x], img_ds[y][x+1]
						x8, x4, x5 = 0, 0, 0
				else:
					if x == 0:
						x7, x2, x6 = 0, img_ds[y-1][x], img_ds[y-1][x+1]
						x3, x0, x1 = 0, img_ds[y][x], img_ds[y][x+1]
						x8, x4, x5 = 0, img_ds[y+1][x], img_ds[y+1][x+1]
					elif x == img_ds.shape[1]-1:
						x7, x2, x6 = img_ds[y-1][x-1], img_ds[y-1][x], 0
						x3, x0, x1 = img_ds[y][x-1], img_ds[y][x], 0
						x8, x4, x5 = img_ds[y+1][x-1], img_ds[y+1][x], 0
					else:
						x7, x2, x6 = img_ds[y-1][x-1], img_ds[y-1][x], img_ds[y-1][x+1]
						x3, x0, x1 = img_ds[y][x-1], img_ds[y][x], img_ds[y][x+1]
						x8, x4, x5 = img_ds[y+1][x-1], img_ds[y+1][x], img_ds[y+1][x+1]

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

	if sys.argv[1] == 'yokoi':
		answer = yokoi(img)
		with open('yokoi.txt','w') as f:
			for y in range(64):
				for x in range(64):
					f.write(str(answer[y][x]))
				f.write('\n')
	else:
		print("wrong command")

if __name__ == '__main__':
    main()