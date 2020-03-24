
import numpy as np
import imageio

width = 2400
height = 2400
check_width = int(2400/32)

output = 255*np.ones((width,height), dtype=np.uint8)
for x in range(width):
	if int(x/check_width)%2 == 0:
		for y in range(height):
			if int(y/check_width)%2 == 0:
				output[x,y] = 0
	else:
		for y in range(height):
			if int(y/check_width)%2 == 1:
				output[x,y] = 0

imageio.imwrite('surfaces/tube.png', output)
