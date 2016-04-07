from PIL import Image, ImageDraw, ImageColor
from math import floor

input_data = []
with open('labels.txt') as input:
    for line in input:
    	line = line[:-1]
    	input_data.append(line.split(" "))

print([len(d) for d in input_data])

normal_info_size = 3
cancer_info_size = 7
for i in range(1, 323):
	directory = "PNG/"
	name = "mdb" + str("0" * (3 - len(str(i)))) + str(i)
	image = Image.open(directory + name + ".png")
	width = image.width
	height = image.height

	image_info = input_data[i - 1]
	if len(image_info) == cancer_info_size:
		x = int(image_info[4])
		y = height - int(image_info[5])
		r = int(image_info[6])

		draw = ImageDraw.Draw(image)
		draw.ellipse((x - r, y - r, x + r, y + r), None, (255, 0, 0, 0))

		directory = "CIR/"
		image.save(directory + name + ".png")
	
	image.close()
	
