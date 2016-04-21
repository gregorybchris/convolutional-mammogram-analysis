from PIL import Image, ImageDraw, ImageColor
from math import floor

input_data = []
with open('labels.csv') as input:
    for line in input:
    	line = line[:-1]
    	input_data.append(line.split(" "))

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

		image.draft("RGB", (width, height))
		draw = ImageDraw.Draw(image)
		draw.ellipse((x - r, y - r, x + r, y + r), fill=None, outline="blue")


		directory = "CIR_2/"
		image.save(directory + name + ".png")

	image.close()

