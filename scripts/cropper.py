from PIL import Image, ImageDraw, ImageColor
from math import floor

input_data = []
with open('labels.csv') as input:
    for line in input:
    	line = line[:-1]
    	input_data.append(line.split(" "))

inputs_directory = "UPNG"
outputs_directory = "CROP_50_Normal"

FILE_IDX = 0
CHARACTER_IDX = 1
CLASS_IDX = 2
SEVERITY_IDX = 3
X_IDX = 4
Y_IDX = 5
RADIUS_IDX = 6

for instance_data in input_data:
	file_name = instance_data[0]
	image = Image.open(inputs_directory + "/" + file_name + ".png")
	width = image.width
	height = image.height

	if instance_data[CLASS_IDX] != "NORM" and len(instance_data) > RADIUS_IDX:
		print(file_name)
		# x = int(instance_data[X_IDX])
		# y = height - int(instance_data[Y_IDX])
		# r = int(instance_data[RADIUS_IDX])

		# cropped_image = image.crop((x - r, y - r, x + r, y + r))

		# crop_size = (50, 50)
		# resized_image = cropped_image.resize(crop_size, Image.ANTIALIAS)

		# # resized_image = cropped_image.thumbnail(crop_size, Image.ANTIALIAS)

		# # paintable = ImageDraw.Draw(image)
		# # paintable.ellipse((x - r, y - r, x + r, y + r), fill=None, outline="blue")

		# resized_image.save(outputs_directory + "/" + file_name + ".png")
	else:
		print(file_name)
		x = 550
		y = height - 450
		r = 25
		cropped_image = image.crop((x - r, y - r, x + r, y + r))
		cropped_image.save(outputs_directory + "/" + file_name + ".png")
	image.close()




