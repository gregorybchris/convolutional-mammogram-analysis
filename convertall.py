from PIL import Image
from math import floor

for i in range(1, 323):
	name = "mdb" + str("0" * (3 - len(str(i)))) + str(i)
	image = Image.open(name + ".pgm")
	image.save(name + ".png")