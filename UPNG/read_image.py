from PIL import Image
im = Image.open("mdb001.png") #Can be many different formats.

pix = im.load()

for i in range(1024):
	for j in range(1024):
		print (pix[(i,j)])#Get the RGBA Value of the a pixel of an image
