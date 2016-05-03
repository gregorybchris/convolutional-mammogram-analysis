from PIL import Image
import csv

# def readData():
f=open("../data/labels_all_masses.csv")
fname = "../data/combined_set_52/"
targ_name = "../data/all_mass_augmented/"
label_file = open('../data/all_mass_augmented/labels_all_calc_augmented.csv', 'w')
for row in csv.reader(f, delimiter=' '):
    name = row[0]
    read_name = fname + name + ".png"
    image = Image.open(read_name)
    image.save(targ_name + name + ".png")
    row[0] = name + ".png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.rotate(90).save(targ_name + name + "_r90.png")
    row[0] = name + "_r90.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.rotate(180).save(targ_name + name + "_r180.png")
    row[0] = name + "_r180.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.rotate(270).save(targ_name + name + "_r270.png")
    row[0] = name + "_r270.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.transpose(Image.FLIP_LEFT_RIGHT).save(targ_name + name + "_flip.png")
    row[0] = name + "_flip.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90).save(targ_name + name + "_r90_flip.png")
    row[0] = name + "_r90_flip.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.transpose(Image.FLIP_LEFT_RIGHT).rotate(180).save(targ_name + name + "_r180_flip.png")
    row[0] = name + "_r180_flip.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270).save(targ_name + name + "_r270_flip.png")
    row[0] = name + "_r270_flip.png"
    label_file.write(' '.join(row))
    label_file.write('\n')
    image.close()