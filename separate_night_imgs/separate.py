import cv2
import os


input_path = 'images/images/'
output_path = 'night/'
file_type = '.jpg'
break_point = 60
sampling_rate = 10

directory = os.fsencode(input_path)
for file in os.listdir(directory):
    
    filename = os.fsdecode(file)

    if filename.endswith(file_type): 

        img = cv2.imread(input_path + filename, 1)
        pixelvalue = 0
        
        for widthpixel in [x*sampling_rate for x in range(int(len(img)/sampling_rate)) if x*sampling_rate < len(img)]:
            for heightpixel in [y*sampling_rate for y in range(int(len(img[0])/sampling_rate)) if y*sampling_rate < len(img[0])]:
                for colour in range(len(img[0][0])):
                    pixelvalue = pixelvalue + img[widthpixel][heightpixel][colour]

        mean = pixelvalue / (len(img) * len(img[0]) * len(img[0][0]) / sampling_rate ** 2)
        if mean < break_point:
            cv2.imwrite(output_path + filename, img)
        continue
    else:
        continue
