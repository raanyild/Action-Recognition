import cv2

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out_vid/action_recognition.avi',fourcc, 8, (1080, 1080))

for i in range(1021):
    image_path = "out_frames/" + str(i) + ".png"
    frame = cv2.imread(image_path, 1)
    out.write(frame)
    
out.release()