import cv2
import numpy as np

path_to_frozen_inference_graph = 'frozen_inference_graph_coco.pb'
path_coco_model = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
image_path = 'exemplification.jpg'
# # Generate random colors
colors = np.random.randint(0, 255, (80, 3))

# #image_path='C:/Users/maind/Downloads/dog1.jpg'
# # Loading Mask RCNN(# model weight file # model config file)
net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
#
#
# print(colors)
#
# # Load image
img = cv2.imread(image_path)
img = cv2.resize(img, (650, 550))
height, width, _ = img.shape
#
# # Create black image
black_image = np.zeros((height, width, 3), np.uint8)
# # set background color of blank image
black_image[:] = (150, 150, 0)
#
# # preprocess & Detect objects
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
#
# # propagate through the network
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
# # # Boxes is a 3 dimensional list of information about the detected objects
# # # can access the i'th object like boxes = [0,0,i-1]
# # #Returns an array like this:
# # # [0.         0.         0.99492836 0.92008555 0.27316627 0.98529506 0.53607523]
# # #Last 4 is the position of the bounding box
#
#
# # no of objects detected
detection_count = boxes.shape[2]
#
# # iterate through the no of objects
for i in range(detection_count):
    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score < 0.5:
        continue
    #     # Get box Coordinates
    #     # box return x,y,w,h according to the aspect ratio thus it needs to be multiplied by
    #     # the width and height of the image
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)
    #
    #     # define the roi which is actually each separate object in every loop
    #     # iteration
    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape
    #
    #     # Get the mask (ith object in the list, class id)
    mask = masks[i, int(class_id)]
    #     # the model returns a 15x15 array which we resize to the shape oof the
    #     # roi to create the mask
    mask = cv2.resize(mask, (roi_width, roi_height))
    #     # use the threshold function to transform the ROI into as mask where
    #     #     # all the pixels with values under 0.5 in the image are zero and above
    #     #     # are 1
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    #
    #     # put bounding boxes around objects
    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
    #     # Get mask coordinates
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = colors[int(class_id)]
    for cnt in contours:
        cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
#
#     # cv2.imshow("roi", roi)
#     # cv2.waitKey(0)
#
#
# # cv2.imshow("Image", img)
# # cv2.imshow("Black image", black_image)
cv2.imshow("Final", np.hstack([img, black_image]))
cv2.waitKey(0)