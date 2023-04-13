import cv2
import numpy as np

path_to_frozen_inference_graph = 'frozen_inference_graph_coco.pb'
path_coco_model = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
VIDEO = 'race.mp4'
net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph, path_coco_model)
colors = np.random.randint(0, 255, (80, 3))

video = cv2.VideoCapture(VIDEO)
while True:
    grabbed, frame = video.read()
    if not grabbed:
        break

    img = cv2.resize(frame, (650, 550))
    height, width, _ = img.shape
    black_image = np.zeros((height, width, 3), np.uint8)
    black_image[:] = (150, 150, 0)
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        if score < 0.5:
            continue

        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        roi = black_image[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape

        mask = masks[i, int(class_id)]

        mask = cv2.resize(mask, (roi_width, roi_height))

        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[int(class_id)]
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

    cv2.imshow("Black image", black_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
