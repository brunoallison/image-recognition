from imageai.Detection import ObjectDetection
import os, cv2

execution_path = os.getcwd()

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "images/object.jpg"
        cv2.imwrite(img_name, frame)
        print("image from camera written! Detecting objects...")

        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "images/object.jpg"), output_image_path=os.path.join(execution_path , "images/newObject.jpg"))

        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

cam.release()

cv2.destroyAllWindows()
