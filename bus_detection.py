# we will need to constantly scan for a bus. And when a bus in view
# we will start scanning for text and will return the text that we have found
# in the bounding box of the bus
from ultralytics import YOLO, RTDETR
import cv2
import time
from PIL import Image
import easyocr
import os


# use llava for text detection


if __name__ == '__main__':
    # To run in IDE (instead of commamnd line), comment out main() and uncomment the block below:
    # model = YOLO('yolov6l6.pt')

    model = RTDETR('rtdetr-l.pt')

    model = model.to("cuda")

    reader = easyocr.Reader(['en'], gpu=True)

    cap = cv2.VideoCapture(0)

    start_rec = False
    bus_bool = False

    text = ""

    while True:
        # Quit condition:
        pressed_key = cv2.waitKey(1)

        ret, frame = cap.read()  # Grabs the most recent frame read by the VideoStream class

        # Resize the cropped image back to original size
        cv2.imwrite(
            "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_bus_detection\\current.png",
            frame)

        if pressed_key == ord('c'):
            start_rec = True

        if start_rec:

            # here we will be getting our object detection result
            results = model(['C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_text_detection\\RealTime-OCR\\current.png'])

            '''''''''
            Here we will be processing all of the yolo information which we will 
            later use in order to get the bounding boxes for each object which will
            be used to crop the image 
            '''
            boxes = None
            classes = None
            for key in results:
                boxes = key.boxes.xyxy.tolist()
                classes = key.boxes.cls.tolist()

            current_image = Image.open(
                'C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_text_detection\\RealTime-OCR\\current.png')
            cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_bus_detection\\cropped_imgs\\" + "cropped_im" + str(
                1) + ".png"
            current_image.save(cropped_image_path)

            index = 2

            global task_queue
            global task_pool

            task = []

            found_bus = False
            for i in range(len(boxes)):
                if classes[i] == 5:

                    found_bus = True

                    box = boxes[i]

                    # here we will cropping the image and placing the image in the directory
                    image = Image.open(
                        'C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_text_detection\\RealTime-OCR\\current.png')
                    crop_area = (
                        int(round(float(box[0]), 2)), int(round(float(box[1]), 2)), int(round(float(box[2]), 2)),
                        int(round(float(box[3]), 2)))
                    cropped_image = image.crop(crop_area)
                    cropped_image_path = "C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_crosswalk\\RealTime-OCR\\cropped_imgs\\" + "cropped_im" + str(
                        index) + ".png"
                    cropped_image.save(cropped_image_path)

                    index = index + 1

            bus_bool = found_bus

            result_text = ""
            index = 1
            if bus_bool:
                for filename in os.listdir("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_bus_detection\\cropped_imgs"):
                    text = reader.readtext("C:\\Users\\davin\\PycharmProjects\\real-world-alt-text_test\\real-time_bus_detection\\cropped_imgs\\" + filename)
                    result_text += "Text " + str(index) + ": " + text
                    index = index + 1
            else:
                text = "No buses detected"

            text = result_text

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.5
        color = (0, 0, 255)  # Green color in BGR
        thickness = 3  # Minimum thickness
        lineType = cv2.LINE_AA

        # Display each line of text
        y_position = 50
        for line in text:
            cv2.putText(frame, line.strip(), (10, y_position), font, fontScale, color, thickness, lineType)
            y_position += 40  # Adjust line spacing as needed

        cv2.imshow("realtime OCR", frame)

        if pressed_key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("OCR stream stopped\n")
            break