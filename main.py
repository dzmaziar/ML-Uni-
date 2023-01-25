import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation
import cv2
import tensorflow as tf


def object_detection():
    segment_image = instance_segmentation()
    segment_image.load_model("mask_rcnn_coco.h5")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        res = segment_image.segmentFrame(frame, show_bboxes=True)
        image = res[1]

        cv2.imshow('Instance Segmentation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.realese()
    cv2.destroyAllWindows()

  # temp = segment_image.select_target_classes(person=True)
  #   segment_image.segmentImage(
  #       image_path="1q.jpg",
  #       show_bboxes=True,
  #       segment_target_classes=temp,
  #       output_image_name="dog1q.jpg"
  #   )

def main():
    object_detection()



if __name__ == '__main__':
    main()