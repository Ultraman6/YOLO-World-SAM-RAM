import os
import cv2
from collections import defaultdict
import supervision as sv
from supervision import Detections
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator


class ObjectCropper:
    """A class to manage the cropping of objects detected in images."""

    def __init__(self, names=None, thickness=2, save_crops=True, save_dir="crops"):
        """
        Initializes the ObjectCropper with class names and saving options.

        Args:
            names (list): Dictionary of class names.
            save_dir (str): Directory where cropped images will be saved.
            line_thickness (int): Line thickness for drawing bounding boxes (if > 0).
            save_crops (bool): Flag to control whether to save cropped objects to disk.
            retain_label (bool): Flag to control whether to retain labels in the cropped images.
        """
        if names is None:
            names = []
        self.names = names  # Class names from YOLO model
        self.save_dir = save_dir  # Directory to save crops
        self.crop_tf = thickness  # Line thickness for annotation (if > 0)
        self.save_crops = save_crops  # Flag to save crops

        # Create directory for saving crops if it doesn't exist
        if self.save_crops and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def crop_objects(self, im0, results, classes_to_crop=None, *params):
        """
        Crops detected objects in the image based on YOLO results.
        Args:
            im0 (np.ndarray): Original image from which objects will be cropped.
            results (ultralytics YOLO results): YOLO model's detection results.
            classes_to_crop (list): List of class names to be cropped. If None, all classes are cropped.

        Returns:
            dict: A dictionary containing cropped images organized by class name.
        """
        print(params)
        if params is not None:
            self._set(*params)
        crop_dict = defaultdict(list)  # Dictionary to store cropped images by class

        # Extract detection results
        boxes, classes, confs = results

        # Iterate through each detected object
        for i, (box, cls_id, conf) in enumerate(zip(boxes, classes, confs)):
            class_name = self.names[int(cls_id)]

            # Only crop objects in the specified classes
            if classes_to_crop is None or class_name in classes_to_crop:
                # Crop the object from the original image
                x1, y1, x2, y2 = map(int, box)
                crop_img = im0[y1:y2, x1:x2].copy()
                new_box = (0, 0, x2-x1, y2-y1)
                # Optionally retain bounding box on the cropped image if tf > 0
                self._draw(crop_img, new_box, cls_id, conf)

                # Save the crop if save_crops is True
                if self.save_crops:
                    crop_filename = f"{self.save_dir}/{class_name}_{i}.png"
                    cv2.imwrite(crop_filename, crop_img)

                # Store the cropped image in the dictionary
                crop_dict[class_name].append(crop_img)

        return crop_dict

    def _draw(self, img, box, cls_id, conf):
        color = colors(int(cls_id), True)
        class_name = self.names[int(cls_id)]
        annotator = Annotator(img, line_width=self.crop_tf, font_size=self.crop_tf)  # Create a copy of the original image
        if self.crop_tf > 0:
            annotator.box_label(box=box, label=f"{class_name} {conf:.2f}", color=color)

    def _set(self, tf):
        if tf is not None:
            self.crop_tf = tf

    def annotate_image(self, results):
        """
        Annotates the original image with bounding boxes and class names.
        Args:
            im0 (np.ndarray): Original image to be annotated.
            results (ultralytics YOLO results): YOLO model's detection results.

        Returns:
            np.ndarray: Annotated image.
        """
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        classes = results[0].boxes.cls.cpu().numpy()  # Class IDs
        confs = results[0].boxes.conf.cpu().numpy()

        cls_ids = []
        # Draw bounding boxes and class names
        for box, cls_id, conf in zip(boxes, classes, confs):
            if cls_id not in cls_ids:
                cls_ids.append(cls_id)
            # self._draw(im0, box, cls_id, conf)

        return results[0].plot(), [self.names[id] for id in cls_ids]


# Example usage of ObjectCropper
if __name__ == "__main__":
    # Example class names (these come from the YOLO model)
    os.chdir("F:/Github/YOLO-World-SAM-RAM")  # 设置工作路径
    model = YOLO("world/ultralytics/solutions/yolov8n.pt")
    # Initialize ObjectCropper
    cropper = ObjectCropper(model.names, 0, 0)
    im0 = cv2.imread('world/ultralytics/assets/bus.jpg')  # Load your image
    results = model(im0)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    res_dict = cropper.crop_objects(im0, (boxes, classes, confs),
                                    classes_to_crop=["person", "car"])  # Crop only "person" and "car" classes
    for class_name, crops in res_dict.items():
        for i, crop in enumerate(crops):
            cv2.imshow(f"{class_name}_{i}", crop)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

