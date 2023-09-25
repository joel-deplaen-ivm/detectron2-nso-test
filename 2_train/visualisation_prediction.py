# Visu
import random
import cv2 as cv
import matplotlib.pyplot as plt
import os

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

def random_visu_prediction(dataset_dicts, metadata, predictor, plot_dir, num_images):
    for d in random.sample(dataset_dicts, num_images):    
        im = cv.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,  
                    instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #plt.imshow(out.get_image()[:, :, ::-1])
        plt.figure(figsize=(20, 20))
        plt.imshow(out.get_image()[:, :, :])
        plt.show()
        #save im + annots
        cv.imwrite(os.path.join(plot_dir, "visu_inference" + d["file_name"].split("/")[-1]), out.get_image()[:, :, ::-1])