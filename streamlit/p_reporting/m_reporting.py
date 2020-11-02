from PIL import Image
import operator
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import *
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px
from resizeimage import resizeimage


# import streamlit as st


class Generators:
    """
    Train, validation and test generators.
    Directory will be none if we have the absolute route on the df. Otherwise, use path to folder.
    """

    def __init__(self, test_df):
        self.batch_size = 50
        self.img_size = (280, 200)
        self.df_y = "shoe"
        self.df_X = "folder"

        # Base train/validation generator
        _datagen = ImageDataGenerator(
            rescale=1. / 255.,
            validation_split=0.25,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        # Test generator
        _test_datagen = ImageDataGenerator(rescale=1. / 255.)
        self.test_generator = _test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=None,
            x_col=self.df_X,
            y_col=self.df_y,
            has_ext=False,
            class_mode="categorical",
            batch_size=self.batch_size,
            seed=42,
            shuffle=False,
            target_size=self.img_size)
        print('Test generator created')


# Create generators
test_df = pd.read_csv('data/test_df_InceptionV3_v11_classweight_504.csv')


class Prediction:
    """
    This is a class where all prediction function will be allocated. Make sure to load
    the generators in order to have the labels.
    """

    def __init__(self, model, generators):
        self.model = model
        self.label_encoded = generators.test_generator.class_indices

    def img_operations(self, img_url):

        image = Image.open(img_url).convert('RGB')
        width, height = image.size
        if height < 200 and width < 280:
            new_image = image.resize((280, 200))
        else:
            new_image = resizeimage.resize_cover(image, [280, 200])
        new_image.save('test.jpg')
        img_saved = 'test.jpg'
        return img_saved

    def from_url(self, img_url, brand=None):
        if img_url:
            url_test = img_url
        else:
            url_test = 'temp/test.jpg'

        # Original image
        image = Image.open(url_test).convert('RGB')

        # Original image
        test_image = Image.open(url_test).convert('RGB')

        # Predict shoe model
        # adjust img to the model size
        test_image = img_to_array(test_image)
        test_image /= 255.
        test_image = np.expand_dims(test_image, axis=0)

        # prediction
        prediction = self.model.predict(test_image)
        prediction_probability = np.amax(prediction)
        a = prediction.flatten()
        indexed = list(enumerate(a))
        top_3 = sorted(indexed, key=operator.itemgetter(1))[-5:]

        percentages = reversed([v for i, v in top_3])
        index = reversed([i for i, v in top_3])
        prediction_idx = np.argmax(prediction)
        # labels
        labels = self.label_encoded
        labels = dict((v, k) for k, v in labels.items())
        class_names = [labels[x] for x in index]
        per = [y for y in percentages]

        graphic_index_pred = [a if a == labels[prediction_idx].replace('_', ' ') else 1 for p, a in
                              enumerate(class_names)]

        position_pred = [b for b, a in enumerate(graphic_index_pred) if isinstance(a, str)]
        colorbh = ["Other Predictions" for i in range(0, 5)]
        colorbh[position_pred[0]] = "Predicted Brand"

        fig = px.bar(x=per, y=class_names, color=colorbh, height=200, color_discrete_map={
            "Real Brand": "#2971b1",
            "Predicted Brand": "#2971b1",
            "Other Predictions": "grey", }, )
        fig.update_layout(
            title_font_color='blue',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=0, t=20, b=20),
            xaxis_title="",
            yaxis_title="",
            legend_title="Predictions",
            # paper_bgcolor = "LightSteelBlue",

        )

        predict = labels[prediction_idx].replace('_', ' ')
        proba = round(prediction_probability * 100, 2)
        return url_test, fig, predict, proba

    def random_prediction(self):
        long = test_df.shape[0]
        randd = random.randint(0, long + 1)
        # find column index
        nbrand = test_df.columns.get_loc("shoe")
        # find brand for index
        real_brand = test_df.iloc[randd, nbrand]
        # find column index
        nfold = test_df.columns.get_loc("folder")
        # find url file for index
        image_file = test_df.iloc[randd, nfold]

        # Original image
        test_image = load_img(image_file, target_size=(280, 200))
        # Predict shoe model
        # adjust img to the model size
        test_image = img_to_array(test_image)
        test_image /= 255.
        test_image = np.expand_dims(test_image, axis=0)

        # prediction
        prediction = self.model.predict(test_image)
        prediction_probability = np.amax(prediction)
        a = prediction.flatten()
        indexed = list(enumerate(a))
        top_3 = sorted(indexed, key=operator.itemgetter(1))[-5:]

        percentages = reversed([v for i, v in top_3])
        index = reversed([i for i, v in top_3])
        prediction_idx = np.argmax(prediction)

        # labels
        labels = self.label_encoded
        labels = dict((v, k) for k, v in labels.items())
        class_names = [labels[x] for x in index]
        per = [y for y in percentages]

        graphic_index_pred = [a if a == labels[prediction_idx].replace('_', ' ') else 1 for p, a in
                              enumerate(class_names)]
        graphic_index_true = [a if a == real_brand.replace('_', ' ') else 1 for p, a in enumerate(class_names)]
        title = "Real Brand = {}\nPredicted Brand = {}\nPrediction probability = {:.2f} %".format(
            real_brand.replace('_', ' '), labels[prediction_idx].replace('_', ' '), prediction_probability * 100)

        position_pred = [b for b, a in enumerate(graphic_index_pred) if isinstance(a, str)]
        position_true = [b for b, a in enumerate(graphic_index_true) if isinstance(a, str)]

        if position_pred == position_true:
            colorbh = ["Real Brand" if isinstance(a, str) else "Other Predictions" for a in graphic_index_true]
        elif (len(position_true) > 0) and (position_true[0] < 5):
            colorbh = ["Other Predictions" for i in range(0, 5)]
            colorbh[position_pred[0]] = "Predicted Brand"
            colorbh[position_true[0]] = "Real Brand"
        else:
            colorbh = ["other predictions" for i in range(0, 5)]
            colorbh[position_pred[0]] = "Predicted Brand"

        fig = px.bar(x=per, y=class_names, color=colorbh, height=200, color_discrete_map={
            "Real Brand": "#2971b1",
            "Predicted Brand": "#da6a55",
            "Other Predictions": "grey", }, )
        fig.update_layout(
            title_font_color='blue' if real_brand.replace('_', ' ') == labels[prediction_idx].replace('_',
                                                                                                      ' ') else 'red',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=0, t=20, b=20),
            xaxis_title="",
            yaxis_title="",
            legend_title="Predictions",
            # paper_bgcolor = "LightSteelBlue",

        )
        image = image_file
        realbrand = real_brand.replace('_', ' ')
        predict = labels[prediction_idx].replace('_', ' ')
        proba = round(prediction_probability * 100, 2)
        return image, fig, realbrand, predict, proba

    def shoe_detection(self, img_url, threshold):
        from object_detection.utils import ops as utils_ops
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as vis_util
        # Import utilites
        #         import tensorflow.compat.v1 as tff
        #         tff.disable_v2_behavior()

        # patch tf1 into `utils.ops`
        utils_ops.tf = tf.compat.v1

        # Patch the location of gfile
        tf.gfile = tf.io.gfile

        # Path to load model weights
        PATH_TO_CKPT = 'data/frozen_inference_graph.pb'
        # Path to label map file
        PATH_TO_LABELS = 'data/label_map.pbtxt'
        #         category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.compat.v1.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value

        filename = img_url
        # image = cv2.imread(os.path.join(PATH_TO_IMAGE_FOLDER, filename))
        image = cv2.imread(filename)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        # output_dict = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})
        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=5,
            min_score_thresh=0.70)

        #         plt.show()
        # crop the detected image
        img_height, img_width, img_channel = image.shape
        absolute_coord = []
        if threshold:
            THRESHOLD = threshold  # adjust your threshold here
        else:
            THRESHOLD = 0.5
        N = len(boxes)
        for i in range(N):
            if scores[i][0] < THRESHOLD:
                continue
            box = boxes[0][0]
            ymin, xmin, ymax, xmax = box
            x_up = int(xmin * img_width)
            y_up = int(ymin * img_height)
            x_down = int(xmax * img_width)
            y_down = int(ymax * img_height)
            absolute_coord.append((x_up, y_up, x_down, y_down))
        bounding_box_img = []
        for c in absolute_coord:
            bounding_box_img.append(image[c[1]:c[3], c[0]:c[2], :])
        im = Image.fromarray(image)
        im.save('imagewithframe.jpg')
        im = Image.fromarray(bounding_box_img[0])
        image_final_url = im.save(f'cropdetected-image.jpg')

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
        axes[0].axis('off')
        axes[0].imshow(image)
        #         axes[0].savefig(f'{filename}-detected.jpg', dpi=254)

        axes[1].axis('off')
        axes[1].imshow(bounding_box_img[0])
        figure, ax = plt.subplots()
        plt.imshow(bounding_box_img[0])
        plt.axis('off')
        # plt.savefig(f'cropdetected-{filename}', dpi=254)
        #         plt.show()

        return 'cropdetected-image.jpg'
