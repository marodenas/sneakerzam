import streamlit as st
import time
import cv2
from p_model import m_model as mmodel
from p_reporting import m_reporting as mreporting
from p_recommender_system import m_rs as mrecommender
import pandas as pd
import tensorflow as tf
from PIL import Image

# Asign GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, enable=True)


def upload_file():
    st.file_uploader('Upload your picture')


def printrandom_prediction(model, generators):
    # evaluate model
    prediction = mreporting.Prediction(model, generators)
    result = prediction.random_prediction()
    return result


def url_prediction(model, generators, file, threshold, agree=None):
    # evaluate model
    new_image = Image.open(file).convert('RGB')
    new_image.save('uploaded_img.jpg')
    img_saved = 'uploaded_img.jpg'
    # instance the model
    prediction = mreporting.Prediction(model, generators)
    # detect the shoe
    if agree:
        to_test = prediction.shoe_detection(img_saved, threshold)
        img_path = prediction.img_operations(to_test)
        result = prediction.from_url(img_path, "adidas iniki runner")
    else:
        img_path = prediction.img_operations(img_saved)
        result = prediction.from_url(img_path, "adidas iniki runner")

    return result


def recommender_system(sneakers_selected, punctuations):
    recommender = mrecommender.Recommender()
    dict_s = recommender.dic_transform(sneakers_selected, punctuations)
    shoesrecommended = recommender.recommender_system(dict_s)
    return shoesrecommended


def main():
    # load the model
    test_df = pd.read_csv('data/test_df_InceptionV3_v11_classweight_504.csv')
    model = mmodel.get_model()

    # Create generators
    generators = mreporting.Generators(test_df)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Choose how you want to predict your sneaker's model:")

    app_mode = st.sidebar.selectbox("Choose how you want to upload the sneaker picture",
                                    ["Random Prediction", "Let me upload an image", "I will take a picture",
                                     "I need a recommendation"])

    if app_mode == "Random Prediction":
        # Create generators
        n = st.sidebar.slider('Number of predictions:', 1, 10, 5)
        st.sidebar.button("Re-run prediction")
        st.title(f'You are running Random Prediction with {n} sneakers')
        # Random prediction based on sidebar number
        for i in range(0, n):

            image, resultado, realbrand, predict, proba = printrandom_prediction(model, generators)
            if realbrand == predict:
                text = f'Real Brand:{realbrand},  \n Predicted Brand:{predict}  \nProbability:{proba}'
                st.info(text)
            else:
                text = f'Real Brand:{realbrand},  \n Predicted Brand:{predict}  \nProbability:{proba}'
                st.warning(text)
            col1, col2 = st.beta_columns((0.5, 2))
            col1.write(" ")
            col1.write(" ")
            col1.write(" ")
            col1.image(image, use_column_width=True)
            col2.plotly_chart(resultado, use_container_width=True)

    elif app_mode == "Let me upload an image":
        st.title("Hey! upload your photo to predict your sneaker's model")
        file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
        agree = st.checkbox('Do you want that we will detect your sneaker?', value=None)
        if agree:
            threshold = st.sidebar.slider(':', 1, 100, 75)
            threshold = threshold / 100
        else:
            threshold = 0.5

        if st.button("Let's predict it "):
            st.spinner("Doing magic")
            url_test, resultado, predict, proba = url_prediction(model, generators, file, threshold, agree)
            text = predict, proba
            st.info(text)
            # plot
            col1, col2 = st.beta_columns((0.5, 1.5))
            col1.write(" ")
            col1.write(" ")
            col1.image(url_test, use_column_width=True)
            col2.plotly_chart(resultado, use_container_width=True)

            # comparative
            col3, col4, col5 = st.beta_columns((1, 1, 1))
            col3.write("Your image")
            col3.image('uploaded_img.jpg', use_column_width=True)
            col4.write("Our prediction")
            image_predicted = test_df.loc[test_df['shoe'] == predict, 'folder'].to_list()
            col4.image(image_predicted[0], use_column_width=True)
            col5.write(" ")
            col5.image(image_predicted[1], use_column_width=True)
    elif app_mode == "I will take a picture":
        image_placeholder = st.empty()
        if st.button('Open the camera'):
            video = cv2.VideoCapture(-1)
            video.set(cv2.CAP_PROP_FPS, 50)
            # Set properties. Each returns === True on success (i.e. correct resolution)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while True:
                success, image = video.read()
                cv2.imwrite('t.jpg', image)
                if not success:
                    break
                image_placeholder.image(image, channels="BGR", use_column_width=True)
                time.sleep(0.01)
        agree = st.checkbox('Do you want that we will detect your sneaker?', value=None)
        if agree:
            threshold = st.sidebar.slider(':', 1, 100, 65, key="take_picture_slider")
            threshold = threshold / 100
        else:
            threshold = 0.5
        if st.button("Take a picture"):
            # cv2.imwrite(f'testttt.jpg', image)
            cv2.destroyAllWindows()
            st.image("t.jpg", use_column_width=True)
            image = "t.jpg"
            st.spinner("Doing magic")
            url_test, resultado, predict, proba = url_prediction(model, generators, image, threshold, agree)
            text = predict, proba
            st.info(text)

            col1, col2 = st.beta_columns((0.5, 1.5))
            col1.write(" ")
            col1.write(" ")
            col1.image(url_test, use_column_width=True)
            col2.plotly_chart(resultado, use_container_width=True)

            col3, col4, col5 = st.beta_columns((1, 1, 1))
            col3.write("Your image")
            col3.image('uploaded_img.jpg', use_column_width=True)
            col4.write("Our prediction")
            image_predicted = test_df.loc[test_df['shoe'] == predict, 'folder'].to_list()
            col4.image(image_predicted[0], use_column_width=True)
            col5.write(" ")
            col5.image(image_predicted[1], use_column_width=True)

    elif app_mode == "I need a recommendation":
        # Create a list of possible values and multiselect menu with them in it.

        st.header("First things first, we need to know: what are your favourite sneakers?")
        st.subheader("Please, select at least 5 models that you love and rate them from 0 to 5")
        sneakers_multi = test_df['shoe'].unique()
        sneakers_selected = st.multiselect('Select sneakers', sneakers_multi)
        if len(sneakers_selected) > 4:
            punctuations = []
            for i in range(0, len(sneakers_selected)):
                st.write("\n \n")
                c0, c1, c2, c3, c4 = st.beta_columns((0.5, 0.5, 0.5, 0.5, 0.5))
                snselected = test_df.loc[test_df['shoe'] == sneakers_selected[i], 'folder'].to_list()
                c0.write(" ")
                c0.write(" ")
                c0.write(sneakers_selected[i])
                number = c1.number_input('', min_value=0, max_value=5, value=4, key=f'{sneakers_selected[i]}')
                c2.image(snselected[0], use_column_width=True)
                c3.image(snselected[1], use_column_width=True)
                c4.image(snselected[2], use_column_width=True)
                punctuations.append(number)
            recommender_bt = st.button('I need a recommendation')
            if recommender_bt:
                data = recommender_system(sneakers_selected, punctuations)
                dfsneakers = pd.read_csv("data/sneakers.csv")
                st.title("Here is our recommendation:")
                for s in data:
                    c0, c1, c2, c3 = st.beta_columns((0.5, 0.5, 0.5, 0.5))
                    rs_title = dfsneakers.loc[dfsneakers['sneakerId'] == s, 'shoe'].to_list()
                    rs_image = dfsneakers.loc[dfsneakers['sneakerId'] == s, 'folder'].to_list()
                    c0.write(" ")
                    c0.write(" ")
                    c0.write(rs_title[0])
                    c1.image(rs_image[0], use_column_width=True)
                    c2.image(rs_image[1], use_column_width=True)
                    c3.image(rs_image[2], use_column_width=True)


if __name__ == "__main__":
    main()
