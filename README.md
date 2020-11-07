# SneakerZam
## Final Proyect - Ironhack

![Image](https://images.unsplash.com/photo-1528669697102-a6edb9b6a282?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&h=500&q=80)  

## Table of Contents  

* [About the Project](#about-the-project) 
  * [Data](#bookmark_tabs:-data)
  * [Random Prediction](#pushpin-randon-prediction)  
  * [Prediction uploading](#pushpin-prediction-uploading)  
  * [Take a picture with your camera ](#pushpin-take-a-picture-with-your-camera)      
  * [Recommendation System](#pushpin-recommendation-system)     

* [How was it made?](#how-to-use-the-pipeline)  
  * [How was built?](#hammer-built-with)  
  * [Prerequisites](#page_with_curl-prerequisites)  
  * [Folder Structure](#file_folder-folder-structure)  
* [Next Stages](#next-stages) 
  
## About the project  

SneakerZam is a python algorithm that predict sneaker's model. Based on a Convolutional Neuronal Network, the algorithm will predict which sneaker model is shown on a given picture.

### :bookmark_tabs: Data
  
To create a big database, I started using [Sneakers Databased Api](https://app.swaggerhub.com/apis-docs/tg4solutions/the-sneaker-database/1.0.0#/sneakers/getSneakers) as the main source of my project. The problem was that in the database there are a lot of sneakers models with few images. In order to create a balanced and robust databease for a convolutional neouronal network, we need more images.

We solved this problem with web scrapping and Selenium. The database grown to 80k images and after several filter and cleaning it ends in a 30k images database group by 504 models.

After creating a Sequential layer CNN, i realized that loss and accuracy was not high enough to go on with the CNN. In that point, we incorporate InceptionV3 as base model. With transfer learingn, we could reach a 88% accuracy for that model. 

Note that for github limitations, i have not update the database. There is a database with url column where you will be able to download the image and create a new column with the absolute paht of the image for this dataset. 

### Streamlit

To create a more friendly interface, i use Streamlit combined with a custom pipeline that resolved all the steps needed for correct functionallities of the CNN. 
  
### :pushpin: Random Prediction

![image info](./readmeims/randomp.png)

For firing a random prediction, you only have to select "Random Prediction" on the dropdown menu located in the sidebar. It will choose randomly n sneakers(you could modify how many you want to predict with the slider button)and will predcit wich models is. Note that if the CNN fails, it will show a yellow box with the predicted brand and the real brand. 

  
###  :pushpin: Prediction uploading img
  
![image info](./readmeims/uploadimg.png)

The key for a CNN is to pass any image and predict. For this key point, i let the user upload any images on their hdd and predict. Just choose "Predition uploading image" on the dropdown menu on the sidebar and you will see and input where you will be able to attach your image. 

Furthemore, there is a checkbox where you could detect the shoe. This is a complementary funtionallity based on a computer vision algorithm. If you check it, the first algorithm will predict where are the sneaker on your image, and after that, it will send the crop image to the CNN in order to predicth which model is. If your image is so clean (there is only a sneaker on it), you can omit the checkbox. 

The outpu will be a plot where you will see which model is base on CNN weights and it will show some database image of the model predict too.
  
  
###  :pushpin: Take a picture with your camera  

![image info](./readmeims/tp.png)

I created an option where you will be able to connect your mobile camera with streamlit. All you need is to download [iriun webcam](https://iriun.com/) for your computer an mobile, and run it before launch streamlit. The process to predict a sneaker is similitar to the previous step. You can take a picture of and image(you will see your mobile screen on your desktop) and the CNN will predict your sneakers.

###  :pushpin: Recommendation System  
  
![image info](./readmeims/rs.png)

The last part it is a Recommender System based on Pearson correlation. Note that no previous database were got, so i've created a random one based on a punctuation system from 0 to 5. The model ask you about 5 sneakers that you have to give marks. After choosing them, you will see other sneaker recommendations. It was created for a learning purpose but can be extrapolated to another system that give points to a product and record and user id to find similarities


  ###  :hammer: Built With   
The core of the project is Python 3.7.3, but you have to install those libraries for run the script.   

- Pandas
- Tensorflow
- plotly express
- cv2
- PIL 
- Streamlit
- Selenium
- Numpy


  
## **How to use the app**
###  **:page_with_curl:Prerequisites**  
Please, install all the libraries mentinoned in [Built With](#built-with) in your enviroment in order to run the script.  

After install them, run streamlit as: 
<code>
streamlit run app.py
</code>

and you will see the app. 

My strongly recommendation is to play with the interface!!
  
### **:computer: Inputs**  
 
  
### :file_folder: **Folder structure**  
```
└── sneakerzam   
    ├── requeriments.txt  
    ├── README.md  
    ├── notebooks  
    |   ├── download_img_from_url.ipynb  
    │   └── cnn_shoe_prediction.ipynb  
    │   
    └── streamlit
        │   ├── __init__.py  
        │   └── m_acquisition.py  
        ├── p_model 
        │   ├── __init__.py  
        │   └── m_model.py  
        ├── p_reporting  
        │   ├── __init__.py  
        │   └── m_reporting.py  
        └── p_recommender_system  
            ├── __init__.py  
            └── m_rs.py  

```
  
 ---  
### ** Next stages**  

- Deploy the streamlit on a python server enviroment.
