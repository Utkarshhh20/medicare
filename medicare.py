from logging import warning
import streamlit as st 
from PIL import Image
from keras.models import load_model
import keras
import tensorflow
from keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import numpy as np
import io
import nltk
import pandas as pd
import textblob as tb
import pickle as pkl
import hydralit_components as hc
import requests
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
st.set_page_config(page_title='MediCare', layout="wide",initial_sidebar_state='collapsed')
#disease=st.sidebar.selectbox(options=['Home', 'Pneumonia', 'Diabetes', 'Skin Diseases', 'Heart Stroke', 'Plant Diseases'], label='Choose a disease from the following')

#train_datagen=ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip= True, preprocessing_function=preprocess_input)
#train= train_datagen.flow_from_directory(directory='C:\\Users\\Utki\\Desktop\\hackathon\\sciencious\\plantdataset\\trainvalid\\trainvalid\\train', target_size=(256,256), batch_size=32)
menu_data = [
    {'icon': "fa fa-plus", 'label':"Pneumonia"},
    {'icon': "fa fa-medkit", 'label':"Diabetes"},
    {'icon': "fa fa-ambulance", 'label':"Skin Diseases"},
    {'icon': "fa fa-heartbeat", 'label':"Heart Stroke"},
    {'icon': "fa fa-leaf", 'label':"Plant Diseases"},
    {'icon': "fa fa-newspaper", 'label':"News"},
]
#    {'icon': "bi bi-telephone", 'label':"Contact us"},
over_theme = {'txc_inactive': "#D3D3D3",'menu_background':'#3948A5','txc_active':'white','option_active':'#3948A5'}
disease = hc.nav_bar(
menu_definition=menu_data,
override_theme=over_theme,
home_name='Home',
hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
sticky_nav=True, #at the top or not
sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
use_animation=True,
key='NavBar'
)
st.write(' ')
st.write(' ')
if disease!='Home' and disease!='News':
    st.title(disease)

if disease=='Home':
    logo='''
        <style>
        .logo{
            width: 300px;
            margin-top:0px;
            margin-left:-30px;
        }
        </style>
        <body>
        <center><img src='https://cdn.myportfolio.com/fd40c2a8-1f6f-47d7-8997-48ba5415a69c/a07d2d55-198d-4faf-b1f8-647ce69e0250_rw_600.png?h=cc9f4428ccd6340a1d7fc50858dcc3f6' alt='logo' class='logo'></img></center> 
        </body>
        '''
    whatcan='''
    <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
    <style>
    .whatcan{
        font-family: 'Montserrat';
        font-size:1.8em;
        color:#0676cc;
        font-weight:600;
        margin-top:;
    }
    </style>
    <body>
        <center><p1 class='whatcan'>What can I do with MediCare?</p1></center>
    </body>
    '''
    tech='''
    <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        .taimg {
        float: center;
        z-index: 1;
        width: 400px;
        position: relative;
        border-radius: 10%;
        margin-left: 10px;
        }
        </style>
        <body>
        <img src='https://elements-cover-images-0.imgix.net/2459234f-4ed8-49d5-b355-229ba34f2440?auto=compress%2Cformat&fit=max&w=1370&s=3185b54ef380d133d5344d637eb9dd2e' alt="House" class='taimg'></img>
        </body>'''
    techtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .techtxt {
            font-family: 'Montserrat';
            font-size: 25px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='techtxt'> DIAGNOSIS </p1> </center>
        </body>
        '''
    techsubtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .techsubtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='techsubtxt'> Use our disease identification model built using AI and ML to identify if there is any possibility of you suffering from an illness. We require you to input your data or an image in order to identify.</p1> </center>
        </body>
        '''
    fundament='''
    <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        .faimg {
        float: center;
        z-index: 1;
        width: 400px;
        position: relative;
        border-radius: 5%;
        margin-left: 10px;
        }
        </style>
        <body>
        <img src='https://elements-cover-images-0.imgix.net/05482da7-2bbf-4a6f-8dfd-e5dde6d3268b?auto=compress%2Cformat&fit=max&w=1370&s=430777fe432e91c9bd8a5e3b14a3ceb3'  alt="House" class='faimg'></img>
        </body>'''
    fundtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .fundtxt {
            font-family: 'Montserrat';
            font-size: 25px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='fundtxt'> RECOMMENDATION </p1> </center>
        </body>
        '''
    fundsubtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .fundsubtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='fundsubtxt'> If you were to qualify for any disease based on your input data or image, we will provide you with reccomendations on how to cure it or take precautions as well as several other symptoms. </p1> </center>
        </body>
        '''
    backt='''
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        .btimg {
        float: center;
        z-index: 1;
        width: 400px;
        position: relative;
        border-radius: 5%;
        }
        </style>
        <body>
        <center><img src='https://elements-cover-images-0.imgix.net/d48042a8-aec1-4074-990a-3089278e673e?auto=compress%2Cformat&fit=max&w=1370&s=d895f403859e914f6d6452300ebc58fd' alt="House" class='btimg'></img></center>
        <p1 class>
        </body>'''
    bttxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .techtxt {
            font-family: 'Montserrat';
            font-size: 25px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='techtxt'> LATEST NEWS </p1> </center>
        </body>
        '''
    btsubtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .btsubtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='btsubtxt'> Get the latest news based on topics such as health care in order to be up to date with the recent advancements being made in this field as well as each headline will also be accompanied with a sentiment score. </p1> </center>
        </body>
        '''
    warninghead='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .warning {
            font-family: 'Montserrat';
            font-size: 32px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='warning'> DISCLAIMER ! </p1> </center>
        </body>
        '''
    warningtxt='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .warningtxt {
            font-family: 'Montserrat';
            font-size: 15px;
            margin-top:20px;
            font-weight: 600;
            margin-bottom: 0px;
        }
        </style>
        <body>
        <center> <p1 class='warningtxt'> Please note that we are not certified by any medical agency and the website is built for educational and information purposes only. We would recommend you to consult with a doctor regardless of our medical results. Since the models trained are not completely accurate, they are viable to error. </p1> </center>
        </body>
        '''
    st.markdown(logo, unsafe_allow_html=True)
    st.write('')
    st.write('______________________________________')
    st.markdown(whatcan, unsafe_allow_html=True)
    technical,fundamental,backtest=st.columns(3)
    with technical:
        st.markdown(tech, unsafe_allow_html=True)
        st.markdown(techtxt, unsafe_allow_html=True)
        st.markdown(techsubtxt, unsafe_allow_html=True)
        st.write('____________________')
    with fundamental:
        st.markdown(fundament, unsafe_allow_html=True)
        st.markdown(fundtxt, unsafe_allow_html=True)
        st.markdown(fundsubtxt, unsafe_allow_html=True)
        st.write('____________________')
    with backtest:
        st.markdown(backt, unsafe_allow_html=True)
        st.markdown(bttxt, unsafe_allow_html=True)
        st.markdown(btsubtxt, unsafe_allow_html=True)
        st.write('____________________')
    st.markdown(warninghead, unsafe_allow_html=True)
    st.write('')
    st.markdown(warningtxt, unsafe_allow_html=True)
    st.write(' ')
    st.write('__________________________')
    acchead='''
        <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
        <style>
        .acc {
            font-family: 'Montserrat';
            font-size: 32px;
            margin-top:0px;
            font-weight: 700;
            margin-bottom: 0px;
        }
        #bullseye {
        margin-top:0px;
        }
        </style>
        <body>
        <center> <p1 class='acc'> OUR MODEL ACCURACIES <svg xmlns="http://www.w3.org/2000/svg" width="35" height="50" fill="currentColor" class="bi bi-bullseye" viewBox="0 0 16 20">
  <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
  <path d="M8 13A5 5 0 1 1 8 3a5 5 0 0 1 0 10zm0 1A6 6 0 1 0 8 2a6 6 0 0 0 0 12z"/>
  <path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"/>
  <path d="M9.5 8a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"/>
    id='bullseye'</svg></p1> </center>
        </body>
        '''
        
    st.markdown(acchead, unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    cards='''
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
    <style>
    .card1 {
    display: inline-block;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 27%;
    border-radius: 5px;
    float:left;
    margin-bottom:50px;
    margin-right:100px;
    }
    .card1:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .card2 {
    display: inline-block;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 27%;
    border-radius: 5px;
    float:left;
    margin-right:10px;
    margin-bottom:50px;
    }
    .card2:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .card3 {
    display: inline-block;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 27%;
    border-radius: 5px;
    float:center;
    margin-bottom:50px;
    margin-right:10px;
    }
    .card3:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .card4 {
    display: inline-block;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 27%;
    border-radius: 5px;
    float:center;
    margin-bottom:50px;
    margin-right:10px;
    }
    .card4:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .card5 {
    display: inline-block;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 27%;
    border-radius: 5px;
    float:center;
    margin-bottom:50px;
    margin-right:10px;
    }
    .card5:hover {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .container {
    padding: 10px 16px;
    width: 90%;
    border-radius: 5px;
    }
    .pneumonia_img {
    border-radius: 5px 5px 0 0;
    margin-bottom: 10px;
    }
    #progress1 {
    margin-left:7px;
    margin-right:7px;
    }
    .diabetes_img {
    border-radius: 5px 5px 0 0;
    margin-bottom: 10px;
    }
    .skin_img {
    border-radius: 5px 5px 0 0;
    margin-bottom: 10px;
    }
    .heart_img {
    border-radius: 5px 5px 0 0;
    margin-bottom: 10px;
    }
    .plant_img {
    border-radius: 5px 5px 0 0;
    margin-bottom: 10px;
    }
    </style>
    <body>
    <center>
    <div id='wrapper'>
    <div class="card1">
        <img src="https://cdn.myportfolio.com/fd40c2a8-1f6f-47d7-8997-48ba5415a69c/8a107197-e397-4faf-955e-61f8aa6c377b_rw_600.jpg?h=e5db59a67b0641095c2a8b64a88e03a1" alt="Avatar" style="width:100%" class='pneumonia_img'>
        <div class="container">
            <h4><b>Pneumonia Accuracy: 80.77%</b></h4>
        </div>
        <div class="w3-light-grey w3-round-xlarge" id='progress1'>
        <div class="w3-container w3-green w3-center w3-round-xlarge" style="width:80.77%">80.77%</div>
        </div><br>
         <div class="container">
        <h5><a href='https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia'> Link to Dataset here</a></h5>
        </div>
    </div>
    <div class="card2">
        <img src="https://cdn.myportfolio.com/fd40c2a8-1f6f-47d7-8997-48ba5415a69c/eaded34b-22d1-445c-8021-b5d8a0086da2_rw_600.jpeg?h=6a3be7fb1670b56a6ecc55bdfd2c95d7" alt="Avatar" style="width:100%" class='diabetes_img'>
        <div class="container">
            <h4><b>Diabetes Accuracy: 75.78%</b></h4>
        </div>
        <div class="w3-light-grey w3-round-xlarge" id='progress1'>
        <div class="w3-container w3-green w3-center w3-round-xlarge" style="width:75.78%">75.78%</div>
        </div><br>
         <div class="container">
        <h5><a href='https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database'> Link to Dataset here</a></h5>
        </div>
    </div>
    <div class="card3">
        <img src="https://cdn.myportfolio.com/fd40c2a8-1f6f-47d7-8997-48ba5415a69c/c011c263-441e-4278-b83e-691afeb95742_rw_600.jpg?h=a1d706fc8b92423619fc4b3763cdb1b8" alt="Avatar" style="width:100%" class='skin_img'>
        <div class="container">
            <h4><b>Skin Disease Accuracy: 76.57%</b></h4>
        </div>
        <div class="w3-light-grey w3-round-xlarge" id='progress1'>
        <div class="w3-container w3-green w3-center w3-round-xlarge" style="width:76.57%">76.57%</div>
        </div><br>
         <div class="container">
        <h5><a href='https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000'> Link to Dataset here</a></h5>
        </div>
    </div>
    <div class="card4">
        <img src="https://cdn.myportfolio.com/fd40c2a8-1f6f-47d7-8997-48ba5415a69c/2508fc4e-9fb6-4a87-89a3-59f097f2e4e2_rw_600.jpg?h=1eb9e88194f7997fefc53636ce808a98" alt="Avatar" style="width:100%" class='heart_img'>
        <div class="container">
            <h4><b>Stroke prediction Accuracy: 95.04%</b></h4>
        </div>
        <div class="w3-light-grey w3-round-xlarge" id='progress1'>
        <div class="w3-container w3-green w3-center w3-round-xlarge" style="width:95.04%">95.04%</div>
        </div><br>
         <div class="container">
        <h5><a href='https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset'> Link to Dataset here</a></h5>
        </div>
    </div>
    <div class="card5">
        <img src="https://cdn.myportfolio.com/fd40c2a8-1f6f-47d7-8997-48ba5415a69c/6674ca46-94ed-4a72-92dd-cb2451af3536_rw_600.jpg?h=7f2995fca6d71f0346a83b7be8eb8bc4" alt="Avatar" style="width:100%" class='plant_img'>
        <div class="container">
            <h4><b>Plant Disease Accuracy: 81.45%</b></h4>
        </div>
        <div class="w3-light-grey w3-round-xlarge" id='progress1'>
        <div class="w3-container w3-green w3-center w3-round-xlarge" style="width:81.45%">81.45%</div>
        </div><br>
         <div class="container">
        <h5><a href='https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset'> Link to Dataset here</a></h5>
        </div>
    </div>
    </div>
    </body>
    '''
    st.markdown(cards, unsafe_allow_html=True) 
st.write(' ')
st.write(' ')
if disease=='Pneumonia':
    ref={0:'You seem to be in the safe zone. Keep it up!', 1:'You may be prone to Pneumonia. Kindly see a doctor at the earliest!'}
    def prediction(image):
        img=load_img(image, target_size=(256,256))
        i=img_to_array(img)
        im=preprocess_input(i)
        img=np.expand_dims(im, axis=0)
        pred=np.argmax(model.predict(img))
        print(f" The image belongs to { ref[pred] } ")
        st.write(f" The image belongs to { ref[pred] } ")
        return ref[pred]
    def load_image(image_file):
        img = Image.open(image_file)
        return img
    
    image_inp = st.file_uploader("Upload your image for identification", type=["png","jpg","jpeg"])

    if image_inp is not None:
            blank1, text, blank2=st.columns([1,5,1])
            with blank1:
                st.write(' ')
            with text:
                file_details = {"filename":image_inp.name, "filetype":image_inp.type, "filesize":image_inp.size}
                st.write(file_details)
                img_input=st.image(load_image(image_inp),width=250)
                image_data = image_inp.getvalue()
                img_input = Image.open(io.BytesIO(image_data))
                model = keras.models.load_model('pneumonia.h5')
                predict=prediction(io.BytesIO(image_data))
            with blank2:
                st.write(' ')
            if predict=='You may be prone to Pneumonia. Kindly see a doctor at the earliest!':
                blank1, text, blank2=st.columns([0.45,5,0.8])
                with blank1:
                    st.write(' ')
                with text:
                    st.title('Diagnosis')
                    st.write('Your doctor will start by taking your medical history. They’ll ask you questions about when your symptoms first appeared and your health in general.')
                    st.write('')
                    st.write('They’ll then give you a physical exam. This will include listening to your lungs with a stethoscope for any abnormal sounds, such as crackling.')
                    st.write('')
                    st.write('Depending on the severity of your symptoms and your risk of complications, your doctor may also order one or more of these tests:')
                    st.write('')
                    st.subheader('Chest X-ray')
                    st.write('An X-ray helps your doctor look for signs of inflammation in your chest. If inflammation is present, the X-ray can also inform your doctor about its location and extent.')
                    st.write(' ') 
                    st.subheader('Blood culture')
                    st.write('This test uses a blood sample to confirm an infection. Culturing can also help identify what may be causing your condition.')
                    st.write(' ') 
                    st.subheader('Sputum culture')
                    st.write('During a sputum culture, a sample of mucus is collected after you’ve coughed deeply. It’s then sent to a lab to be analyzed to identify the cause of the infection.')
                    st.write(' ') 
                    st.subheader('Pulse oximetry')
                    st.write('A pulse oximetry measures the amount of oxygen in your blood. A sensor placed on one of your fingers can indicate whether your lungs are moving enough oxygen through your bloodstream.')
                    st.write(' ') 
                    st.subheader('CT scan')
                    st.write('CT scans provide a clearer and more detailed picture of your lungs.')
                    st.write(' ') 
                    st.subheader('Fluid sample')
                    st.write('If your doctor suspects there’s fluid in the pleural space of your chest, they may take a fluid sample using a needle placed between your ribs. This test can help identify the cause of your infection.')
                    st.write(' ') 
                    st.subheader('Bronchoscopy')
                    st.write('A bronchoscopy looks into the airways in your lungs. It does this using a camera on the end of a flexible tube that’s gently guided down your throat and into your lungs.')
                    st.write(' ')
                    st.write('Your doctor may do this test if your initial symptoms are severe, or if you’re hospitalized and not responding well to antibiotics.')
                    st.write(' ')                    
                    st.title('Causes')
                    st.write('Pneumonia happens when germs get into your lungs and cause an infection. The immune system’s reaction to clear the infection results in inflammation of the lung’s air sacs (alveoli). This inflammation can eventually cause the air sacs to fill up with pus and liquids, causing pneumonia symptoms.')
                    st.write(' ')
                    st.write('Several types of infectious agents can cause pneumonia, including bacteria, viruses, and fungi.')
                    st.write(' ')
                    st.title('Treatment')
                    st.write('Your treatment will depend on the type of pneumonia you have, how severe it is, and your general health.')
                    st.write(' ')
                    st.subheader('Prescription medications')
                    st.write(' ')
                    st.write('Your doctor may prescribe a medication to help treat your pneumonia. What you’re prescribed will depend on the specific cause of your pneumonia.')
                    st.write(' ')
                    st.write('Oral antibiotics can treat most cases of bacterial pneumonia. Always take your entire course of antibiotics, even if you begin to feel better. Not doing so can prevent the infection from clearing, and it may be harder to treat in the future.')
                    st.write('')
                    st.write('Antibiotic medications don’t work on viruses. In some cases, your doctor may prescribe an antiviral. However, many cases of viral pneumonia clear on their own with at-home care')
                    st.write('')
                    st.write('Antifungal medications are used to treat fungal pneumonia. You may have to take this medication for several weeks to clear the infection.')
                    st.write('')
                    st.subheader('Home remedies')
                    st.write('Although home remedies don’t actually treat pneumonia, there are some things you can do to help ease symptoms.')
                    st.write('')
                    st.write('Coughing is one of the most common symptoms of pneumonia. Natural ways to relieve a cough include gargling salt water or drinking peppermint tea.')
                    st.write('')
                    st.write('Cool compresses can work to relieve a fever. Drinking warm water or having a nice warm bowl of soup can help with chills. Here are more home remedies to try.')
                    st.write('')
                    st.write('You can help your recovery and prevent a recurrence by getting a lot of rest and drinking plenty of fluids.')
                    st.write('')
                    st.write('Although home remedies can help ease symptoms, it’s important to stick to your treatment plan. Take any prescribed medications as directed.')
                    st.write('')
                with blank2:
                    st.write(' ')
    else:
        st.subheader('Sample image')
        st.image(
            "https://github.com/Utkarshhh20/medicare/blob/master/person1946_bacteria_4874.jpeg?raw=true",
        )

elif disease=='Diabetes':
    op=''
    input,blank,images=st.columns([1,0.1,1])
    with input:
        st.subheader('Please fill in the following details accurately to get an estimation of your current condition')
        ref={0:'You seem to be in the safe zone. Keep it up!', 1:'You may be prone to diabetes. Kindly see a doctor at the earliest!'}
        filename = 'diabetes.pkl'
        loaded_model = pkl.load(open(filename, 'rb'))
        age=st.number_input('Please enter your age: ', step=1, value=32)
        st.write(' ')
        glucose=st.number_input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test: ', step=1, value=183)
        st.write(' ')
        pressure=st.number_input('Diastolic blood pressure (mm Hg): ', step=1, value=64)
        st.write(' ')
        thickness=st.number_input('Triceps skin fold thickness (mm): ', step=1, value=30)
        st.write(' ')
        insulin=st.number_input('2-Hour serum insulin (mu U/ml): ', step=1, value=120)
        st.write(' ')
        bmi=st.number_input('Body mass index (weight in kg/(height in m)^2): ', value=23.3)
        st.write(' ')
        pedigreefunc=st.number_input('Diabetes pedigree function: ', step=0.001, value=0.67)
        st.write(' ')
        pregnancy=st.number_input('Number of times pregnant: ', step=1, value=8)
        st.write(' ')
        calculate=st.button('Calculate')
        if calculate==True:
            prediction=loaded_model.predict([[pregnancy, glucose, pressure, thickness, insulin, bmi, pedigreefunc, age]])
            for i in prediction:
                op=ref[i]
                st.subheader(ref[i])
    if op=='You may be prone to diabetes. Kindly see a doctor at the earliest!':
                blank1, text, blank2=st.columns([0.45,5,0.8])
                with blank1:
                    st.write(' ')
                with text:
                    st.title('Symptoms of diabetes')
                    st.write('Diabetes symptoms are caused by rising blood sugar.')
                    st.write(' ')
                    st.header('General symptoms')
                    st.write("The general symptoms of diabetes include:")
                    st.write(' ')
                    st.write('''
● increased hunger \n
● increased thirst \n
● weight loss \n
● frequent urination \n
● blurry vision \n
● extreme fatigue \n
● sores that dont heal''')
                    st.header('The bottom line')
                    st.write("Diabetes symptoms can be so mild that they're hard to spot at first. Learn which signs should prompt a trip to the doctor.")
                    st.write(' ')
                    st.header('Diabetes and exercise')
                    st.write('Along with diet and treatment, exercise plays an essential role in diabetes management. This is true for all types of diabetes.')
                    st.write('')
                    st.write('Staying active helps your cells react to insulin more effectively and lower your blood sugar levels. Exercising regularly can also help you:')
                    st.write('''
● reach and maintain a healthy weight\n
● reduce your risk of diabetes-related health complications\n
● boost mood\n
● get better sleep\n
● improve memory''')
                    st.write(' ')
                    st.write('Diabetes-friendly exercises include:')
                    st.write('''
● walking\n
● swimming\n
● dancing\n
● cycling''')
                with blank2:
                    st.write(' ')
    with blank:
        st.write(' ')
    with images:
        st.header('Dataset insights')
        st.image('https://serving.photos.photobox.com/046066339fd9cad0187a493e5f9161cc13c91339c31d4eac09526f3a19c91f8bece550d4.jpg') 
        st.write(' ')           
        st.image('https://serving.photos.photobox.com/740891662fee02fb126ab54e458aad825bb84e509b10d489c5fe81d540b9755e4483bba1.jpg')
        st.write(' ')
elif disease=='Skin Diseases':
    ref={ 
    1: 'Melanocytic nevi',
    2: 'Melanoma',
    3: 'Benign keratosis-like lesions',
    4: 'Basal cell carcinoma',
    5: 'Actinic keratoses',
    6: 'Vascular lesions',
    7: 'Dermatofibroma'
    }

    def load_image(image_file):
        img = Image.open(image_file)
        return img

    def prediction(path):
        img=load_img(path, target_size=(75,100))
        i=img_to_array(img)
        im=preprocess_input(i)
        img=np.expand_dims(im, axis=0)
        pred=np.argmax(model.predict(img))
        print(f" You may be infected with { ref[pred] } ")
        st.subheader(f" You may be infected with { ref[pred] } ")
        op=ref[pred]
        return op
    image_inp = st.file_uploader("Upload your image for identification", type=["png","jpg","jpeg"])
    if image_inp is not None:
            file_details = {"filename":image_inp.name, "filetype":image_inp.type, "filesize":image_inp.size}
            st.write(file_details)
            img_input=st.image(load_image(image_inp),width=250)
            image_data = image_inp.getvalue()
            img_input = Image.open(io.BytesIO(image_data))
            model = keras.models.load_model('skin.h5')
            op=prediction(io.BytesIO(image_data))
            blank1, text, blank2=st.columns([0.45,5,0.8])
            with blank1:
                st.write(' ')
            with text:
                    st.title('Symptoms of skin disorders')
                    st.write('Skin conditions have a wide range of symptoms. Symptoms on your skin that appear due to common problems aren’t always the result of a skin disorder. Such symptoms can include blisters from new shoes or chafing from tight pants. However, skin problems that have no obvious cause may indicate the presence of an actual skin condition that requires treatment.')
                    st.write('')
                    st.write('Skin irregularities that are typically symptoms of a skin disorder include:')
                    st.write('')
                    st.write('''
● raised bumps that are red or white
● a rash, which might be painful or itchy
● scaly or rough skin
● peeling skin
● ulcers
● open sores or lesions
● dry, cracked skin
● discolored patches of skin
● fleshy bumps, warts, or other skin growths
● changes in mole color or size
● a loss of skin pigment
● excessive flushing''')
                    st.write('')
                    st.title('Causes of skin disorders')
                    st.write('Common known causes of skin disorders include:')
                    st.write(' ') 
                    st.write('''
● bacteria trapped in skin pores and hair follicles\n
● fungus, parasites, or microorganisms living on the skin\n
● viruses\n
● a weakened immune system\n
● contact with allergens, irritants, or another person’s infected skin\n
● genetic factors\n
● illnesses affecting the thyroid, immune system, kidneys, and other body systems\n
● Numerous health conditions and lifestyle factors can also lead to the development of certain skin disorders. Some skin conditions have no known cause.''')
                    st.write(' ') 
                    st.title('Preventing skin disorders')
                    st.write('Certain skin disorders aren’t preventable, including genetic conditions and some skin problems due to other illnesses. However, it’s possible to prevent some skin disorders.')
                    st.write(' ') 
                    st.write('Follow these tips to prevent infectious skin disorders:')
                    st.write('''
● Wash your hands with soap and warm water frequently.\n
● Avoid sharing eating utensils and drinking glasses with other people.\n
● Avoid direct contact with the skin of other people who have an infection.\n
● Clean things in public spaces, such as gym equipment, before using them.\n
● Don’t share personal items, such as blankets, hairbrushes, or swimsuits.\n
● Sleep for at least seven hours each night.\n
● Drink plenty of water.\n
● Avoid excessive physical or emotional stress.\n
● Eat a nutritious diet.\n
● Get vaccinated for infectious skin conditions, such as chickenpox.''')
    else:
        st.subheader('Sample image')
        st.image(
            "https://github.com/Utkarshhh20/medicare/blob/master/ISIC_0029306.jpg?raw=true",
        )
elif disease=='Heart Stroke':
    inp,blank,images=st.columns([1,0.1,1])
    with inp:
        filename = 'heart.pkl'
        loaded_model = pkl.load(open(filename, 'rb'))
        st.subheader('Please fill in the following details accurately to get an estimation of your current condition')
        st.write(' ')
        ref={0:'You seem to be in the safe zone', 1:'You have a chance of getting a stroke'}
        gender=st.selectbox(label='Please select one of the genders below: ', options=['Male', 'Female'])
        st.write(' ')
        age=st.number_input('Please enter your age: ', step=1, value=32)
        st.write(' ')
        hypertension=st.selectbox('Do you have hypertension: ', options=['Yes', 'No'])
        st.write(' ')
        heartdisease=st.selectbox('Do you have a heart disease: ', options=['Yes', 'No'])
        st.write(' ')
        married=st.selectbox('Are you or were you married: ', options=['Yes', 'No'])
        st.write(' ')
        work=st.selectbox('Please select your work type: ', options=[ "Child", "Govtjob", "Never worked", "Private", "Self-employed"])
        st.write(' ')
        residence=st.selectbox('Residence area: ', options=['Rural', 'Urban'])
        st.write(' ')
        bmi=st.number_input('Body mass index (weight in kg/(height in m)^2): ', step=0.1, value=23.3)
        st.write(' ')
        glucose=st.number_input('Average glucose level in blood: ', step=0.1, value=174.1)
        st.write(' ')
        smoking=st.selectbox('Smoking status: ', options=["formerly smoked", "never smoked", "smokes" , "Unknown"])
        st.write(' ')
        calculate=st.button('Calculate')
    with blank:
        st.write(' ')
    with images:
        st.header('Dataset insights')
        st.image('https://github.com/Utkarshhh20/medicare/blob/master/scatterplot_heart.png?raw=true') 
        st.write(' ')           
        st.image('https://github.com/Utkarshhh20/medicare/blob/master/heatmap_heart.png?raw=true')
        st.write(' ')
    if calculate==True:
        if gender=='Male':
            gender=1
        else:
            gender=9
        if hypertension=='No':
            hypertension=0
        else:
            hypertension=1
        if heartdisease=='No':
            heartdisease=0
        else:
            heartdisease=1
        if married=='No':
            married=0
        else:
            married=1
        if work=='Private':
            work=2
        elif work=='Self-employed':
            work=3
        elif work=='Child':
            work=4
        elif work=='govt job':
            work=0
        elif work=='Never worked':
            work=1
        if residence=='Rural':
            residence=0
        else:
            residence=1
        if smoking=='formerly smoked':
            smoking=1
        elif smoking=='never smoked':
            smoking=2
        elif smoking=='smokes':
            smoking=3
        else:
            smoking=0
        prediction=loaded_model.predict([[gender,	age,	hypertension,	heartdisease,	married,	work,	residence,	glucose,	bmi,	smoking]])
        for i in prediction:
            op=i
            st.write(ref[i])
        if op=='You have a chance of getting a stroke!':
                blank1, text, blank2=st.columns([0.45,5,0.8])
                with blank1:
                    st.write(' ')
                with text:
                    st.title('Stroke symptoms')
                    st.write('The loss of blood flow to the brain damages tissues within the brain. Symptoms of a stroke show up in the body parts controlled by the damaged areas of the brain.')
                    st.write(' ')
                    st.header('The sooner a person having a stroke gets care, the better their outcome is likely to be. For this reason, it’s helpful to know the signs of a stroke so you can act quickly. Stroke symptoms can include:')
                    st.write("The general symptoms of diabetes include:")
                    st.write(' ')
                    st.write('''
● paralysis\n
● numbness or weakness in the arm, face, and leg, especially on one side of the body\n
● trouble speaking or understanding others\n
● slurred speech\n
● confusion, disorientation, or lack of responsiveness\n
● sudden behavioral changes, especially increased agitation\n
● vision problems, such as trouble seeing in one or both eyes with vision blackened or blurred, or double vision\n
● trouble walking\n
● loss of balance or coordination\n
● dizziness\n
● severe, sudden headache with an unknown cause\n
● seizures\n
● nausea or vomiting''')
                    st.header('How to prevent a stroke')
                    st.write("Lifestyle changes can’t prevent all strokes. But many of these changes can make a radical difference when it comes to lowering your risk of stroke.")
                    st.write(' ')
                    st.header('These changes include the following:')
                    st.write('Quit smoking. If you smoke, quitting now will lower your risk of stroke. You can reach out to your doctor to create a quit plan.')
                    st.write('')
                    st.write('Limit alcohol use. Heavy alcohol consumption can raise your blood pressure, which in turn raises the risk of stroke. If reducing your intake is difficult, reach out to your doctor for help.')
                    st.write('''Keep a moderate weight. Overweight and obesity increases the risk of stroke. To help manage your weight, eat a balanced diet and stay physically active more often than not. Both steps can also reduce blood pressure and cholesterol levels.''')
                    st.write(' ')
                    st.write('Diabetes-friendly exercises include:')
                    st.write('''Get regular checkups. Talk with your doctor about how often to get a checkup for blood pressure, cholesterol, and any conditions you may have. They can also support you in making these lifestyle changes and offer guidance.''')
                with blank2:
                    st.write(' ')
elif disease=='Plant Diseases':
    op=''
    ref= {"0":"Apple scab",
    "1":"Apple Black rot",
    "2":"Apple Cedar rust",
    "3":"Apple healthy",
    "4":"Blueberry healthy",
    "5":"Cherry (including_sour) Powdery_mildew",
    "6":"Cherry (including_sour) healthy",
    "7":"Corn(maize) Cercospora_leaf spot Gray",
    "8":"Corn_(maize) Common rust",
    "9":"Corn_(maize) Northern Leaf Blight",
    "10":"Corn (maize) healthy",
    "11":"Grape Black rot",
    "12":"Grape Esca (Black Measles)",
    "13":"Grape Leaf blight (Isariopsis Leaf Spot)",
    "14":"Grape healthy",
    "15":"Orange Haunglongbing (Citrus greening)",
    "16":"Peach Bacterial spot",
    "17":"Peach healthy",
    "18":"Bell Pepper Bacterial spot",
    "19":"Bell Pepper healthy",
    "20":"Potato Early blight",
    "21":"Potato Late blight",
    "22":"Potato healthy",
    "23":"Raspberry healthy",
    "24":"Soybean healthy",
    "25":"Squash Powdery mildew",
    "26":"Strawberry Leaf scorch",
    "27":"Strawberry healthy",
    "28":"Tomato Bacterial spot",
    "29":"Tomato Early blight",
    "30":"Tomato Late blight",
    "31":"Tomato Leaf Mold",
    "32":"Tomato Septoria leaf spot",
    "33":"Tomato Spider mites, Two-spotted spider mite",
    "34":"Tomato Target Spot",
    "35":"Tomato Yellow Leaf Curl Virus",
    "36":"Tomato mosaic virus",
    "37":"Tomato healthy"}

    def prediction(img_input):
        #img=load_img(path, target_size=(256,256))
        img=img_input
        img=img.resize((256, 256))
        i=img_to_array(img)
        im=preprocess_input(i)
        img=np.expand_dims(im, axis=0)
        pred=np.argmax(model.predict(img))
        pred=str(pred)
        print(f" The image is most likely classified under { ref[pred] } ")
        st.write(f" The image is most likely classified under { ref[pred] } ")
        op=ref[pred]
        return op

    def load_image(image_file):
        img = Image.open(image_file)
        return img

    image_inp = st.file_uploader("Upload your image for identification", type=["png","jpg","jpeg"])
    if image_inp is not None:
            file_details = {"filename":image_inp.name, "filetype":image_inp.type, "filesize":image_inp.size}
            st.write(file_details)
            img_input=st.image(load_image(image_inp),width=250)
            image_data = image_inp.getvalue()
            #st.image(image_data)
            img_input = Image.open(io.BytesIO(image_data))
            model = keras.models.load_model('plantsmodel.h5')
            op=prediction(img_input)
            op=op.split(' ')
            for i in op:
                if i=='healthy':
                    op='healthy'
            if op!='healthy':
                blank1, text, blank2=st.columns([0.45,5,0.8])
                with blank1:
                    st.write(' ')
                with text:
                    st.title('Signs and symptoms of plant disease: Is it fungal, viral or bacterial?')
                    st.write('Most plant diseases – around 85 percent – are caused by fungal or fungal-like organisms. However, other serious diseases of food and feed crops are caused by viral and bacterial organisms. Certain nematodes also cause plant disease. Some plant diseases are classified as “abiotic,” or diseases that are non-infectious and include damage from air pollution, nutritional deficiencies or toxicities, and grow under less than optimal conditions. For now, we’ll look at diseases caused by the three main pathogenic microbes: fungus, bacteria and virus. If plant disease is suspected, careful attention to plant appearance can give a good clue regarding the type of pathogen involved.')
                    st.write('')
                    st.write('A sign of plant disease is physical evidence of the pathogen. For example, fungal fruiting bodies are a sign of disease. When you look at powdery mildew on a lilac leaf, you’re actually looking at the parasitic fungal disease organism itself (Microsphaera alni). Bacterial canker of stone fruits causes gummosis, a bacterial exudate emerging from the cankers. The thick, liquid exudate is primarily composed of bacteria and is a sign of the disease, although the canker itself is composed of plant tissue and is a symptom.')
                    st.write('')
                    st.write('A symptom of plant disease is a visible effect of disease on the plant. Symptoms may include a detectable change in color, shape or function of the plant as it responds to the pathogen. Leaf wilting is a typical symptom of verticilium wilt, caused by the fungal plant pathogens Verticillium albo-atrum and V. dahliae. Common bacterial blight symptoms include brown, necrotic lesions surrounded by a bright yellow halo at the leaf margin or interior of the leaf on bean plants. You are not actually seeing the disease pathogen, but rather a symptom that is being caused by the pathogen.')
                    st.write('')
                    st.subheader('Fungal disease signs:')
                    st.write('''
● Leaf rust (common leaf rust in corn)\n
● Stem rust (wheat stem rust)\n
● Sclerotinia (white mold)\n
● Powdery mildew''')
                    st.write(' ') 
                    st.subheader('Fungal disease symptoms:')
                    st.write('''
● Birds-eye spot on berries (anthracnose)\n
● Damping off of seedlings (phytophthora)\n
● Leaf spot (septoria brown spot)\n
● Chlorosis (yellowing of leaves)
''')
                    st.write(' ') 
                    st.subheader('Viral disease signs:')
                    st.write('None – the viruses themselves can’t be seen')
                    st.write(' ') 
                    st.subheader('Viral disease symptoms:')
                    st.write('''
● Mosaic leaf pattern\n
● Crinkled leaves\n
● Yellowed leaves\n
● Plant stunting''')
                with blank2:
                    st.write(' ')
    else:
        st.subheader('Sample image')
        st.image(
            "https://github.com/Utkarshhh20/medicare/blob/master/PotatoEarlyBlight3.JPG?raw=true",
        )
elif disease=='News':
    vader = SentimentIntensityAnalyzer()
    blank1, head, checkbox, blank2=st.columns([0.1,2,0.5,0.3])
    with head:
        newstxt='''
                <link href='https://fonts.googleapis.com/css?family=Montserrat' rel="stylesheet">
                <style>
                .tweepyhead {
                    font-family:Montserrat;
                    font-size:40px;
                    font-weight:1000;
                    font-style: bold;
                    float:left;
                    margin-left:60px;
                    margin-top: 0px;
                    margin-right: 20px;
                            }
                #newsicon {
                    margin-top: 0px;
                }
                </style>

                <body>
                <center><p1 class='tweepyhead'> Latest News</p1></center>
                <svg xmlns="http://www.w3.org/2000/svg" width="55" height="55" fill="currentColor" class="bi bi-newspaper" id='newsicon' viewBox="0 0 16 16">
                <path d="M0 2.5A1.5 1.5 0 0 1 1.5 1h11A1.5 1.5 0 0 1 14 2.5v10.528c0 .3-.05.654-.238.972h.738a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 1 1 0v9a1.5 1.5 0 0 1-1.5 1.5H1.497A1.497 1.497 0 0 1 0 13.5v-11zM12 14c.37 0 .654-.211.853-.441.092-.106.147-.279.147-.531V2.5a.5.5 0 0 0-.5-.5h-11a.5.5 0 0 0-.5.5v11c0 .278.223.5.497.5H12z"/>
                <path d="M2 3h10v2H2V3zm0 3h4v3H2V6zm0 4h4v1H2v-1zm0 2h4v1H2v-1zm5-6h2v1H7V6zm3 0h2v1h-2V6zM7 8h2v1H7V8zm3 0h2v1h-2V8zm-3 2h2v1H7v-1zm3 0h2v1h-2v-1zm-3 2h2v1H7v-1zm3 0h2v1h-2v-1z"/>
                </svg>
                </body>
                            '''
        st.markdown(newstxt, unsafe_allow_html=True)
    with checkbox:
        sentiment=st.checkbox('Sentimental Score of news')
    blank1, news, blank2=st.columns([0.5,2,0.7])
    with blank1:
        st.write(' ')
    with news:
        st.write(' ')
        st.write(' ')
        url = 'https://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=f92e03e9f1b5497b96117c9ed2bad6b7'
        headers1 = {"user-agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:53.0) Gecko/20100101 Firefox/53.0"}
        results = requests.get(url=url, headers=headers1)
        newsdata=results.json()['articles']
        print(newsdata)
        for i in newsdata:
            for j in i:
                if j=='title':
                    st.header(i[j])
                elif j=='description':
                    if i[j]!=None:
                        st.subheader(i[j])
                elif j=='url':
                    link=i[j]
                    st.write(f"[More on this article]({link})")
                    if sentiment==True:
                        scores = vader.polarity_scores(i['title'])
                        if i['description']!=None:
                            scores2 = vader.polarity_scores(i['description'])
                            for k in scores:
                                scores[k]=(scores[k]+scores2[k])/2
                        st.write('Sentimental Score (Vader Analysis): ', scores)
                        x=i['title']
                        x=tb.TextBlob(x)
                        scorestb=x.sentiment.polarity
                        st.write('Sentimental Score (TextBlob Analysis): ', scorestb)
                    st.write(' ')
                    st.write('_______________________')
                # Iterate through the headlines and get the polarity scores using vader
    with blank2:
        st.write(' ')
