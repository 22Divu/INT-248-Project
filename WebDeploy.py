import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2

title="VRL Skin Cancer Detector"
st.set_page_config(
        page_title=title,
        page_icon="MyLogoLight.png",
        layout="centered"
    )

st.image("MyLogoLight.png")


head='''
        <style>
        h1{
            color: aquamarine;
            font-size:30;
            
        }
        h2{
            color:red;
            text-align:center;
            font-size:20;
        }
        h1:hover{
               color:blue;
               font-size:40px;
        }
        h2:hover{
            color: blue;
            font-size:50px;
        }
        #cname{
            
            margin:2px;
            padding:3px;
            margin-bottom:5px;
            border-radius:10%;
            background-color: aqua;
            color: blue;
            animation: 20s mt infinite;
            text-align:center;
        }
        @keyframes mt
        {
            0%{
               background-color: aqua; 
            }
            30%{
                background-color: pink; 
            }
            60%{
                background-color: lime; 
            }
        }
        #cname:hover{
            margin:6px;
            padding:9px;
            margin-bottom:9px;
            font-size:10px;
        }
        button{
        border: 1px solid grey;
        border-radius: 10%;
        padding: 10px 50px; 
        background-color: white;
        text-align: center;
        font-size: 14px;
        }
    button:hover{
        border:1px solid red;
        color: red;
        }

    </style>
    <body>
    <center> <h1> Skin Cancer </h1> </center>  <h2>Classifying Engine</h2>  
    </body>
        
'''
st.markdown(head,unsafe_allow_html=True)

classes={2:"Melanoma",0:"Melanocytic Nevi",1:"Others"}

def ModelLoader(path):
    return load_model(path)

def Predict_Class(img):
    path_model="model.h5"
    
    #img_n=tf.image.resize(img,(28,28,3))
    model=ModelLoader(path_model)
    class_prob= model.predict(np.expand_dims(img,axis=0))
    cname=classes[np.argmax(class_prob,axis=1)[0]]
    return cname,class_prob
    #return 1,np.array([0.1,0.9,0.2])


    return classes[label]

def Layout(img,live=False):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(28,28))
    classname,class_percent= Predict_Class(img)
    class_percent*=100

    if live:
        
        st.image(img,"Catured Spot")
    with st.expander("Class Probability"):
        names= list(classes.values())
        print(names)
        #probs=np.array([0.9,0.1,0.2])*100
        #classes=["A","B","C"]

        #prob=pd.DataFrame(probs.reshape(-1,1),index=["A","B","C"])
            
        prob=pd.DataFrame(class_percent.reshape(-1,1),index=["Melanocytic Nevi","Others","Melanoma"])
        st.bar_chart(prob)
        st.header("So, You have {}".format(classname))

pages=["Upload Image","Take The Image"]
option=st.selectbox("Ways to Use",pages)




if option in pages[0]:
    img_read = st.file_uploader("Insert An Image of your Cancer Wound",["jpg","jpeg","png","jfif"])
    if img_read is not None:
        img = Image.open(img_read)
        img = img.resize((28,28))
        img = np.array(img)
        print("Image: ",img.shape)
        Layout(img)
        
elif option in pages[1]:
    if st.button("Capture The Spot and Then Analyze"):
        cap = cv2.VideoCapture(0)
        while True:
            _,img=cap.read()
            #blur = cv2.blur(img,(200,200))
            
            blur = cv2.imread("bg1.jpg")
            blur=cv2.resize(blur,(1000,750))
            img=cv2.resize(img,(1000,750))

            x,y,xlen,ylen=250,250,600,750

            cv2.rectangle(img,(y,x),(ylen,xlen),(25,0,240),10)
            blur[x:xlen,y:ylen,:]=img[x:xlen,y:ylen,:]
            
            cv2.putText(blur,"Press C to click spot",(125,125),cv2.FONT_HERSHEY_SIMPLEX,2,(255,191,0),5)
            cv2.putText(blur,"Press Q to quit",(200,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,69,255),5)
            
            cv2.imshow("Capture Cancer Spot",blur)
            
            
            k = cv2.waitKey(1)
            flag=True
            if k & 0xFF in [ord('c'),ord('C')]:
                extract_img=img[x:xlen,y:ylen,:]
                break
            elif k & 0xFF in [ord('q'),ord('Q')]:
                flag=False
                break
        cap.release()
        cv2.destroyAllWindows()

        if flag:
            Layout(extract_img,True)





            
        


        

    







