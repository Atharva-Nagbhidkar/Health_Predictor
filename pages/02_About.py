import streamlit as st
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie


def about():
    col1, col2 = st.columns([0.3,0.7])
    with col1:
        url = requests.get("https://lottie.host/cfa6d512-d818-4b19-9cef-63e4fbfeffc4/yIe8LWxglL.json") 

        url_json = dict()

        if url.status_code == 200: 
            url_json = url.json() 
        else: 
            print("Error in the URL") 

        st_lottie(url_json,height=200, width=200, speed=1, loop=True) 

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.write("<h1 style='text-align: left;'>Health Predictor </h1>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: left; color:grey'>Predicting Health, Protecting Privacy </h5>", unsafe_allow_html=True)
        
    st.markdown("")

    st.write("""## Approach""")
    st.markdown("")
    st.write("Federated learning is a decentralized approach to training machine learning models. In traditional machine learning, data is collected from various sources, centralized in one location, and then used to train a model. However, in federated learning, instead of bringing all the data to one central server, the model is sent to where the data is located.")

    
    st.write("Here's how it works:")
    st.markdown(" **1) Initialization:** ""Central model is created and sent to all the participating devices or nodes.")
    st.markdown(" **2) Training on Local Data:** ""Each device or node trains the model locally using its own data. This training is done using techniques like stochastic gradient descent (SGD).")
    st.markdown(" **3) Model Update:** ""After training on local data, each device computes the model's updates. However, instead of sending the raw data back to the central server, only the model updates (gradients) are sent.")
    st.markdown(" **4) Aggregation:** ""The updates from all the devices are aggregated, typically by computing their average or weighted average. This aggregated update represents the collective knowledge learned from all the devices' data.")
    st.markdown(" **5) Iterative Process:** ""Steps 2-4 are repeated iteratively, with the updated model being sent back to the devices for further training on their local data.")
    st.markdown("")

    st.write("""## Data Set Information""")
    st.markdown("")
    st.write("The main idea of this data set is to prepare the algorithm of the expert system, which will perform the presumptive diagnosis of two diseases of urinary system. It will be the example of diagnosing of the acute inflammations of urinary bladder and acute nephritises. For better understanding of the problem, let us consider definitions of both diseases given by medics.")
    st.markdown(" **Acute inflammation of urinary bladder** ""is characterised by sudden occurrence of pains in the abdomen region and the urination in form of constant urine pushing, micturition pains and sometimes lack of urine keeping. Temperature of the body is rising, however most often not above 38°C. The excreted urine is turbid and sometimes bloody. At proper treatment, symptoms decay usually within several days. However, there is inclination to returns. At persons with acute inflammation of urinary bladder, we should expect that the illness will turn into protracted form.")
    st.markdown(" **Acute nephritis of renal pelvis origin** ""occurs considerably more often at women than at men. It begins with sudden fever, which reaches, and sometimes exceeds 40°C. The fever is accompanied by shivers and one- or both-side lumbar pains, which are sometimes very strong. Symptoms of acute inflammation of urinary bladder appear very often. Quite not infrequently there are nausea and vomiting and spread pains of whole abdomen.")


    st.write("""## Results""")
    st.markdown("")

    st.markdown("We define our machine learning model, which is a logistic regression model. Why? Because this medical dataset is linearly separable, which simplifies things a lot.")
    st.markdown("In this demo, there are 4 hospitals. (The dataset will be split in 4, randomly.) There could be more hospitals. The 4 hospitals cannot share the cases of their patients because they are competitors and it is necessary to protect the privacy of patients. Hence, the ML model will be learned in a federated way.")
    st.markdown("How?")
    st.markdown("Federated learning is iterated 1600 times. At each iteration, a copy of the shared model is sent to all the 4 hospitals. Each hospital trains its own local model with its own local dataset, in 5 local iterations. Each local model improves a little bit in its own direction. Then we compute the local losses and local accuracies to keep track of them and to make graphs of them. We send the local models to the trusted aggregator that will average all the model updates. This averaged model is the shared model that is sent to all the 4 hospitals at the begining of each iteration.")
    st.markdown("We actually train the machine learning model to diagnose the Nephritis of Renal Pelvis Origin. As you can see in the graphs, the training loss drops quickly to almost zero and the training accuracy reaches the 97.17%. The testing accuracy is also 94.34%.")
    st.markdown("")
    st.image('Neph.png', caption='Nephritis of Renal Pelvis Origin (Federated Process)')
    st.markdown("")
    st.markdown("")


    st.markdown("We actually train the machine learning model to diagnose the Inflammation of Urinary Bladder. As you can see in the graphs, the training loss drops quickly to almost zero and the training accuracy reaches the 96.23%. The testing accuracy is also 93.40%.")
    st.markdown("")   
    st.image('UrinaryB.png', caption='Inflamation of Urinary Bladder (Federated Process)')
    st.markdown("")
    st.markdown("The learning curves Training Losses versus Iterations and Training Accuracies versus Iterations have 4 colors for all 4 hospitals. Each graph has 4 curves of different colors: Blue, orange, green, and red. The curves are not lines; they are rather regions. Why? Because each iteration of federated learning is complex: First, 5 local iterations in each virtual worker (each hospital) to train each local model. Each local model improves a little bit in its own direction. Then, the 4 different models are sent to the trusted aggregator that averages them. Finally, the averaged model is sent back to the 4 hospitals. Such averaged model can have lower performance in comparison to the local models, which are more locally adapted to the local datasets. That's why the progress in the learning curves goes back and forth. Moreover, the graph has 1600 iterations. That's why the curves becomes regions. Because the curves go back and forth too often and are quite dense.")


    st.markdown("")

    st.write("""### Tech Stack""")
    

    labels = ['Python', 'Torch', 'Jupyter Notebook', 'Streamlit']
    sizes = [55, 10, 25 , 10]
    explode = (0.08, 0.08, 0.08, 0.08)
    woodsmoke="#0e1117"

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,explode=explode,shadow=True,textprops={'color':"w"} ,labels=labels, autopct='%1.1f%%', startangle=90)
    plt.gcf().patch.set_facecolor(woodsmoke)
    ax1.axis('equal') 

    st.pyplot(fig1)

about()
