import streamlit as st 
import requests
from streamlit_lottie import st_lottie


def home():
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
    st.markdown("")
    st.write("Health Predictor is an online platform that utilizes federated learning techniques to predict inflammation of the urinary bladder and nephritis of renal pelvis origin. Our platform ensures privacy and data security by training the prediction model directly on user data without the need to share sensitive medical information. By leveraging the distributed nature of federated learning, we provide predictions while preserving the confidentiality of user data.")

    st.write("""## What is Federated Learning ?""")
    st.write("Federated learning is a machine learning approach where a model is trained across multiple decentralized edge devices or servers holding local data samples, such as smartphones, IoT devices, or servers in different locations. Instead of sending raw data to a central server for training, which can raise privacy concerns or consume significant bandwidth, federated learning allows for model training to occur locally on each device/server.")

    st.markdown("")
    st.markdown("")

    st.image('Fedlearn.png', caption='Federated Learning')
    st.markdown("")
    st.markdown("")

    st.write("""## Federated learning offers several benefits""")
    st.markdown("")
    st.markdown(" **Privacy Preservation:** ""Raw data remains on the local device/server, reducing privacy concerns associated with centralizing data.")
    st.markdown(" **Reduced Communication Costs:** ""Only model updates are communicated, reducing bandwidth usage.")
    st.markdown(" **Decentralization:** ""It enables learning from distributed data sources without centralizing them.")
    st.markdown(" **Edge Computing:** ""It leverages local computation power, making it suitable for edge devices with limited connectivity.")
    st.markdown("")
    
    st.markdown("")



home()






