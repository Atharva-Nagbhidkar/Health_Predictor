# Health Predictor

Health Predictor is an online platform utilizing federated learning techniques to predict inflammation of the urinary bladder and nephritis of renal pelvis origin. Our platform ensures privacy and data security by training the prediction model directly on user data without the need to share sensitive medical information. Leveraging the distributed nature of federated learning, we provide predictions while preserving the confidentiality of user data.This project extensively employs federated learning methodologies.

## What is Federated Learning?

Federated learning is a machine learning approach where a model is trained across multiple decentralized edge devices or servers holding local data samples, such as smartphones, IoT devices, or servers in different locations. Instead of sending raw data to a central server for training, which can raise privacy concerns or consume significant bandwidth, federated learning allows for model training to occur locally on each device/server.

### Federated Learning Benefits:
1. **Privacy Preservation**: Raw data remains on the local device/server, reducing privacy concerns associated with centralizing data.
2. **Reduced Communication Costs**: Only model updates are communicated, reducing bandwidth usage.
3. **Decentralization**: Enables learning from distributed data sources without centralizing them.
4. **Edge Computing**: Leverages local computation power, making it suitable for edge devices with limited connectivity.

## Approach

Federated learning is a decentralized approach to training machine learning models. In traditional machine learning, data is collected from various sources, centralized in one location, and then used to train a model. However, in federated learning, instead of bringing all the data to one central server, the model is sent to where the data is located.

### How it Works:
1. **Initialization**: Central model is created and sent to all the participating devices or nodes.
2. **Training on Local Data**: Each device or node trains the model locally using its own data, typically using techniques like stochastic gradient descent (SGD).
3. **Model Update**: After training, each device computes model updates (gradients), sending only these updates back to the central server.
4. **Aggregation**: Updates from all devices are aggregated, often by computing their average or weighted average, representing collective knowledge.
5. **Iterative Process**: Steps 2-4 are repeated iteratively, with updated models sent back for further training on local data.

## Project Overview

1. **Data Preparation**: Dataset includes medical diagnosis information with 6 input features and 2 output targets.
2. **Model Training**: Logistic regression implemented using PyTorch, optimizing binary cross-entropy loss function with stochastic gradient descent (SGD). Achieved training accuracy: 97.17%, testing accuracy: 93.40%.
3. **Federated Learning**: Implemented to train the model across multiple hospitals while keeping data decentralized. Hospitals trained local models using own data and shared updates with a secure central server. Federated accuracy achieved: 92.45%.

## Prerequisites

1. PyTorch
2. Pandas
3. Numpy
4. Streamlit
5. Streamlit Lottie

## How to Run App

1. **Install virtualenv**:
    - **MacOS**: 
        ```
        sudo python2 -m pip install virtualenv
        ```
    - **Windows**: 
        ```
        py -2 -m pip install virtualenv
        ```

2. **Create an Environment (Should be in same directory as the Project files)**:
    - **MacOS Python 3**: 
        ```
        python3 -m venv <name of environment>
        ```
    - **MacOS Python 2**: 
        ```
        python -m virtualenv <name of environment>
        ```
    - **Windows Python 3**: 
        ```
        py -3 -m venv <name of environment>
        ```
    - **Windows Python 2**: 
        ```
        py -2 -m virtualenv <name of environment>
        ```

3. **Activate the Environment**:
    - **MacOS**: 
        ```
        . <name of environment>/bin/activate
        ```
    - **Windows**: 
        ```
        <name of environment>\Scripts\activate
        ```

4. **Run the App**:
    ```
    streamlit run <appname>.py
    ```
