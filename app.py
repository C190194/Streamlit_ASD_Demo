import streamlit as st

import pandas as pd
import numpy as np

from PIL import Image

from AE import AE_application as ae
from MB import MB_application as mb



# Upload audio files for testing
import pydub
import glob
import os
from pathlib import Path




# Overall setting

st.set_page_config(layout="wide")

# CSS
# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     background-color: rgb(204, 49, 49);
# }
# </style>""", unsafe_allow_html=True)

# Main GUI

st.title("Anomalous Sound Detection Demo Prototype") 

side_selectbox = st.sidebar.selectbox(
    "Which page would you like to go to?",
    ("Introduction", "Autoencoder", "MobileNetV2",
     "Auxiliary Classification", "Density Estimation",
     "Experiment with Trained Models")
)

# Page 1
if side_selectbox == "Introduction":
    st.header("Introduction")
    st.write("This application will briefly introduce the baseline methods and Team JKU's methods\
        for the task, Anomalous Sound Detection with Shifted Domains.")
    st.write("In this task, all the training data is from the normal machines and only contains\
        few samples from the target domain, where the machine operates with a different spin rate\
        or in different conditions from the source domain.")
    st.write("The aim of this task is to produce models which can detect abnormal sounds from both\
        source and target domains.")
    st.write("For each method, one model will be trained for each of the 7 machine types in the\
        training dataset.")
    st.write("All the models will be evaluated using the area under curve (AUC) and partial-AUC\
        (pAUC) scores defined below. The higher the scores are, the better the models are.")
    
    auc_img = Image.open('./pic/AUC_def.png')
    st.image(auc_img, caption='Evaluation Metric of the Models')

# Page 2
elif side_selectbox == "Autoencoder":
    st.header("Baseline 1: Autoencoder-based Method")
    st.write("Before training, each 10-second-long audio file is converted to a 313*640\
        Log Mel-spectrogram vector using the Librosa package.")
    st.write("The Autoencoder model is trained to minimize the input audio file's reconstruction error,\
        which is used as its anomaly score.")
    st.write("The architecture of the Autoencoder model is shown below.")

    ae_bl_archi_df = pd.read_csv("./csv/ae-bl-archi.csv", delimiter=',')
    st.write(ae_bl_archi_df)

    st.write("- Learning (epochs: 100, batch size: 512, data shuffling between epochs)")
    st.write("- Optimizer: Adam (learning rate: 0.001)")

    ae_anomaly_score_img = Image.open('./pic/ae_anml_score.png')
    st.image(ae_anomaly_score_img, width = 500)


# Page 3
elif side_selectbox == "MobileNetV2":
    st.header("Baseline 2: MobileNetV2-based Method")
    st.write("The MobileNetV2 model will output the softmax value for each section\
         of the selected machine type. The anomaly score is calculated as the averaged\
         negative logit of the predicted probabilities for the correct section.")
    st.write("The architecture of the MobileNetV2 model is shown below.")

    c1, c2 = st.columns(2)
    with c1:
        mb_bl_archi_1_img = Image.open('./pic/mb-bl-archi-1.png')
        st.image(mb_bl_archi_1_img)
        st.write("- Learning (epochs: 20, batch size: 32, data shuffling between epochs)")
        st.write("- Optimizer: Adam (learning rate: 0.00001)")
        st.write("")
        st.write("")
        st.write("")
        st.write("The anomaly score is calculated as:")
        mb_anomaly_score_img = Image.open('./pic/mb_anml_score.png')
        st.image(mb_anomaly_score_img, width = 500)
    with c2:
        mb_bl_archi_img = Image.open('./pic/mb-bl-archi.png')
        st.image(mb_bl_archi_img, caption="Architecture of MobileNetV2", width = 400)
    
# Page 4
elif side_selectbox == "Auxiliary Classification":
    st.header("Auxiliary Classification Method from Team JKU")
    st.write("For the Auxiliary Classifier, the Team JKU used a receptive-field-regularized,\
        fully convolutional, residual network (ResNet). The receptive field was tuned such that\
        the initial anomaly detection performance across all machine types without outlier\
        exposure was maximized.")
    st.write("Each training audio data was given a label (integer from 1 to 6) denoting its domain\
        (source or target) and section (00, 01 or 02).")
    st.write("During training, the Auxiliary Classifier was trained to minimize the cross entropy\
        loss.")
    AC_CEL_img = Image.open('./pic/JKU_AC_CEL.png')
    st.image(AC_CEL_img)
    st.write("The negative probability of the true class (label) of the input audio file was used as\
        its anomaly score.")
    AC_AS_img = Image.open('./pic/JKU_AC_AS.png')
    st.image(AC_AS_img)

    st.subheader("Proxy Outliers")
    st.write("Additionally, during training\
        for the selected machine type, audio files from other machine types could be used\
        as proxy outliers. ")
    st.write("For Auxiliary Classifier, the loss for a proxy outlier example\
        is calculated by enforcing a close-to-uniform class probability distribution via\
        the cross-entropy loss H:")
    AC_PO_CEL_img = Image.open('./pic/JKU_AC_PO_CEL.png')
    st.image(AC_PO_CEL_img)
    AC_PO_CEL_ex_img = Image.open('./pic/JKU_AC_PO_CEL_ex.png')
    st.image(AC_PO_CEL_ex_img, width=250)

# Page 5
elif side_selectbox == "Density Estimation":
    st.header("Density Estimation Method from Team JKU")  
    st.write("For Density Estimators, the Team JKU applied Masked Autoencoder for Distribution\
        Estimation (MADE) and Masked Autoregressive Flows (MAF).")
    st.write("The models were trained to maximize the log likelihood on the normal data.")
    DE_L_img = Image.open('./pic/JKU_DE_L.png')
    st.image(DE_L_img)
    st.write("Then, the negative log probabilities were used as anomaly scores of the input\
        audio files.")
    DE_AS_img = Image.open('./pic/JKU_DE_AS.png')
    st.image(DE_AS_img)
    
    st.subheader("Proxy Outliers")
    st.write("Silimar to Auxiliary Classification,\
        audio files from other machine types could also be used as proxy outliers when training\
        density estimators. ")
    st.write("A margin ranking loss was computed for each proxy outlier example so that it\
        receives a lower log probability. ")
    st.write("The margin loss between a normal sample (x, y) and a proxy outlier (x', y') is\
        computed by:")
    DE_PO_ML_img = Image.open('./pic/JKU_DE_PO_ML.png')
    st.image(DE_PO_ML_img)
    st.write("Then, the margin ranking loss is computed by:")
    DE_PO_MRL_img = Image.open('./pic/JKU_DE_PO_MRL.png')
    st.image(DE_PO_MRL_img)

# Page 6
elif side_selectbox == "Experiment with Trained Models":

    with st.form("uploader-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Choose the audio files for testing",
                                type=['wav', 'mp3'], accept_multiple_files=True)
        submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_files is not None:
        st.write("New file added!")
        # do stuff with your uploaded files
        save_dir = "./temp_audio_file"
        save_paths = []
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                if uploaded_file.name.endswith('wav'):
                    audio = pydub.AudioSegment.from_wav(uploaded_file)
                    file_type = 'wav'
                elif uploaded_file.name.endswith('mp3'):
                    audio = pydub.AudioSegment.from_mp3(uploaded_file)
                    file_type = 'mp3'

                save_path = Path(save_dir) / uploaded_file.name
                save_paths.append(save_path)
                audio.export(save_path, format=file_type)
        stored_files = glob.glob('./temp_audio_file/*.wav')


    # Display the list of paths of the uploaded files
    stored_files = glob.glob('./temp_audio_file/*.wav')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Uploaded files:")
    with c2:
        if st.button('Clear the uploaded files'):
            # removed previously stored files
            stored_files = glob.glob('./temp_audio_file/*.wav')
            for f in stored_files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            stored_files = glob.glob('./temp_audio_file/*.wav')
    if len(stored_files) > 0:
        stored_files.sort()
        for file_path in stored_files:
            st.write(file_path)
    else:
        st.write("None")

    # Options for loading the models
    col1, col2 = st.columns(2)
        
    with col1:
        model_option = st.selectbox(
            'Which model would you like to use?',
            ('Baseline 1: Autoencoder', 'Baseline 2: MobileNetV2', 
            'Auxiliary Classification', 'Density Estimation'))

    with col2:
        machine_type_option = st.selectbox(
            'Which machine type does the uploaded audio file belong to?',
            ('fan', 'pump', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'))

    if st.button('Get prediction'):
        if machine_type_option is None:
            st.write('Please choose the machine type of the audio file.')
        elif model_option == 'Baseline 1: Autoencoder':
            model, decision_threshold = ae.load_model(machine_type_option)
            # get prediction for each uploaded file
            result_list = []
            for file_path in stored_files:
                prediction = ae.get_prediction(model, decision_threshold, file_path)
                if prediction == 1:
                    result_list.append('Anomalous')
                else:
                    result_list.append('Normal')
            # show results as a form
            result_dic = {"File path:": stored_files,
                        "Result:": result_list}
            result_df = pd.DataFrame(data=result_dic)
            st.dataframe(result_df)
        elif model_option == 'Baseline 2: MobileNetV2':
            model, trained_section_names, decision_threshold = mb.load_model(machine_type_option)
            # get prediction for each uploaded file
            result_list = []
            for file_path in stored_files:
                prediction = mb.get_prediction(model, trained_section_names, decision_threshold, file_path)
                if prediction == 1:
                    result_list.append('Anomalous')
                else:
                    result_list.append('Normal')
            # show results as a form
            result_dic = {"File path:": stored_files,
                        "Result:": result_list}
            result_df = pd.DataFrame(data=result_dic)
            st.dataframe(result_df)
            
    
    

