# qc_streamlit
# Libraries
import os
import cv2
import random
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

################################
#######   FUNCTIONS  ###########
################################
# Function to extract image size
def image_size(img_array_list):
    img_shape = []
    for img in img_array_list:
        img_shape.append(img.shape[:2])

    return set(img_shape)

# Function to extract annotation information
def label_info(image_list):
    label_list = sorted([f[:f.rfind('.')] + '.txt' for f in image_list if os.path.exists(os.path.join(label_folder_path, f[:f.rfind('.')] + '.txt'))])
    total_annotations = 0
    classes = []
    for labels in label_list:
        with open(os.path.join(label_folder_path, labels), 'r') as file:
            annotations = file.readlines()
            for annotation in annotations:
                annotation = annotation.strip().split()
                if len(annotation) < 9:
                    continue  # Skip if the annotation does not have enough values

                classes.append(annotation[8])
                total_annotations += 1
    
    return set(classes), int(total_annotations)

# Function to check if given paths are valid directories
def isDir(image_folder_path, label_folder_path):
    if os.path.isdir(image_folder_path):
        st.sidebar.success(f"Valid Image Folder!")
    else:
        st.sidebar.error("Please enter a valid image folder path.")

    if os.path.isdir(label_folder_path):
        st.sidebar.success(f"Valid Label Folder!")
    else:
        st.sidebar.error("Please enter a valid label folder path.")    

# Function to plot the annotations on the image and return the image 
def annotate_image(image_path, label_folder_path, view_labels=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filename = os.path.basename(image_path)
    label_path = os.path.join(label_folder_path, filename[:filename.rfind('.')] + '.txt')

    with open(label_path, 'r') as file:
        annotations = file.readlines()

    for annotation in annotations:
        annotation = annotation.strip().split()
        if len(annotation) < 9:
            continue  # Skip if the annotation does not have enough values

        label = annotation[8]
        try:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, annotation[:8])
        except ValueError:
            continue  # Skip if the conversion to float fails

        coords = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        cv2.polylines(image, [coords], isClosed=True, color=(0, 255, 0))
        if view_labels:
            cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Function to extract the no. of annotations in single image
def single_image_ann_info(image_path):
    no_of_annotations = 0
    filename = image_path[:image_path.rfind('.')] + '.txt'
    with open(os.path.join(label_folder_path, filename), 'r') as f:
        annotations = f.readlines()
        for annotation in annotations:
            if len(annotation) > 1:
                no_of_annotations += 1

    return no_of_annotations


################################
#######      UI      ###########
################################

tab1, tab2 = st.tabs(["QC", "Stats"])

# Date
todays_date = datetime.datetime.now()
jan_1 = datetime.date(todays_date.year, 1, 1)

custom_date = st.sidebar.checkbox('Custom date')
if custom_date:
    dataset_date = st.sidebar.date_input("Select date", jan_1, format="DD/MM/YYYY")
else:
    dataset_date = todays_date.strftime("%d/%m/%Y")
    st.sidebar.write("Date:", dataset_date)

# Dataset name input
dataset_name = st.sidebar.text_input('Dataset Name')
# st.sidebar.write("Dataset:", dataset_name)

# Annotator name input
annotator_name = st.sidebar.text_input('Annotator Name')
qc_name = st.sidebar.text_input('QC Name')
# st.sidebar.write("Annotator:", annotator_name)

# Image and Label folder paths selection
image_folder_path = st.sidebar.text_input("Enter image folder path:")
label_folder_path = st.sidebar.text_input("Enter label folder path:")
output_folder_path = st.sidebar.text_input("Enter output path:")
qc_percentage = st.sidebar.slider("QC Percentage (%)", 0, 100) / 100
startQC = st.sidebar.toggle("Start QC'ing", value=False)

# List of image names
image_list = sorted([f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
num_samples = int(len(image_list) * qc_percentage)
random.seed(8)
qc_image_list = sorted(random.sample(image_list, num_samples))

# Dataframe
df = pd.DataFrame({
    "Image": qc_image_list,
    "Image Quality": [None] * len(qc_image_list),
    "Remarks": [None] * len(qc_image_list),
    "No_of_Objects_Annotated": [single_image_ann_info(image) for image in qc_image_list],
    "Misclassification": [0] * len(qc_image_list),
    "Incorrect_Annotation": [0] * len(qc_image_list),
    "Missed": [0] * len(qc_image_list),
}) 

if "df" not in st.session_state:
    st.session_state["df"] = df

numeric_columns = ["No_of_Objects_Annotated", "Misclassification", "Incorrect_Annotation", "Missed"]
st.session_state["df"][numeric_columns] = st.session_state["df"][numeric_columns].apply(pd.to_numeric, errors="coerce")

# Function to update image index by typing
def update_index():
    if st.session_state.image_index != st.session_state.counter:  
        st.session_state.counter = st.session_state.image_index

def img_quality_change():
    st.session_state["df"].at[st.session_state.counter, "Image Quality"] = st.session_state.img_quality

def remarks_change():
    st.session_state["df"].at[st.session_state.counter, "Remarks"] = st.session_state.remarks

def misclassification_change():
    st.session_state["df"].at[st.session_state.counter, "Misclassification"] = st.session_state.misclass

def incorrect_annotation_change():
    st.session_state["df"].at[st.session_state.counter, "Incorrect_Annotation"] = st.session_state.incorr_ann

def missed_change():
    st.session_state["df"].at[st.session_state.counter, "Missed"] = st.session_state.missed

def calculate_accuracy(total_number_of_objects_annotated, incorrect_annotation):
    correct_annotations = total_number_of_objects_annotated - incorrect_annotation
    if total_number_of_objects_annotated == 0:
        return 0
    annotation_accuracy = (correct_annotations / total_number_of_objects_annotated) * 100
    return annotation_accuracy

with tab1:
    if startQC:
        if len(image_list) > 0:  # Ensure there are images in the list
            if 'counter' not in st.session_state:
                st.session_state.counter = 0

            image = annotate_image(os.path.join(image_folder_path, qc_image_list[st.session_state.counter]), label_folder_path, view_labels=True)
            st.image(image, use_column_width=True, caption=f"{qc_image_list[st.session_state.counter]}")
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.session_state.counter > 0:
                    prevButton = st.button("Prev", use_container_width=True, key="previous")
                    if prevButton:
                        st.session_state.counter -= 1

            with c2:
                if st.session_state.counter >= 0 and st.session_state.counter <= len(qc_image_list) - 1:
                    # Replace the existing index section with an input box/slider for index
                    index = st.number_input(
                        f"{st.session_state.counter} / {len(qc_image_list)}",
                        value=st.session_state.counter,
                        min_value=0,
                        max_value=len(qc_image_list) - 1, 
                        label_visibility='collapsed',
                        key='image_index',
                        on_change=update_index)
                    
            with c3:
                if st.session_state.counter < len(qc_image_list) - 1:
                    nextButton = st.button("Next", use_container_width=True, key="next")
                    if nextButton:
                        st.session_state.counter += 1

            # Creating a dataframe for rest of the info
            col1, col2 = st.columns([2, 2])  # Define columns 
            obects_in_image = 2
            no_of_objects_annotated = 3

            # First column content
            with col1:
                img_quality = st.text_input("Image Quality", value=df.at[st.session_state.counter, 'Image Quality'], key='img_quality', on_change=img_quality_change)
                misclass = st.text_input("Misclassification", value=df.at[st.session_state.counter, 'Misclassification'], key='misclass', on_change=misclassification_change) 
                remarks = st.text_input("Remarks", value=df.at[st.session_state.counter, 'Remarks'], key='remarks', on_change=remarks_change)

            # Second column content
            with col2:
                # obj_cls_acc = st.text_input("Object_Classification_Accuracy")
                # obj_ann_acc = st.text_input("Object_Annotation_Accuracy")
                incorr_ann = st.text_input("Incorrect_Annotation", value=df.at[st.session_state.counter, 'Incorrect_Annotation'], key='incorr_ann', on_change=incorrect_annotation_change)
                no_obj_ann = st.text_input("No_of_Objects_Annotated", value=single_image_ann_info(qc_image_list[st.session_state.counter]))
                missed = st.text_input("Missed", value=df.at[st.session_state.counter, 'Missed'], key='missed', on_change=missed_change)

            st.sidebar.write("COUNTER: ", st.session_state.counter)


        edited_df = st.data_editor(st.session_state["df"], 
                                   hide_index=False, 
                                   width=1600, 
                                   disabled=(["Image"]),
                                   key='df_editor',
                                #    on_change=df_on_change,
                                   )

with tab2:
    # Dataset Statistics
    st.title("Dataset Statistics")

    # MAIN DATASET
    st.subheader("Main Dataset")
    classes, total_annotations = label_info(image_list)
    # st.sidebar.write("Dataset Image Shape(s):", image_shape)
    st.write("Images & Labels: ", len(image_list))
    st.write("Total Annotations: ", total_annotations)
    st.write("Classes: ", classes)

    # QC DATASET
    qc_classes, qc_total_annotations = label_info(qc_image_list)
    st.subheader(f"{int(qc_percentage * 100)}% of Dataset: ")
    st.write("QC Images & Labels: ", len(qc_image_list))
    st.write("Total Annotations: ", qc_total_annotations)
    st.write("Classes: ", qc_classes)
    annotation_accuracy = calculate_accuracy(qc_total_annotations, sum(st.session_state["df"]["Incorrect_Annotation"]))
    # st.write("Annotation Accuracy: ", annotation_accuracy)
    st.metric(label="Annotation Accuracy", value=f'{annotation_accuracy:.2f}')
    classification_accuracy = calculate_accuracy(qc_total_annotations, sum(st.session_state["df"]["Misclassification"]))
    st.metric(label="Classification Accuracy", value=f'{classification_accuracy:.2f}')


    # Pie-chart for dataset distribution
    total_images = len(image_list)
    qc_images = len(qc_image_list)

    # Calculating the counts
    non_qc_images = total_images - qc_images

    # Creating a pie chart
    fig = px.pie(names=['QC Images', 'Non-QC Images'],
                 values=[qc_images, non_qc_images],)
                #  title='Data Distribution')
    fig.update_traces(textinfo='percent')

    # Displaying the chart using Streamlit
    st.plotly_chart(fig)
