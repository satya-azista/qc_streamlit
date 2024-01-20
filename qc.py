# Libraries
import os
import cv2
import random
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from multiprocessing.pool import ThreadPool

################################
#######   FUNCTIONS  ###########
################################

# Function to extract image size
def load_image(image_path):
    return cv2.imread(image_path).shape[:2]

def image_size(image_folder_path, image_list):
    img_shape = set()
    pool = ThreadPool()
    image_paths = [os.path.join(image_folder_path, img) for img in image_list]
    results = pool.map(load_image, image_paths)
    pool.close()
    pool.join()
    img_shape.update(results)
    return img_shape

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

# # Function to check if given paths are valid directories
# def isDir(image_folder_path, label_folder_path):
#     valid_image_path, valid_label_path = False, False
#     if os.path.isdir(image_folder_path):
#         st.toast(f"Valid Image Folder! :white_check_mark: ")
#         valid_image_path = True
#     else:
#         st.toast("Please enter a valid image folder path. :x: ")
#         valid_image_path = False

#     if os.path.isdir(label_folder_path):
#         st.toast(f"Valid Label Folder! :white_check_mark: ")
#         valid_label_path = True
#     else:
#         st.toast("Please enter a valid label folder path. :x: ")
#         valid_label_path = False

#     return valid_image_path and valid_label_path

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

# Function to update image index by typing
def update_index():
    if st.session_state.image_index != st.session_state.counter:
        st.session_state.counter = st.session_state.image_index
    elif st.session_state.image_index > len(qc_image_list) - 1:
        st.session_state.image_index = len(qc_image_list) - 1
    elif st.session_state.image_index < 0:
        st.session_state.image_index = 0

def onNext():
    if st.session_state.counter <= len(qc_image_list) - 1:
        st.session_state.counter += 1 
        # st.session_state.img_quality, st.session_state.remarks = None, None
        # st.session_state.misclass, st.session_state.incorr_ann, st.session_state.missed = 0, 0, 0
    elif st.session_state.counter == len(qc_image_list) - 1:
        st.session_state.counter = len(qc_image_list) - 1

def onPrev():
    if st.session_state.counter > 0:
        st.session_state.counter -= 1 

# Functions to update the dataframe       
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

# Function to calculate accuracy
def calculate_accuracy(total_number_of_objects_annotated, incorrect_annotation):
    correct_annotations = total_number_of_objects_annotated - incorrect_annotation
    if total_number_of_objects_annotated == 0:
        return 0
    annotation_accuracy = (correct_annotations / total_number_of_objects_annotated) * 100
    return annotation_accuracy

def updateQCperc():
    st.session_state["qc_perc"] = st.session_state.qc_percentage / 100

# Function to update the DataFrame based on text input changes
def update_dataframe(row, column, new_value):
    st.session_state['df'].at[row, column] = new_value
    # st.session_state.edited_df = st.session_state['df'].copy()
    
################################
#######      UI      ###########
################################

st.title("Annotation QC Tool! :rocket:")
tab1, tab2 = st.tabs(["QC Information", "QC"])

with tab1:

    # Date
    todays_date = datetime.datetime.now()
    jan_1 = datetime.date(todays_date.year, 1, 1)
    custom_date = st.checkbox('Custom date')
    if custom_date:
        dataset_date = st.date_input("Select date", jan_1, format="DD/MM/YYYY")
    else:
        dataset_date = todays_date.strftime("%d/%m/%Y")
        st.write("Date:", dataset_date)

    # Dataset name input
    dataset_name = st.text_input('Dataset Name')
    # st.sidebar.write("Dataset:", dataset_name)

    # Annotator name input
    annotator_name = st.text_input('Annotator Name', help="Record the name of the person who performed the annotations.")
    qc_name = st.text_input('QC Name', help="Record the name of the person responsible for quality control.")
    # st.sidebar.write("Annotator:", annotator_name)

    # Image and Label folder paths selection
    valid_image_path, valid_label_path = False, False
    image_folder_path = st.text_input("Enter image folder path:", help="Define the folder location for images")
    if os.path.isdir(image_folder_path):
        st.success(f"Valid Image Folder!", icon="✅")
        valid_image_path = True
    elif image_folder_path == "":
        st.warning('Enter directory of Images', icon="ℹ️")
        valid_image_path = False
    else:
        st.toast("Please enter a valid image folder path", icon='❌')
        valid_image_path = False

    label_folder_path = st.text_input("Enter label folder path:", help="Define the folder location for labels")
    if os.path.isdir(label_folder_path):
        st.success(f"Valid Label Folder!", icon="✅")
        valid_label_path = True
    elif label_folder_path == "":
        st.warning('Enter directory of Labels', icon="ℹ️")
        valid_label_path = False
    else:
        st.error("Please enter a valid label folder path", icon='❌')
        valid_label_path = False

    # output_folder_path = st.sidebar.text_input("Enter output path:")
    qc_percentage = st.slider("QC Percentage (%)", 0, 100, help="Select a percentage of data for quality control", on_change=updateQCperc, key="qc_percentage")
    if "qc_perc" not in st.session_state:
        st.session_state["qc_perc"] = st.session_state.qc_percentage / 100
    st.write("qc_perc", st.session_state["qc_perc"])
    startQC = st.toggle("Start QC'ing", value=False, help="Toggle ON/OFF to Start/Stop QC", key="startQC")

# List of image names
image_list = None
if valid_image_path and valid_label_path:
    image_list = sorted([f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Calculate no. of samples and create a new list for QC
    num_samples = int(len(image_list) * st.session_state["qc_perc"])
    random.seed(8)
    qc_image_list = sorted(random.sample(image_list, num_samples))

    # Dataframe
    st.session_state['df'] = pd.DataFrame(data={
        "Image": qc_image_list,
        "Image Quality": [None] * len(qc_image_list),
        "Remarks": [None] * len(qc_image_list),
        "No_of_Objects_Annotated": [single_image_ann_info(image) for image in qc_image_list],
        "Misclassification": [0] * len(qc_image_list),
        "Incorrect_Annotation": [0] * len(qc_image_list),
        "Missed": [0] * len(qc_image_list),
    }) 

    # Convert the columns to numeric
    numeric_columns = ["No_of_Objects_Annotated", "Misclassification", "Incorrect_Annotation", "Missed"]
    st.session_state["df"][numeric_columns] = st.session_state["df"][numeric_columns].apply(pd.to_numeric, errors="coerce")

with tab2:
    if startQC:
        if len(image_list) > 0:  # Ensure there are images in the list
            if 'counter' not in st.session_state:
                st.session_state.counter = 0

            image = annotate_image(os.path.join(image_folder_path, qc_image_list[st.session_state.counter]), label_folder_path, view_labels=True)
            st.image(image, use_column_width=True, caption=f"{qc_image_list[st.session_state.counter]}")
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.session_state.counter > 0:
                    prevButton = st.button("Prev", use_container_width=True, key="previous", on_click=onPrev)

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
                    nextButton = st.button("Next", use_container_width=True, key="next", on_click=onNext)

            # Creating a dataframe for rest of the info
            col1, col2 = st.columns([2, 2])  # Define columns

            # First column content
            with col1:
                img_quality = st.text_input("Image Quality", value=st.session_state.edited_df.at[st.session_state.counter, 'Image Quality'], key='img_quality', on_change=img_quality_change)
                misclass = st.text_input("Misclassification", value=st.session_state.edited_df.at[st.session_state.counter, 'Misclassification'], key='misclass', on_change=misclassification_change) 
                remarks = st.text_input("Remarks", value=st.session_state.edited_df.at[st.session_state.counter, 'Remarks'], key='remarks', on_change=remarks_change)
                update_dataframe(st.session_state.counter, 'Image Quality', img_quality)
                update_dataframe(st.session_state.counter, 'Misclassification', misclass)
                update_dataframe(st.session_state.counter, 'Remarks', remarks)

            # Second column content
            with col2:
                incorr_ann = st.text_input("Incorrect_Annotation", value=st.session_state.edited_df.at[st.session_state.counter, 'Incorrect_Annotation'], key='incorr_ann', on_change=incorrect_annotation_change)
                no_obj_ann = st.text_input("No_of_Objects_Annotated", value=single_image_ann_info(qc_image_list[st.session_state.counter]))
                missed = st.text_input("Missed", value=st.session_state["df"].at[st.session_state.counter, 'Missed'], key='missed', on_change=missed_change)
                update_dataframe(st.session_state.counter, 'Incorrect_Annotation', incorr_ann)
                update_dataframe(st.session_state.counter, 'Missed', missed)


        elif len(image_list) == 0:
            st.error("No images in the list! Please input another directory!!")
            # st.button("View Table", use_container_width=True)
            # st.sidebar.write("COUNTER: ", st.session_state.counter)

        # def updateDF():
        #     if not st.session_state.df_editor["edited_rows"].empty:
        #         for no_of_changes in range(len([i for i in st.session_state.df_editor["edited_rows"].items()])):
        #             st.session_state['df'] = st.session_state['df'].at[]

        if not st.session_state['df'].empty:
            st.session_state.edited_df = st.data_editor(st.session_state["df"],
                                        hide_index=False, 
                                        use_container_width=True, 
                                        disabled=(["Image"]),
                                        key='df_editor',
                                        # on_change=update_dataframe,
                                        )
            
        st.session_state["df"][numeric_columns] = st.session_state["df"][numeric_columns].apply(pd.to_numeric, errors="coerce")
            
        # # st.write([i for i in st.session_state.items()])
        # st.write([i for i in st.session_state.df_editor["edited_rows"].items()])
        # st.divider()
        # st.write(st.session_state.df_editor)

if image_list != None and startQC:
    # Dataset Statistics
    c1, c2 = st.columns(2)
    with c1:
        # MAIN DATASET
        st.sidebar.subheader("Main Dataset")
        classes, total_annotations = label_info(image_list)
        st.sidebar.write("Dataset Image Shape(s):", image_size(image_folder_path, image_list))
        st.sidebar.write("Images & Labels: ", len(image_list))
        st.sidebar.write("Total Annotations: ", total_annotations)
        st.sidebar.write("Classes: ", classes)
        st.divider()
        qc_classes, qc_total_annotations = label_info(qc_image_list)
        annotation_accuracy = calculate_accuracy(qc_total_annotations, sum(st.session_state["df"]["Incorrect_Annotation"]))
        classification_accuracy = calculate_accuracy(qc_total_annotations, sum(st.session_state["df"]["Misclassification"]))
        
    with c2:
        # QC DATASET
        st.sidebar.subheader(f"{int(qc_percentage)}% of Dataset: ")
        st.sidebar.write("QC Dataset Image Shape(s):", image_size(image_folder_path, qc_image_list))
        st.sidebar.write("QC Images & Labels: ", len(qc_image_list))
        st.sidebar.write("Total Annotations: ", qc_total_annotations)
        st.sidebar.write("Classes: ", qc_classes)
        st.sidebar.divider()
        st.sidebar.metric(label="Annotation Accuracy", value=f'{annotation_accuracy:.2f}')
        st.sidebar.metric(label="Classification Accuracy", value=f'{classification_accuracy:.2f}')

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
    st.sidebar.plotly_chart(fig)
