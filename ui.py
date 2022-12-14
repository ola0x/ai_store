from xmlrpc.client import Boolean
import streamlit as st
import cv2
import numpy as np

from inference import inference, initialization
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

metadata = MetadataCatalog.get("_")
metadata.thing_classes = ['None', 'space']

# Title the page
st.title("AI out of stock Replenishment App")


@st.cache
def draw_img(img, metadata, outputs):
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()


@st.cache(persist=True)
def load_img(path, resize=Boolean):
    # original image loaded from the image path
    ori_img = cv2.imread(path)
    #if true, resize image
    if resize:
        height, width = ori_img.shape[:2]
        resize_img = cv2.resize(ori_img, (round(width / 10), 300), interpolation=cv2.INTER_AREA)
        return ori_img, resize_img
    else:
        return ori_img

@st.cache(persist=True)
def run_img_inference(predictor, input_img):
    outputs = inference(predictor, input_img)
    out_img = draw_img(input_img, metadata,outputs)
    # st.image(out_img, caption='Processed Image')
    return out_img

def load_sample_img(predictor):
    sam_img = st.sidebar.selectbox('Sample Images',
                                   ('img1', 'img2', 'img3', 'img4'))
    if sam_img == "img1":
        test_img, resize_img = load_img('test_img/test1.jpg', resize=True)
        st.sidebar.image(resize_img,use_column_width=True, channels="BGR")
    elif sam_img == "img2":
        test_img, resize_img = load_img("test_img/test2.jpg", resize=True)
        st.sidebar.image(resize_img,use_column_width=True, channels='BGR')
    elif sam_img == "img3":
        test_img, resize_img = load_img("test_img/test3.jpg", resize=True)
        st.sidebar.image(resize_img,use_column_width=True, channels='BGR')
    elif sam_img == "img4":
        test_img, resize_img = load_img("test_img/test4.jpg", resize=True)
        st.sidebar.image(resize_img,use_column_width=True, channels='BGR')

    sidebar_predict_btn = st.sidebar.button('Process Image')
    if sidebar_predict_btn:
        # st.image(test_img, channels='BGR', width=400)
        output_img = run_img_inference(predictor, test_img)
        st.image(output_img, caption='Processed Image')
    
    # return test_img
        # pred_button = st.button('Predict')
        # if pred_button:
        #     outputs = inference(predictor, test_img)
        #     out_img = draw_img(img, metadata,outputs)
        #     st.image(out_img, caption='Processed Image')

    # if st.sidebar.button('Select Image'):
    #     st.sidebar.write('Running inference....', sam_img)
    # else:
    #     st.sidebar.write('')


def main():

    predictor = initialization()

    # Initialise the sample image
    load_sample_img(predictor)

    # Create a FileUploader so that the user can upload an image to the UI
    uploaded_img = st.file_uploader("Choose an image...",
                                type=['jpg', 'png', 'jpeg'])

       # Display the predict button just when an image is being uploaded
    if not uploaded_img:
        st.warning("Please upload an image before proceeding!")
        st.stop()
    else:
        try:
            file_bytes = np.asarray(
                bytearray(uploaded_img.read()), dtype=np.uint8)
            uploaded_img = cv2.imdecode(file_bytes, 1)
                # Display uploaded image
            st.image(uploaded_img, channels='BGR', width=500)
            pred_button = st.button('Predict')
        except:
            st.subheader("Please, reupload an image to see the changes")
    
    #Create the predict button
    # pred_button = st.button('Predict')
    
    if pred_button:
        outputs = inference(predictor, uploaded_img)
        out_img = draw_img(uploaded_img, metadata, outputs)
        st.image(out_img, caption='Processed Image')


if __name__ == "__main__":
    main()
