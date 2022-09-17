import os
import streamlit as st

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = "model/model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    predictor = DefaultPredictor(cfg)
    return predictor

@st.cache
def inference(predictor, img):
    return predictor(img)