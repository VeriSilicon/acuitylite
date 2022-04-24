#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.lite.acuitymodel import AcuityModel
from acuitylib.lite.importer import CaffeLoader
from acuitylib.lite.exporter import TimVxExporter
from acuitylib.lite.exporter import TFLiteExporter
from acuitylib.lite.quantizer import QuantizeType

prototxt = "model/lenet.prototxt"
caffe_model = "model/lenet.caffemodel"
image0 = "data/0.jpg"
image1 = "data/1.png"

# data generator
def get_data():
    for image in [image0, image1]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 1, 28, 28).astype(np.float32)
        # arr = arr/255.0  # preprocess
        yield np.array(arr, dtype=np.float32)

def test_caffe_lenet():
    # load caffe model
    importcaffe = CaffeLoader(model=prototxt, weights=caffe_model)
    acuitynet = importcaffe.load()
    model = AcuityModel(acuitynet)

    # inference
    data_list = [get_data()]
    results = model.inference(data_list, batch_size=1, iterations=2)
    for i, result in enumerate(results):
        for _, outs in result.items():
            assert outs[0][i] > 0.99

    # export tim-vx float16 case
    TimVxExporter(model).export('export_timvx/float16/lenet')
    # export tflite float16 case
    TFLiteExporter(model).export('export_tflite/float16/lenet.tflite')

    # quantize perchanneli8
    data_list = [get_data()]
    model.quantize(data_list, quantizer=QuantizeType.pcq_symi8, iteration=1)

    # inference with quantized model
    data_list = [get_data()]
    results = model.inference(data_list, iterations=2)
    for i, result in enumerate(results):
        for _, outs in result.items():
            assert outs[0][i] > 0.99

    # export tim-vx perchanneli8 case
    TimVxExporter(model).export('export_timvx/pcq_symi8/lenet')
    # export tflite perchanneli8 case
    TFLiteExporter(model).export('export_tflite/pcq_symi8/lenet.tflite')

if __name__ == '__main__':
    test_caffe_lenet()
