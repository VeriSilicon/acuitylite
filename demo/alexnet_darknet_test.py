#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.lite.acuitymodel import AcuityModel
from acuitylib.lite.importer import DarknetLoader
from acuitylib.lite.exporter import TFLiteExporter
from acuitylib.lite.quantizer import QuantizeType

alexnet_model = "model/alexnet2.cfg"
alexnet_weights = "model/alexnet2.weights"
image0 = "data/space_shuttle_227x227.jpg"

# data generator
def get_data():
    for image in [image0]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 3, 227, 227).astype(np.float32)
        arr = (arr-128.0)/128.0  # preprocess
        yield np.array(arr, dtype=np.float32)

def test_darknet_alexnet():
    # load darknet model
    importdarknet = DarknetLoader(model=alexnet_model, weights=alexnet_weights)
    acuitynet = importdarknet.load()
    model = AcuityModel(acuitynet)

    # inference
    data_list = [get_data()]
    results = model.inference(data_list, batch_size=1, iterations=1)
    for i, result in enumerate(results):
        for _, outs in result.items():
            print(outs.shape)

    # quantize perchanneli8
    data_list = [get_data()]
    model.quantize(data_list, quantizer=QuantizeType.pcq_symi8, iteration=1)

    # inference with quantized model
    data_list = [get_data()]
    results = model.inference(data_list, iterations=1)
    for i, result in enumerate(results):
        for _, outs in result.items():
            print(outs.shape)

    # export tflite perchanneli8 case
    TFLiteExporter(model).export('export_tflite/pcq_symi8/alexnet.tflite')

if __name__ == '__main__':
    test_darknet_alexnet()
