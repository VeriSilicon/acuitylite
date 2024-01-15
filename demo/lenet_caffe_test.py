#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.interface.importer import CaffeLoader
from acuitylib.interface.exporter import TimVxExporter, TFLiteExporter
from acuitylib.interface.quantization import Quantization, QuantizerType
from acuitylib.interface.inference import Inference


prototxt = "model/lenet.prototxt"
caffe_model = "model/lenet.caffemodel"
image0 = "data/0.jpg"
image1 = "data/1.png"


# data generator
def get_data():
    for image in [image0, image1]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 1, 28, 28).astype(np.float32)
        # arr = arr/255.0  # preprocess
        inputs = {'input': np.array(arr, dtype=np.float32)}
        yield inputs


def test_caffe_lenet():
    # load caffe model
    model = CaffeLoader(model=prototxt, weights=caffe_model).load()

    # inference with float model
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        assert outs[0].flatten()[i] > 0.99

    # export tim-vx float16 case
    TimVxExporter(model).export('export_timvx/float16/lenet')
    # export tflite float16 case
    TFLiteExporter(model).export('export_tflite/float16/lenet.tflite')

    # perchanneli8 quantization
    quantizer = QuantizerType.PERCHANNEL_SYMMETRIC_AFFINE
    qtype = 'int8'
    Quantization(model).quantize(input_generator_func=get_data, quantizer=quantizer, qtype=qtype, iterations=1)

    # inference with quantized model
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        assert outs[0].flatten()[i] > 0.99

    # export tim-vx perchanneli8 case
    TimVxExporter(model).export('export_timvx/pcq_symi8/lenet')
    # export tflite perchanneli8 case
    TFLiteExporter(model).export('export_tflite/pcq_symi8/lenet.tflite')


if __name__ == '__main__':
    test_caffe_lenet()
