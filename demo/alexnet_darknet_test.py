#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from acuitylib.interface.importer import DarknetLoader
from acuitylib.interface.exporter import TFLiteExporter
from acuitylib.interface.quantization import Quantization, QuantizerType
from acuitylib.interface.inference import Inference


alexnet_model = "model/alexnet2.cfg"
alexnet_weights = "model/alexnet2.weights"
image0 = "data/space_shuttle_227x227.jpg"


# data generator
def get_data():
    for image in [image0]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 3, 227, 227).astype(np.float32)
        arr = (arr-128.0)/128.0  # preprocess
        inputs = {'input': np.array(arr, dtype=np.float32)}
        yield inputs


def test_darknet_alexnet():
    # load darknet model
    model = DarknetLoader(model=alexnet_model, weights=alexnet_weights).load()

    # inference
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        print(outs[0])

    # perchanneli8 quantization
    quantizer = QuantizerType.PERCHANNEL_SYMMETRIC_AFFINE
    qtype = 'int8'
    Quantization(model).quantize(input_generator_func=get_data, quantizer=quantizer, qtype=qtype, iterations=1)

    # inference with quantized model
    infer = Inference(model)
    infer.build_session()  # build inference session
    for i, data in enumerate(get_data()):
        ins, outs = infer.run_session(data)  # run inference session
        print(outs[0])

    # export tflite perchanneli8 case
    TFLiteExporter(model).export('export_tflite/pcq_symi8/alexnet.tflite')


if __name__ == '__main__':
    test_darknet_alexnet()
