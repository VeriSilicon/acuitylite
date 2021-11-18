import tensorflow as tf
import numpy as np
from acuitylib.lite.acuitymodel import AcuityModel
from acuitylib.lite.importer import TFliteLoader
from acuitylib.lite.exporter import TimVxExporter

# wget https://storage.googleapis.com/download.tensorflow.org/
# models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz

mobilenet = "model/mobilenet_v1_1.0_224_quant.tflite"
image0 = "data/697.jpg"
image1 = "data/813.jpg"
labels = [697, 813]

# data generator
def get_data():
    for image in [image0, image1]:
        arr = tf.io.decode_image(tf.io.read_file(image)).numpy().reshape(1, 224, 224, 3).astype(np.float32)
        arr = (arr-128.0)/128.0  # preprocess
        yield np.array(arr, dtype=np.float32)

def main():
    # no need to quantize using acuity lite for quant model
    # load tflite quant model
    importtflite = TFliteLoader(mobilenet)
    acuitynet = importtflite.load()
    quantmodel = AcuityModel(acuitynet)

    # quant model inference
    data_list = [get_data()]
    results = quantmodel.inference(data_list, batch_size=1, iterations=2)
    for i, result in enumerate(results):
        for _, outs in result.items():
            assert outs[0][labels[i]] > 0.9

    # export tim-vx quant case
    TimVxExporter(quantmodel).export('export_timvx/quant/mobilenet')

if __name__ == '__main__':
    main()
