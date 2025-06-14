# -*- coding: utf-8 -*-
"""
Spyder Editor
J�zef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""

"""
Konwersja modelu Keras do formatu ONNX dla zwi�kszenia przeno�no�ci i kompatybilno�ci.
Skrypt wczytuje wcze�niej wytrenowany model sieci neuronowej zapisany w formacie Keras
i przekszta�ca go do standardu ONNX (Open Neural Network Exchange). Format ONNX umo�liwia
uruchamianie modelu w r�nych �rodowiskach i platformach, niezale�nie od frameworka
u�ytego do trenowania. Konwersja zachowuje architektur� sieci oraz wagi modelu.
Proces obejmuje:
- Wczytanie modelu Keras z pliku kontrolnego
- Okre�lenie specyfikacji wej�ciowej na podstawie wymiar�w modelu
- Konwersj� do formatu ONNX z wykorzystaniem operator�w w wersji 13
- Zapisanie skonwertowanego modelu do pliku binarnego
"""

import tensorflow as tf
import tf2onnx

model_path = "checkpoints/USGEquina-Pregna_v1_0.keras"
model = tf.keras.models.load_model(model_path)

spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open("checkpoints/USGEquina-Pregna_v1_0.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
