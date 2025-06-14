# -*- coding: utf-8 -*-
"""
Spyder Editor
Józef Sroka, 67195-CKP, 2025, 67195-ckp@kozminski.edu.pl
"""

"""
Konwersja modelu Keras do formatu ONNX dla zwiêkszenia przenoœnoœci i kompatybilnoœci.
Skrypt wczytuje wczeœniej wytrenowany model sieci neuronowej zapisany w formacie Keras
i przekszta³ca go do standardu ONNX (Open Neural Network Exchange). Format ONNX umo¿liwia
uruchamianie modelu w ró¿nych œrodowiskach i platformach, niezale¿nie od frameworka
u¿ytego do trenowania. Konwersja zachowuje architekturê sieci oraz wagi modelu.
Proces obejmuje:
- Wczytanie modelu Keras z pliku kontrolnego
- Okreœlenie specyfikacji wejœciowej na podstawie wymiarów modelu
- Konwersjê do formatu ONNX z wykorzystaniem operatorów w wersji 13
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
