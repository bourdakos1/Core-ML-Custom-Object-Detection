# Core-ML-Custom-Object-Detection

```
git clone https://github.com/bourdakos1/Core-ML-Custom-Object-Detection.git
cd Core-ML-Custom-Object-Detection
```

```
pip3 install -r requirements.txt
```

```
python3 object_detection/create_tf_record.py
```

```
python3 object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=ssd.config
```

```
python3 object_detection/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path train/pipeline.config \
        --trained_checkpoint_prefix train/model.ckpt-NUMBER \
        --output_directory output_inference_graph
```

```
pip2 install tfcoreml tensorflow numpy protobuf
python2 core_ml_conversion/convert.py
```
