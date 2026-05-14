#! /bin/sh
python src/preprocess_pipeline.py
python src/fit_predict_pipeline.py
python src/postprocess_pipeline.py
#python src/validation_pipeline.py
