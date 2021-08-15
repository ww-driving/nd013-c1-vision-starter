#!/bin/bash

set -x

mkdir -p $1/movies

for f in data/test/*.tfrecord; do
  python inference_video.py --labelmap_path label_map.pbtxt --model_path "$1"/exported_model/saved_model --tf_record_path "$f" --config_path "$1"/pipeline_new.config --output_path $1/movies/${f##*/}.mp4
done
