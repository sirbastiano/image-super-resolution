docker run --gpus all -p 8889:8888 --ip=0.0.0.0 -w /tf/workspace/ --rm -it -v $(pwd):/tf/workspace  tensorflow/tensorflow:2.9.3-gpu
python setup.py install
pip install pyaml
pip install tqdm
pip install imageio
