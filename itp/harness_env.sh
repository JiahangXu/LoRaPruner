wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash  Anaconda3-2021.05-Linux-x86_64.sh -f -b
source /home/aiscuser/anaconda3/bin/activate
conda create -n py39 python==3.9.0 -y
conda activate py39
pip install sentencepiece
pip install deepspeed
pip install accelerate
pip install datasets
pip install evaluate
pip install mlflow