wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash  Anaconda3-2021.05-Linux-x86_64.sh -f -b
source /home/aiscuser/anaconda3/bin/activate
conda create -n instruct-eval python=3.8 -y
conda activate instruct-eval

cd instruct-eval
pip install -r requirements.txt
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
cd ..

pip install scikit-learn