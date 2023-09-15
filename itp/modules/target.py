sing_octo_target = \
"""
target:
  service: aisc
  name: msroctovc

environment:
  image: amlt-sing/pytorch-1.11.0-cuda11.6
  #image: amlt-sing/pytorch-1.8.0
  setup:
  - set -x
  #- cp -r /mnt/data/LPM/note3.txt /dev/shm
  #- cp -r /dev/shm/Anote2.txt /mnt/data/Lpm
  #- tar -xvf /mnt/ramdisk/imagenet/ILSVRC2012_img_train.tar -C /mnt/ramdisk/imagenet/
  #- rm -f ILSVRC2012_img_train.tar 
  #- tar -xvf /mnt/ramdisk/imagenet/ILSVRC2012_img_val.tar -C /mnt/ramdisk/imagenet/
  # - git clone https://github.com/microsoft/nn-Meter.git
  # - cd nn-Meter/ && git switch dev/transformer && pip install .
  image_setup:
  - pip install accelerate==0.15.0
  - pip install transformers==4.25.1
  - pip install mpi4py
  - pip install scipy
  - pip install scikit-learn
  - pip install numpy
  - pip install tqdm
  - pip install mlflow azureml-mlflow
  - pip install protobuf==3.19.0
  - pip install deepspeed==0.7.7 
  - pip install evaluate 
  - pip install datasets==2.8.0 
  - pip install sentencepiece==0.1.97 
  - pip install scipy 
  - pip install viztracer
  - pip install py-spy
  - pip install pydantic==1.10.7
  - pip install zstandard

"""

sing_research_target = \
"""
target:
  service: aisc
  name: msrresrchvc

environment:
  image: amlt-sing/pytorch-1.11.0-cuda11.6
  #image: amlt-sing/pytorch-1.10.0-cuda11.4-a100
  setup:
  - set -x
  # - cp -r /mnt/data/EdgeDL/imagenet2012/tar/* /mnt/ramdisk/imagenet/
  # - tar -xvf /mnt/ramdisk/imagenet/ILSVRC2012_img_train.tar -C /mnt/ramdisk/imagenet/
  # - rm -f ILSVRC2012_img_train.tar 
  # - tar -xvf /mnt/ramdisk/imagenet/ILSVRC2012_img_val.tar -C /mnt/ramdisk/imagenet/
  # - git clone https://github.com/microsoft/nn-Meter.git
  # - cd nn-Meter/ && git switch dev/transformer && pip install .
  image_setup:
  - pip install accelerate==0.15.0
  - pip install transformers==4.25.1
  - pip install mpi4py
  - pip install scipy
  - pip install scikit-learn
  - pip install numpy
  - pip install tqdm
  - pip install mlflow azureml-mlflow
  - pip install protobuf==3.19.0
  - pip install deepspeed==0.7.7 
  - pip install evaluate 
  - pip install datasets==2.8.0 
  - pip install sentencepiece==0.1.97 
  - pip install scipy 
  - pip install viztracer
  - pip install py-spy
  - pip install pydantic==1.10.7
  - pip install zstandard
"""


sing_mrh_target = \
"""
target:
  service: aisc
  name: ads

environment:
  image: amlt-sing/pytorch-1.11.0-cuda11.6
  #image: amlt-sing/pytorch-1.8.0
  setup:
  - set -x
  image_setup:
  - pip install accelerate==0.15.0
  - pip install transformers==4.25.1
  - pip install mpi4py
  - pip install scipy
  - pip install scikit-learn
  - pip install numpy
  - pip install tqdm
  - pip install mlflow azureml-mlflow
  - pip install protobuf==3.19.0
  - pip install deepspeed==0.7.7 
  - pip install evaluate 
  - pip install datasets==2.8.0 
  - pip install sentencepiece==0.1.97 
  - pip install scipy 
  - pip install viztracer
  - pip install py-spy
  - pip install pydantic==1.10.7
  - pip install zstandard

"""


itp_rr1_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itphyperbj1cl1  
                          
  vc: hcbj03  #hcbj01

environment:
  image: azureml/curated/pytorch-1.8-ubuntu18.04-py37-cuda11-gpu:39

  setup:
    - pip install accelerate==0.15.0
    - pip install transformers==4.25.1
    - pip install mpi4py
    - pip install scipy
    - pip install numpy
    - pip install tqdm
    - pip install mlflow azureml-mlflow
    - pip install protobuf==3.19.0
    - pip install deepspeed==0.7.7 
    - pip install evaluate 
    - pip install datasets==2.8.0 
    - pip install sentencepiece==0.1.97 
    - pip install scipy 
    - pip install scikit-learn 
    - pip install viztracer
    - pip install py-spy
    - pip install pydantic==1.10.7
    - pip install zstandard



"""

itp_rr1_target1 = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itplabrr1cl1 # choices:
                     # MSR-Lab-RR1-V100-32GB: itplabrr1cl1, 
                     # ads: v100-8x-eus-1
                          
  vc: resrchvc

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io

  setup:
    - sudo mkdir /mnt/ramdisk/
    - sudo mount -t tmpfs -o size=320g tmpfs /mnt/ramdisk/
    - pip install accelerate==0.15.0
    - pip install transformers==4.25.1
    - pip install mpi4py
    - pip install scipy
    - pip install scikit-learn
    - pip install numpy
    - pip install tqdm
    - pip install mlflow azureml-mlflow
    - pip install protobuf==3.19.0
    - pip install deepspeed==0.7.7 
    - pip install evaluate 
    - pip install datasets==2.8.0 
    - pip install sentencepiece==0.1.97 
    - pip install scipy 
    - pip install viztracer
    - pip install py-spy
    - pip install pydantic==1.10.7
    - pip install zstandard

"""

itp_p100_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: itpeusp100cl
  vc: resrchvc

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./itp/setup.sh
"""
itp_ads_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: v100-8x-eus-1 # choices:
                      # MSR-Lab-RR1-V100-32GB: itplabrr1cl1, 
                      # ads: v100-8x-eus-1
                          
  vc: ads

environment:
  image: azureml/curated/pytorch-1.8-ubuntu18.04-py37-cuda11-gpu:39

  setup:
    - sudo mkdir /mnt/ramdisk/
    - sudo mount -t tmpfs -o size=320g tmpfs /mnt/ramdisk/
    - pip install -U scikit-learn
    - pip install pyyaml
    - pip install munch
    - pip install dataclasses
    - pip install timm
    - pip install yacs
    - pip install pytorchcv
    - pip install git+https://github.com/ildoonet/pytorch-randaugment
    - pip install protobuf==3.20.0
    - mkdir /mnt/ramdisk/imagenet
    - cp -r /mnt/data/EdgeDL/imagenet2012/tar/* /mnt/ramdisk/imagenet/
    - tar -xvf /mnt/ramdisk/imagenet/ILSVRC2012_img_val.tar -C /mnt/ramdisk/imagenet/
    - rm -f /mnt/ramdisk/imagenet/ILSVRC2012_img_val.tar 
    - tar -xvf /mnt/ramdisk/imagenet/ILSVRC2012_img_train.tar -C /mnt/ramdisk/imagenet/
    - rm -f /mnt/ramdisk/imagenet/ILSVRC2012_img_train.tar 

"""


itp_ads_gpt3_target = \
"""
target:
  service: amlk8s
  name: v100-8x-wus2
  vc: Ads-GPT3

environment:
  image: azureml/curated/pytorch-1.8-ubuntu18.04-py37-cuda11-gpu:39
"""

itp_ads_a100_target = \
"""
target:
  service: amlk8s
  # run "pt target list amlk8s" to list the names of available AMLK8s targets
  name: a100-8x-wus2
  vc: ads

environment:
  image: v-xudongwang/pytorch:cddlyf_a100
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
"""

target_dict = dict(
    sing_octo=sing_octo_target,
    sing_research=sing_research_target,
    sing_ads=sing_mrh_target,
    itp_rr1=itp_rr1_target,
    itp_rr1_1=itp_rr1_target1,
    itp_p100=itp_p100_target,
    itp_ads_v100=itp_ads_target,
    itp_ads_v100_gpt3=itp_ads_gpt3_target,
    itp_ads_a100=itp_ads_a100_target
)