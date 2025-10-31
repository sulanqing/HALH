# HALH
A Lightweight Image Hashing Retrieval Method Based on Hybrid Neural Networks and Asymmetric Learning
<img src="model.png" alt="ViT_SIR" style="width: 70%;"/>
# Installation
pip install -r requirements.txt
# Train
python train.py  \
eg.python train.py --dataset cifar-10 \
                --root ./data/data-cifar10 \
                --batch-size 64 \
                --lr 0.0005 \
                --code-length 32 \
                --max-epoch 5 \
                --gpu 0
# Test
python test.py 
eg.python test.py 

