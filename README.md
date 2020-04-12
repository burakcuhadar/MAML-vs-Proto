# Implementation of MAML and Prototypical Networks 
- Training and testing Prototypical Networks: 
  > python run ProtoNet.py ./omniglot resized/ --n-way=5 --k-shot=1 --n-query=5 --n-meta-test-way=5 --k-meta-test-shot=4 --n-meta-test-query=4
- Training MAML: 
  > python run maml.py --n way=5 --k shot=1 --inner update lr=0.4 --num inner updates=1
- Testing MAML:
  > python run maml.py --n way=5 --k shot=4 --inner update lr=0.4 --num inner updates=1 --meta train=False --meta test set=True --meta train k shot=1
