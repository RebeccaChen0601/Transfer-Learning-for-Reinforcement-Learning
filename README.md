# SuperGo For CZ4041  

## Environment:  

Ubuntu 18.04  
Python 3.5  
CUDA 10.1  
Local Mongo DB Server  

## Python Library Required:  

pachi_pi   
numpy  
timeit  
pickle  
pytorch 1.4  
pymongo  



## Training Configuration:

LR: 0.01  
BoardSize 9*9  
Data Batch Size: 4000 steps  
MiniBatch: 64  
Iteration for one generation: 12*MiniBatch 15*MiniBatch  



## To start Traning from zero:

```python main.py```

## To start training from break point

```pythoin main.py --folder your_folder --verion version_number```
