#SuperGo For CZ4041  

##Environment:  

Ubuntu 18.04  
Python 3.5  
CUDA 10.1  
Local Mongo DB Server  

##Python Library Required:  

pachi_pi


Training Configuration:

LR: 0.01
BoardSize 9*9
Data Batch Size: 4000 steps
MiniBatch: 64
Iteration for one generation: 12*MiniBatch 15*MiniBatch


1. Winning rate of every generation against a random agent as well as against the previous generation

2. Loss Function drops from 6.0 to 4.2



General Software FLow:


ALWAYS:

Self Play Process  -- Parallel Alway on going process (keep pooling the best model)


START

--wait for enough self play data
Training Process -- Spawn Evaluation Process after Training
Evalutaion Process --Spawn child self-play process that compare newly trained model with last best player to see whether to accpet the newly trained model 


JUMP START 