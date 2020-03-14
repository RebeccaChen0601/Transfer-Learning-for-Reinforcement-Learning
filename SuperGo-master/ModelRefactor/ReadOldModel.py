import torch
import signal
import torch.nn as nn
import numpy as np
import pickle
import time
import torch.nn.functional as F
import multiprocessing
import multiprocessing.pool
from lib.process import MyPool
from lib.dataset import SelfPlayDataset
from lib.evaluate import evaluate
from lib.utils import load_player
from copy import deepcopy
from pymongo import MongoClient
from torch.autograd import Variable
from torch.utils.data import DataLoader
from const import *
from models.agent import Player
from models.feature import Extractor
from models.policy import PolicyNet
from models.value import ValueNet

class Transfer:


    def transferAtoB(self, model_a, model_b):
        return





class GoModel:

    def __init__(self):
        """ Create an agent and initialize the networks """

        self.extractor = Extractor(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.value_net = ValueNet(OUTPLANES_MAP, OUTPLANES).to(DEVICE)
        self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES).to(DEVICE)
        self.passed = False


    def loadModel(self, folderName, version):

        names = ["extractor", "policy_net", "value_net"]
        for i in range(0, 3):
            checkpoint = torch.load("../saved_models/" + folderName + "/" + str(version) + "-" + names[i] + ".pth.tar")
            model = getattr(self, names[i])
            model.load_state_dict(checkpoint['model'])
            print(checkpoint['model'])
            print(model)


instance = GoModel()
instance.loadModel("1584165886",3)