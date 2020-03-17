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
import os

class GoModel:

    def __init__(self):
        """ Create an agent and initialize the networks """

        self.extractor = Extractor(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.value_net = ValueNet(OUTPLANES_MAP, OUTPLANES).to(DEVICE)
        self.policy_net = PolicyNet(OUTPLANES_MAP, OUTPLANES).to(DEVICE)

        self.newModelExtractor = Extractor(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.newModelValue = ValueNet(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.newModelPolicy = PolicyNet(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.passed = False

    def loadModel(self, folderName, version):
        names = ["extractor", "policy_net", "value_net"]
        for i in range(0, 3):
            checkpoint = torch.load("../saved_models/" + folderName + "/" + str(version) + "-" + names[i] + ".pth.tar")
            model = getattr(self, names[i])
            model.load_state_dict(checkpoint['model'])
            print(checkpoint['model'])
            print(model)

        # a is 13 x 13; b is 9 x 9

        aFolderName = "1584177784"
        bFolderName = "1584187381"

        a_extractor_path = os.path.join("../saved_models/" + aFolderName + "/" + str(version) + "-" + "extractor" + ".pth.tar")
        a_value_path = os.path.join("../saved_models/" + aFolderName + "/" + str(version) + "-" + "value_net" + ".pth.tar")
        a_policy_path = os.path.join("../saved_models/" + aFolderName + "/" + str(version) + "-" + "policy_net" + ".pth.tar")

        b_extractor_path = os.path.join("../saved_models/" + bFolderName + "/" + str(version) + "-" + "extractor" + ".pth.tar")
        b_value_path = os.path.join("../saved_models/" + bFolderName + "/" + str(version) + "-" + "extractor" + ".pth.tar")
        b_policy_path = os.path.join("../saved_models/" + bFolderName + "/" + str(version) + "-" + "extractor" + ".pth.tar")

        b_extractor = torch.load(b_extractor_path)
        b_value = torch.load(b_value_path)
        b_policy = torch.load(b_policy_path)

        # load all a parameters
        self.newModelExtractor.load(a_extractor_path)
        self.newModelValue.load(a_value_path)
        self.newModelPolicy.load(a_policy_path)

        # overwrite non-transferrable parameters from b
        with torch.no_grad():
            # load b extractor first layer
            self.newModelExtractor.fc1.weight.copy_(
                b_extractor['fc1.weight'])  # change the name in quotation marks to b extractor first layer
            # load b value last layer?
            self.newModelValue.fc1.bias.copy_(
                b_value['fc1.weight'])  # change the name in quotation marks to b value last layer
            # load b policy last layer?
            self.newModelPolicy.fc1.bias.copy_(
                b_policy['fc1.weight'])  # change the name in quotation marks to b policy last layer

        # freeze parameters loaded from model 1
        self.newModelExtractor.layer.weight.requires_grad = False

        # train


instance = GoModel()
instance.loadModel("1584165886", 3)
