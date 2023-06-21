import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertModel

from src.algos.bbb_layers import BBBLinear
from src.algos.rank1 import Rank1Linear

class BertClassifier(nn.Module):
    def __init__(self, ty, classes, prior=None, drop_p=None, components=None):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if ty == "map":
            self.classifier = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.2), # Should be only 0.2
                nn.Linear(768, classes),
            )
        elif ty == "drop":
            self.classifier = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(768, classes),
            )
        elif ty == "bbb":
            self.classifier = nn.Sequential(
                BBBLinear(768, 768, prior, prior),
                nn.ReLU(),
                nn.Dropout(0.2),
                BBBLinear(768, classes, prior, prior),
            )
        elif ty == "rank1":
            self.classifier = nn.Sequential(
                Rank1Linear(768, 768, prior, components=components),
                nn.ReLU(),
                nn.Dropout(0.2),
                Rank1Linear(768, classes, prior, components=components),
            )
        else:
            raise ValueError(f"Unknown classifier type '{ty}'")
    
    def forward(self, input):
        input_ids = input[:, :, 0]
        attention_mask = input[:, :, 1]
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_out[0]
        return self.classifier(hidden_state[:,0])
