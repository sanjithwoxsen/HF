from huggingface_hub import login

login('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')

from FWModule.config import FeelWiseConfig
from FWModule.model import FeelWiseModel
import torch
from transformers import AutoTokenizer


tk_path = "/Users/sanjithrj/PycharmProjects/HF/Tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tk_path)


conf = FeelWiseConfig()
HF_Model = FeelWiseModel(conf) # instantiate the model using the config

# load the weights
weights = torch.load("model.pth",map_location=torch.device('cpu'))
HF_Model.load_state_dict(weights['state_dict'])

conf.register_for_auto_class()
HF_Model.register_for_auto_class("AutoModel")
conf.push_to_hub('FeelWise')
HF_Model.push_to_hub('FeelWise')
tokenizer.push_to_hub('FeelWise')