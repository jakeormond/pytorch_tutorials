# from https://pytorch.org/text/stable/tutorials/sst2_classification_non_distributed.html

# This tutorial demonstrates how to train a text classifier on SST-2 binary 
# dataset using a pre-trained XLM-RoBERTa (XLM-R) model. We will show how 
# to use torchtext library to:
# build text pre-processing pipeline for XLM-R model
# read SST-2 dataset and transform it using text and label transformation
# instantiate classification model using pre-trained XLM-R encoder

import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)


from torch.utils.data import DataLoader


# Alternately we can also use transform shipped with pre-trained model that does 
# all of the above out-of-the-box
# text_transform = XLMR_BASE_ENCODER.transform()


# Dataset
# torchtext provides several standard NLP datasets. For complete list, 
# refer to documentation at https://pytorch.org/text/stable/datasets.html. 
# These datasets are build using composable torchdata datapipes and hence support 
# standard flow-control and mapping/transformation using user defined functions 
# and transforms. Below, we demonstrate how to use text and label processing 
# transforms to pre-process the SST-2 dataset.

from torchtext.datasets import SST2

batch_size = 16

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")


# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
def apply_transform(x):
    return text_transform(x[0]), x[1]


train_datapipe = train_datapipe.map(apply_transform)
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(apply_transform)
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)

# Alternately we can also use batched API (i.e apply transformation on the whole batch)