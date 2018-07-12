
from functions import*
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch flower classification')

parser.add_argument('checkpoint')

parser.add_argument('image_path')

parser.add_argument('--topk', type=int)

parser.add_argument('--gpu',action='store_true')
