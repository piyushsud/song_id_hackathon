import pandas as pd
import numpy as np
import argparse
import NNmodel


parser = argparse.ArgumentParser(description='description')
parser.add_argument('dataset', type=str, help='dataset file')
args = parser.parse_args()

OUTPUT_PATH = '/home/piyushsud/Desktop/'

print('loading datasets')
datasets = pd.read_csv('/home/piyushsud/Desktop/fma_metadata/features.csv')

ds_name = args.dataset

try:
    file = pd.read_csv(OUTPUT_PATH + ds_name + '-results.csv')    
except FileNotFoundError:
    file = None