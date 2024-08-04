import pandas as pd
from datetime import datetime 
import numpy as np

def force_submission_data(loss, data):
    sample = pd.read_csv('./data/sample_submission.csv')
    data = np.array(data).reshape(len(sample), 96,3).tolist()
    sample['forces'] = list(data)
    
    return sample

if __name__ == "__main__":
    force_submission_data(1, 0)
