import pandas as pd
import torch

def sentence_features_to_device(sentence_features, target_device):
    # moves sentence feature tensors to the device
    for key in sentence_features:
        if isinstance(sentence_features[key], torch.Tensor):
            sentence_features[key] = sentence_features[key].to(target_device)
    return sentence_features

def get_train_test_dataframes(dataset_path):
    similariti_dataframe = pd.read_csv(dataset_path, index_col=0).reset_index(drop=True)
    similariti_dataframe.dropna(inplace=True)
    similariti_dataframe = similariti_dataframe.reset_index(drop=True)

    DATASET_LENGTH = len(similariti_dataframe)
    SPLIT_POINT = DATASET_LENGTH - 1000 # 1000 samples for val set


    similariti_dataframe = similariti_dataframe.sample(frac=1, random_state=1).reset_index(drop=True) # randomize order
    train_dataframe = similariti_dataframe.iloc[:SPLIT_POINT].reset_index(drop=True)
    test_dataframe = similariti_dataframe.iloc[SPLIT_POINT:].reset_index(drop=True)
    
    return train_dataframe, test_dataframe
