import pandas as pd
import torch

def sentence_features_to_device(sentence_features, target_device):
    for key in sentence_features:
        if isinstance(sentence_features[key], torch.Tensor):
            sentence_features[key] = sentence_features[key].to(target_device)
    return sentence_features


def get_train_test_dataframes(dataset_path):
    similariti_dataframe = pd.read_csv(dataset_path, index_col=0).reset_index(drop=True)
    similariti_dataframe.dropna(inplace=True)
    similariti_dataframe = similariti_dataframe.reset_index(drop=True)

    # TRAIN_DATASET_RATIO = 0.9
    DATASET_LENGTH = len(similariti_dataframe)
    # SPLIT_POINT = int(DATASET_LENGTH * TRAIN_DATASET_RATIO)
    SPLIT_POINT = DATASET_LENGTH - 1000


    similariti_dataframe = similariti_dataframe.sample(frac=1, random_state=1).reset_index(drop=True)
    train_dataframe = similariti_dataframe.iloc[:SPLIT_POINT].reset_index(drop=True)
    test_dataframe = similariti_dataframe.iloc[SPLIT_POINT:].reset_index(drop=True)
    
    return train_dataframe, test_dataframe
