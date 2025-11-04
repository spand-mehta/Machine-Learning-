# this validation file computes binary balanced accuracy

import argparse
import numpy as np
import pickle
import bz2
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# build parser 
parser = argparse.ArgumentParser(description='Generate kaggle labels for ece5307 project')
parser.add_argument("model_path",help="path to model file")
parser.add_argument("--Xtr_path",default="Xtr.csv",help="path to training-feature file")
parser.add_argument("--ytr_path",default="ytr.csv",help="path to training-label file")
parser.add_argument("--Xka_path",default="Xka.csv",help="path to kaggle-feature file")
parser.add_argument("--yka_hat_path",default="yka_hat.csv",help="path to kaggle-label-prediction file")
parser.add_argument("--verbose",default="True",help="logical for verbosity")

# parse input arguments
args = parser.parse_args()
model_path = args.model_path
Xtr_path = args.Xtr_path
ytr_path = args.ytr_path
Xka_path = args.Xka_path
yka_hat_path = args.yka_hat_path
if not yka_hat_path.endswith('.csv'): 
    print("Error: Argument of --yka_hat_path must end in .csv")
    quit()
verbose_str = args.verbose
if verbose_str == 'False' or verbose_str == 'false' or verbose_str == '0':
    verbose = False
else:
    verbose = True

# load data
Xtr = np.loadtxt(Xtr_path, delimiter=",")
ytr = np.loadtxt(ytr_path, delimiter=",")
Xka = np.loadtxt(Xka_path, delimiter=",")

# load model
if model_path.endswith('.json'):
    # XGBOOST
    from xgboost import XGBClassifier 
    model = XGBClassifier()
    model.load_model(model_path)
    ytr_hat = model.predict(Xtr)
    yka_hat = model.predict(Xka)
elif model_path.endswith('.pth'):
    # PYTORCH
    import torch 
    model = torch.jit.load(model_path)
    with torch.no_grad():
        ytr_hat = model(torch.Tensor(Xtr)).detach().numpy().argmax(axis=1)
        yka_hat = model(torch.Tensor(Xka)).detach().numpy().argmax(axis=1)
elif model_path.endswith('.bz2'): 
    # SKLEARN
    with bz2.BZ2File(model_path,'r') as f:
        model = pickle.load(f)
    name = type(model).__name__
    if name in [
        'LogisticRegression',
        'LinearSVC',
        'SVC',
        'NuSVC',
        'RidgeClassifier',
        'AdaBoostClassifier',
        'BaggingClassifier',
        'GradientBoostingClassifier',
        'StackingClassifier',
        'HistGradientBoostingClassifier',
        'RandomForestClassifier',
        'ExtraTreesClassifier',
        'VotingClassifier',
        'Pipeline' 
    ]:
        ytr_hat = model.predict(Xtr)
        yka_hat = model.predict(Xka)
    else:
        raise ValueError('model type '+name+' not supported')
else:
    print("Error: Unrecognized extension on model_path.  Should be .bz2 for SkLearn models, or .pth for PyTorch models, or .json for XGBoost models")
    quit()


# print training score 
acc = balanced_accuracy_score(ytr,ytr_hat)
if verbose:
    print('training balanced-accuracy = ',acc)

# save kaggle-label predictions in a csv file 
df = pd.DataFrame(data={'Id':np.arange(len(yka_hat)),
                        'Label':np.int64(yka_hat)})
df.to_csv(yka_hat_path, index=False)
if verbose:
    print('kaggle label predictions saved in',yka_hat_path)
