import numpy as np
import gbst
from gbst.sklearn import gbstModel
from gbst.metrics import evalauc
import pandas as pd
from sklearn.metrics import roc_auc_score


def custom_load(path, return_dmatrix=True):
    """Loads csv data.
    """
    print('Load Data...')
    ratio = 0.8
    seed = 123
    data = pd.read_csv(path)
    data = data.drop(['appl_no', 'cust_no', 'gapdays', 'day', 'second_label'], axis=1)
    features = data.values[0:, 0:-1].astype(np.float32)
    np.random.seed(seed)
    np.random.shuffle(features)
    ys = []
    for y in data.values[0:, -1]:
        ys.append(list(eval(y).values()).count(0))
    ys_np = np.array(ys).astype(np.int)
    np.random.seed(seed)
    np.random.shuffle(ys_np)
    if return_dmatrix:
        dmatrix1 = sbst.DMatrix(data=features[0:int(0.8*len(features))], label=ys_np[0:int(0.8*len(ys_np))])
        dmatrix2 = sbst.DMatrix(data=features[int(0.8*len(features)):], label=ys_np[int(0.8*len(ys_np)):])
        return dmatrix1, dmatrix2
    else:
        return (features[0:int(0.8*len(features))], ys_np[0:int(0.8*len(ys_np))]), (features[int(0.8*len(features)):], ys_np[int(0.8*len(ys_np)):])


train, test = custom_load('bmj_gbst_on_loan_sample_apart_xdata_selected_xy_train.csv', return_dmatrix=False)
train_x, train_y = train
test_x, test_y = test

print("Testing compatibility with sklearn")

classifier = gbstModel(n_estimators=40, num_class = 26)
# n_estimators = num_boost_rounds.
# num_class is #total_timewindows + 1
classifier.fit(X=train_x, y=train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric=evalauc, verbose=True)
# data format:
# X: numpy array. [dataset_size, num_features]
# y: 1-d numpy int array. Each index means "the number of timewindows that this sample survives". [dataset_size]

a = classifier.predict_hazard(data=test_x)
# the output of predict_hazard() is h(t). Accumulated surviving probability should be calculated manually.

