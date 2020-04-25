import pandas as pd  
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import dispatcher
import argparse
import joblib 

TRAINING_DATA = "../input/train_folds.csv"
Folds = [0,1,2,3,4]

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}


if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("--model",type=str)
    args = my_parser.parse_args()
    model = args.model
    print("Model : {}".format(model))
    for FOLD in Folds:
        print('Fold : {}'.format(FOLD))
        df = pd.read_csv(TRAINING_DATA)
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
        valid_df = df[df.kfold == FOLD]

        ytrain = train_df.grade.values
        yvalid = valid_df.grade.values

        train_df = train_df.drop(["grade", "kfold"], axis = 1)
        valid_df = valid_df.drop(["grade", "kfold"], axis = 1)

        valid_df = valid_df[train_df.columns]

        #data ready to train

        clf = dispatcher.MODELS[model]
        clf.fit(train_df, ytrain)
        preds = clf.predict_proba(valid_df)
        print("log_loss:")
        print(metrics.log_loss(yvalid,preds))

    print("Savind the model: {} into models folder".format(model))
    joblib.dump(clf, f'../models/{model}.pkl')    
    print("-----------------------------")
    


