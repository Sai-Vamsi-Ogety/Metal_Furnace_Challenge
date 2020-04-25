from sklearn import ensemble

MODELS = {
    "randomforest" : ensemble.RandomForestClassifier(n_jobs = -1, verbose = 2),
    "extratrees":  ensemble.ExtraTreesClassifier(n_jobs = -1, verbose = 2)
}