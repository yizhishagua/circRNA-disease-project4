import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.preprocessing import MinMaxScaler

def make_datasets():
    """

    :return:
    """

    return [make_moons(n_samples=200, noise=0.3, random_state=0),
            make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
            make_linearly_separable()]


def make_classifiers(number_of_hidden):
    """

    :return:
    """

    names = ["ELM(tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)", "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = number_of_hidden

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
    log_reg = LogisticRegression()

    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                   GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                   GenELMClassifier(hidden_layer=srhl_sinsq),
                   GenELMClassifier(hidden_layer=srhl_tribas),
                   GenELMClassifier(hidden_layer=srhl_hardlim),
                   GenELMClassifier(hidden_layer=srhl_rbf)]

    return names, classifiers


def load_samples(file, col_num):
    df = pd.read_csv(file)
    X = np.array(df.iloc[:,col_num:])
    y = np.array(df['label'].tolist())
    return X, y

def make_linearly_separable():
    """

    :return:
    """

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)


if __name__ == '__main__':
    # generate some datasets
    file = r"D:\Study_Shihao_Li\circRNA_disease_2020_thesis\data\2020\Prototype_CircRNA_Disease_Association\third_data\final_dataset\pick_method1\feature_file\dataset_feature_file\dataset_disease_sim_4_mer.csv"
    X, y = load_samples(file, 5) #加载样本.
    datasets = [[X, y]]
    names, classifiers = make_classifiers(40)

    # iterate over datasets
    scaler = MinMaxScaler()
    scaler.fit(X)
    # X = scaler.transform(X)
    X = StandardScaler().fit_transform(X,y)
    # pre-process dataset, split into training and test part
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    y_test = y_test.reshape(-1, )
    y_train = y_train.reshape(-1, )

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('Model %s score: %s' % (name, score))