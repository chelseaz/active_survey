import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def compute_na_proportion(responses):
    return responses.isnull().sum() / responses.shape[0]

def plot_na_proportion(responses):
    na_proportion = compute_na_proportion(responses)

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 15)
    na_proportion.plot.barh(title="Proportion of null responses by question")
    plt.show()

def plot_nonresponse_freq(responses):
    plt.rcParams["figure.figsize"] = (10, 6)
    nbins = responses.shape[1]
    responses.isnull().sum(axis=1).plot.hist(bins=nbins,
        title="Nonresponse frequency across respondents")
    plt.xlabel("Number of null responses for respondent")
    plt.ylabel("Number of respondents")
    plt.show()

def response_violin_plots(responses):
    plt.rcParams["figure.figsize"] = (10,6)
    n_questions = responses.shape[1]
    d = 8
    for slice_num in range(int(np.ceil(n_questions/d))):
        responses_slice = responses.iloc[:,(slice_num*d):((slice_num+1)*d)]
        plt.figure()
        sns.violinplot(data=responses_slice, orient='h')

def center_and_scale(responses, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("unrecognized standardization method %s" % method)

    scaled_array = scaler.fit_transform(responses)
    return pd.DataFrame(scaled_array, index=responses.index, columns=responses.columns)

def replace_values(responses, to_replace_by_column):
    n = responses.shape[0]
    for column, values in to_replace_by_column.items():
        replace_count = responses[column].isin(values).sum()
        print("Replacing {:d}/{:d} ({:.2f}%) values of {} with NA".format(replace_count, n, 100.*replace_count/n, column))
        responses[column].replace(values, np.nan, inplace=True)
