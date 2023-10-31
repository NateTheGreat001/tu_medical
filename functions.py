import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from joblib import Parallel, delayed
import os
import shutil
from tqdm import tqdm


# hardcoded stuff
root_dir = "/home/nate/tu_medical"

df_path = "data/otu_table_example.csv"
metadata_path = "data/metadata_example.csv"
metadata_cols = ["subject_id", "sampling_day", "sampling_gap", "ind_time"]


#def forward_rolling_average(values, window_size):
#    # Description: applies a forward rolling average function e.g.:
#    #   sequence = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#    #   forward_rolling_average(values=sequence, window_size=3).tolist()
#    #   out [1, 2, 3, 4, 5, 6, 7, 8]
#
#    values = pd.Series(values)
#    original_idxs = values.index
#
#    series = values.reset_index(drop=True)
#
#    for idx in range(len(series) - window_size):
#        next_vals = series[idx:idx + window_size]
#
#        # print(idx, next_vals.tolist())
#        # print(next_vals.mean())
#
#        series[idx] = next_vals.mean()
#
#    series.index = original_idxs
#
#    series = series.iloc[:-window_size]
#
#    return series


#def smooth_it_out(df, window_size=5, n_jobs=16):
#    # Description: applies forward_rolling_average to every column in df in parallel
#
#    smooth_cols = Parallel(n_jobs=n_jobs)(delayed(forward_rolling_average)(df[col], window_size) for col in df.columns)
#
#    for idx, col in enumerate(df.columns):
#        df[col] = smooth_cols[idx]
#
#    df = df.iloc[:-window_size]
#
#    return df


def load_and_merge():

    # Description: loads the main dateset and the metadata datasets, merges their contents

    df = pd.read_csv(df_path, index_col="Unnamed: 0").T
    meta_df = pd.read_csv(metadata_path, index_col="sample_id")

    df["subject_id"] = [idx.split(".")[0] for idx in df.index]
    df["sampling_day"] = meta_df.loc[df.index]["Sampling_day"].astype(int)
    df["ind_time"] = meta_df.loc[df.index]["ind_time"].astype(float)

    return df


def calculate_non_zero_value_percentages(df):

    # Description: for each taxa calculate how much percentage of the total entries are not zeros

    non_zero_counts = pd.Series([sum(df[col] != 0) for col in df.columns], index=df.columns)
    population_rates = non_zero_counts / len(df)

    return population_rates


def remove_underpopulated_taxa(df, min_non_zero_pcts):

    # get rid of the taxa that are populated too sparsely

    non_zero_pcts = calculate_non_zero_value_percentages(df)
    populated_feats = non_zero_pcts[non_zero_pcts > min_non_zero_pcts].index
    df = df[populated_feats]

    return df


def standard_rolling_average(df, window_size):

    # For each taxa apply a rolling average function. Ignores the non-taxa column e.g. sampling_day

    for column in df.columns:
        if column in metadata_cols: continue
        df[column] = df[column].rolling(window=window_size).mean()

    df = df.dropna()  # Drop NaN values introduced by the rolling operation
    return df


def feature_wise_scaling(df):
    # Description: scales every column to between 0 and 1

    for col in df.columns:
        if str(col) in ["subject_id"]:
            continue

        _min_ = df[col].min()
        _max_ = df[col].max()

        df[col] = (df[col] - _min_) / (_max_ - _min_)

    return df


#def preprocess(zero_values_percentage_cutoff, smoothing_window_size):
#
#    # Description: reads the data; removes underpopulated taxa; smoothes and scales
#
#    # load the data
#    df = pd.read_csv(df_path, index_col="Unnamed: 0").T
#
#    # get rid of features that are way too sparse
#    zero_counts = pd.Series([sum(df[col] == 0) for col in df.columns], index=df.columns)
#    zero_pcts = zero_counts / len(df)
#    populated_feats = zero_pcts[zero_pcts < zero_values_percentage_cutoff].index
#    df = df[populated_feats]
#
#    # smooth the data my forward-rolling averaging
#    df = smooth_it_out(df=df, window_size=smoothing_window_size)
#
#    # scale the data
#    df = feature_wise_scaling(df)
#
#    return df


def plot_a_taxa_sequence(sequence, color, title, figsize=(10,5)):

    plt.figure(figsize=figsize)
    sns.lineplot(sequence, color=color)
    plt.title(title)

    plt.show()


def cut_to_sequences(feats_df, seq_length, mode):
    # Description: cuts the dataframe into X_sequences of shape (seq_length, n_features) and y_targets
    # if mode is test then metadata cols are kept in the targets for evaluation
    # elif mode is test then matadata cols are not kept in targets because they are not predicted

    # Example:
    # Our example data
    #    one  two  three
    # 0    1   11     21
    # 1    2   12     22
    # 2    3   13     23
    # 3    4   14     24
    # 4    5   15     25
    # 5    6   16     26
    # 6    7   17     27
    # 7    8   18     28
    # 8    9   19     29
    #
    # X_sequences, y_targets = cut_to_sequences(feats_df=example_df, seq_length=3)
    #
    # X_sequences[0] is [[ 1 11 21]
    #                    [ 2 12 22]
    #                    [ 3 13 23]]
    #
    # y_targets[0] is    [ 4 14 24]

    num_features = len(feats_df.columns)

    # get the column names for the subject_id encodings
    subject_id_dummies = feats_df.columns[pd.Series(feats_df.columns).apply(lambda x: str(x).startswith("subject_id"))]

    X_sequences = []
    y_targets = pd.Series(dtype=float)

    for i in range(len(feats_df) - seq_length):
        target_idx = feats_df.iloc[i + seq_length].name

        feats = feats_df.iloc[i:i + seq_length].drop(columns=["subject_id"])

        targets = feats_df.loc[target_idx]
        targets = targets[~targets.index.isin(subject_id_dummies)]  # remove subject_id encodings from targets

        if mode == "train":
            targets = targets[~targets.index.isin(metadata_cols)]

        X_sequences.append(feats)
        y_targets[target_idx] = targets

    X_sequences = np.asarray(X_sequences)

    return X_sequences, y_targets


def feats_and_targets(df, seq_length, n_test_seq):

    # Description: Prepares the features. For each test subject applies cut_to_sequences to get X_sequences and y_targets
    # Each test sequence is a set of all taxa observations for one subject

    subjects = df.subject_id.unique()

    # encode subject id
    subject_ids = df["subject_id"]  # to preserve the original column through one hot encoding
    df = pd.get_dummies(df, columns=['subject_id'], prefix='subject_id')  # encode subject_id using one hot encoding
    df["subject_id"] = subject_ids

    # create the sampling_gap_feature
    df['sampling_gap'] = df['sampling_day'].diff()
    df = df.fillna(0)
    df = df.drop(columns=["sampling_day"])

    df_subject_grp = df.groupby("subject_id")

    # select the test subjects
    test_subjects_idx = np.random.choice(len(subjects), size=n_test_seq, replace=False)
    test_subjects = subjects[test_subjects_idx]

    print(f"The test subjects are {test_subjects}")

    train_feats = []
    train_targets = []

    test_feats = {test_subject: [] for test_subject in test_subjects}
    test_targets = {test_subject: [] for test_subject in test_subjects}

    for subject_id in subjects:
        subject_df = df_subject_grp.get_group(subject_id)

        feats_mode = "test" if subject_id in test_subjects else "train"
        subject_feats, subject_targets = cut_to_sequences(subject_df, seq_length=seq_length, mode=feats_mode)

        for sequence_idx in range(len(subject_feats)):

            if subject_id in test_subjects:
                test_feats[subject_id].append(subject_feats[sequence_idx])
                test_targets[subject_id].append(subject_targets[sequence_idx])
            else:
                train_feats.append(subject_feats[sequence_idx])
                train_targets.append(subject_targets[sequence_idx])

    # shuffle the train features and targets
    random_order = np.random.permutation(len(train_feats))

    train_feats = np.asarray(train_feats)[random_order].astype("float32")
    train_targets = np.asarray(train_targets)[random_order].astype("float32")

    return train_feats, train_targets, test_feats, test_targets, test_subjects


class mae_ignore_zeros(tf.keras.losses.Loss):

    # Description: a version of batch MAE that only accounts the errors where the target is not zero.
    #              Thus if y_true is 0 and y_pred is 0 the nodel is not rewarded and not punished.
    #              A lot of taxa are sparsely populated and this loss allows the model to focus on the relevant errors.

    # Example:
    #
    # y_true = [0,0,10]
    # y_pred = [0,0,0]
    #
    # keras.losses.mae(y_true, y_pred)
    # out: 3
    #
    # mae_ignore_zeros(y_true, y_pred)
    # out: 10
    #

    def __init__(self, false_positives_penalty_factor, name='mae_ignore_zeros',
                 reduction=tf.keras.losses.Reduction.AUTO, ):
        super(mae_ignore_zeros, self).__init__()
        self.false_positives_penalty_factor = false_positives_penalty_factor
        self.name = name

    def call(self, y_true, y_pred):

        # Find indices where y_true is not zero
        non_zero_indices = tf.where(tf.not_equal(y_true, 0))

        # Gather the non-zero elements from y_true and y_pred using the indices
        y_true_non_zero = tf.gather_nd(y_true, non_zero_indices)
        y_pred_non_zero = tf.gather_nd(y_pred, non_zero_indices)

        y_true_non_zero = tf.cast(y_true_non_zero, tf.float64)
        y_pred_non_zero = tf.cast(y_pred_non_zero, tf.float64)

        # Calculate MAE on the non-zero elements
        mae_non_zero = tf.reduce_mean(tf.abs(y_pred_non_zero - y_true_non_zero))

        # Find indices where y_true is zero
        zero_indices = tf.where(tf.equal(y_true, 0))

        # Gather the corresponding y_pred values
        y_pred_zero = tf.gather_nd(y_pred, zero_indices)

        y_pred_zero = tf.cast(y_pred_zero, tf.float64)

        # Calculate the average of false positives
        false_positives_avg = tf.reduce_mean(y_pred_zero)

        # Combine the MAE on non-zero elements with the average of false positives
        mae_ignore_zeros = (mae_non_zero + (false_positives_avg * self.false_positives_penalty_factor)) * 100

        return mae_ignore_zeros


def calculate_percentage_errors(y_pred_df, y_test_df):
    # Description: calculate percentage errors on on all taxa

    y_pred_df = y_pred_df.reset_index(drop=True)
    y_test_df = y_test_df.reset_index(drop=True)

    errors_df = []
    for col in y_pred_df.columns:
        if col in metadata_cols: continue
        errors = abs((y_test_df[col] - y_pred_df[col]) / (y_test_df[col] + 1e-10))
        errors_df.append(errors)

    errors_df = pd.concat(errors_df, axis=1)

    return errors_df


def percentile_graph(errors_df, label, y_top_lim=3, step=0.1):
    # Description: Solves the problem of representing accuracy on a lot of different taxa in one single graph
    # Produces a graph of percentiles on medians in (y_pred - y_true) on taxa sequences

    # hardcoded stuff
    percentile_range = np.arange(0, 101)
    x_ticks_range = range(0, 110, 10)

    median_errors = errors_df.median()
    median_error_percentiles = np.percentile(median_errors, percentile_range)

    sns.set()
    plt.figure(figsize=(8, 6))
    sns.lineplot(median_error_percentiles)
    plt.title(f"Percentiles of median absolute errors {label}")
    plt.xlabel("Percentile")
    plt.ylabel("Median Error")
    plt.xticks(x_ticks_range)
    plt.yticks(np.arange(0, float(y_top_lim + step), step))
    plt.ylim(0, y_top_lim)

    percentile_50 = median_error_percentiles[50]
    sns.lineplot([percentile_50 for _ in range(len(percentile_range))], color="red", label="50 percentile line")

    plt.show()


def sequence_comparisson_graphs(true_sequence, pred_sequence, target_taxa):

    # hardcoded stuff
    figsize = (10,5)
    true_colour = "blue"
    pred_colour = "green"

    sns.set()
    plt.figure(figsize=figsize)
    sns.lineplot(true_sequence, color=true_colour)
    plt.title(f"True sequence for taxa_idx {target_taxa}")
    plt.xticks([])
    plt.show()

    sns.set()
    plt.figure(figsize=figsize)
    sns.lineplot(pred_sequence, color=pred_colour)
    plt.title(f"Pred sequence for taxa_idx {target_taxa}")
    plt.xticks([])
    plt.show()

    sns.set()
    plt.figure(figsize=figsize)

    sns.lineplot(data=true_sequence, label='True Sequence', color=true_colour)
    sns.lineplot(data=pred_sequence, label='Predicted Sequence', color=pred_colour)

    plt.title(f"True and Predicted sequences for taxa_idx {target_taxa}")

    plt.legend()
    plt.xticks([])
    plt.show()



def compile_model(model, loss):
    model.compile(optimizer="Adam", loss=loss, metrics=["mae", "mape"])
    return model


class ensemble():

    # Creates an ensemble of models where one every model intakes all the features but only estimates counts for one taxa

    def __init__(self, ensemble_name, loss, overwrite_on_train=False):

        self.models_out_dir = f"{root_dir}/models/{ensemble_name}"
        self.overwrite_on_train = overwrite_on_train
        self.loss = loss

    def train(self, X_sequences_train, y_targets_train, n_epochs):

        if not os.path.exists(self.models_out_dir):
            os.mkdir(self.models_out_dir)
        else:
            if self.overwrite_on_train is False:
                raise Exception("This model dir already exists")
            else:
                print("Overwriting an existing model dir")
                shutil.rmtree(self.models_out_dir)
                os.mkdir(self.models_out_dir)

        for taxa_idx in tqdm(y_targets_train.columns, desc="Training models"):
            model = fetch_model()
            model = compile_model(model, self.loss)
            y_targets = y_targets_train[taxa_idx]
            model.fit(x=X_sequences_train, y=y_targets, validation_split=0.05, epochs=n_epochs, verbose=0)

            model.save(f"{self.models_out_dir}/{taxa_idx}.model")

            del model

    def load(self):

        self.model_dic = {}

        for model_dir in tqdm(os.listdir(self.models_out_dir), desc="Loading the models"):
            taxa_idx = int(model_dir.replace(".model", ""))

            if isinstance(self.loss, mae_ignore_zeros):
                model = tf.keras.models.load_model(f"{self.models_out_dir}/{taxa_idx}.model", compile=False)
                model = compile_model(model, self.loss)
            else:
                model = tf.keras.models.load_model(f"{self.models_out_dir}/{taxa_idx}.model")

            self.model_dic[taxa_idx] = model

    def predict(self, X_sequences):

        self.load()

        n_sequences = len(X_sequences)

        pred_list = []
        for taxa_idx in tqdm(self.model_dic.keys(), desc="Predicting values"):
            model = self.model_dic[taxa_idx]
            pred_list.append(model.predict(X_sequences, verbose=0).reshape(n_sequences, ))
            del (model)

        pred_df = pd.DataFrame(pred_list).T

        return pred_df


def create_flat_sequences(df, seq_length):
    # Solves the problem of representing sequences of taxa counts in 2d space
    # Using seq_length previous values for each column predict the next one

    # Example:

    # seq_length = 3

    # input data:
    #     one  two  three
    # 0    0   10     20
    # 1    1   11     21
    # 2    2   12     22
    # 3    3   13     23
    # 4    4   14     24
    # 5    5   15     25
    # 6    6   16     26
    # 7    7   17     27
    # 8    8   18     28
    # 9    9   19     29
    #
    # feats_df.iloc[0]:
    # 0  1  2   3   4   5   6   7   8
    # 0  1  2  10  11  12  20  21  22
    #
    # targets_df.iloc[0]:
    # one  two  three
    #  3   13     23

    # get the column names for the subject_id encodings
    subject_id_dummies = df.columns[pd.Series(df.columns).apply(lambda x: str(x).startswith("subject_id"))]

    df = df.reset_index(drop=True)

    feats_list = []
    targets = []
    for top_sample_idx in df.index[seq_length - 1: len(df) - 1]:
        feats_row = []
        for taxa_idx in df.columns:
            taxa_sequence = df.loc[top_sample_idx - seq_length + 1: top_sample_idx, taxa_idx]
            feats_row.append(taxa_sequence)

        target = df[df.columns[(~df.columns.isin(subject_id_dummies) & (~df.columns.isin(metadata_cols)))]].loc[top_sample_idx + 1]
        targets.append(target)

        feats_row = pd.concat(feats_row, ignore_index=True)
        feats_list.append(feats_row)

    feats_df = pd.concat(feats_list, axis=1).T
    targets_df = pd.concat(targets, axis=1).T

    return feats_df, targets_df


def xgboost_flat_feats_and_targets(df, n_test_seq, seq_length, validation_pct=0.1):

    # Follows the same logic as feats_and_targets, but with two major differences: the first is it applies create_flat_sequences because XGBoost expects flat output
    # secondly it outputs also a validation set as it is required to properly train an XGBoost model

    # encode subject id
    subject_ids = df["subject_id"]  # to preserve the original column through one hot encoding
    df = pd.get_dummies(df, columns=['subject_id'], prefix='subject_id')  # encode subject_id using one hot encoding
    df["subject_id"] = subject_ids

    # create the sampling_gap_feature
    df['sampling_gap'] = df['sampling_day'].diff()
    df = df.fillna(0)
    df = df.drop(columns=["sampling_day"])

    subjects = df.subject_id.unique()
    df_subject_grp = df.groupby("subject_id")

    test_subjects_idx = np.random.choice(len(subjects), size=n_test_seq, replace=False)
    test_subjects = subjects[test_subjects_idx]

    print(f"The test subjects are {test_subjects}")

    train_feats = []
    train_targets = []
    test_feats = {test_subject: [] for test_subject in test_subjects}
    test_targets = {test_subject: [] for test_subject in test_subjects}

    for subject_id in subjects:
        subject_df = df_subject_grp.get_group(subject_id).drop(columns=["subject_id"])
        subject_feats, subject_targets = create_flat_sequences(subject_df, seq_length=seq_length)

        if subject_id in test_subjects:
            test_feats[subject_id] = subject_feats
            test_targets[subject_id] = subject_targets
        else:
            train_feats.append(subject_feats)
            train_targets.append(subject_targets)

    train_feats = pd.concat(train_feats).reset_index(drop=True)
    train_targets = pd.concat(train_targets).reset_index(drop=True)

    # get the validation sets
    validation_size = int(np.round(validation_pct * len(train_feats)))
    validation_idxs = list(np.random.choice(range(len(train_feats)), size=validation_size, replace=False))

    val_feats = train_feats.loc[validation_idxs]
    val_targets = train_targets.loc[validation_idxs]

    train_feats = train_feats.drop(validation_idxs).reset_index(drop=True)
    train_targets = train_targets.drop(validation_idxs).reset_index(drop=True)

    random_order = np.random.permutation(len(train_feats))

    train_feats = train_feats.loc[random_order].reset_index(drop=True)
    train_targets = train_targets.loc[random_order].reset_index(drop=True)

    return train_feats, train_targets, val_feats, val_targets, test_feats, test_targets, test_subjects


def median_errors_by_population_rate(df, only_predicted_errors, only_predicted_taxa):

    # Description: creates a scatterplot where median errors in taxa are compared to their population rates

    population_rates = calculate_non_zero_value_percentages(df).drop(['subject_id', 'sampling_day'])

    population_rates_only_predicted = population_rates[only_predicted_taxa]
    populated_taxa_errors = only_predicted_errors.median()[population_rates_only_predicted.index]
    # clip populated taxa errors for plotting
    populated_taxa_errors = populated_taxa_errors.clip(0, 10)

    sns.set()
    plt.figure(figsize=(10, 8))

    sns.regplot(x=population_rates_only_predicted, y=populated_taxa_errors)
    plt.yticks(range(0, 11))
    plt.xlabel("Feature population rates")
    plt.ylabel("Median Error")

    plt.title(f"Median errors by population rates")
    plt.show()




