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
    df = pd.read_csv(df_path, index_col="Unnamed: 0").T
    meta_df = pd.read_csv(metadata_path, index_col="sample_id")

    df["subject_id"] = [idx.split(".")[0] for idx in df.index]
    df["sampling_day"] = meta_df.loc[df.index]["Sampling_day"].astype(int)
    df["ind_time"] = meta_df.loc[df.index]["ind_time"].astype(float)

    return df


def remove_underpopulated_taxa(df, max_zeros_pct):
    # get rid of features that are way too sparse
    zero_counts = pd.Series([sum(df[col] == 0) for col in df.columns], index=df.columns)
    zero_pcts = zero_counts / len(df)
    populated_feats = zero_pcts[zero_pcts < max_zeros_pct].index
    df = df[populated_feats]

    return df


def standard_rolling_average(df, window_size):
    # hardcoded stuff
    ignore_cols = ["subject_id", "sampling_day", "ind_time"]

    for column in df.columns:
        if column in ignore_cols: continue
        df[column] = df[column].rolling(window=window_size).mean()

    df = df.dropna()  # Drop NaN values introduced by the rolling operation
    return df


def feature_wise_scaling(df):
    ignore_cols = "subject_id"
    # Description: scales every column to between 0 and 1

    for col in df.columns:
        if str(col) in ignore_cols:
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


def cut_to_sequences(feats_df, seq_length):
    # Description: cuts the dataframe into X_sequences of shape (seq_length, n_features) and y_targets

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

    X_sequences = []
    y_targets = []

    for i in range(len(feats_df) - seq_length):
        X_sequences.append(feats_df.iloc[i:i + seq_length])
        y_targets.append(feats_df.iloc[i + seq_length])

    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    X_sequences = X_sequences.reshape(-1, seq_length, num_features)

    return X_sequences, y_targets


def feats_and_targets(df, seq_length):
    feats = []
    targets = []

    subjects = df.subject_id.unique()
    df_subject_grp = df.groupby("subject_id")

    for subject_id in subjects:
        subject_df = df_subject_grp.get_group(subject_id).drop(columns=["subject_id"])
        subject_feats, subject_targets = cut_to_sequences(subject_df, seq_length=seq_length)

        for sequence_idx in range(len(subject_feats)):
            feats.append(subject_feats[sequence_idx])
            targets.append(subject_targets[sequence_idx])

    feats = np.asarray(feats)
    targets = np.asarray(targets)

    return feats, targets


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

    # hardcoded stuff
    ignore_cols = ["subject_id", "sampling_day", "ind_time"]

    y_pred_df = y_pred_df.reset_index(drop=True)
    y_test_df = y_test_df.reset_index(drop=True)

    errors_df = []
    for col in y_pred_df.columns:
        if col in ignore_cols: continue
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

    df = df.reset_index(drop=True)

    feats_list = []
    targets = []
    for top_sample_idx in df.index[seq_length - 1: len(df) - 1]:
        feats_row = []
        for taxa_idx in df.columns:
            taxa_sequence = df.loc[top_sample_idx - seq_length + 1: top_sample_idx, taxa_idx]
            feats_row.append(taxa_sequence)

        target = df.loc[top_sample_idx + 1]
        targets.append(target)

        feats_row = pd.concat(feats_row, ignore_index=True)
        feats_list.append(feats_row)

    feats_df = pd.concat(feats_list, axis=1).T
    targets_df = pd.concat(targets, axis=1).T

    return feats_df, targets_df




