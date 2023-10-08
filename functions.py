import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from joblib import Parallel, delayed


# hardcoded stuff
root_dir = "/home/nate/tu_medical"
df_path = "data/otu_table_example.csv"


def forward_rolling_average(values, window_size):
    # Description: applies a forward rolling average function e.g.:
    #   sequence = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #   forward_rolling_average(values=sequence, window_size=3).tolist()
    #   out [1, 2, 3, 4, 5, 6, 7, 8]

    values = pd.Series(values)
    original_idxs = values.index

    series = values.reset_index(drop=True)

    for idx in range(len(series) - window_size):
        next_vals = series[idx:idx + window_size]

        # print(idx, next_vals.tolist())
        # print(next_vals.mean())

        series[idx] = next_vals.mean()

    series.index = original_idxs

    series = series.iloc[:-window_size]

    return series


def smooth_it_out(df, window_size=5, n_jobs=16):
    # Description: applies forward_rolling_average to every column in df in parallel

    smooth_cols = Parallel(n_jobs=n_jobs)(delayed(forward_rolling_average)(df[col], window_size) for col in df.columns)

    for idx, col in enumerate(df.columns):
        df[col] = smooth_cols[idx]

    df = df.iloc[:-window_size]

    return df


def feature_wise_scaling(df):
    # Description: scales every column to between 0 and 1

    for col in df.columns:
        _min_ = df[col].min()
        _max_ = df[col].max()

        df[col] = (df[col] - _min_) / (_max_ - _min_)

    return df


def preprocess(zero_values_percentage_cutoff, smoothing_window_size):

    # load the data
    df = pd.read_csv(df_path, index_col="Unnamed: 0").T

    # get rid of features that are way too sparse
    zero_counts = pd.Series([sum(df[col] == 0) for col in df.columns], index=df.columns)
    zero_pcts = zero_counts / len(df)
    populated_feats = zero_pcts[zero_pcts < zero_values_percentage_cutoff].index
    df = df[populated_feats]

    # smooth the data my forward-rolling averaging
    df = smooth_it_out(df=df, window_size=smoothing_window_size)

    # scale the data
    df = feature_wise_scaling(df)

    return df


def plot_a_taxa_sequence(sequence, color, title, figsize=(10,5)):

    plt.figure(figsize=figsize)
    sns.lineplot(sequence, color=color)
    plt.title(title)

    plt.show()


def cut_to_sequences(feats_df, seq_length):
    # Description: cuts the dataframe into X_sequences of shape (seq_length, n_features) and y_targets

    # Example:
    # Our example features are
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


def batch_mae_ignore_zeros(y_true, y_pred, false_positives_penalty_factor=0.1):
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
    mae_ignore_zeros = (mae_non_zero + (false_positives_avg * false_positives_penalty_factor)) * 100

    return mae_ignore_zeros


def calculate_percentage_errors(y_pred_df, y_test_df):
    # Description: calculate percentage errors on on all taxa

    y_pred_df = y_pred_df.reset_index(drop=True)
    y_test_df = y_test_df.reset_index(drop=True)

    errors_df = []
    for col in y_pred_df.columns:
        errors = abs((y_pred_df[col] - y_test_df[col]) / (y_pred_df[col] + 1e-10))
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
    plt.show()

    sns.set()
    plt.figure(figsize=figsize)
    sns.lineplot(pred_sequence, color=pred_colour)
    plt.title(f"Pred sequence for taxa_idx {target_taxa}")
    plt.show()

    sns.set()
    plt.figure(figsize=figsize)

    sns.lineplot(data=true_sequence, label='True Sequence', color=true_colour)
    sns.lineplot(data=pred_sequence, label='Predicted Sequence', color=pred_colour)

    plt.title(f"True and Predicted sequences for taxa_idx {target_taxa}")
    plt.xlabel('X-axis Label')  # Add your x-axis label here
    plt.ylabel('Y-axis Label')  # Add your y-axis label here

    plt.legend()

    plt.show()




