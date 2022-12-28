# Written by serhatsoyer

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential, load_model
from keras.layers import Input, Flatten, LSTM, Dropout, Dense
from keras.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import time as timer # To be able to measure execution time


class Const_Sets:
    layerSize = 32
    dropoutProb = 1 / 16
    maxEpochs = 128
    patienceNum = 8


class Paths:
    dataset_path = Path.cwd().parent.parent.parent / 'datasets' / 'time_series'
    id = ''
    case_path = ''
    model_path = ''
    learn_curve_path = ''
    test_path = ''


class ResetStatesCallback(Callback):
    def __init__(self):
        super().__init__()
        self.resetStates = False

    def set_resetStates(self, resetStates):
        self.resetStates = resetStates

    def on_epoch_begin(self, epoch, logs=None):
        if self.resetStates: self.model.reset_states()
    
    def on_test_batch_begin(self, batch, logs=None):
        if self.resetStates: self.model.reset_states()
    
    def on_predict_batch_begin(self, batch, logs=None):
        if self.resetStates: self.model.reset_states()


input = np.load(Paths.dataset_path / 'input.npy')
output = np.load(Paths.dataset_path / 'output.npy')

reset_states_callback = ResetStatesCallback()

def init(Train_Sets):
    Paths.id = f'{Train_Sets.vers}_{Train_Sets.within}_{Train_Sets.inter}_' + \
        f'{Train_Sets.numOfSens}_{Train_Sets.batchSize}_' + \
        f'{int(Train_Sets.useHalf)}_' + \
        f'{int(Train_Sets.dense)}_{int(Train_Sets.shuffle)}_' + \
        f'{int(Train_Sets.stateful)}_{int(Train_Sets.resetStates)}'

    Paths.case_path = Paths.dataset_path / 'cases' / Paths.id
    Paths.case_path.mkdir(parents=True, exist_ok=True)
    Paths.model_path = Paths.case_path / 'model'
    Paths.history_path = Paths.case_path /'history.csv'
    Paths.learn_curve_path = Paths.case_path / 'learn_curve.pdf'
    Paths.test_path = Paths.case_path / 'test.pdf'
    reset_states_callback.set_resetStates(Train_Sets.resetStates)
    print(f'ID: {Paths.id}')
    return Train_Sets, Paths


def split_data(Train_Sets):
    """
    batchSize = (2 ** 0) or ... or (2 ** 8)
    Train: 60%
    Val:   20%
    Test:  20%
    numOfWins = 256 * (6 + 2 + 2) * 2 * 6 # Only change the integer at the end with an integer
    """
    winNum = round(input.shape[1] / 2) if Train_Sets.useHalf else input.shape[1]
    X_train, X_val, y_train, y_val = \
        train_test_split(input[Train_Sets.vers][:winNum, :, :Train_Sets.numOfSens], \
            output[Train_Sets.within][Train_Sets.inter][Train_Sets.vers][:winNum], \
                random_state=1, test_size=0.20, shuffle=Train_Sets.shuffle)

    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, \
            random_state=2, test_size=0.25, shuffle=Train_Sets.shuffle)

    print(f"{'X_train shape:':<20}{X_train.shape}")
    print(f"{'y_train shape:':<20}{y_train.shape}")
    print(f"{'X_val shape:':<20}{X_val.shape}")
    print(f"{'y_val shape:':<20}{y_val.shape}")
    print(f"{'X_test shape:':<20}{X_test.shape}")
    print(f"{'y_test shape:':<20}{y_test.shape}")
    print(f"{'Common data type:':<20}{type(y_val[0])}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_model(Train_Sets, X_train, X_val, y_train, y_val, summarize=False):
    if Paths.model_path.is_dir():
        model = load_model(Paths.model_path)
        df = pd.read_csv(Paths.history_path)
        train_time = 0
        if summarize: model.summary()
    else:
        model = Sequential(name=Paths.id)
        if Train_Sets.dense:
            model.add(Input((X_val.shape[1], X_val.shape[2]), name='input'))
            model.add(Flatten(name='flatten'))
            model.add(Dense(2 * Const_Sets.layerSize, activation='relu', name='dense1'))
            model.add(Dense(3 * Const_Sets.layerSize, activation='relu', name='dense2'))
        else:
            model.add(LSTM(Const_Sets.layerSize, \
                batch_input_shape=(Train_Sets.batchSize, X_val.shape[1], X_val.shape[2]), \
                return_sequences=True, stateful=Train_Sets.stateful, name='lstm1'))

            model.add(LSTM(Const_Sets.layerSize, name='lstm2'))
        
        model.add(Dense(Const_Sets.layerSize, activation='relu', name='dense3'))
        model.add(Dense(Const_Sets.layerSize, activation='relu', name='dense4'))
        model.add(Dense(Const_Sets.layerSize, activation='relu', name='dense5'))
        model.add(Dropout(Const_Sets.dropoutProb, name=f'drop{Const_Sets.dropoutProb}'))
        model.add(Dense(Const_Sets.layerSize, name='dense6'))
        model.add(Dense(1, name='out'))
        model.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError', 'MeanAbsoluteError'])
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, \
            patience=Const_Sets.patienceNum, restore_best_weights=True)

        if summarize: model.summary()
        start_time = timer.time()
        model.fit(X_train, y_train, batch_size=Train_Sets.batchSize, epochs=Const_Sets.maxEpochs, \
            validation_data=(X_val, y_val), shuffle=Train_Sets.shuffle, verbose=1, \
                callbacks=[early_stop, reset_states_callback])
        
        train_time = timer.time() - start_time
        model.save(Paths.model_path)
        if Paths.history_path.is_file(): Paths.history_path.unlink()
        df = pd.DataFrame.from_dict(model.history.history)
        df.to_csv(Paths.history_path, index=False)
    
    return model, df, train_time


def train_info(df, train_time):
    if train_time==0: print('Model loaded! Not recently trained!')
    print(f"{'Total training time:':<30}{(train_time / 60):3.1f} min")
    print(f"{'Num of epochs:':<30}{df.shape[0]}")
    print(f"{'Best val RMSE epoch:':<30}{df['val_root_mean_squared_error'].argmin() + 1}")
    print(f"{'Training time per epoch:':<30}{(train_time / df.shape[0]):3.2f} sec")


def plot_learn_curve(df):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)
    axes[0].plot(np.arange(1, df.shape[0] + 1), df['root_mean_squared_error'])
    axes[0].plot(np.arange(1, df.shape[0] + 1), df['val_root_mean_squared_error'])
    axes[0].scatter(df['root_mean_squared_error'].argmin() + 1, df['root_mean_squared_error'].min())
    axes[0].scatter(df['val_root_mean_squared_error'].argmin() + 1, df['val_root_mean_squared_error'].min())
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Learning Curves')
    axes[0].legend(['Train', 'Val', \
        f"Min Train {df['root_mean_squared_error'].min():3.2f} " + \
            f"@ Epoch #{df['root_mean_squared_error'].argmin() + 1}", \
        f"Min Val {df['val_root_mean_squared_error'].min():3.2f} " + \
            f"@ Epoch #{df['val_root_mean_squared_error'].argmin() + 1}"])

    axes[0].grid(True)
    axes[1].plot(np.arange(1, df.shape[0] + 1), df['mean_absolute_error'])
    axes[1].plot(np.arange(1, df.shape[0] + 1), df['val_mean_absolute_error'])
    axes[1].scatter(df['mean_absolute_error'].argmin() + 1, df['mean_absolute_error'].min())
    axes[1].scatter(df['val_mean_absolute_error'].argmin() + 1, df['val_mean_absolute_error'].min())
    axes[1].set_xlabel('Number of Epochs')
    axes[1].set_ylabel('MAE')
    axes[1].legend(['Train', 'Val', \
        f"Min Train {df['mean_absolute_error'].min():3.2f} " + \
            f"@ Epoch #{df['mean_absolute_error'].argmin() + 1}", \
        f"Min Val {df['val_mean_absolute_error'].min():3.2f} " + \
            f"@ Epoch #{df['val_mean_absolute_error'].argmin() + 1}"])

    axes[1].grid(True)
    if Paths.learn_curve_path.is_file(): Paths.learn_curve_path.unlink()
    plt.savefig(Paths.learn_curve_path, bbox_inches='tight')
    plt.show()
    plt.close()


def get_predictions(Train_Sets, model, X_train, X_val, X_test):
    y_train_pred = np.squeeze(model.predict(X_train, batch_size=Train_Sets.batchSize, verbose=0, \
        callbacks=[reset_states_callback]))
    
    y_val_pred = np.squeeze(model.predict(X_val, batch_size=Train_Sets.batchSize, verbose=0, \
        callbacks=[reset_states_callback]))
    
    y_test_pred = np.squeeze(model.predict(X_test, batch_size=Train_Sets.batchSize, verbose=0, \
        callbacks=[reset_states_callback]))
    
    print(f"{'y_train_pred shape:':<20}{y_train_pred.shape}")
    print(f"{'y_val_pred shape:':<20}{y_val_pred.shape}")
    print(f"{'y_test_pred shape:':<20}{y_test_pred.shape}")
    print(f"{'Common data type:':<20}{type(y_test_pred[0])}")
    return y_train_pred, y_val_pred, y_test_pred


def evaluate_model(y_train, y_val, y_test, y_train_pred, y_val_pred, y_test_pred):
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_val = np.sqrt(mse_val)
    rmse_test = np.sqrt(mse_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"{'mse_train:':<15}{mse_train:3.2f}")
    print(f"{'mse_val:':<15}{mse_val:3.2f}")
    print(f"{'mse_test:':<15}{mse_test:3.2f}")
    print(f"{'rmse_train:':<15}{rmse_train:3.2f}")
    print(f"{'rmse_val:':<15}{rmse_val:3.2f}")
    print(f"{'rmse_test:':<15}{rmse_test:3.2f}")
    print(f"{'mae_train:':<15}{mae_train:3.2f}")
    print(f"{'mae_val:':<15}{mae_val:3.2f}")
    print(f"{'mae_test:':<15}{mae_test:3.2f}")
    plt.style.use('dark_background')
    def_size = plt.rcParams.get('figure.figsize')
    fig, axes = plt.subplots(2, 1, \
        figsize=(def_size[0], def_size[1]), gridspec_kw={'height_ratios': [2, 1]})
    
    fig.subplots_adjust(hspace=.5)
    sns.regplot(ax=axes[0], x=y_test, y=y_test_pred, \
        scatter_kws={'color': plt.rcParams['axes.prop_cycle'].by_key()['color'][1]})
    
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Estimation')
    axes[0].set_title('Test Data')
    axes[0].text(np.min(y_test), np.max(y_test_pred), \
        f'MSE:   {mse_test:3.2f}\nRMSE: {rmse_test:3.2f}\nMAE:   {mae_test:3.2f}', \
        ha='left', va='top')

    sns.histplot(ax=axes[1], data=y_test_pred - y_test)
    axes[1].set_xlabel('Prediction Error (Estimation - G.T.)')
    axes[1].set_ylabel("Num. of Win's")
    axes[1].set_title('Error Distribution')
    if Paths.test_path.is_file(): Paths.test_path.unlink()
    plt.savefig(Paths.test_path, bbox_inches='tight')
    plt.show()
    plt.close()