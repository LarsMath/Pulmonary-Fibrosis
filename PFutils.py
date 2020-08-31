import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm
import seaborn as sns
import wandb
from wandb.keras import WandbCallback
import keras
from keras.models import Sequential


def get_test_data(files):
    test = pd.read_csv(files)
    submission = pd.DataFrame( columns=['Patient_Week', 'FVC', 'Confidence'])

    count = 0
    #Submission File
    for patient in test["Patient"]:
        for i in range(-12,133+1):
            count += 1
            submission.loc[count, "Patient_Week"] = patient + "_" + str(i)
            submission.loc[count, "FVC"] = 0
            submission.loc[count, "Confidence"] = 0

    test_data = pd.DataFrame(columns = ["Weeks", "FVC", "Percent", "Age", "Sex", 
                                        "Weekdiff_target", 'SmokingStatus'])

    count = 0
    for patient in test["Patient"]:
        for i in range(-12,133+1):
            count+=1
            test_data.loc[count, "Patient_Week"] = patient + "_" + str(i)
            test_data.loc[count, "Weekdiff_target"] = i - test[test["Patient"] == patient]["Weeks"].values[0] 
            test_data.loc[count, "Weeks"] = test[test["Patient"] == patient]["Weeks"].values[0]
            test_data.loc[count, "FVC"] = test[test["Patient"] == patient]["FVC"].values[0]
            test_data.loc[count, "Percent"] = test[test["Patient"] == patient]["Percent"].values[0]
            test_data.loc[count, "Sex"] = test[test["Patient"] == patient]["Sex"].values[0]
            test_data.loc[count, 'SmokingStatus'] = test[test["Patient"] == patient]['SmokingStatus'].values[0]
            test_data.loc[count, 'Age'] = test[test["Patient"] == patient]['Age'].values[0]

    test_data["Sex"] = (test_data['Sex']=="Male").astype(int)
    test_data = pd.concat([test_data,pd.get_dummies(test_data['SmokingStatus'])],axis = 1).reset_index(drop = True)
    test_data = test_data.drop(["SmokingStatus", "Patient_Week"],axis = 1)

    Check = ["Currently smokes", "Ex-smoker", "Never smoked"]

    for col in Check:
        if col not in test_data.columns:
            test_data[col] = 0

    test_data = test_data[["Weeks", "FVC", "Percent", "Age", "Sex", "Currently smokes",
               "Ex-smoker", "Never smoked", "Weekdiff_target"]]
    test_data = test_data.astype("Float32")
    
    return test_data, submission

def get_train_data(files, pseudo_test_patients):
    df = pd.read_csv(files)
    patients = []
    for patient in df.Patient.unique()[pseudo_test_patients:]:
        weekcombinations = []
        weeks = df.loc[df.Patient == patient]['Weeks']
        for week1 in weeks:
            dfnew = df.loc[(df.Patient == patient) & (df.Weeks != week1)]
            dfnew = dfnew.assign(Weekdiff_target = week1)
            dfnew = dfnew.assign(TargetFVC = df.loc[(df.Patient == patient)&(df.Weeks == week1)]['FVC'].values[0])
            weekcombinations.append(dfnew)
            patients.append(pd.concat(weekcombinations))

    train = pd.DataFrame(pd.concat(patients))    
    train["Sex"] = (train['Sex']=="Male").astype(int)

    train = pd.concat([train,pd.get_dummies(train['SmokingStatus'])],axis = 1).reset_index(drop = True)
    for i in range(len(train)):
        train.loc[i, "Weekdiff_target"] = train.loc[i, "Weekdiff_target"] - train.loc[i, "Weeks"]
    
    labels = pd.DataFrame(train[["TargetFVC","Weekdiff_target","FVC"]])
    labels = labels.astype("float32")
    
    train = train[["Weeks", "FVC", "Percent", "Age", "Sex", "Currently smokes",
                   "Ex-smoker", "Never smoked", "Weekdiff_target", "Patient"]]

    data = {"input_features": train[["Weeks", "FVC", "Percent", "Age", "Sex", 
                                     "Currently smokes", "Ex-smoker", "Never smoked", "Weekdiff_target"]],
            "FVC_Start_Weeks_from_start": train["Weekdiff_target"]}
    
    return train, data, labels

def build_model(config):
    
    size = config["NUMBER_FEATURES"]
    actfunc = config["ACTIVATION_FUNCTION"]
    predict_slope = config["PREDICT_SLOPE"]
    
    if(actfunc == 'swish'):
        actfunc = tf.keras.activations.swish

    inp = tf.keras.layers.Input(shape=(1,size), name = "input_features")
    
    inp2 = tf.keras.layers.Input(shape = (1,1), name = "FVC_Start_Weeks_from_start")
    
    inputs = [inp]
    outputs = []
    
    x = tf.keras.layers.Dense(100, activation=actfunc)(inp)
    x = tf.keras.layers.Dense(75, activation=actfunc)(x)
    x = tf.keras.layers.Dense(50, activation=actfunc)(x)
    x = tf.keras.layers.Dense(25, activation=actfunc)(x)
    x = tf.keras.layers.Dense(15, activation=actfunc)(x)
    x = tf.keras.layers.Dense(10, activation=actfunc)(x)

    
    # output : [slope/FVC_pred, s/sigma, FVC_start, weeks_from_start]
    outputs += [tf.keras.layers.Dense(2, name = "Output_a_s")(x)]

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    
    def Laplace_log_likelihood(y_true, y_pred):
        # y_pred : [slope/FVC_pred, s/sigma, FVC_start, weeks_from_start]
        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
        FVC_true = y_true[:,0]
        
        if(predict_slope):
            slope = y_pred[:,0]
            s = y_pred[:,1]

            weeks_from_start = y_true[:,1]
            FVC_start = y_true[:,2]
            
            sigma = s * weeks_from_start
            # Kan probleem worden by ReLu omdat slope negatief wordt door minimalisering Loss
            FVC_pred = weeks_from_start * slope + FVC_start
        else:
            FVC_pred = tf.abs(y_pred[:,0])
            sigma = tf.abs(y_pred[:,1])
        
        ## ** Hier kan een fout komen doordat de afgeleide moeilijker te berekenen is
        sigma_clip = tf.maximum(tf.abs(sigma), 70)
        delta = tf.abs(FVC_true - FVC_pred)
        ## **
        
        sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
        loss = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip * sq2)
        return K.mean(loss)
    
    def Laplace_metric(y_true, y_pred):
        # y_pred : [slope/FVC_pred, s/sigma, FVC_start, weeks_from_start]
        y_true = tf.dtypes.cast(y_true, tf.float32)
        y_pred = tf.dtypes.cast(y_pred, tf.float32)
        FVC_true = y_true[:,0]
        
        if(predict_slope):
            slope = y_pred[:,0]
            s = y_pred[:,1]

            weeks_from_start = y_true[:,1]
            FVC_start = y_true[:,2]
            
            sigma = s * weeks_from_start
            # Kan probleem worden by ReLu omdat slope negatief wordt door minimalisering Loss
            FVC_pred = weeks_from_start * slope + FVC_start
        else:
            FVC_pred = tf.abs(y_pred[:,0])
            sigma = tf.abs(y_pred[:,1])
        
        ## ** Hier kan een fout komen doordat de afgeleide moeilijker te berekenen is
        sigma_clip = tf.maximum(tf.abs(sigma), 70)
        delta = tf.abs(FVC_true - FVC_pred)
        delta = tf.minimum(delta, 1000)
        ## **
        
        sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
        loss = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip * sq2)
        return K.mean(loss)
    
    
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=Laplace_log_likelihood, metrics = [Laplace_metric])

    return model

def get_cosine_annealing_lr_callback(lr_max=1e-4, n_epochs= 10000, n_cycles= 10):
    epochs_per_cycle = np.floor(n_epochs / n_cycles)

    def lrfn(epoch):
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return lr_max / 2 * (np.cos(cos_inner) + 1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    
    return lr_callback

def get_fold_indices(folds, train):
    
    n_patients = len(np.unique(train.Patient))
    patients_per_fold = n_patients//folds
    remainder = n_patients % folds
    fold_pos = [0]

    count = 0
    for i in np.unique(train["Patient"]):
        count += 1
        if count == patients_per_fold:
            count = 0
            fold_pos.append(np.max(np.where(train["Patient"] == i))+1)
            
    return fold_pos