import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import keras

SQRT2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))

def get_test_data(files, input_normalization):
    test = pd.read_csv(files)
    submission = pd.DataFrame(columns=['Patient_Week', 'FVC', 'Confidence'])

    if input_normalization:
        test["Age"] = test["Age"]/100
        test["Percent"] = test["Percent"]/100
        test["Weeks"] = test["Weeks"]/100
        test["FVC"] = test["FVC"]/5000 
    
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
            test_data.loc[count, "Weekdiff_target"] = i/100 - test[test["Patient"] == patient]["Weeks"].values[0] 
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

def get_train_data(files, pseudo_test_patients, input_normalization, train_on_backward_weeks):
    df = pd.read_csv(files)
    patients = []
    for patient in df.Patient.unique()[pseudo_test_patients:]:
        weekcombinations = []
        weeks = df.loc[df.Patient == patient]['Weeks']
        for week1 in weeks:
            dfnew = df.loc[(df.Patient == patient) & (df.Weeks != week1) & ((df.Weeks<week1) | (train_on_backward_weeks))]
            dfnew = dfnew.assign(Weekdiff_target = week1)
            dfnew = dfnew.assign(TargetFVC = df.loc[(df.Patient == patient)&(df.Weeks == week1)]['FVC'].values[0])
            weekcombinations.append(dfnew)
        patients.append(pd.concat(weekcombinations))

    train = pd.DataFrame(pd.concat(patients))    
    train["Sex"] = (train['Sex']=="Male").astype(int)
    train = pd.concat([train,pd.get_dummies(train['SmokingStatus'])],axis = 1).reset_index(drop = True)
    train = train.drop(columns=["SmokingStatus"])
        
    for i in range(len(train)):
        train.loc[i, "Weekdiff_target"] = train.loc[i, "Weekdiff_target"] - train.loc[i, "Weeks"]
    
    non_normalized_FVC = train["FVC"]
    non_normalized_Weekdiff = train["Weekdiff_target"]
    non_normalized_FVC = non_normalized_FVC.astype("float32")
    non_normalized_Weekdiff = non_normalized_Weekdiff.astype("float32")
    
    labels = pd.DataFrame(train[["TargetFVC","Weekdiff_target", "FVC"]])
    labels = labels.astype("float32")
        
    if input_normalization:
        train["Age"] = train["Age"]/100
        train["Percent"] = train["Percent"]/100
        train["Weeks"] = train["Weeks"]/100
        train["Weekdiff_target"] = train["Weekdiff_target"]/100
        train["FVC"] = train["FVC"]/5000 
    
    train = train[["Weeks", "FVC", "Percent", "Age", "Sex", "Currently smokes",
                   "Ex-smoker", "Never smoked", "Weekdiff_target", "Patient"]]

    data = {"input_features": train[["Weeks", "FVC", "Percent", "Age", "Sex", 
                                     "Currently smokes", "Ex-smoker", "Never smoked", "Weekdiff_target"]],
            "slope_FVC": non_normalized_FVC, "slope_Weekdiff": non_normalized_Weekdiff}
    
    return train, data, labels

def get_pseudo_test_data(files, pseudo_test_patients, input_normalization, random_seed = 42):
    np.random.seed(random_seed)
    
    df = pd.read_csv(files)
    
    if input_normalization:
        df["Age"] = df["Age"]/100
        df["Percent"] = df["Percent"]/100
        df["Weeks"] = df["Weeks"]/100
        df["FVC"] = df["FVC"]/5000    
    
    patients = df.Patient.unique()[:pseudo_test_patients]

    test_data = pd.DataFrame(columns = ["Weeks", "FVC", "Percent", "Age", "Sex", 
                                        "Weekdiff_target", 'SmokingStatus'])
    test_check = pd.DataFrame(columns = ["TargetFVC","Weekdiff_target","FVC"])

    count = 0
    for patient in patients:
        init_choice = int((len(df[df.Patient == patient])-3)*np.random.rand())
        basecase = df[df.Patient == patient].iloc[init_choice]
        for testcase in df[df.Patient == patient].iloc[-3:].iterrows():
            count+=1
            test_data.loc[count, "Patient_Week"] = patient + "_" + str(testcase[1]["Weeks"])
            test_data.loc[count, "Weekdiff_target"] = testcase[1]["Weeks"] - basecase["Weeks"]
            test_data.loc[count, "Weeks"] = basecase["Weeks"]
            test_data.loc[count, "FVC"] = basecase["FVC"]
            test_data.loc[count, "Percent"] = basecase["Percent"]
            test_data.loc[count, "Sex"] = basecase["Sex"]
            test_data.loc[count, 'SmokingStatus'] = basecase['SmokingStatus']
            test_data.loc[count, 'Age'] = basecase['Age']
            test_check.loc[count, "TargetFVC"] = testcase[1]["FVC"]*5000
            test_check.loc[count, "Weekdiff_target"] = (testcase[1]["Weeks"] - basecase["Weeks"])*100
            test_check.loc[count, "FVC"] = basecase["FVC"]*5000

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
    test_check = test_check.astype("Float32")
    
    return test_data, test_check

def build_model(config):
    size = config["NUMBER_FEATURES"]
    actfunc = config["ACTIVATION_FUNCTION"]
    predict_slope = config["PREDICT_SLOPE"]
    drop_out_rate = config["DROP_OUT_RATE"]
    l2_regularization = config["L2_REGULARIZATION"]
    output_normalization = config["OUTPUT_NORMALIZATION"]
    hidden_layers = config["HIDDEN_LAYERS"]
    regularization_constant = config["REGULARIZATION_CONSTANT"]
    drop_out_layers = config["DROP_OUT_LAYERS"]
    modified_loss = config["MODIFIED_LOSS"]
    
    if actfunc == 'swish':
        actfunc = tf.keras.activations.swish

    inp = tf.keras.layers.Input(shape=(size), name = "input_features")
    inp2 = tf.keras.layers.Input(shape=(1), name = "slope_FVC")
    inp3 = tf.keras.layers.Input(shape=(1), name = "slope_Weekdiff")
    
    inputs = [inp,inp2,inp3]

    x = inp
    
    for j,n_neurons in enumerate(hidden_layers):
        if l2_regularization:
            x = tf.keras.layers.Dense(n_neurons, activation=actfunc,
                                      kernel_regularizer = tf.keras.regularizers.l2(regularization_constant))(x)
        else:
            x = tf.keras.layers.Dense(n_neurons, activation=actfunc)(x)
        if j in drop_out_layers:
            x = tf.keras.layers.Dropout(drop_out_rate)(x)
    
    FVC_output = tf.keras.layers.Dense(1, name = "FVC_output")(x)
    sigma_output = tf.keras.layers.Dense(1, name = "sigma_output")(x)
    
    if output_normalization:
        FVC_output = tf.math.scalar_mul(tf.constant(50,dtype = 'float32'), FVC_output)
        sigma_output = tf.math.scalar_mul(tf.constant(5,dtype = 'float32'), sigma_output)
        if not predict_slope:
            FVC_output = tf.math.scalar_mul(tf.constant(100,dtype = 'float32'), FVC_output)
            sigma_output = tf.math.scalar_mul(tf.constant(100,dtype = 'float32'), sigma_output)

    if predict_slope:
        FVC_output = tf.add(tf.keras.layers.multiply([FVC_output, inp2]),inp3)
        sigma_output = tf.keras.layers.multiply([sigma_output, inp2])
        
    outputs = tf.keras.layers.concatenate([FVC_output,sigma_output])

    model = tf.keras.Model(inputs = inputs, outputs = outputs)    
    opt = tf.keras.optimizers.Adam()
    if modified_loss:
        model.compile(optimizer=opt, loss=experimental_loss_function,
                      metrics = [Laplace_metric, sigma_cost, delta_over_sigma, absolute_delta_error])
    else:
        model.compile(optimizer=opt, loss=Laplace_log_likelihood,
                      metrics = [Laplace_metric, sigma_cost, delta_over_sigma, absolute_delta_error])
    
    return model

def get_cosine_annealing_lr_callback(config):
    n_epochs = config["EPOCHS"]
    lr_max = config["MAX_LEARNING_RATE"]
    n_cycles = config["COSINE_CYCLES"]
    
    epochs_per_cycle = np.floor(n_epochs / n_cycles)

    def lrfn(epoch):
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return lr_max / 2 * (np.cos(cos_inner) + 1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    
    return lr_callback

def get_exponential_decay_lr_callback(config):
    lr_max = config["MAX_LEARNING_RATE"]
    decay = config["EPOCHS_PER_OOM_DECAY"]

    def lrfn(epoch):
        return lr_max * np.power(0.1,(epoch/decay))

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    
    return lr_callback

def get_fold_indices(folds, train):
    
    fold_pos = [0]
    count = 0
    for i in np.unique(train["Patient"]):
        count += 1
        if count >= (len(fold_pos)*len(np.unique(train.Patient))/folds):
            fold_pos.append(np.max(np.where(train["Patient"] == i))+1)
            
    return fold_pos

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, config, validation = False, number_of_labels = 3,
                 batch_size = 128, shuffle = True):
        self.number_features = int(config["NUMBER_FEATURES"])
        self.validation = validation
        self.gauss_std = config["VALUE_GAUSSIAN_NOISE_ON_FVC"]
        self.list_IDs = list_IDs
        self.batch_size = config["BATCH_SIZE"]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_size = number_of_labels
        self.normalized = config["INPUT_NORMALIZATION"]
        self.correlated = config["GAUSSIAN_NOISE_CORRELATED"]
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.number_features))
        y = np.empty((self.batch_size, self.label_size), dtype=int)
        
        data = np.load("./train_data.npy", allow_pickle = True)
        lab = np.load("./train_labels.npy", allow_pickle = True)
        
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.asarray(data[ID], dtype = "float32")
            y[i,] = np.asarray(lab[ID], dtype = "float32")
        y = np.asarray(y,dtype = "float32")
        
        if not self.validation:
            gauss_X = np.random.normal(0, self.gauss_std, size = self.batch_size)

            if self.correlated:
                gauss_y = gauss_X
            else:
                gauss_y = np.random.normal(0, self.gauss_std, size = self.batch_size)
            if self.normalized:
                gauss_X = gauss_X/5000 

            X[:,2] += gauss_X.astype("float32")*X[:,2]/X[:,1]
            X[:,1] += gauss_X.astype("float32")
            y[:,2] += gauss_X.astype("float32")
            y[:,0] += gauss_y.astype("float32")
        
        return X, y

def absolute_delta_error(y_true, y_pred):

    FVC_true = y_true[:,0]
    FVC_pred = tf.abs(y_pred[:,0])

    return K.mean(tf.abs(FVC_true - FVC_pred))

def sigma_cost(y_true, y_pred):

    sigma_clip = tf.maximum(tf.abs(y_pred[:,1]), 70)

    loss = tf.math.log(tf.maximum(tf.abs(y_pred[:,1]), 70) * SQRT2)

    return K.mean(loss)

def delta_over_sigma(y_true, y_pred):

    FVC_true = y_true[:,0]
    FVC_pred = tf.abs(y_pred[:,0])
    sigma = tf.abs(y_pred[:,1])

    sigma_clip = tf.maximum(tf.abs(sigma), 70)
    delta = tf.abs(FVC_true - FVC_pred)
    delta = tf.minimum(delta, 1000)

    loss = (delta / sigma_clip)*SQRT2

    return K.mean(loss)

def Laplace_metric(y_true, y_pred):

    FVC_true = y_true[:,0]
    FVC_pred = tf.abs(y_pred[:,0])

    sigma_clip = tf.maximum(tf.abs(y_pred[:,1]), 70)
    delta = tf.abs(FVC_true - FVC_pred)
    delta = tf.minimum(delta, 1000)

    loss = (delta / sigma_clip)*SQRT2 + tf.math.log(sigma_clip * SQRT2)
    return K.mean(loss)

def Laplace_log_likelihood(y_true, y_pred):

    FVC_true = y_true[:,0]
    FVC_pred = tf.abs(y_pred[:,0])

    ## ** Hier kan een fout komen doordat de afgeleide moeilijker te berekenen is
    sigma = tf.maximum(tf.abs(y_pred[:,1]), 70)
    delta = tf.abs(FVC_true - FVC_pred)
    ## **

    loss = (delta / sigma)*SQRT2 + tf.math.log(sigma * SQRT2)
    return K.mean(loss)

def experimental_loss_function(y_true, y_pred):

    FVC_true = y_true[:,0]
    FVC_pred = tf.abs(y_pred[:,0])

    ## ** Hier kan een fout komen doordat de afgeleide moeilijker te berekenen is
    sigma = tf.maximum(tf.abs(y_pred[:,1]), 70)
    delta = tf.abs(FVC_true - FVC_pred)
    ## **

    loss = (delta / 70)*SQRT2 + (delta / sigma)*SQRT2 + tf.math.log(sigma * SQRT2)
    return K.mean(loss)

