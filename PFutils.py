import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import keras
import pathlib
import pydicom
import cv2
import pathlib
from os import listdir

from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure


def get_test_data(files, input_normalization):
    test = pd.read_csv(files)
    submission = pd.DataFrame(columns=['Patient_Week', 'FVC', 'Confidence'])

    if input_normalization:
        test["Age"] = (test["Age"]-50)/50
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

def get_train_data(files, pseudo_test_patients, train_on_backward_weeks, apply_lungmask, dim):
    df = pd.read_csv(files)
    
    train = pd.DataFrame()
    images = []
    index = 0
    for patient in df.Patient.unique()[pseudo_test_patients:]:
        if patient == "ID00011637202177653955184":
            continue
        weeks = df.loc[df.Patient == patient]['Weeks']
        for weektarget in weeks:
            dftemp = df.loc[(df.Patient == patient) & (df.Weeks != weektarget) & ((df.Weeks<weektarget) | (train_on_backward_weeks))]
            dftemp = dftemp.assign(WeekTarget = weektarget)
            dftemp = dftemp.assign(TargetFVC = df.loc[(df.Patient == patient)&(df.Weeks == weektarget)]['FVC'].values[0])
            dftemp = dftemp.assign(PatientIndex = index)
            train = train.append(dftemp, ignore_index = True)
        index += 1        
        
        img_nmbrs = [np.int(file.split(".")[0]) for file in listdir("../input/osic-pulmonary-fibrosis-progression/train/" + patient)]
        img_nmbrs = np.sort(img_nmbrs)
        middle = len(img_nmbrs)//2
        img_nmbrs = img_nmbrs[middle-1:middle+2]
        center_images = ([pydicom.dcmread("../input/osic-pulmonary-fibrosis-progression/train/" + patient + "/" 
                                    + str(index) + ".dcm").pixel_array for index in img_nmbrs])
        if apply_lungmask:
             center_images = [make_lungmask(image) for image in center_images]
        center_images = np.array([cv2.resize(image, dsize = (dim,dim)) for image in center_images])
        center_images = np.moveaxis(center_images, 0, -1)
        images.append(center_images)
        
    train = train.rename(columns={"Weeks":"WeekInit"})
    train["Sex"] = (train['Sex']=="Male").astype(int)
    train = pd.concat([train,pd.get_dummies(train['SmokingStatus'])],axis = 1).reset_index(drop = True)
    train["WeekDiff"] = train["WeekTarget"] - train["WeekInit"]
        
    images = np.array(images)
    
    labels = pd.DataFrame(train[["TargetFVC"]])
    labels = labels.astype("float32")
    
    train = train[["WeekInit", "WeekTarget", "WeekDiff", "FVC", "Percent",
                   "Age", "Sex", "Currently smokes", "Ex-smoker", "Never smoked", "PatientIndex"]]

    return train, images, labels

def get_pseudo_test_data(files, pseudo_test_patients, input_normalization, random_seed = 42):
    np.random.seed(random_seed)
    
    df = pd.read_csv(files)
    
    if input_normalization:
        df["Age"] = (df["Age"]-50)/50
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
    for i in np.unique(train["PatientIndex"]):
        count += 1
        if count >= (len(fold_pos)*len(np.unique(train.PatientIndex))/folds):
            fold_pos.append(np.max(np.where(train["PatientIndex"] == i))+1)
            
    return fold_pos

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, config, validation = False, number_of_labels = 1,
                 batch_size = 128, shuffle = True):
        self.number_features = int(config["NUMBER_FEATURES"])
        self.validation = validation
        self.noise_SDs = config["NOISE_SDS"]
        self.list_IDs = list_IDs
        self.batch_size = config["BATCH_SIZE"]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.label_size = number_of_labels
        self.input_normalization = config["INPUT_NORMALIZATION"]
        self.negative_normalization = config["NEGATIVE_NORMALIZATION"]
        self.correlated = config["GAUSSIAN_NOISE_FVC_CORRELATED"]
        self.percentcorrelated = config["ADD_NOISE_FVC_TO_PERCENT"]
        self.dim = config["DIM"]
        self.use_images = config["USE_IMAGES"]
        self.predict_slope = config["PREDICT_SLOPE"]
        if validation and not self.use_images:
            self.batch_size = len(self.list_IDs)
    
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
        X = np.zeros((self.batch_size, self.number_features))
        y = np.zeros((self.batch_size, self.label_size))
        
        inpdata = np.load("./train_data.npy", allow_pickle = True)
        lab = np.load("./train_labels.npy", allow_pickle = True)
        
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.asarray(inpdata[ID][0:-1], dtype = "float32")
            y[i,] = np.asarray(lab[ID], dtype = "float32")
        
        if self.predict_slope:
            WeekDiff = X[:,2]
            InitFVC = X[:,3]
        
        if not self.validation:
            gaussian_noise = np.zeros((self.batch_size, self.number_features))
            for i, sigma in enumerate(self.noise_SDs):
                gaussian_noise[:,i] = np.random.normal(0, sigma, size = self.batch_size)
                
            X += gaussian_noise.astype("float32")
            
            # Index 3 represents FVC, index 4 represents Percent
            if self.percentcorrelated:
                X[:,4] += gaussian_noise[:,3].astype("float32")*X[:,4]/X[:,3]
            
            if self.correlated:
                gauss_y = gaussian_noise[:,3].copy()
            else:
                gauss_y = np.random.normal(0, self.noise_SDs[3], size = self.batch_size)
                
            y[:,0] += gauss_y
            
        if self.input_normalization:
            X[:,0:2] = X[:,0:2]/100 #Weeks
            X[:,3] = X[:,3]/5000    #FVC
            X[:,4] = X[:,4]/100     #Percent
            X[:,5] = (X[:,5]-50)/50 #Age
                
        if self.negative_normalization:
            X = X * 2 - 1
            
        data = {"meta_data": X}
        
        if self.use_images:
            imgs = np.load("./train_images.npy", allow_pickle = True)
            images = np.zeros((self.batch_size, self.dim, self.dim, 3))
            for i, ID in enumerate(list_IDs_temp):
                images[i,] = np.asarray(imgs[int(inpdata[i,-1])], dtype = "float32")
            data["image"] = images
        
        if self.predict_slope:
            data["WeekDiff"] = WeekDiff
            data["InitFVC"] = InitFVC
        
        return data, y
    
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    # Find the average pixel value near the lungs
        # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0


    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    return mask*img    
    
SQRT2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))

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
    loss_modification = config["LOSS_MODIFICATION"]
    optimal_sigma_loss = config["OPTIMAL_SIGMA_LOSS"]
    batch_normalization = config["BATCH_NORMALIZATION"]
    pre_batch_normalization = config["PRE_BATCH_NORMALIZATION"]
    batch_renormalization = config["BATCH_RENORMALIZATION"]
    dim = config["DIM"]
    img_features = config["IMG_FEATURES"]
    use_imgs = config["USE_IMAGES"]
    effnet = config["EFFNET"]
    
    if actfunc == 'relu':
        actfunc = tf.keras.activations.relu
    if actfunc == 'swish':
        actfunc = tf.keras.activations.swish
    if actfunc == 'leakyrelu':
        actfunc = lambda x: tf.keras.activations.relu(x, alpha=0.3)

    metadata = tf.keras.layers.Input(shape=(size), name = "meta_data")
    inputs = [metadata]
    
    x = metadata
    
    if use_imgs:
        image = tf.keras.layers.Input(shape=(dim,dim,3), name = "image")
        inputs.append(image)
        base = EFNS[effnet](input_shape=(dim,dim,3), weights='imagenet', include_top=False, pooling = 'avg')
        y = base(image)
        y = tf.keras.layers.Dense(img_features)(y)
        y = actfunc(y)
        x = tf.keras.layers.concatenate([x,y])    
    
    if pre_batch_normalization:
        x = tf.keras.layers.BatchNormalization(renorm = batch_renormalization)(x)
    
    for j,n_neurons in enumerate(hidden_layers):
        if l2_regularization:
            x = tf.keras.layers.Dense(n_neurons, kernel_regularizer = tf.keras.regularizers.l2(regularization_constant))(x)
        else:
            x = tf.keras.layers.Dense(n_neurons)(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization(renorm = batch_renormalization)(x)
        x = actfunc(x)
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
        WeekDiff = tf.keras.layers.Input(shape = (1), name = "WeekDiff")
        InitFVC = tf.keras.layers.Input(shape = (1), name = "InitFVC")
        inputs.extend([WeekDiff,InitFVC])
        FVC_output = tf.add(tf.keras.layers.multiply([FVC_output, WeekDiff]),InitFVC)
        sigma_output = tf.keras.layers.multiply([sigma_output, WeekDiff])
        
    outputs = tf.keras.layers.concatenate([FVC_output,sigma_output])

    model = tf.keras.Model(inputs = inputs, outputs = outputs)    
    opt = tf.keras.optimizers.Adam()
    if optimal_sigma_loss:
        model.compile(optimizer=opt, loss=optimal_sigma_loss_function,
                      metrics = [Laplace_metric, sigma_cost, delta_over_sigma, absolute_delta_error])
    else:
        model.compile(optimizer=opt, loss=(lambda x,y:(Laplace_log_likelihood(x,y) + absolute_delta_error(x,y)*loss_modification*SQRT2/70)),
                      metrics = [Laplace_metric, sigma_cost, delta_over_sigma, absolute_delta_error])
    
    return model

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

def optimal_sigma_loss_function(y_true, y_pred):

    FVC_true = y_true[:,0]
    FVC_pred = tf.abs(y_pred[:,0])
    
    delta = tf.abs(FVC_true - FVC_pred)
    sigma = tf.maximum(SQRT2*delta, 70)

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