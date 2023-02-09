import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from data import load_data, print_summary
import time
import json

class PoseNet():
    def __init__(self, conv_activation, dropout_rate,
                    l1_rate, l2_rate, learning_rate, decay_rate, dense_width, loss_func,
                    batch_bool = True, N_convs=4, N_filters=512):
        self.model = None
        self.history = None

        # Model Hyperparameters
        self.conv_activation = conv_activation
        self.dropout_rate = dropout_rate
        self.l1_rate = l1_rate
        self.l2_rate = l2_rate
        self.dense_width = dense_width
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.loss_func = loss_func
        self.batch_bool = batch_bool
        self.N_convs = N_convs
        self.N_filters = N_filters

        # Get the time to be used for saving the model
        stamp = str(time.ctime())
        stamp=stamp.replace(' ', '_')
        self.stamp=stamp.replace(':', '-')


    def create_network(self, input_height, input_width, num_outputs):
        '''
        Create the CNN model as a sequential model in Keras.

        Parameters
        ----------
        input_height : int
            The height in pixels of the input images.
        input_width : int
            The width in pixels of the input images.
        num_outputs : int
            The length of the target vectors.
        '''
        self.model = Sequential()

        # 1st convolution
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                        input_shape=(input_height, input_width, 1)))
        if self.batch_bool:
            self.model.add(BatchNormalization())
        self.model.add(Activation(self.conv_activation))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # 2nd convolution
        for i in range(self.N_convs):
            self.model.add(Conv2D(filters=self.N_filters, kernel_size=(3,3), padding='same'))
            if self.batch_bool:
                self.model.add(BatchNormalization())
            self.model.add(Activation(self.conv_activation))
            self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Flatten the convolved images
        self.model.add(Flatten())

        #model.add(Dense(256, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Dense(self.dense_width, activation='relu', 
                        kernel_regularizer=regularizers.L1L2(l1=self.l1_rate, l2=self.l2_rate)))

        # Output Layer
        self.model.add(Dropout(rate=self.dropout_rate))
        self.model.add(Dense(num_outputs, activation='linear', 
                    kernel_regularizer=regularizers.L1L2(l1=self.l1_rate, l2=self.l2_rate)))

        self.model.compile(loss=self.loss_func, 
                        optimizer=Adam(learning_rate=self.learning_rate, decay = self.decay_rate), 
                        metrics=['mae', 'mse'])
    

    def fit(self, x_train, y_train, epochs, batch_size, 
            x_val=None, y_val=None):
        '''
        Fit (train) the NN with early stopping if a validation set is provided.

        Parameters
        ----------
        x_train : np.array
            The training set's images.
        y_train : np.array
            The training set's target images.
        epochs : int
            The maximum number of epochs to train for.
        batch_size : int
            The batch size to train the network with.
        x_val : np.array / None
            The validation set's images (if provided).
        y_val : np.array / None
            The validations set's targets (if provided).
        '''
        # Save details on the training data
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.epochs = epochs

        # Train the network
        if (isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray)) or (isinstance(x_val, list) and isinstance(y_val, np.ndarray)):
            callback = EarlyStopping(monitor='val_loss', patience=10)
            self.history = self.model.fit(x_train, y_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        callbacks = [callback],
                                        validation_data = (x_val,y_val))

        else:
            callback = EarlyStopping(monitor='loss', patience=10)
            self.history = self.model.fit(x_train, y_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        callbacks = [callback])



    def evaluate(self, x_test, y_test):
        '''
        Evaluate the Model on the test set.
        '''
        # Save details on the testing set
        self.x_test = x_test
        self.y_test = y_test

        # Evaluate the network
        self.loss, self.mae, self.mse = self.model.evaluate(x_test, y_test)
        print('Evaluation MAE: ', self.mae)


    def summary(self):
        '''
        Print the summary of the NN
        '''
        self.model.summary()


    def plot_learning_curves(self, finger_name, img_type):
        '''
        Plot the loss and the validation loss
        '''
        plt.plot(self.history.history['loss'], label='Loss')
        #plt.plot(self.history.history['val_loss'], label= 'Val Loss')
        plt.savefig('saved_nets/'+finger_name+'_'+img_type+'_'+self.stamp+'/learning-curve.pdf')
        plt.legend(), plt.title('Loss Curve'), plt.show()


    def return_history(self):
        return self.model.history, self.model


    def save_network(self, finger_name, img_type):
        '''
        Save the network and a JSON of parameters.
        '''
        param_dict = {
            'finger' : finger_name,
            'training examples': self.x_train.shape[0],
            'test examples': self.x_test.shape[0],
            'epochs': self.epochs,
            'conv activation': self.conv_activation,
            'dropout rate': self.dropout_rate,
            'l1 rate': self.l1_rate,
            'l2 rate': self.l2_rate,
            'Dense width': self.dense_width,            
            'learning rate': self.learning_rate,
            'decay rate': self.decay_rate,
            'loss function': self.loss_func,
        }

        if self.loss:
            param_dict['MAE'] = self.mae
            param_dict['loss'] = self.loss
            param_dict['MSE'] = self.mse

        self.model.save('saved_nets/'+finger_name+'_'+img_type+'_'+self.stamp+'/CNN.h5')
        with open('saved_nets/'+finger_name+'_'+img_type+'_'+self.stamp+'/params.json', 'w') as fp:
            json.dump(param_dict, fp)
            fp.close()


    def load_net(self, path):
        '''
        Load a pre-trained model from a .h5 file.
        
        Parameters
        ----------
        path : str
            The path to the folder containing the saved model.
        '''
        print('[INFO] Loading Model')
        self.model = load_model(path+'/CNN.h5')

    def predict(self, input):
        '''
        Generate predictions to a set of inputs.

        Parameters
        ----------
        input : np.array
            Array containing the set of tactile images to generate predictions for.
        '''
        angles = self.model.predict(input)
        return angles






def main():

    batch_size = 32 # from paper
    finger_name = 'Middle'
    img_type = 't2'

    print('Loading Data...')
    df, t1, t2, t3, blob_locs = load_data(finger_name)
    print(finger_name+':')
    print_summary(df)

    if img_type == 't1':
        images = t1[1:]/255 # remove default image and normalise
    elif img_type == 't2':
        images = t2[1:]
    elif img_type == 't3':
        images = t3[1:]

    X_train, X_test, y_train, y_test = train_test_split(images, df, test_size=0.2, random_state=42)

    y_train = np.array(y_train['fz']).reshape(-1, 1)
    y_test = np.array(y_test['fz']).reshape(-1, 1)
    # Normalise the labels
    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)


    CNN = PoseNet(  conv_activation = 'elu',
                    dropout_rate = 0.001,
                    l1_rate = 0.0001,
                    l2_rate = 0.01,
                    learning_rate = 0.00001,
                    decay_rate = 0.000001,
                    dense_width = 16,
                    loss_func = 'mse',
                    batch_bool = False,
                    N_convs = 4,
                    N_filters = 512
                     )

    CNN.create_network(240, 100, 1) # create the NN
    CNN.summary()
    CNN.fit(X_train, y_train, epochs=150, batch_size=8, x_val=None, y_val=None) # train the NN
    CNN.evaluate(X_test, y_test) # evaluate the NN
    CNN.save_network(finger_name, img_type)
    CNN.plot_learning_curves()


if __name__ =='__main__':
    main()