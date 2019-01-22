#------ IMPORTING DATA SET ---------#

#importing libraries
import numpy as np
import pandas as pd

#importing dataset
X_train = pd.read_csv('/home/nikhila/Desktop/sentiment/aclImdb/train/data.csv').iloc[:,:].values.ravel()
X_test = pd.read_csv('/home/nikhila/Desktop/sentiment/aclImdb/test/test_data.csv').iloc[:,:].values.ravel()

#length  of dataset
train_len = len(X_train)
test_len = len(X_test)


#------ PRE PROCESSING CORPUS---------#

#creating appended dataset
X = np.concatenate((X_train,X_test))

#finding top words
from sklearn.feature_extraction.text import CountVectorizer
features = 5000
cv = CountVectorizer(max_features = features)
X = cv.fit_transform(X).toarray()

#splitting back dataset
X_train = X[:train_len,:]
X_test = X[train_len:,:]


#------ DEPENDENT VARIABLE ---------#

y_train = np.append(np.ones((12500,)),np.zeros((12500,)))
y_test = np.append(np.ones((12500,)),np.zeros((12500,)))


#------ GANS ---------#

#importing libraries
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from keras.models import Model,Sequential
from keras.layers import Dense,Dropout
from keras import initializers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
import keras

#initializing variables
input_size = 100
batch_size = 32
epoch_size = 100
sample_size = 5000

#generator network
def create_generator(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = input_size,output_dim=100,
                         kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    classifier.add(LeakyReLU(0.2))
    
    #internal layers
    classifier.add(Dense(output_dim =32))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dense(output_dim = 32))
    classifier.add(LeakyReLU(0.2))
    
    #output layer
    classifier.add(Dense(output_dim =len(X_train[0]), activation = 'tanh'))
    
    #compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' )
    return classifier

#discriminator network
def create_discriminator(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = len(X_train[0]) , output_dim = 10,
                         kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.5))
    
    #internal layers
    classifier.add(Dense(output_dim =20))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 20))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.2))
    
    #output layer
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' )
    return classifier  

#gan network  
def create_gan_network():
    
    discriminator.trainable = False
    gan_input = Input(shape=(input_size,))
    
    x = generator(gan_input)
    gan_output = discriminator(x)
    
    #model network gan    
    gan = Model(inputs=gan_input, outputs=gan_output)
    
    #compiling
    gan.compile(loss='binary_crossentropy',optimizer = keras.optimizers.Adam(lr=0.001,amsgrad=True))
    return gan

X_pos = X_train[:12500,:]
X_neg = X_train[12500:,:]

#positive training
def train_gan_pos():
    
   for epoch in range(1,epoch_size+1):
       j=0
       
       for i in range(batch_size,len(X_pos),batch_size):
            
            #discriminator training
            X_curr = X_pos[j:i,:]
            noise = np.random.normal(0, 1, size=[batch_size,input_size])
            generated_output = generator.predict(noise)
        
            X_dis = np.concatenate([X_curr, generated_output])
            y_dis = np.zeros(len(X_dis))
            y_dis[:batch_size]=0.9
            
            discriminator.trainable = True
            discriminator.train_on_batch(X_dis,y_dis)
            
            #generator training
            discriminator.trainable = False
            gan_input = np.random.normal(0, 1, size=[batch_size,input_size])
            gan_output = np.ones(batch_size)
            gan.train_on_batch(gan_input, gan_output)
            
            j=i      
            
#negative training
def train_gan_neg():
    
   for epoch in range(1,epoch_size+1):
       j=0
       
       for i in range(batch_size,len(X_neg),batch_size):
            
            #discriminator training
            X_curr = X_neg[j:i,:]
            noise = np.random.normal(0, 1, size=[batch_size,input_size])
            generated_output = generator.predict(noise)
        
            X_dis = np.concatenate([X_curr, generated_output])
            y_dis = np.zeros(len(X_dis))
            y_dis[:batch_size]=0.1
            
            discriminator.trainable = True
            discriminator.train_on_batch(X_dis,y_dis)
            
            #generator training
            discriminator.trainable = False
            gan_input = np.random.normal(0, 1, size=[batch_size,input_size])
            gan_output = np.zeros(batch_size)
            gan.train_on_batch(gan_input, gan_output)
            
            j=i    


#positive data generation
generator = create_generator()
discriminator = create_discriminator()
gan = create_gan_network()

train_gan_pos()

generated = generator.predict(np.random.normal(0, 1, size=[sample_size,input_size]))
X_train = np.concatenate((X_train,generated))
y_train = np.concatenate((y_train,np.ones(sample_size)))

#negative data generation
generator = create_generator()
discriminator = create_discriminator()
gan = create_gan_network()

train_gan_neg()

generated = generator.predict(np.random.normal(0, 1, size=[sample_size,input_size]))
X_train = np.concatenate((X_train,generated))
y_train = np.concatenate((y_train,np.zeros(sample_size)))



#------ CLASSIFICATION ---------#


#building function
def build_fn():

    d = Sequential()
    
    d.add(Dense(input_dim = len(X_train[0]), output_dim =100,init = 'uniform',activation = 'relu'))
    d.add(Dense(256,init = 'uniform',activation = 'relu'))
    d.add(Dense(32,init = 'uniform',activation = 'relu'))
    d.add(Dense(1,init = 'uniform',activation = 'sigmoid'))
    
    d.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

    return d

#model building 
def build_model():
    
    estimators = []
    
    #neural networks
    classifier = KerasClassifier(build_fn = build_fn,batch_size = 25,epochs=5)
    model2 = BaggingClassifier(base_estimator = classifier, n_estimators = 5)
    estimators.append(('bagging',model2))
    
    #random forest
    model1 = RandomForestClassifier(n_estimators = 300)
    estimators.append(('rf',model1))
    
    # create the ensemble model
    model = VotingClassifier(estimators,voting = 'soft')
    
    return model


#training
model = build_model()
model.fit(X_train,y_train)

#testing
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)