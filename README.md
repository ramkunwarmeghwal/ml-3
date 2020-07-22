1. Create container image that’s has Python3 and Keras or numpy  installed  using dockerfile 

2. When we launch this image, it should automatically starts train the model in the container.

3. Create a job chain of job1, job2, job3, job4 and job5 using build pipeline plugin in Jenkins 

4.  Job1 : Pull  the Github repo automatically when some developers push repo to Github.

5.  Job2 : By looking at the code or program file, Jenkins should automatically start the respective machine learning software installed interpreter install image container to deploy code  and start training( eg. If code uses CNN, then Jenkins should start the container that has already installed all the softwares required for the cnn processing).

6. Job3 : Train your model and predict accuracy or metrics.

7. Job4 : if metrics accuracy is less than 80%  , then tweak the machine learning model architecture.

8. Job5: Retrain the model or notify that the best model is being created

9. Create One extra job job6 for monitor : If container where app is running. fails due to any reason then this job should automatically start the container again from where the last trained model left


from keras.datasets import mnist

dataset = mnist.load_data('mymnist.db')

train , test = dataset

X_train , y_train = train

X_test , y_test = test


import matplotlib.pyplot as plt

X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)

X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')

from keras.utils.np_utils import to_categorical

y_train_cat = to_categorical(y_train)

y_train_cat

from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(units=512, input_dim=28*28, activation='relu'))

model.summary()

def layers():
 
 import random
 x=0
 y=255
    
 model.add(Dense(random.randint(x, y), activation='relu'))

 model.add(Dense(units=random.randint(x, y), activation='relu'))

 model.add(Dense(units=random.randint(x, y), activation='relu'))



layers()

model.summary()

model.add(Dense(units=10, activation='softmax'))

model.summary()

from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )

h = model.fit(X_train, y_train_cat, epochs=1)


scores = model.evaluate(X_train, y_train_cat, verbose=1)
print('Test loss:', scores[0])
print('accuracy:', scores[1])


if scores[1] <= 0.80:
    layers()
else:
    pass
