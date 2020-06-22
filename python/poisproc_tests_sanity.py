import pkgutil
from pathlib import Path
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from SHMModels.fitted_models import ContextModel
from keras import optimizers, Input, Model
from keras.models import Sequential
from keras.layers import (
    Dense,
    TimeDistributed,
    SimpleRNN,
    Dropout,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Conv1D,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
from genDat import *
from itertools import permutations 
import time
import matplotlib.pyplot as plt
##### USER INPUTS (Edit some of these to be CLI eventually)

# Path to germline sequence
germline_sequence = "data/sanity.fasta"
# Context model length and pos_mutating
context_model_length = 3
context_model_pos_mutating = 2
# Path to aid model
aid_context_model = "data/aid_logistic_3mer.csv"
# Num seqs and n_mutation rounds
n_seqs = 1
n_mutation_rounds = 1
# step size
step_size = 0.0001
# batch size num epochs
batch_size = 100
num_epochs = 1000
steps_per_epoch = 1

start_time = time.time()
seqdict = ['A','C','G','T']
# Let's get a list of all the 4-mers!
fourmers = []
for i in range(256):
      p4 = i // 64
      p3 = (i-p4*64) // 16
      p2 = (i-p4*64-p3*16) // 4
      p1 = (i-p4*64-p3*16-p2*4)
      fourmers.append(''.join([seqdict[p1],seqdict[p2], seqdict[p3],seqdict[p4]]))
fourmers = np.asarray(fourmers)
freq = np.zeros(256)
les_mean = np.zeros((256,4))
# Load sequence into memory
sequence = list(
    SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
)[0]
# Load aid model into memory
aid_model_string = pkgutil.get_data("SHMModels", aid_context_model)
aid_model = ContextModel(
    context_model_length, context_model_pos_mutating, aid_model_string
)

t_batch_labels, t_batch_data, t_letters = gen_batch_letters(sequence.seq, 300000)
t_letters = [''.join(t_letters[i]) for i in range(np.shape(t_letters)[0])]

test_batch_labels, test_batch_data = gen_batch(sequence.seq, 200)
for i in range(256):
      freq[i] = np.sum([t_letters[j]== fourmers[i] for  j in range(np.shape(t_letters)[0])])
      les_mean[i,:] = np.mean(t_batch_labels[[t_letters[j]== fourmers[i] for j in range(np.shape(t_letters)[0])],:],axis = 0)
# Get ones which actually appeared
obs_fourmer = fourmers[[freq[i]!= 0 for i in range(256)]]
obs_freq = freq[[freq[i]!= 0 for i in range(256)]]
obs_les_mean = les_mean[[freq[i]!= 0 for i in range(256)],:]
print(time.time()- start_time)
# Let's build our encoder. Seq is of length 308.
input_seq = Input(shape=(4, 4, 1))

# We add 2 convolutional layers.
x = Conv2D(16, (3, 6), activation="relu", padding="same")(input_seq)
x = MaxPooling2D((2, 1), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

# Now we decode back up
x = Conv2DTranspose(
    filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="relu"
)(x)
x = Flatten()(x)
# I think ReLU is fine here because the values are nonnegative?
decoded = Dense(units=4, activation="relu")(x)

# at this point the "decoded" representation is a 308 vector indicating our predicted # of lesions at each site.
autoencoder = Model(input_seq, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

print(autoencoder.summary(90))

# Create iterator for simulation
def genTraining(batch_size):
    while True:
        # Get training data for step
        les,mut = gen_batch(sequence.seq,batch_size)
        yield mut,les

# Train
history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(test_batch_data, test_batch_labels),
    verbose=2,
)

test_fourmers = fourmers[freq>20]
test_les_mean = les_mean[freq>20,:]
pred_les_mean = np.zeros(np.shape(test_les_mean))
for i in range(np.shape(test_fourmers)[0]):
      pred_les_mean[i,:] = autoencoder.predict(hot_encode_2d(test_fourmers[i]).reshape(1,4,4,1))
np.corrcoef(pred_les_mean[:,[1,3]].flatten(),test_les_mean[:,[1,3]].flatten())
fig, ax = plt.subplots()
ax.scatter(pred_les_mean[:,[1,3]].flatten(),test_les_mean[:,[1,3]].flatten(), s=25)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
#temp = np.mean(t_batch_labels)
#means = np.repeat(temp, 308)
#
#print("Null Model Loss:")
#print(np.mean(np.square(t_batch_labels - means)))
#print("Conv/Deconv Model Loss:")
#print(min(history.history["val_loss"]))
#
## Save predictions and labels
#np.savetxt(Path("../sims/", "labels"), t_batch_labels, delimiter=",")
#np.savetxt(Path("../sims/", "preds"), autoencoder.predict(t_batch_data))
## Save  model loss
#np.savetxt(Path("..sims/", "loss"), history.history["val_loss"])
#autoencoder.save(Path("..sims/", "model"))