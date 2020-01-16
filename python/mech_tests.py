import pkgutil
from pathlib import Path
import numpy as np
from Bio import SeqIO
from Bio.Alphabet import IUPAC
from SHMModels.fitted_models import ContextModel
from keras import optimizers, Input, Model
from genDat import hot_encode_2d, gen_batch
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

##### USER INPUTS (Edit some of these to be CLI eventually)

# Path to germline sequence
germline_sequence = "data/gpt.fasta"
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
batch_size = 300
num_epochs = 10000
steps_per_epoch = 1
# flag to include ber_pathway
ber_pathway = 1

# Means and sds from set of 5000 prior samples (logit transform 4:8)
means = [
    0.50228154,
    26.8672,
    0.08097563,
    0.07810973,
    -1.52681097,
    -1.49539369,
    -1.49865018,
    -1.48759332,
    0.50265601,
]
sds = [
    0.29112116,
    12.90099082,
    0.1140593,
    0.11241542,
    1.42175933,
    1.43498051,
    1.44336424,
    1.43775417,
    0.28748498,
]

# Load sequence into memory
sequence = list(
    SeqIO.parse(germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
)[0]
# Load aid model into memory
aid_model_string = pkgutil.get_data("SHMModels", aid_context_model)
aid_model = ContextModel(
    context_model_length, context_model_pos_mutating, aid_model_string
)
orig_seq = hot_encode_2d(sequence)
# Create testing data
junk = gen_batch(
    batch_size,
    sequence,
    aid_model,
    n_seqs,
    n_mutation_rounds,
    orig_seq,
    means,
    sds,
    2,
    4,
    ber_pathway,
)
t_batch_data = junk["seqs"][:, 0, :, :, :]
# Just use the lesion sites (1st argument in 3rd position) as label
t_batch_labels = junk["mechs"][:, 0, 0, :]

# Let's build our encoder. Seq is of length 308.
input_seq = Input(shape=(308, 4, 1))

# We add 2 convolutional layers.
x = Conv2D(16, (3, 4), activation="relu", padding="same")(input_seq)
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
decoded = Dense(units=308, activation="relu")(x)

# at this point the "decoded" representation is a 308 vector indicating our predicted # of lesions at each site.
autoencoder = Model(input_seq, decoded)
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

print(autoencoder.summary(90))

# Create iterator for simulation
def genTraining(batch_size):
    while True:
        # Get training data for step
        dat = gen_batch(
            batch_size,
            sequence,
            aid_model,
            n_seqs,
            n_mutation_rounds,
            orig_seq,
            means,
            sds,
            2,
            4,
            ber_pathway,
        )
        # Get lesion sites and 2d encoded sequence
        batch_labels = dat["mechs"][:, 0, 0, :]
        batch_data = dat["seqs"][:, 0, :, :, :]
        yield batch_data, batch_labels


history = autoencoder.fit_generator(
    genTraining(batch_size),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    verbose=2,
)
temp = np.mean(t_batch_labels)
means = np.repeat(temp, 308)

print("Null Model Loss:")
print(np.mean(np.square(t_batch_labels - means)))
print("Conv/Deconv Model Loss:")
print(min(history.history["val_loss"]))

# Save predictions and labels
np.savetxt(Path("../sims/", "labels"), t_batch_labels, delimiter=",")
np.savetxt(Path("../sims/", "preds"), autoencoder.predict(t_batch_data))
# Save  model loss
np.savetxt(Path("..sims/", "loss"), history.history["val_loss"])
autoencoder.save(Path("..sims/", "model"))
