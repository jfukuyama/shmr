# We would like to simulate sequences in place, with the fasta file and context model loaded into memory
# This way we can simulate A LOT of training data without writing to disk (slow)
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import scale
import pkgutil
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from SHMModels.mutation_processing import *
from SHMModels.fitted_models import *
from SHMModels.simulate_mutations import *

# Set Model Parameters (they don't change across prior samples)
pol_eta_params = {
            "A": [0.9, 0.02, 0.02, 0.06],
            "G": [0.01, 0.97, 0.01, 0.01],
            "C": [0.01, 0.01, 0.97, 0.01],
            "T": [0.06, 0.02, 0.02, 0.9],
        }
ber_params = np.random.dirichlet([1, 1, 1, 1])
cm = aid_context_model = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
# Get a 2d hot encoding of a sequence
def hot_encode_2d(seq):
    seq_hot = np.zeros((len(seq), 4, 1))
    for j in range(len(seq)):
        seq_hot[j, 0, 0] = seq[j] == "A"
        seq_hot[j, 1, 0] = seq[j] == "T"
        seq_hot[j, 2, 0] = seq[j] == "G"
        seq_hot[j, 3, 0] = seq[j] == "C"
    return seq_hot


# Get a 1d hot encoding of a sequence
def hot_encode_1d(seq):
    seq_hot = np.zeros((len(seq) * 4))
    for j in range(len(seq)):
        seq_hot[4 * j] = seq[j] == "A"
        seq_hot[4 * j + 1] = seq[j] == "T"
        seq_hot[4 * j + 2] = seq[j] == "G"
        seq_hot[4 * j + 3] = seq[j] == "C"
    return seq_hot

# Get batch
def gen_batch(seq,batch_size):
      mut = np.zeros((batch_size,np.shape(seq)[0],4,1))
      les = np.zeros((batch_size,np.shape(seq)[0]))
      for i in range(batch_size):
            mp = MutationProcess(seq,
                           aid_context_model = cm,
                           ber_params = ber_params,
                           pol_eta_params = pol_eta_params,
                           ber_lambda = .0100,
                           mmr_lambda = .0100,
                           overall_rate = 10,
                           show_process = False)
            mp.generate_mutations()
            mut[i,:,:,:] = hot_encode_2d(mp.repaired_sequence[0,:])
            les[i,:] = np.sum(mp.aid_lesions_per_site, axis = 0)
      return les, mut


# Get batch
def gen_batch_letters(seq,batch_size):
      mut = np.zeros((batch_size,np.shape(seq)[0],4,1))
      les = np.zeros((batch_size,np.shape(seq)[0]))
      let = []
      for i in range(batch_size):
            mp = MutationProcess(seq,
                           aid_context_model = cm,
                           ber_params = ber_params,
                           pol_eta_params = pol_eta_params,
                           ber_lambda = .0100,
                           mmr_lambda = .0100,
                           overall_rate = 10,
                           show_process = False)
            mp.generate_mutations()
            
            mut[i,:,:,:] = hot_encode_2d(mp.repaired_sequence[0,:])
            les[i,:] = np.sum(mp.aid_lesions_per_site, axis = 0)
            let.append(mp.repaired_sequence[0,:])
      return les, mut, let
      