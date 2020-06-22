from SHMModels.simulate_mutations import simulate_sequences_abc

simulate_sequences_abc(
    "data/gpt.fasta",
    "data/aid_logistic_3mer.csv",
    context_model_length=3,
    context_model_pos_mutating=2,
    n_seqs=2,
    n_mutation_rounds=3,
    ss_file="for_nnet_ss.csv",
    param_file="for_nnet_params.csv",
    sequence_file="for_nnet_sequences.csv",
    n_sims=2,
    write_ss=False,
    write_sequences=True,
)
