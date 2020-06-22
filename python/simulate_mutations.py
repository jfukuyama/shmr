#!/usr/bin/python2.7
# This driver is meant to give an interface to the EasyABC package in R. It simulates a fixed number of sequences, computes summary statistics, and writes them to the disk.
from SHMModels.simulate_mutations import MutationRound
from SHMModels.summary_statistics import write_all_stats
from SHMModels.fitted_models import ContextModel
import pkgutil
import sys
import argparse
import ast
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-summary-stats",
        type=bool,
        help="Write summary statistics?",
        default=True,
    )
    parser.add_argument(
        "--summary-stat-output",
        type=str,
        help="File to write summary statistics to",
        default="output",
    )
    parser.add_argument(
        "--write-seqs", type=bool, help="Write simulated sequences?", default=False
    )
    parser.add_argument(
        "--sequence-output",
        type=str,
        help="File to write the simulated sequences to",
        default="",
    )
    parser.add_argument("--ber-lambda", type=float, default=0.5)
    parser.add_argument("--bubble-size", type=int, default=20)
    parser.add_argument("--exo-left", type=float, default=0.2)
    parser.add_argument("--exo-right", type=float, default=0.2)
    parser.add_argument(
        "--aid-context-model",
        type=str,
        help="csv file containing context-specific probabilities for AID in the SHMModels package",
        default="data/aid_logistic_3mer.csv",
    )
    parser.add_argument(
        "--context-model-length",
        type=int,
        help="Length of the context for the context model",
        default=3,
    )
    parser.add_argument(
        "--context-model-pos-mutating",
        type=int,
        help="Position mutating in the context model",
        default=2,
    )
    parser.add_argument(
        "--germline-sequence",
        type=str,
        help="Starting sequence",
        default="/Users/juliefukuyama/GitHub/HyperMutationModels/data/gpt.fasta",
    )
    parser.add_argument("--n-mutation-rounds", type=int, default=3)
    parser.add_argument("--n-seqs", type=int, default=500)
    parser.add_argument("--ber-params", type=str, default="[.25,.25,.25,.25]")
    parser.add_argument("--pol-eta-a", type=str)
    parser.add_argument("--pol-eta-g", type=str)
    parser.add_argument("--pol-eta-c", type=str)
    parser.add_argument("--pol-eta-t", type=str)
    parser.add_argument("--p-fw", type=float, default=0.5)
    args = parser.parse_args()
    return args


def main(args=sys.argv[1:]):
    args = parse_args()
    args.ber_params = ast.literal_eval(args.ber_params)
    pol_eta_params = {
        "A": ast.literal_eval(args.pol_eta_a),
        "G": ast.literal_eval(args.pol_eta_g),
        "C": ast.literal_eval(args.pol_eta_c),
        "T": ast.literal_eval(args.pol_eta_t),
    }
    # We start with a germline sequence, simulate one round of deamination plus repair
    sequence = list(
        SeqIO.parse(args.germline_sequence, "fasta", alphabet=IUPAC.unambiguous_dna)
    )[0]
    mutated_seq_list = []
    mmr_length_list = []
    aid_model_string = pkgutil.get_data("SHMModels", args.aid_context_model)
    aid_model = ContextModel(
        args.context_model_length, args.context_model_pos_mutating, aid_model_string
    )
    for i in range(args.n_seqs):
        mr = MutationRound(
            sequence.seq,
            ber_lambda=args.ber_lambda,
            mmr_lambda=1 - args.ber_lambda,
            replication_time=100,
            bubble_size=args.bubble_size,
            aid_time=10,
            exo_params={"left": args.exo_left, "right": args.exo_right},
            pol_eta_params=pol_eta_params,
            ber_params=args.ber_params,
            p_fw=args.p_fw,
            aid_context_model=aid_model,
        )
        for j in range(args.n_mutation_rounds):
            mr.mutation_round()
            mr.start_seq = mr.repaired_sequence
        mutated_seq_list.append(SeqRecord(mr.repaired_sequence, id=""))
        if len(mr.mmr_sizes) > 0:
            mmr_length_list.append(np.mean(mr.mmr_sizes))
    if args.write_summary_stats:
        write_all_stats(
            sequence,
            mutated_seq_list,
            np.mean(mmr_length_list),
            file=args.summary_stat_output,
        )
    if args.write_seqs:
        SeqIO.write(mutated_seq_list, args.sequence_output, "fasta")


if __name__ == "__main__":
    main(sys.argv[1:])
