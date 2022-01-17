#' PCA filtering of simulations
#'
#' @param abc_output The output from ABC_rejection.
#' @param sum_stats The observed summary statistic vector.
#' @param nf The number of principal components to use for distance computations.
#' @param tol The fraction of simulations to keep.
#' @param cols The columns of summary statistics on which to perform PCA. The other columns are left unchanged.
#'
#' @importFrom ade4 dudi.pca
#' @return Filtered version
#' @export
pca_filter <- function(abc_output, sum_stats, nf, tol, cols = 1:length(sum_stats), scale = NULL) {
    df = rbind(sum_stats, abc_output$stats)
    rownames(df) = 1:nrow(df)
    out.pca = dudi.pca(df[,cols],
                       row.w = c(0, rep(1, nrow(abc_output$stats))),
                       scannf = FALSE, nf = nf)
    observed_projection = as.matrix(out.pca$li)[1,]
    simulation_projections = as.matrix(out.pca$li)[-1,]
    if(length(cols) < length(sum_stats)) {
        non_projected = which(!(1:length(sum_stats) %in% cols))
        observed_projection = c(observed_projection, scale * sum_stats[non_projected])
        simulation_projections = cbind(simulation_projections,
                                       sweep(abc_output$stats[,non_projected, drop = FALSE], 2, STATS = scale, FUN = "*"))
    }
    dist_from_observed = apply(sweep(simulation_projections, 2, STATS = observed_projection, FUN = "-"), 1, function(x) sqrt(sum(x^2)))
    to_keep = order(dist_from_observed)[1:(length(dist_from_observed) * tol)]
    abc_filtered = abc_output
    abc_filtered$param = abc_filtered$param[to_keep,]
    abc_filtered$stats = abc_filtered$stats[to_keep,]
    abc_filtered$weights = rep(1 / length(to_keep), length(to_keep))
    abc_filtered$stats_normalization = rep(NA, length(abc_filtered$stats_normalization))
    return(abc_filtered)
}

#' Make SeqSet from partis output
#'
#' @param partis_file
#'
#' @return SeqSet object, contains the mutated sequences and their
#'     corresponding germline sequences. Sequences are stored as
#'     character vectors.
#' @importFrom seqinr s2c
#' @export
seq_set_from_partis <- function(partis_df, trim_to_v = FALSE, v_gene_fasta = NULL) {
    if(trim_to_v & is.null(v_gene_fasta)) {
        stop("If you want to trim to V genes, you need to supply V gene sequences")
    }
    out = list()
    if(!trim_to_v) {
        germline = apply(partis_df[,c("naive_seq", "indel_reversed_seqs")], 1, function(x) ifelse(x[1] == "", x[2], x[1]))
        germline_unique = unique(germline)
                out$germline = lapply(germline_unique, s2c)
        out$mutated_seqs = lapply(partis_df$input_seqs, s2c)
        out$germline_mapping = sapply(germline, function(x) which(germline_unique == x))
    } else {
        v_gene_sequences = read.fasta(v_gene_fasta, forceDNAtolower = FALSE)
        trim_lengths = sapply(partis_df$v_gene, function(x) length(v_gene_sequences[[x]]))
        
        out$germline = v_gene_sequences
        out$mutated_seqs = lapply(1:nrow(partis_df), function(i) s2c(partis_df$input_seqs[i])[1:trim_lengths[i]])
        out$germline_mapping = sapply(partis_df$v_gene, function(vg) which(names(out$germline) == vg))
    }
    ## TODO: check that the mutated sequences all have the same length as their corresponding germline sequences
    class(out) = "seq_set"
    return(out)
}

#' Make SeqSet from fasta file plus germline
#'
#' @param mutated_seq_fasta A fasta file containing the mutated sequences.
#' @param germline_fasta A fasta file containing the germline sequence.
#' @param germline_string A string giving the germline sequence.
#'
#' @importFrom seqinr read.fasta
#' @return SeqSet object, contains the mutated sequences and their
#'     corresponding germlien sequences. Sequences are stored as
#'     character vectors.
#' @export
seq_set_from_fasta <- function(mutated_seq_fasta, germline_fasta = NULL, germline_string = NULL) {
    if(is.null(germline_fasta) & is.null(germline_string)) {
        stop("You must specify either a germline fasta file or a string containing the germline sequence.")
    }
    if(!is.null(germline_fasta)) {
        fasta = read.fasta(germline_fasta, forceDNAtolower = FALSE)
        germline_seq = fasta[[names(fasta)[1]]]
    }
    mutated_seqs = read.fasta(mutated_seq_fasta, forceDNAtolower = FALSE)
    ## check whether the mutated sequences all have the same length as the germline sequence
    mutated_seq_lengths = sapply(mutated_seqs, length)
    if(any(mutated_seq_lengths != length(germline_seq))) {
        err = sprintf("The mutated sequences need to be the same length as the germline sequence, your germline sequence is length %i and there are mutated sequences of length(s) %s",
                      length(germline_seq),
                      paste(mutated_seq_lengths[mutated_seq_lengths != length(germline_seq)], collapse = ","))
        stop(err)
    }
    out = list()
    out$germline[[1]] = germline_seq
    out$mutated_seqs = mutated_seqs
    out$germline_mapping = rep(1, length(mutated_seqs))
    class(out) = "seq_set"
    return(out)
}

#' Computes per-base mutation rates for each germline sequence in a sequence set
#'
#' @param seq_set seq_set object.
#'
#' @return A data frame: One column gives the germline sequence id,
#'     one column gives the position within the germline sequence, one
#'     column gives the mutation rate at that position in the germline
#'     sequence, and one column gives the germline base.
#' @export
mutation_rates <- function(seq_set) {
    frames = lapply(unique(seq_set$germline_mapping), function(glmap) {
        mf = get_mutation_freqs(seq_set$germline[[glmap]], seq_set$mutated_seqs[which(seq_set$germline_mapping == glmap)])
        mf$germline_id = glmap
        return(mf)
    })
    return(Reduce(rbind, frames))
}


#' Finds the positions of mutations
#'
#' @param germline_seq Germline sequence as a character vector.
#' @param mutated_seq Mutated sequence as a character vector.
#' @return If as_vector, an indicator vector of mutation at each
#'     position. Otherwise a vector containing the indices of the
#'     mutations. The indices of the mutations.
#' @export
get_mutation_indices <- function(seq1, seq2, as_indicator = TRUE) {
    if(length(seq1) != length(seq2))
        stop(sprintf("The input sequences need to be the same length, yours are %i and %i", length(seq1), length(seq2)))
    mutations = rep(FALSE, length(seq1))
    for(i in 1:length(seq1)) {
        if(seq1[i] != seq2[i])
            mutations[i] = TRUE
    }
    if(as_indicator) {
        return(mutations)
    } else {
        return(which(mutations))
    }
}

#' Finds the number of mutations.
#'
#' @param mutated_seq String giving the mutated sequence.
#' @param germline_seq String giving the germline sequence germline
#' sequence.
#' @return The number of mutations.
#' @export
get_num_mutations <- function(mutated_seq, germline_seq) {
    if(length(mutated_seq) != length(germline_seq))
        stop(sprintf("The input sequences need to be the same length, yours are %i and %i", length(mutated_seq), length(germline_seq)))
    mvec = strsplit(mutated_seq, "", fixed = TRUE)[[1]]
    glvec = strsplit(germline_seq, "", fixed = TRUE)[[1]]
    return(sum(mvec != glvec))
}


#' Finds per-position mutation frequences for a set of sequences
#'
#' @param germline_seq Germline sequence as a character vector.
#' @param mutated_seqs A list of mutated sequences, each one as a
#'     character vector.
#'
#' @return A data frame with one column for position, one column for
#'     mutation count, one column for mutation frequency, and one
#'     column for germline base.
#' @export
get_mutation_freqs <- function(germline_seq, mutated_seqs) {
    mutations = sapply(mutated_seqs, function(ms) {
        get_mutation_indices(germline_seq, ms)
    })
    mutation_counts = rowSums(mutations)
    return(data.frame(position = 1:length(mutation_counts),
                      mutation_counts = mutation_counts,
                      mutation_rates = mutation_counts / length(mutated_seqs),
                      germline_base = as.vector(germline_seq)))
}

#' Make a predictor matrix from a set of sequences
#'
#' Presumably to be used as input to a neural net
#'
#' @param seqs A vector, length equal to the number of sequences, each
#' element a string giving the sequence.
#'
#' @return A matrix, number of rows equal to length(seqs), each column
#' a predictor.
#' @export
make_predictors_from_seqs <- function(seqs) {
    expand_sequence = function(seq) {
        s_vec = strsplit(seq, split = "", fixed = TRUE)[[1]]
        match_list = lapply(s_vec, function(s) s == c("A", "T", "G", "C"))
        expanded_representation = as.numeric(Reduce(c, match_list))
    }
    seq_matrix = t(sapply(seqs, expand_sequence))
    return(seq_matrix)
}

#' Make a predictor matrix from a sequence
#'
#' Presumably to be used as input to a neural net
#'
#' @param seq A string, length equal to the number of sequences, each
#' element a string giving the sequence.
#'
#' @return A matrix, number of rows equal to length(seqs), each column
#' a predictor.
#' @export
sequence_matrix_representation <- function(seq) {
    s_vec = strsplit(seq, split = "", fixed = TRUE)[[1]]
    match_list = lapply(s_vec, function(s) s == c("A", "T", "G", "C"))
    seq_matrix = Reduce(rbind, match_list)
    rownames(seq_matrix) = s_vec
    colnames(seq_matrix) = c("A", "T", "G", "C")
    return(seq_matrix + 0)
}

#' Computes spatial colocalization statistic
#'
#' @param gl_seq A string containing the germline sequence.
#' @param mutated_seqs A list of strings containing the mutated sequences.
#' @param r_max The maximum value of r at which to compute spatial colocalization.
#' @param r_min The minimum value of r at which to compute spatial colocalization.
#' @export
spatial_colocalization <- function(seq_set, r_max = 25, r_min = 0) {
    frames = lapply(unique(seq_set$germline_mapping), function(glmap) {
        gr = get_gr(seq_set$germline[[glmap]],
                    seq_set$mutated_seqs[which(seq_set$germline_mapping == glmap)],
                    r_max = r_max, r_min = r_min)
        gr$germline_id = glmap
        return(gr)
    })
    return(Reduce(rbind, frames))
}

#' Computes g(r) function
#'
#' @param germline_seq The germline sequence, as a character vector.
#' @param mutated_seqs The mutated sequences, as a list of character vectors.
#' @param r_max Maximum distance between mutations to calculate g(r)
#'     for.
#' @param r_min Minimum distance between mutations to calculate g(r)
#'     for.
#' @param mutation_freqs If NULL, will estiamte the per-position
#'     mutation probabilities from germline_seq and
#'     mutated_seqs. Otherwise, mutation_freqs is a vector where
#'     mutation_freqs[i] gives the probability of mutation at position
#'     i.
#'
#' @return A data frame with columns dist (the distance between
#'     mutations) and gr (spatial colocalization statistic for the
#'     corresponding distance between mutations.)
#'
#' @importFrom dplyr group_by
#' @importFrom dplyr summarise
#' @importFrom magrittr %>%
#' @export
get_gr <- function(germline_seq, mutated_seqs, r_max, r_min, mutation_probs = NULL, n_boot = NULL) {
    mutations = sapply(mutated_seqs, function(ms) {
        get_mutation_indices(germline_seq, ms, as_indicator = FALSE)
    })
    pairfreq = get_pair_frequency(mutations, length(mutated_seqs[[1]]), max_dist = r_max, min_dist = r_min, mutation_probs = mutation_probs)
    gr = pairfreq %>% subset(expected_freq > 0) %>%
        group_by(dist) %>%
        summarise(gr = mean(freq_pair / expected_freq))
    if(!is.null(n_boot)) {
        boot_dist = list()
        for(b in 1:n_boot) {
            boot_sample = sample(length(mutations), replace = TRUE)
            pairfreq = get_pair_frequency(mutations[boot_sample], length(mutated_seqs[[1]]), max_dist = r_max, min_dist = r_min, mutation_probs = mutation_probs)
            gr_boot = pairfreq %>% subset(expected_freq > 0) %>%
                group_by(dist) %>%
                summarise(gr = mean(freq_pair / expected_freq))
            gr_boot$boot_rep = b
            boot_dist[[b]] = gr_boot
        }
        boot_dist = Reduce(rbind, boot_dist)
        boot_summarised = boot_dist %>% group_by(dist) %>%
            summarise(sd_boot = sd(gr), lo = quantile(gr, .025), hi = quantile(gr, .975))
        return(list(gr, boot_dist, boot_summarised))
    }
    return(gr)
}

#' @export
get_gr_cross <- function(germline_seq, mutated_seqs, r_max, r_min, from_base, to_base, mutation_probs = NULL, n_boot = NULL) {
    mutations = sapply(mutated_seqs, function(ms) {
        get_mutation_indices(germline_seq, ms, as_indicator = FALSE)
    })
    pairfreq = get_pair_frequency(mutations, length(mutated_seqs[[1]]), max_dist = r_max, min_dist = r_min, mutation_probs = mutation_probs)
    pairfreq$b1_identity = germline_seq[pairfreq$b1]
    pairfreq$b2_identity = germline_seq[pairfreq$b2]
    gr = pairfreq %>%
        subset(expected_freq > 0 &
               ((b1_identity == from_base & b2_identity == to_base) |
                (b1_identity == to_base & b2_identity == from_base))
               ) %>%
        group_by(dist) %>%
        summarise(gr = mean(freq_pair / expected_freq))
    if(!is.null(n_boot)) {
        boot_dist = list()
        for(b in 1:n_boot) {
            boot_sample = sample(length(mutations), replace = TRUE)
            pairfreq = get_pair_frequency(mutations[boot_sample], length(mutated_seqs[[1]]), max_dist = r_max, min_dist = r_min, mutation_probs = mutation_probs)
            pairfreq$b1_identity = germline_seq[pairfreq$b1]
            pairfreq$b2_identity = germline_seq[pairfreq$b2]
            gr_boot = pairfreq %>%
                subset(expected_freq > 0 &
                       ((b1_identity == from_base & b2_identity == to_base) |
                        (b1_identity == to_base & b2_identity == from_base))
                       ) %>%
                group_by(dist) %>%
                summarise(gr = mean(freq_pair / expected_freq))
            gr_boot$boot_rep = b
            boot_dist[[b]] = gr_boot
        }
        boot_dist = Reduce(rbind, boot_dist)
        boot_summarised = boot_dist %>% group_by(dist) %>%
            summarise(sd_boot = sd(gr), lo = quantile(gr, .025), hi = quantile(gr, .975))
        return(list(gr, boot_dist, boot_summarised))
    }
    return(gr)
}

#' @export
get_gr_at_position <- function(germline_seq, mutated_seqs, r_max, r_min, position, to_base, mutation_probs = NULL, n_boot = NULL) {
    mutations = sapply(mutated_seqs, function(ms) {
        get_mutation_indices(germline_seq, ms, as_indicator = FALSE)
    })
    pairfreq = get_pair_frequency(mutations, length(mutated_seqs[[1]]), max_dist = r_max, min_dist = r_min, mutation_probs = mutation_probs)
    pairfreq$b1_identity = germline_seq[pairfreq$b1]
    pairfreq$b2_identity = germline_seq[pairfreq$b2]
    gr = pairfreq %>%
        subset(expected_freq > 0 & b1 == position & b2_identity %in% to_base) %>%
        group_by(dist) %>%
        summarise(gr = mean(freq_pair / expected_freq))
        if(!is.null(n_boot)) {
        boot_dist = list()
        for(b in 1:n_boot) {
            boot_sample = sample(length(mutations), replace = TRUE)
            pairfreq = get_pair_frequency(mutations[boot_sample], length(mutated_seqs[[1]]), max_dist = r_max, min_dist = r_min, mutation_probs = mutation_probs)
            pairfreq$b1_identity = germline_seq[pairfreq$b1]
            pairfreq$b2_identity = germline_seq[pairfreq$b2]
            gr_boot = pairfreq %>%
                subset(expected_freq > 0 & b1 == position & b2_identity %in% to_base) %>%
                group_by(dist) %>%
                summarise(gr = mean(freq_pair / expected_freq))
            gr_boot$boot_rep = b
            boot_dist[[b]] = gr_boot
        }
        boot_dist = Reduce(rbind, boot_dist)
        boot_summarised = boot_dist %>% group_by(dist) %>%
            summarise(sd_boot = sd(gr), lo = quantile(gr, .025), hi = quantile(gr, .975))
        return(list(gr, boot_dist, boot_summarised))
    }

    return(gr)
}


#' Finds the frequencies of all pairs of mutations within a certain distance.
#'
#' The idea here is to compare the number of times we see each pair of
#' mutations with the number of times we would expect to see it under
#' an independence model. Pairs of mutations are described as a pair
#' (i,j), with i < j. If mutations are independent, the probability of
#' seeing mutations i and j in a sequence with n mutations total is n
#' * (n-1) * p(i) * p(j), where p(i) and p(j) are the marginal
#' probabilities of mutations at positions i and j. Therefore, the
#' expected number of times we see mutation pair (i,j) in the set of
#' sequences is p(i) * p(j) * sum_seq n_seq (n_seq - 1).
#'
#' @param mutations A list, each element of which is a vector
#'     containing the positions of mutations in the corresponding
#'     sequence.
#' @param npositions The number of positions (length of the germline
#'     sequence).
#' @param max_dist The maximum distance to compute.
#' @param min_dist The minimum distance to compute.
#' @param mutation_probs mutation_probs[i] gives the probability of
#'     mutation at position i.
#' @export
get_pair_frequency <- function(mutations, npositions, max_dist, min_dist = 0, mutation_probs = NULL) {
    n_pairs = sum(sapply(mutations, function(x) length(x) * (length(x) - 1) / 2))
    n_mutations = sum(sapply(mutations, length))
    out = subset(expand.grid(b1 = 1:npositions, b2 = 1:npositions), b2 - b1 > 0 & b2 - b1 <= max_dist & b2 - b1 >= min_dist)
    out$freq_pair = rep(0, nrow(out))
    for(i in 1:length(mutations)) {
        mut_idx = mutations[[i]]
        for(j in 1:(length(mut_idx) - 1)) {
            for(k in (j+1):length(mut_idx)) {
                pair_idx = which(out$b1 == mut_idx[j] & out$b2 == mut_idx[k])
                out[pair_idx, "freq_pair"] = out[pair_idx, "freq_pair"] + 1
            }
        }
    }
    if(is.null(mutation_probs)) {
        mutations_overall = unlist(mutations)
        ## mutation_freqs[i] gives the frequency of mutation at position i
        mutation_probs = sapply(1:npositions, function(pos) sum(mutations_overall == pos)) / n_mutations
    }
    out$prob_b1 = mutation_probs[out$b1]
    out$prob_b2 = mutation_probs[out$b2]
    out$expected_freq = 2 * n_pairs * out$prob_b1 * out$prob_b2
    out$dist = out$b2 - out$b1
    return(out)
}

#' Wrapper function for the python simulator
#'
#' @param write_summary_stats
#' @param summary_stat_output
#' @param write_seqs
#' @param sequence_output
#' @param aid_context_model
#' @param context_model_length
#' @param context_model_pos_mutating
#' @param germline_sequence
#' @param ber_lambda
#' @param bubble_size
#' @param exo_length
#' @param pol_eta_params A matrix with giving nucleotide substitution
#'     rates, nucleotide order is AGCT, row indicates germline base,
#'     column indicates target base (so the substitution rates
#'     starting from A are given by the first row of the matrix).
#' @param ber_params A vector giving substitution rates, nucleotide
#'     order is AGCT.
#' @param germline_sequence A string with the location of a fasta file describing the germline sequence.
#' @param python_driver A string with the location of the python driver.
#'
#' @export
shm_sim <- function(write_summary_stats = TRUE,
                    summary_stat_output = "output",
                    write_seqs = FALSE,
                    sequence_output= "output.fasta",
                    aid_context_model = "data/logistic_fw_3mer.csv",
                    context_model_length = 3,
                    context_model_pos_mutating = 2,
                    germline_sequence = system.file("extdata", "gpt.fasta", package = "shmr"),
                    ber_lambda = .5,
                    bubble_size = 20,
                    exo_left = 20,
                    exo_right = 20,
                    n_mutation_rounds = 3,
                    n_seqs = 500,
                    ber_params = c(.25, .25, .25, .25),
                    pol_eta_params = diag(rep(1,4)),
                    p_fw = .5,
                    python_driver = system.file("src", "simulate_mutations.py", package = "shmr")) {
    #nucleotides = c('A', 'G', 'C', 'T')
    call = paste(c(python_driver,
                   '--write-summary-stats',
                   ifelse(write_summary_stats, 'True', 'False'),
                   '--write-seqs',
                   ifelse(write_seqs, 'True', 'False'),
                   '--sequence-output',
                   sequence_output,
                   '--aid-context-model',
                   aid_context_model,
                   '--context-model-length',
                   context_model_length,
                   '--context-model-pos-mutating',
                   context_model_pos_mutating,
                   '--germline-sequence',
                   germline_sequence,
                   '--ber-lambda',
                   ber_lambda,
                   '--bubble-size',
                   bubble_size,
                   '--exo-left',
                   exo_left,
                   '--exo-right',
                   exo_right,
                   '--n-mutation-rounds',
                   n_mutation_rounds,
                   '--n-seqs',
                   n_seqs,
                   '--ber-params',
                   python_vec_string(ber_params),
                   '--pol-eta-a',
                   python_vec_string(pol_eta_params[1,]),
                   '--pol-eta-g',
                   python_vec_string(pol_eta_params[2,]),
                   '--pol-eta-c',
                   python_vec_string(pol_eta_params[3,]),
                   '--pol-eta-t',
                   python_vec_string(pol_eta_params[4,]),
                   '--p-fw',
                   p_fw),
                 collapse = ' ')
    system(call)
}

#' Wrapper for the python simulator compatible with EasyABC
#'
#' @param x A vector with the parameters.
#' @return A vector of summary statistics.
#' @export
shm_sim_abc <- function(x) {
    ## call the simulator
    shm_sim(write_summary_stats = TRUE,
            summary_stat_output = "output",
            write_seqs = FALSE,
            aid_context_model = "data/logistic_model_c_fwd_5mer.csv",
            context_model_length = 5,
            context_model_pos_mutating = 2,
            germline_sequence = "/Users/juliefukuyama/GitHub/HyperMutationModels/data/gpt.fasta",
            ber_lambda = x[1],
            bubble_size = x[2],
            exo_length = x[3])
    ss = read.delim("output", header = FALSE)
    ## return a vector of summary statistics
    return(ss)
}

#' String representation of a vector.
#'
#' Creates a string that, when loaded into python with
#' ast.literal_eval, will make a list.
#'
#' @param vec The vector to be encoded.
#' @return A string.
python_vec_string <- function(vec) {
    return(paste('[', paste(vec, collapse = ','), ']', sep = ''))
}

#' String representation of pol eta params
#'
#' Creates a string that, when loaded into python with
#' ast.literal_eval, will make a dictionary with nucleotides as keys
#' and substitution probabilities as values.
#'
#' @param substitution_matrix A 4x4 matrix, rows as germline bases,
#'     columns as target bases, with substitution probabilities.
#' @param nucleotides A vector giving the nucleotides.
#'
#' @return A string that can be turned into a python dictionary with
#'     ast.literal_eval.
make_pol_eta_string <- function(substitution_matrix, nucleotides) {
    substitutions = apply(substitution_matrix, 1, python_vec_string)
    string = paste(apply(cbind(rep("\\'", 4), nucleotides, rep("\\':", 4), substitutions), 1, paste, collapse = ""), collapse = ",")
    return(paste("{", string, "}", sep = ""))
}


