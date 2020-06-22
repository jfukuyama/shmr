import numpy as np
import math
import scipy.stats

#' Make SE kernel at a set of points
#' @param x The set of points for kernel evaluation
#' @param lengthscale Lengthscale in GP
#' @param sigma Overall variance in GP
#' @param gp_ridge The ridge value
def make_se_kernel(x, lengthscale, sigma, gp_ridge):
    D = np.zeros([len(x), len(x)])
    upper_tri = np.triu_indices(len(x), 1)
    D[upper_tri] = (np.array(x)[upper_tri[0]] - np.array(x)[upper_tri[1]])**2
    D += D.T
    K = sigma**2 * np.exp(-D / (2 * lengthscale))
    np.fill_diagonal(K, K.diagonal() + gp_ridge)
    return K


#' Samples from a SGCP on [0,1]
#'
#' @param lambda_star Mean number of points in the unthinned process
#' @param lengthscale The lengthscale parameter for the SE kernel in
#'     the GP
#' @param gp_sigma The variance parameter for the SE kernel in the GP
#' @param gp_ridge The ridge value
#' Returns A dict with A = lesions, A_tilde = prelesions, g =
#' values of the GP on [A, A_tilde], in that order.
def forward_sample_sgcp(lambda_star, lengthscale, gp_sigma, gp_ridge):
    N = np.random.poisson(lam = lambda_star)
    x = np.random.uniform(low = 0, high = 1, size = N)
    K = make_se_kernel(x, lengthscale, gp_sigma, gp_ridge)
    lambda_of_x = np.random.multivariate_normal(mean = np.zeros(N), cov = K)
    sigma_lambda_of_x = 1 / (1 + np.exp(-lambda_of_x))
    uniforms = np.random.uniform(low = 0, high = 1, size = len(x))
    A_and_g = [(xi, li) for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms) if u < si]
    A_tilde_and_g = [(xi, li) for (xi, si, li, u) in zip(x, sigma_lambda_of_x, lambda_of_x, uniforms) if u >= si]
    A = [a for (a, g) in A_and_g]
    A_tilde = [at for (at, g) in A_tilde_and_g]
    g = [g for (a, g) in A_and_g + A_tilde_and_g]
    return({ "A" : A, "A_tilde" : A_tilde, "g" : g})

#' Sample one position from BER distribution
#' @param ber_params A dictionary giving the probabilities of A, T, C, G
#' @return A character, either A, T, C, or G.
def sample_ber(ber_params):
    rm = np.random.multinomial(n = 1, pvals = ber_params.values())
    base = [b for (b, m) in zip(ber_params.keys(), rm) if m == 1]
    return base

def interval_to_discrete(a, seq_length):
    return int(np.floor(a * seq_length))

def discrete_to_interval(i, seq_length):
    return (i - np.random.uniform(low = 0, high = 1, size = 1)) / seq_length

#' @param lambda_star: Rate for the unthinned poisson process
#' @param lengthscale: Lengthscale parameter for the GP
#' @param gp_sigma: Scale for the GP
#' @param gl_seq: A character vector with the germline sequence_complete_data_log_prob
#' @param ber_params: A dictionary giving the probabilities of each base substitution for ber
#' @param gp_ridge: Ridge for the GP
def forward_sample_sequences(lambda_star, lengthscale, gp_sigma, gl_seq, ber_params, gp_ridge):
    output_seq = gl_seq
    # AID lesions, continuous
    A, _, _ = forward_sample_sgcp(lambda_star, lengthscale, gp_sigma, gp_ridge = gp_ridge)
    for a in A:
        i = interval_to_discrete(a, seq_length = len(gl_seq))
        output_seq[i] = sample_ber(ber_params)
    return output_seq

#' @param g: The values of the GP, the first K corresponding to lesions, last M corresponding to prelesions
#' @param K: The number of lesions
#' @param M: The number of prelesions
#' @return: Log probability of thinning out the last M potential
#' lesions and keeping the first K, given values of the GP given in g
def thinning_log_prob(g, K, M):
    if(K == 0 and M == 0):
        return(0)
    elif(K == 0 and M > 0):
        return(sum(np.log(logistic(-g))))
    elif(K > 0 and M == 0):
        return(sum(np.log(logistic(g))))
    else:
        return(sum(np.log(logistic(g[range(K)]))) + sum(np.log(logistic(-g[range(K, K+M)]))))

#' @param A: A vector, with the locations of the lesions
#' @param A_tilde: A vector, with the locations of the prelesions
#' @param g: A vector with the values of the GP at [A, A_tilde]
#' @param lengthscale: The lengthscale for the GP
#' @param sigma: The scale for the GP
#' @param gp_ridge: Ridge for the GP
#' @return: Log probability of the complete data for the SGCP
def sgcp_complete_data_log_prob(A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge):
    K = len(A)
    M = len(A_tilde)
    pois = (K + M) * np.log(lambda_star) - lambda_star - math.lgamma(K + M + 1)
    thinning = thinning_log_prob(g, K, M)
    gp_kern_ridged = make_se_kernel(np.append(np.array(A), A_tilde), lengthscale, sigma, gp_ridge)
    gp = np.log(scipy.stats.multivariate_normal.pdf(g, mean = np.zeros(len(g)), cov = gp_kern_ridged))
    return(pois + thinning + gp)


def logistic(x):
    return(1 / (1 + np.exp(-x)))

def logit(p):
    return(np.log(p / (1 - p)))

#' @param x_list: A list, each element corresponding to the lesions + prelesions in one sample
#' @param g_list: A list, each element corresponding to the values of the GP in one sample
#' @param weights: A list, each element corresponding to the weight for one sample
#' @param l_test_grid: The lengthscale points at which to evaluate the likelihood
#' @param model_params: The model parameters
def lengthscale_inference(x_list, g_list, weights, l_test_grid, model_params):
    log_probs = np.zeros(len(l_test_grid))
    for (ls, ls_idx) in zip(l_test_grid, range(len(l_test_grid))):
        g_log_p_at_test = 0.
        for i in range(len(x_list)):
            g = g_list[i]
            K = make_se_kernel(x_list[i], lengthscale = ls, sigma = model_params['gp_sigma'], gp_ridge = model_params['gp_ridge'])
            g_log_p_at_test += weights[i] * np.log(scipy.stats.multivariate_normal.pdf(g, cov = K))
        log_probs[ls_idx] = g_log_p_at_test
    return(l_test_grid[np.argmax(log_probs)])


#' @param complete_data: A dictionary containing the true lesions (A),
#' prelesinons (A_tilde), and values of the GP at the lesions/prelesions (g)
#' @param current_model_params: A dict with the current model parameters.
#' @param sampling_sd: Instead of the true GP values in g, we use g +
#' multivariate normal noise, sampling_sd is the sd of that noise
#' @return: A dict, with the complete data (lesions = A, prelesions =
#' A_tilde, GP values = g) and the weight for that sample (w)
def complete_data_sample_around_true_states_sgcp(complete_data,
                                                 current_model_params,
                                                 sampling_sd = .1):
    ## each element of observed_data is a list with A (the thinned
    ## points) lambda (the gp at the unthinned points), and x (the
    ## unthinned points).
    A = complete_data['A']
    A_tilde = complete_data['A_tilde']
    g = complete_data['g']
    g_sample = g + np.random.normal(size = len(g), loc = 0, scale = sampling_sd)
    is_log_prob = sum(np.log(scipy.stats.norm.pdf(g_sample - g, loc = 0, scale = sampling_sd)))
    complete_data_log_prob = sgcp_complete_data_log_prob(A, A_tilde, g_sample, current_model_params['lambda_star'], current_model_params['lengthscale'], current_model_params['gp_sigma'], current_model_params['gp_ridge'])
    log_w = complete_data_log_prob - is_log_prob
    return ({ "A" : A, "A_tilde" : A_tilde, "g" : g_sample, "w" : np.exp(log_w)})

#' @param seq: The observed sequence
#' @param gl_seq: The germline sequence
#' @param A: The lesions
#' @param A_tilde: The prelesions
#' @param g: The values of the GP at [A, A_tilde]
#' @param model_params: A dict with the current model params
#' @return The complete-data log probability of seq, A, A_tilde, g given the current model parameters.
def sequence_complete_data_log_prob(seq, gl_seq, A, A_tilde, g, model_params):
    lambda_star = model_params['lambda_star']
    lengthscale = model_params['lengthscale']
    sigma = model_params['sigma']
    gp_ridge = model_params['gp_ridge']
    ber_params = model_params['ber_params']
    sgcp_log_prob = sgcp_complete_data_log_prob(A, A_tilde, g, lambda_star, lengthscale, sigma, gp_ridge)
    sequence_given_sgcp_log_prob = sequence_log_prob_given_lesions(seq, gl_seq, A, ber_params)
    return(sgcp_log_prob + sequence_given_sgcp_log_prob)

#' @param seq: The observed sequence
#' @param gl_seq: The germline sequence
#' @param A: The lesions
#' @param ber_params: A dict with the substitution probabilities for BER
#' @return The log probability of seq given gl_seq, A, and the BER parameters
def sequence_log_prob_given_lesions(seq, gl_seq, A, ber_params):
    A_indices = [interval_to_discrete(a, len(seq)) for a in A]
    ## if mutation but no lesion at any position, log prob is -Inf
    for i in range(len(seq)):
        if((seq[i] != gl_seq[i]) and not np.isin(i, A_indices)):
            return(-Inf)
    log_p = 0
    for i in A_indices:
        log_p = log_p + np.log(ber_params[seq[i]])
    return(log_p)




n_samples = 10;max_steps = 4;n_imp_samples = 1;sampling_noise_sd = .01
start_model_params = { "lambda_star" : 10,
                       "lengthscale" : .01,
                       "gp_sigma" : 2,
                       "gp_ridge" : .0001
}
true_model_params = { "lambda_star" : 10,
                       "lengthscale" : .03,
                       "gp_sigma" : 2,
                       "gp_ridge" : .0001
}

fs = []
current_model_params = start_model_params
## simulating from the SGCP
for i in range(n_samples):
    fs.append(forward_sample_sgcp(lambda_star = true_model_params['lambda_star'], lengthscale = true_model_params['lengthscale'], gp_sigma = true_model_params['gp_sigma'], gp_ridge = true_model_params['gp_ridge']))

## Importance-sampling EM on the simulated data
for step in range(max_steps):
    x_list = []
    g_list = []
    w_list = []
    for i in range(n_samples):
        for j in range(n_imp_samples):
            ## sample complete data around the oracle (we know what the true complete data is)
            imp_sam = complete_data_sample_around_true_states_sgcp(complete_data = fs[i],
                                                                   current_model_params = current_model_params,
                                                                   sampling_sd = sampling_noise_sd)
            x_list.append(fs[i]["A"] + imp_sam["A_tilde"])
            g_list.append(imp_sam["g"])
            w_list.append(imp_sam["w"])
    l_test = np.linspace(.001, .2, 40)
    current_model_params['lengthscale'] = lengthscale_inference(x_list, g_list, w_list, l_test_grid = l_test, model_params = current_model_params)
print(np.sqrt(true_model_params['lengthscale']))
print(np.sqrt(current_model_params['lengthscale']))

