import tensorflow as tf
import pandas as pd

# import model specific functions
from plot_diagnostics import plot_diagnostics_lm, plot_diagnostics_binomial, plot_diagnostics_poisson, plot_diagnostics_negbinom, plot_diagnostics_mlm  
from models import LM_Model, Binomial_Model, Poisson_Model, MLM_Model, Poisson_Model_sensi, MLM_weibull_Model

# define model specific input-variables
def model_specs(selected_model):
    """ returns all case study specific variables """
    if selected_model == "linreg":
        stats_model, plot_diag = LM_Model, plot_diagnostics_lm
    
        true_hyp_dict = {
            # hyperparameter for beta (mean)
            "mus": [0.12, 0.15, -0.02, -0.03, -0.02, -0.04],
            # hyperparameter for beta (sd)
            "sigmas": tf.math.log([0.02, 0.02, 0.06, 0.06, 0.03, 0.03]),
            # hyperparameter for sigma
            "lambda0": tf.math.log([9.]),
            "sigma_taus": None,
            "alpha_LKJ": None
        }
        X_idx = None
        threshold = None
        X = None
        sigma_prior = "gamma-avg"  # gamma, gamma-avg, exponential-avg
        model_specific = {
            "Nobs_cell": 30,
            "fct_a_lvl": 2,
            "fct_b_lvl": 3
        }
        learning_hyp = {
            "epochs":1500, #1000,
            "B": 2**8,
            "rep_exp": 300,
            "rep_mod": 200,
            "lr0": 0.1,
            "lr_min": 0.000001,
            "lr_decay": 0.97,
            "lr_decay_step": 5
            }
        loss_hyp = {
            "names":       ["ma", "mb",  "effects1", "R2", "gm"], #"effects1",
            "format_type": ["quantiles", "quantiles",  "quantiles", "hist", "hist"], #"quantiles",
            "tensor_type": ["sliced", "sliced",  "sliced", "unsliced", "unsliced"], #"sliced",
            "format_meth": ["summary","summary","summary","summary", "summary"], #"summary",
            }
        
    if selected_model == "binom": 
        stats_model, plot_diag = Binomial_Model, plot_diagnostics_binomial
        
        true_hyp_dict = {
            # hyperparameter for beta (mean)
            "mus": [-0.51, 0.26],
            # hyperparameter for beta (sd)
            "sigmas": tf.math.log([0.06, 0.04]),
            # hyperparameter for sigma
            "lambda0": None,
            "sigma_taus": None,
            "alpha_LKJ": None
        }
        X_idx = [0, 5, 10, 15, 20, 25, 30]
        threshold = None
        X = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/haberman_prep.csv')
        sigma_prior = None
        model_specific = {"size": 30}
        learning_hyp = {
            "epochs": 1000,
            "B": 2**8,
            "rep_exp": 300,
            "rep_mod": 200,
            "lr0": 0.01,
            "lr_min": 0.001,
            "lr_decay": 0.95, 
            "lr_decay_step": 18
            }
        loss_hyp = {
            "names":       ["y_idx"],
            "format_type": ["quantiles"],
            "tensor_type": ["sliced"],
            "format_meth": ["summary"],
            }
    
    if selected_model == "pois":
        stats_model, plot_diag = Poisson_Model, plot_diagnostics_poisson
    
        true_hyp_dict = {
            # hyperparameter for beta (mean)
            "mus": [2.91, 0.23, -1.51, -0.610],
            # hyperparameter for beta (sd)
            "sigmas": tf.math.log([0.07, 0.05, 0.135, 0.105]),
            # hyperparameter for sigma
            "lambda0": None,
            "sigma_taus": None,
            "alpha_LKJ": None
        }
        X_idx = [1, 11, 17, 22, 35, 44]
        threshold = 110
        X = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/antidis_laws.csv')
        sigma_prior = None
        model_specific = None
        learning_hyp = {
            "epochs": 600,
            "B": 2**8,
            "rep_exp": 300,
            "rep_mod": 150,
            "lr0": 0.1,
            "lr_min": 0.0001,
            "lr_decay": 0.95,
            "lr_decay_step": 7
            }
        loss_hyp = {
            "names":       ["y_obs", "y_groups"],
            "format_type": ["hist","quantiles"],
            "tensor_type": ["sliced", "sliced"],
            "format_meth": ["summary", "summary"],
            }
        
    if selected_model == "pois_sensitivity":
        # this is for the plot showing the distirbution of count data
        # therefore the rep_exp number is here very high and we have a
        # different model, which only focus on y_obs (Poisson_Model_sensi)
        stats_model, plot_diag = Poisson_Model_sensi, plot_diagnostics_poisson
    
        true_hyp_dict = {
            # hyperparameter for beta (mean)
            "mus": [2.91, 0.23, -1.51, -0.610],
            # hyperparameter for beta (sd)
            "sigmas": tf.math.log([0.07, 0.05, 0.135, 0.105]),
            # hyperparameter for sigma
            "lambda0": None,
            "sigma_taus": None,
            "alpha_LKJ": None
        }
        X_idx = [1, 11, 17, 22, 35, 44]
        threshold = 30
        X = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/antidis_laws.csv')
        sigma_prior = None
        model_specific = None
        learning_hyp = {
            "epochs": 300,
            "B": 2**8,
            "rep_exp": 1000,
            "rep_mod": 150,
            "lr0": 0.1,
            "lr_min": 0.0001,
            "lr_decay": 0.95,
            "lr_decay_step": 7
            }
        loss_hyp = {
            "names":       ["y_obs", "y_groups", "R2"],
            "format_type": ["hist","quantiles","quantiles"],
            "tensor_type": ["sliced", "sliced", "unsliced"],
            "format_meth": ["summary", "summary", "summary"],
            }
        
    if selected_model == "mlm":
        stats_model, plot_diag = MLM_Model, plot_diagnostics_mlm
    
        true_hyp_dict = {
            # tf.reduce_mean(tfd.Exponential(0.04).sample(10000))
            "lambda0":tf.math.log(0.04),
            "mus": [250.40, 30.26],
            "sigmas": tf.math.log([7.27, 4.82]),
            # tf.reduce_mean(tfd.TruncatedNormal(0.,33.,0.,500.).sample(10000))v
            "sigma_taus": tf.math.log([33., 23.]), # tau <- 26.3, 18.3
            "alpha_LKJ": 1.
        }
        X_idx = [0,2,4,6,8]
        threshold = None
        sigma_prior = "gamma-avg"
        model_specific = {
            "N_sim": 100,
            "N_exp": 600,
            "N_days": 10
        }
        X = None
        learning_hyp = {
            "epochs": 800,
            "B": 2**8,
            "rep_exp": 200,
            "rep_mod": 200,
            "lr0": 0.1,
            "lr_min": 0.001,
            "lr_decay": 0.95,
            "lr_decay_step": 7
            }
        loss_hyp = {
            "names":       ["days_sd", "days", "target_sd_day0","sd_mu_day0", "target_sd_day1", "sd_mu_day1"],
            "format_type": ["moments","quantiles","model_only","model_only","model_only","model_only"],
            "tensor_type": ["unsliced", "sliced", "model_only", "model_only", "model_only", "model_only"],
            "format_meth": ["summary", "summary","summary", "summary","summary", "summary"]
            }
    
    if selected_model == "weibull":
        stats_model, plot_diag = MLM_weibull_Model, plot_diagnostics_mlm
    
        true_hyp_dict = {
            # tf.reduce_mean(tfd.Exponential(0.069).sample(10000))
            "lambda0":tf.math.log(0.069), #shape = 14.68
            "mus": [5.52, 0.10],
            "sigmas": tf.math.log([0.03, 0.02]),
            # tf.reduce_mean(tfd.TruncatedNormal(0.,0.09,0.,500.).sample(10000))
            "sigma_taus": tf.math.log([0.15, 0.09]), # tau <- 0.12, 0.07
            "alpha_LKJ": 1.
        }
        X_idx = [0,2,4,6,8]
        X = None
        threshold = None
        sigma_prior = "gamma-avg"
        model_specific = {
            "N_sim": 200,
            "N_exp": 600,
            "N_days": 10
        } 
        learning_hyp = {
            "epochs": 400,
            "B": 2**8,
            "rep_exp": 200,
            "rep_mod": 200,
            "lr0": 0.1,
            "lr_min": 0.0001,
            "lr_decay": 0.9,
            "lr_decay_step": 7
            }
        loss_hyp = {
            "names":       ["days_sd", "days", "target_sd_day0","sd_mu_day0", "target_sd_day1", "sd_mu_day1"],
            "format_type": ["moments","quantiles","model_only","model_only","model_only","model_only"],
            "tensor_type": ["unsliced", "sliced", "model_only", "model_only", "model_only", "model_only"],
            "format_meth": ["summary", "summary","summary", "summary","summary", "summary"]
            }
        
    return (stats_model, plot_diag, true_hyp_dict,X_idx, threshold, X, sigma_prior,
     model_specific,learning_hyp,loss_hyp) 