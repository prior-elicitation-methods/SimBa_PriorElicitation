import os
from configuration_models import model_specs

def settings(selected_model):
    # import model specific settings
    (stats_model, plot_diag, true_hyp_dict,X_idx, threshold, X,sigma_prior,
    model_specific,learning_hyp,loss_hyp) = model_specs(selected_model)

    #%% global settings
    input_settings_model = {
        # decide on which prior sigma should get: gamma, exponential-avg, exponential
        "sigma_prior": sigma_prior,
        # design matrix
        "X": X,
        # selected design points
        "X_idx": X_idx,
        # upper bound for Poisson model
        "threshold_max": threshold,
        # model hyperparameter
        "hyperparameter": true_hyp_dict,
        # further input information depending on model
        "model_specific": model_specific
    }
    input_settings_learning = {
        # number of epochs
        "epochs": learning_hyp["epochs"],              
        # batch size
        "B": learning_hyp["B"],
        # number of replications
        # for expert model          
        "rep_exp": learning_hyp["rep_exp"],
        # for data simulation model
        "rep_mod": learning_hyp["rep_mod"],
        # initial learning rate
        "lr0": learning_hyp["lr0"],
        # minimal learning rate
        "lr_min": learning_hyp["lr_min"],
        # learning rate decay of exponential decay schedul
        "lr_decay": learning_hyp["lr_decay"],
        # after how many steps next decay is applied to learning rate
        "lr_decay_step": learning_hyp["lr_decay_step"],
        # temperature for DWA (hyperparameter multi-task weighting algo.)
        "a_task_balancing": 1.6,
        # temperature for Gumbel-Softmax-Trick
        "temp": 1.,
        # normalize predictor(s)
        "normalize": True,  
        # show expert sample checks (plots)
        "checks": False
    }
    input_settings_global = {
        # seed: None, value
        "seed": 2023,                         # include set_seed fct in function
        # show learning progress? (1-yes, 0-no)
        "verbose": 1,
        # show loss after how many epochs
        # None = no output, value
        "show_ep": 1,
        # learned parameter is average over x last values (how many last values?)
        "l_values": 30
    }
    input_settings_loss = {
        # name of the simulated predicted data
        "name_sim": loss_hyp["names"],
        # name of loss component (if model_only: [name1, name2])
        "name_loss": [f"{loss_hyp['names'][i]}_loss" for i in range(len(loss_hyp["names"]))],
        # kernel used for optimization: energy, gaussian
        "kernel": ["energy"]*len(loss_hyp["names"]),
        # input format of info by expert: hist, quantiles, moments
        # [loss_format]*len(loss_names),
        "format_type": loss_hyp["format_type"],
        # which method shall be used: None, sampling, summary
        "format_meth": loss_hyp["format_meth"],
        # if format_type = quantile: specify quantiles
        "quantile_list": [[10, 20, 30, 40, 50, 60, 70, 80, 90] for i in range(len(loss_hyp["names"]))],
        # specify tensor type for further internal rearrangement
        # sliced, unsliced, model_only
        "tensor_type": loss_hyp["tensor_type"]
    }

    return (stats_model, plot_diag, input_settings_global, input_settings_learning, 
            input_settings_loss, input_settings_model)