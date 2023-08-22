from loss_components import extract_loss_components
from losses import energy_loss
from trainer import trainer
from trainer_step import trainer_step
from configuration_global import settings


# binom, linreg, pois, mlm, weibull
selected_model = "pois"

# import global and model-specific configurations according to 'selected model'
(prob_model, plot_diag, input_settings_global, input_settings_learning, 
 input_settings_loss, input_settings_model) = settings(selected_model)

# initialize the probabilistic (generative) model
generative_model = prob_model(input_settings_model, input_settings_learning,
                              input_settings_global)

# sample data from generative model representing the information elicited 
# from an ideal expert 
target_quant_exp = generative_model.data_generating_model(
                        mus = input_settings_model["hyperparameter"]["mus"],
                        sigmas = input_settings_model["hyperparameter"]["sigmas"],
                        sigma_taus = input_settings_model["hyperparameter"]["sigma_taus"],
                        alpha_LKJ = input_settings_model["hyperparameter"]["alpha_LKJ"],
                        lambda0 = input_settings_model["hyperparameter"]["lambda0"],
                        input_settings_global = input_settings_global,
                        input_settings_learning = input_settings_learning,
                        input_settings_model = input_settings_model,
                        model_type = "expert")

# Only for inconsistent information
## case 1: double random noise "s"
    #target_quant_exp["days_sd"] = target_quant_exp["days_sd"]*2
## case 2: halve R2  
    #target_quant_exp["R2_0_mod_input"] = target_quant_exp["R2_0_mod_input"]*0.5
    #target_quant_exp["R2_1_mod_input"] = target_quant_exp["R2_1_mod_input"]*0.5

if selected_model == "mlm" or selected_model == "weibull":
    input_settings_model["model_specific"]["R2_0"] = target_quant_exp["R2_0_mod_input"]
    input_settings_model["model_specific"]["R2_1"] = target_quant_exp["R2_1_mod_input"]  
    
# initialize gradient-based optimization
training = trainer(selected_model, generative_model, target_quant_exp, 
                   trainer_step, energy_loss, extract_loss_components)

# run gradient-based optimization in order to learn model hyperparameter 
(out, var, target_quant_sim, elicited_quant_sim,  elicited_quant_sim_ini, elicited_quant_exp, weights, 
 time_per_epoch, final_time) = training(input_settings_learning, 
                                        input_settings_global,  input_settings_loss)
                                                                     

