import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import time

from elicitation_techniques import _apply_elicitation_technique

class trainer(tf.keras.Model):
    """
    runs gradient-based optimization and returns learned hyperparameter values
    """
    def __init__(self, selected_model, generative_model, target_quant_exp,  
                 trainer_step, energy_loss, extract_loss_components):
        
        super(trainer, self).__init__()
        
        self.selected_model = selected_model
        self.generative_model = generative_model
        self.target_quant_exp = target_quant_exp
        self.trainer_step = trainer_step
        self.mmd_energy_kernel = energy_loss
        self._extract_loss_components = extract_loss_components
        self._f = _apply_elicitation_technique()
        
    def __call__(self, 
                 input_settings_learning, 
                 input_settings_global,
                 input_settings_loss, 
                 **kwargs):
             
        # transform the target quantities elicited from the expert according to
        # elicitation technique (histogram, quantile-based, or moment-based elicitation)
        elicited_quant_exp = self._f(self.target_quant_exp, 
                                     input_settings_loss, 
                                     input_settings_learning, 
                                     model_type = "expert")  

        # initialize the gradient-descent optimization approach
        training = self.trainer_step(self.generative_model,  self._f) 
    
        # compile training model
        training.compile(loss_fn = self.mmd_energy_kernel,
                         extract_loss_components = self._extract_loss_components)
        
        # run iterative optimization 
        final_time = []
        start_time = time.time()

        (out, var, target_quant_sim, elicited_quant_sim, 
        elicited_quant_sim_ini, weights, time_per_epoch) = training.fit(
                    selected_model = self.selected_model,
                    input_settings_loss = input_settings_loss,
                    input_settings_learning = input_settings_learning,
                    elicited_quant_exp = elicited_quant_exp,
                    epochs = input_settings_learning["epochs"]+1,
                    task_balance_factor = input_settings_learning["a_task_balancing"],
                    custom_weights = None,
                    learned_weights = None,
                    user_weights = False,
                    lr_min = input_settings_learning["lr_min"],
                    normalize=input_settings_learning["normalize"],
                    show_ep = input_settings_global["show_ep"],
                    verbose = input_settings_global["verbose"],
                    lr_initial = input_settings_learning["lr0"],
                    lr_decay_step = input_settings_learning["lr_decay_step"],
                    lr_decay_rate = input_settings_learning["lr_decay"],
                    clip_value = 1.0,
                    **kwargs) 

        final_time = (time.time() - start_time)
    
        return (out, var, target_quant_sim, elicited_quant_sim, 
                elicited_quant_sim_ini, elicited_quant_exp, weights, 
                time_per_epoch, final_time)
    
  
        
    
  
    
    
   