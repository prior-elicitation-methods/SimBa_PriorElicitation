import tensorflow as tf

from helper_functions import loss_list_sliced, loss_list_moments, loss_list_hist_quant, loss_list_mlm

def extract_loss_components(selected_model, elicited_quant_exp,  elicited_quant_sim,
                            input_settings_loss, input_settings_learning, loss_fn, normalize):
    
    losses_all = []
    # compute number of loss components in objective function
    if selected_model == "mlm":
        no_loss_comp = len(input_settings_loss["name_loss"])-4
    else:
        no_loss_comp = len(input_settings_loss["name_loss"])
    
    
    # for each loss component, do:
    for idx in range(no_loss_comp):
        # Is the loss component grouped? (e.g., factor with x groups)  
        # Yes the loss component is grouped
        if input_settings_loss["tensor_type"][idx] == "sliced":
            losses_sliced = loss_list_sliced(elicited_quant_sim, elicited_quant_exp, input_settings_loss, 
                                             input_settings_learning, idx, normalize, loss_fn)
            losses_all.append(tf.concat(losses_sliced,0))

        # No, the loss component is not grouped
        losses_unsliced = []
        if input_settings_loss["tensor_type"][idx] == "unsliced":
            # Which elicitation technique has been used?
            # Moment-based elicitation
            if input_settings_loss["format_type"][idx] == "moments":
                losses_unsliced = loss_list_moments(elicited_quant_sim, elicited_quant_exp, input_settings_loss, 
                                                    input_settings_learning, idx, normalize, loss_fn)
                losses_all.append(tf.concat(losses_unsliced,0))
            
            # Quantile-based elicitation or histogram elicitation
            if (input_settings_loss["format_type"][idx] == "hist" or 
                input_settings_loss["format_type"][idx] == "quantiles"):

                losses_unsliced = loss_list_hist_quant(elicited_quant_sim, elicited_quant_exp, input_settings_loss, 
                                                       input_settings_learning, idx, normalize, loss_fn)     
                losses_all.append(tf.stack(losses_unsliced,0))
    
    # for the multi-level model we train the model with a model-internal variable 
    # therefore we need extra treatment for MLM
    if selected_model == "mlm" or selected_model == "weibull":
        losses_mlm = loss_list_mlm(elicited_quant_sim, input_settings_learning, normalize, loss_fn)
        losses_all.append(tf.stack(losses_mlm,0))

    # concatenate all losses together in order to get the final loss list    
    total_loss = tf.concat(losses_all,0)
    
    return total_loss

