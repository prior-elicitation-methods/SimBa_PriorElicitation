import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def _histogram_elicitation(model_type, S, input_settings_learning, n_groups):
    if model_type == "model":
        # return samples without changes
        output = S
    else:
        # broadcast output to align tensor shape with 'simulation model' tensor 
        # shape (B, rep_exp, n_gr) or (B, rep_exp)
        if n_groups == 1:
            # note: purpose of transpose: (rep_mod, 1) -> (1, rep_mod)
            output = tf.broadcast_to(S, shape=(input_settings_learning["B"],
                                            input_settings_learning["rep_exp"]))
        else:
            output = tf.broadcast_to(S,shape=(input_settings_learning["B"],
                                            input_settings_learning["rep_exp"],
                                            n_groups))
    return output

def _quantile_elicitation(model_type, S, i, input_settings_learning, input_settings_loss, n_groups):
    # initialize quantile vector
    quantiles_gr = []
    for n_gr in range(n_groups):
        # extract group if groups exists
        if n_groups > 1:
            S_gr = S[:,:,n_gr]
        else: 
            S_gr = S
        # compute pre-specified quantiles for samples
        quantiles_gr.append(tfp.stats.percentile(x=S_gr, 
                                                    q=input_settings_loss["quantile_list"][i],
                                                    axis = -1))
    # stack quantiles per group into one tensor
    quantiles = tf.stack(quantiles_gr, -1)
    # reshape tensor from expert: (n_qt, 1, n_gr) to (1, n_qt, n_gr)
    # model: (n_qt, B, n_gr) to (B, n_qt, n_gr)
    output = tf.transpose(quantiles, [1,0,2])
    
    if model_type == "expert":
        # broadcast output to align tensor shape with 'simulation model' tensor 
        # shape (B, n_qt, n_gr) or (B, n_qt)
        if n_groups == 1:
            output = tf.broadcast_to(tf.squeeze(output, axis = -1), 
                                        shape=(input_settings_learning["B"],
                                            len(input_settings_loss["quantile_list"][i])))
        else:
            output = tf.broadcast_to(output, 
                                        shape=(input_settings_learning["B"],
                                            len(input_settings_loss["quantile_list"][i]),
                                            n_groups)) 
    return output        
        
def _moment_elicitation(model_type, S, i, input_settings_learning, n_groups):
    # initialize summary-stats vectors
    S_mean_gr = []
    S_sd_gr = []
    for n_gr in range(n_groups):
        # extract group if groups exists
        if n_groups > 1:
            S_gr = S[:,:,n_gr]
        else: 
            S_gr = S
        
        # compute summary statistics (here: mean and sd)
        S_mean_gr.append(tf.reduce_mean(S_gr, -1))
        S_sd_gr.append(tf.math.reduce_std(S_gr, -1))
    
    # stack grouped summary statistics into one tensor
    output = {
        f"mean_{i}": tf.stack(S_mean_gr, -1),
        f"sd_{i}": tf.stack(S_sd_gr, -1)
        }
    
    if model_type == "expert":
        # broadcast output to align tensor shape with 'simulation model' tensor 
        # do this for all summary statistics
        # shape for mean and sd: (B, n_gr) or (B,1)
        
        for stats in ["mean", "sd"]:
            if n_groups == 1:
                output[f"{stats}_{i}"] = tf.broadcast_to(output[f"{stats}_{i}"], 
                                         shape=(input_settings_learning["B"],1))
            else:
                output[f"{stats}_{i}"] = tf.broadcast_to(output[f"{stats}_{i}"], 
                                         shape=(input_settings_learning["B"],
                                                n_groups))  

    return output


class _apply_elicitation_technique(tf.keras.Model):
    def __init__(self):
        super(_apply_elicitation_technique, self).__init__() 
 
    def __call__(self, generated_samples, input_settings_loss, 
                 input_settings_learning, model_type):
        
        # if loss_format_type: quantiles/moments -> compute summary statistics from generated samples 
        # if loss_format: hist -> return input
        elicited_quant = self._f(
                     samples = [generated_samples[input_settings_loss["name_sim"][i]] for i in range(len(input_settings_loss["name_sim"]))], 
                     name_trans_loss = tf.stack([input_settings_loss["name_loss"][i] for i in range(len(input_settings_loss["name_loss"]))],-1), 
                     loss_input_type = tf.stack([input_settings_loss["format_type"][i] for i in range(len(input_settings_loss["format_type"]))],-1), 
                     input_settings_learning = input_settings_learning,
                     input_settings_loss = input_settings_loss,
                     model_type = model_type
                     )
        
        return elicited_quant
    
    def _f(self, samples, name_trans_loss, loss_input_type, 
            input_settings_learning, input_settings_loss,
            model_type):
        """
        function computes summary statistics from generated samples depending 
        on specified loss_input_type
            + hist: no summary statistics are computed (input = output)
            + quantiles: quantiles are computed (output tensor: (B,q,groups))
            + moments: mean and std are computed (output dict: mean_i, sd_i (B,groups))
        whereby i stands for the ith orig_loss component
        """
        # check that loss_input_type is either "hist, quantiles or moments"
        assert [{loss_input_type[i].numpy().decode()}.issubset(
            {"quantiles","hist","moments"}) for i in range(len(samples))], "specified format_type of each loss component must be either hist, quantiles, or moments"
        
        # initialize dict for returning all final loss components
        losses = {}
        
        # for each loss quantity 
        for i, S in enumerate(samples):
            
            if S is not None:
                # detect whether groups exists and extract n_gr if yes
                if tf.rank(S) == 3: n_groups = S.shape[-1]
                if tf.rank(S) == 2: n_groups = 1            # only one iteration of for-loop 
                
                if input_settings_loss["tensor_type"][i] == "model_only":
                    output = S
                
                if loss_input_type[i].numpy().decode() == "hist":
                    output = _histogram_elicitation(model_type, S, input_settings_learning, n_groups)
    
                if loss_input_type[i].numpy().decode() == "quantiles":          
                    output = _quantile_elicitation(model_type, S, i, input_settings_learning, input_settings_loss, n_groups)
                    
                if loss_input_type[i].numpy().decode() == "moments":                    
                    output = _moment_elicitation(model_type, S, i, input_settings_learning, n_groups)
          
                # save all transformed samples
                losses[f"{name_trans_loss[i].numpy().decode()}"] = output        
        
        return losses  

