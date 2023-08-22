import tensorflow as tf

def normalizer(expert_d, model_d):
        x_min = tf.math.reduce_min(model_d)
        x_max = tf.math.reduce_max(model_d)
        expert = (expert_d - x_min)/(x_max-x_min)
        model = (model_d - x_min)/(x_max-x_min)
        return expert, model

def loss_weighting_DWA(epoch, loss_t, loss_tasks, task_balance_factor):
    n_loss = len(loss_t)
    # initialize weights
    if epoch < 2:
        w_t = tf.Variable(tf.ones(n_loss), trainable=False)
    
    # w_t (epoch-1) = L_t (epoch-1) / L_t (epoch-2)
    if epoch > 1:
        w_t = tf.math.divide(
            tf.stack(loss_tasks[-1]), tf.stack(loss_tasks[0]))
    
    # T*exp(w_t(epoch-1)/a)
    numerator = tf.math.multiply(
        n_loss, tf.exp(tf.math.divide(w_t, task_balance_factor)))
    
    # sum_i exp(w_i(epoch-1)/a)
    denominator = tf.math.reduce_sum(numerator)
    
    # softmax operator
    lambda_t = tf.math.divide(numerator, denominator)
    
    # total loss: L = sum_t lambda_t*L_t
    loss = tf.math.reduce_sum(tf.math.multiply(lambda_t, loss_t)) 

    return loss, lambda_t

def lr_schedule(lr, decay_steps, decay_rate):
    return tf.keras.optimizers.schedules.ExponentialDecay(
                lr, decay_steps, decay_rate, 
                staircase = True)

def extract_model_expert(elicited_quant_sim, elicited_quant_exp, input_settings_loss, input_settings_learning, idx):
    if  input_settings_loss["format_type"][idx] == "moments" :
        keys = list(elicited_quant_exp[input_settings_loss["name_loss"][idx]].keys())
        loss_component_expert_m = elicited_quant_exp[input_settings_loss["name_loss"][idx]][keys[0]]
        loss_component_expert_sd = elicited_quant_exp[input_settings_loss["name_loss"][idx]][keys[1]]
        loss_component_model_m = elicited_quant_sim[input_settings_loss["name_loss"][idx]][keys[0]]
        loss_component_model_sd = elicited_quant_sim[input_settings_loss["name_loss"][idx]][keys[1]]
        # retrieve batch size
        B = input_settings_learning["B"]
        # combine components
        model_components = [loss_component_model_m, loss_component_model_sd]
        expert_components = [loss_component_expert_m, loss_component_expert_sd]
    
    else:
        loss_component_expert = elicited_quant_exp[input_settings_loss["name_loss"][idx]]
        loss_component_model = elicited_quant_sim[input_settings_loss["name_loss"][idx]]
        # retrieve batch size
        B = input_settings_learning["B"]
        # combine components
        model_components = [loss_component_model]
        expert_components = [loss_component_expert]

    return (model_components, expert_components, B)

def extract_model_expert_sample(expert, model,group,k):
    # initialize loss of expert and model 
    if tf.rank(expert[k]) == 2:
        expert_sample = expert[k][:,group]
        model_sample = model[k][:,group]
        N = 1
        M = 1
    if tf.rank(expert[k]) == 3:
        expert_sample = expert[k][:,:,group]
        model_sample = model[k][:,:,group]
        N = model_sample.shape[-1]
        M = expert_sample.shape[-1]

    return (expert_sample, model_sample, N, M)

def loss_list_sliced(elicited_quant_sim, elicited_quant_exp, input_settings_loss, input_settings_learning, idx, normalize, loss_fn):
    losses_sliced = []
    (model, expert, B) = extract_model_expert(elicited_quant_sim, elicited_quant_exp, input_settings_loss, 
                                              input_settings_learning, idx)
    # reset for each loss component    
    list_loss = []
    # select group
    for group in range(expert[0].shape[-1]):
        # reset for each group
        list_loss = []
        # select slice
        for k in range(len(expert)):
            (expert_sample, model_sample, N, M) = extract_model_expert_sample(expert, model, group, k)
            # transfrom samples if normalize = True
            if normalize: expert_sample, model_sample = normalizer(expert_sample,  model_sample)
            # compute loss 
            list_loss.append(loss_fn(expert_sample, model_sample, B = B, N = N, M = M))
        losses_sliced.append(list_loss)

    return losses_sliced

def loss_list_moments(elicited_quant_sim, elicited_quant_exp, input_settings_loss, input_settings_learning, idx, normalize, loss_fn):
    keys_mod = list(elicited_quant_exp[input_settings_loss["name_loss"][idx]].keys())
    losses_unsliced = []            
    list_loss2 = []
    for key in keys_mod: 
            # retrieve batch size and sample size
            B = input_settings_learning["B"]
            N = elicited_quant_sim[input_settings_loss["name_loss"][idx]][key].shape[1]
            M = elicited_quant_exp[input_settings_loss["name_loss"][idx]][key].shape[1]
            
            expert_sample = elicited_quant_exp[input_settings_loss["name_loss"][idx]][key]
            model_sample = elicited_quant_sim[input_settings_loss["name_loss"][idx]][key]
    
            # transfrom samples if normalize = True
            if normalize: expert_sample, model_sample = normalizer(expert_sample, model_sample)
            # compute loss 
            list_loss2.append(loss_fn(expert_sample, model_sample, B = B, N = N, M = M))
            losses_unsliced = tf.stack(list_loss2,0)
    
    return losses_unsliced

def loss_list_hist_quant(elicited_quant_sim, elicited_quant_exp, input_settings_loss, input_settings_learning, idx, normalize, loss_fn):
    losses_unsliced = []
    # retrieve batch size and sample size
    B = input_settings_learning["B"]
    N = elicited_quant_sim[input_settings_loss["name_loss"][idx]].shape[1]
    M = elicited_quant_exp[input_settings_loss["name_loss"][idx]].shape[1]
    
    expert_sample = elicited_quant_exp[input_settings_loss["name_loss"][idx]]
    model_sample = elicited_quant_sim[input_settings_loss["name_loss"][idx]]

    # transfrom samples if normalize = True
    if normalize: expert_sample, model_sample = normalizer(expert_sample, model_sample)
    
    # compute loss 
    losses_unsliced.append(loss_fn(expert_sample, model_sample,  
                                    B = B, N = N, M = M)) 
    
    return losses_unsliced

def loss_list_mlm(elicited_quant_sim, input_settings_learning, normalize, loss_fn):
        
    losses_mlm = []
    # retrieve batch size and sample size
    B = input_settings_learning["B"]
    N = M = elicited_quant_sim["target_sd_day0_loss"].shape[1]
    
    exp = ["target_sd_day0_loss", "target_sd_day1_loss"]
    mod = ["sd_mu_day0_loss", "sd_mu_day1_loss"]
    
    for e, m in list(zip(exp, mod)):
        
        expert_sample = elicited_quant_sim[e]
        model_sample = elicited_quant_sim[m]
        
        # transfrom samples if normalize = True
        if normalize: expert_sample, model_sample = normalizer(expert_sample,  model_sample)
            
        # compute loss 
        losses_mlm.append(loss_fn(expert_sample, model_sample, B = B,  N = N, M = M))

    return losses_mlm

def plot_emp_prior(axs, x_range, pdf, label, color):
    return axs.plot(x_range, pdf, label=label, color=color, linewidth=3)
def plot_true_prior(axs, x_range, pdf):
    return axs.plot(x_range, pdf, linestyle = "dotted", color ="black", linewidth=2)
def plot_titles(axs, title, x_pos):
    return axs.set_title(title, fontweight = "bold", fontsize = "large", x = x_pos)
def plot_handles(axs, locs):
    return axs.legend(handlelength=1, loc = locs, ncol=2, labelspacing=0.2, columnspacing=1., fontsize="large")
def plot_alab(axs, label, axis): 
    if axis == "x":
        return axs.set_xlabel(label, labelpad=1)
    else:
        return axs.set_ylabel(label, labelpad=1)