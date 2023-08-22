import tensorflow as tf
import time 

from helper_functions import loss_weighting_DWA, lr_schedule


class trainer_step(tf.keras.Model):
    
    def __init__(self, generative_model, _f):
        super(trainer_step, self).__init__()
        
        self.generative_model = generative_model
        self._f= _f
        
    def compile(self, loss_fn, extract_loss_components):
        super(trainer_step, self).compile()
        self.loss_fn = loss_fn()
        self._extract_loss_components = extract_loss_components
       
    def train_step(self, 
                   selected_model,
                   input_settings_loss,
                   input_settings_learning,
                   elicited_quant_exp, 
                   epoch, 
                   loss_tasks, 
                   task_balance_factor,
                   custom_weights, 
                   learned_weights, 
                   user_weights, 
                   normalize,
                   opt,
                   **kwargs): 
        """     
        Function that specifies computational steps within one iteration in 
        gradient-descent optimization

        1) simulate from generative model > model-implied target quantities
        2) apply elicitation technique > model-implied elicited quantities
        3) compute loss per loss component
        4) apply dynamic weight averaging in multi-objective optimization function and compute total loss
        5) compute gradients of total loss wrt. hyperparameters
        6) update hyperperameters
        """

        with tf.GradientTape() as g:
            
            # simulate target quantities from generative model
            target_quant_sim = self.generative_model(**kwargs)
          
            # compute elicited quantities by applying selected elicitation technique
            elicited_quant_sim = self._f(target_quant_sim, 
                                        input_settings_loss, 
                                        input_settings_learning, 
                                        model_type = "model")
            
            # compute all single loss components and stack them together in one list
            loss_t = self._extract_loss_components(selected_model,
                                          elicited_quant_exp, 
                                          elicited_quant_sim,
                                          input_settings_loss,
                                          input_settings_learning,
                                          self.loss_fn, 
                                          normalize= normalize) 
            
            # custom weights specified by user?
            if not user_weights:
                # dummy var "loss_tasks" for first epoch
                if epoch < 2: loss_tasks = 0
                # compute total loss as weighted sum 
                loss, lambda_t = loss_weighting_DWA(epoch, loss_t, loss_tasks, task_balance_factor)
            
            else: 
                # user specific weights
                lambda_t = learned_weights
                # total loss: L = sum_t lambda_t*L_t
                loss = tf.math.reduce_sum(tf.multiply(custom_weights, tf.math.multiply(lambda_t, loss_t)))
                
        # compute gradients of parameters wrt total loss
        grads = g.gradient(loss, self.generative_model.trainable_variables) 
        
        # update parameters with computed gradient
        opt.apply_gradients(zip(grads, self.generative_model.trainable_variables))
        
        return {"loss":loss, "gradients":grads, "loss_tasks":loss_t, 
                "weights_tasks":lambda_t}, target_quant_sim, elicited_quant_sim
    
    def fit(self,
            selected_model,
            input_settings_loss,
            input_settings_learning,
            elicited_quant_exp,
            epochs,
            task_balance_factor,
            custom_weights, 
            learned_weights, 
            user_weights,
            lr_min,
            normalize,
            show_ep,
            verbose, 
            lr_initial,
            lr_decay_step,
            lr_decay_rate,
            clip_value,
            **kwargs):
        """
        Function that runs gradient-descent optimization until maximum number 
        of epochs is reached

        1) initialize exponential decay learning rate schedule for Adam optimizer
        2) for each epoch: 
            2.1) get learning rate and initialize Adam optimizer
            2.2) run single gradient-descent step 
            2.3) print progress
            2.4) save results
        """
        # for tracking the learned parameter values
        var_names = [v.name[:-2] for v in self.generative_model.trainable_variables] 
        no_var = len(var_names)
        for i in range(no_var): locals()[var_names[i]] = []
        
        # initialize lists
        res, var, loss_tasks, weights_tasks, time_per_epoch = ([] for i in range(5))
        
        # initialize learning rate schedule: exponential decay
        lr_fn = lr_schedule(lr_initial, lr_decay_step, lr_decay_rate) 
        
        for epoch in range(epochs):
            # track time need to finish one epoch
            start_time = time.time()
            # extract learning rate for current epoch
            lr = lr_fn(epoch)
            # if actual learning rate is smaller than min. lr use min. lr
            if lr < lr_min: lr = lr_min
            # define Adam optimizer with clipping by norm and actual lr
            opt = tf.keras.optimizers.Adam(learning_rate = lr, 
                                           clipnorm = clip_value)
            # do single training step
            outputs, target_quant_sim, elicited_quant_sim = self.train_step(
                                                                    selected_model,
                                                                    input_settings_loss,
                                                                    input_settings_learning,
                                                                    elicited_quant_exp, 
                                                                    epoch, 
                                                                    loss_tasks, 
                                                                    task_balance_factor,
                                                                    custom_weights, 
                                                                    learned_weights, 
                                                                    user_weights, 
                                                                    normalize,
                                                                    opt,
                                                                    **kwargs)
            # save for results initial data set 
            if epoch == 0: elicited_quant_sim_ini = elicited_quant_sim
            
            # Log every "show_ep" epochs
            time_e = time.time() - start_time
            if verbose == 1:
                if epoch % show_ep == 0:
                    print(
                        "Total training loss at end of epoch %d: %.4f"
                        % (epoch, float(outputs["loss"])))
                    print("Time per epoch: %.2fs" % (time_e))
                    print("learning rate: %.5f" % lr)
            
            # save results
            time_per_epoch.append(time_e)
            res.append(outputs) 
            # save hyperparameter values
            for i in range(no_var): locals()[var_names[i]].append(self.generative_model.trainable_variables[i].numpy())
            loss_tasks.append(outputs["loss_tasks"])
            weights_tasks.append(outputs["weights_tasks"])
            
        for i in range(no_var): var.append([var_names[i], locals()[var_names[i]]])
        
        return (res, var, target_quant_sim, elicited_quant_sim, 
                elicited_quant_sim_ini, weights_tasks, time_per_epoch)
