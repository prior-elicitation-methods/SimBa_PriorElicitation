# -*- coding: utf-8 -*-
"""
Binomial Model
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import patsy as pa
import pandas as pd

class Binomial_Model(tf.Module):
    """ This tf.Module implements a Binomial model with one continuous predictor
    
    Parameters
    ----------
    input_settings_model : dict
        model specific variables (e.g., design matrix, sample size, selected design points)
    input_settings_learning : dict
        learning algorithm parameters (e.g., epochs, batch size, number of simulations, learning rate schedule)
    input_settings_global : dict
        global settings (e.g., seed, verbose of learning algo.)
    """
    def __init__(self, input_settings_model, input_settings_learning, input_settings_global):
        """Initialize input parameters and learnable tf.Variables
        """
        super(Binomial_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        self.mus = tf.Variable(initial_value=tf.random.uniform((2,),0., 1.),
                               trainable = True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([0.1, 0.1]),
                                  trainable = True, name = "sigmas")
    
    def __call__(self):
        """Generates samples from the data-generating model  

        Returns
        -------
        d_samples : dict
            model-implied target quantities

        """
        d_samples = self.data_generating_model(self.mus, self.sigmas, sigma_taus=None,
                                               alpha_LKJ=None, lambda0=None,
                                        input_settings_global = self.input_settings_global,
                                        input_settings_learning = self.input_settings_learning, 
                                        input_settings_model = self.input_settings_model,
                                        model_type = "model")
        return d_samples
        
    
    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model, 
                              model_type): 
        """Generates samples from the data-generating model

        Parameters
        ----------
        mus : int
            means of beta coefficients $\beta_0$ and $\beta_1$.
        sigmas : int
            standard deviation of beta coefficients $\beta_0$ and $\beta_1$.
        sigma_taus : None
            Not applicable for Binomial model.
        alpha_LKJ : None
            Not applicable for Binomial model.
        lambda0 : None
            Not applicable for Binomial model.
        input_settings_global : dict
            global settings (e.g., seed, verbose)
        input_settings_learning : dict
            learning algorithm parameters (e.g., batch size, epochs, number of simulations)
        input_settings_model : dict
            model specific variables (e.g., design matrix, selected design points)
        model_type : string, either "expert" or "model"
            specifies whether samples are generated from the ideal expert model or used for model training

        Returns
        -------
        dict
            simulated target quantities.

        """
        # set seed
        tf.random.set_seed(input_settings_global["seed"])
        
        # initialize variables
        if model_type == "expert":
            rep = input_settings_learning["rep_exp"]
            B = 1
        else:
            rep = input_settings_learning["rep_mod"]
            B = input_settings_learning["B"]
            
        X = input_settings_model["X"] 
        X_idx = input_settings_model["X_idx"]
        temp = input_settings_learning["temp"] 
        size = input_settings_model["model_specific"]["size"]
   
        X["no_axillary_nodes"] = tf.constant(X["no_axillary_nodes"], 
                                             dtype=tf.float32)
        x_sd = tf.math.reduce_std(X["no_axillary_nodes"])
        # scale predictor
        X_scaled = tf.constant(X["no_axillary_nodes"],
                               dtype=tf.float32)/x_sd
        
        #select only data points that were selected from expert
        X_scaled = tf.gather(X_scaled,X_idx)
   
        # reshape predictor 
        X = tf.broadcast_to(tf.constant(X_scaled, tf.float32)[None,None,:], (B,rep,len(X_scaled)))
   
        # sample from priors
        beta0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).sample((B,rep,1))
        beta1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).sample((B,rep,1))
        
        betas = tf.stack([beta0, beta1],-1)
        
        # linear predictor
        mu = beta0 + beta1*X
        # map linear predictor to theta
        theta = tf.sigmoid(mu)
        
        # constant outcome vector (including zero outcome)
        c = tf.ones((B,rep,1,size+1))*tf.cast(tf.range(0,size+1), tf.float32)
        # compute pmf value
        pi = tfd.Binomial(total_count=size, probs=theta[:,:,:,None]).prob(c) 
        # prevent underflow
        pi = tf.where(pi < 1.8*10**(-30), 1.8*10**(-30), pi)
        ## sample n-dimensional one-hot-like
        # Gumbel-Max Softmax trick
        # sample from uniform
        u = tfd.Uniform(0,1).sample((B,rep,X.shape[-1],size+1))
        # generate a gumbel sample
        g = -tf.math.log(-tf.math.log(u))
        # softmax trick
        w = tf.nn.softmax(tf.math.divide(tf.math.add(tf.math.log(pi),g),temp))
       
        # reparameterization/ apply linear transformation
        # shape: (B, rep, len(X_idx))
        y_idx = tf.reduce_sum(tf.multiply(w,c), -1)  
        
        # R2
        R2 = tf.math.reduce_variance(mu, -1)/tf.math.reduce_variance(y_idx, -1)
        
        return {"y_idx":y_idx,
                "R2": R2,
                "betas": tf.squeeze(betas)}

class LM_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning, 
                 input_settings_global): 
        super(LM_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.learning_settings = input_settings_learning
        self.model_settings = input_settings_model
        
        self.mus = tf.Variable(initial_value= tf.random.uniform((6,), 0., 1.), 
                               trainable=True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log(tf.random.uniform((6,), 0., 1.)),
                                  trainable=True, name = "sigmas") 
        self.lambda0 = tf.Variable(initial_value=tf.math.log(tf.random.uniform((1,), 0., 1.)), 
                                   trainable=True, name="lambda0") 
    
    def __call__(self):
        
        d_samples =  self.data_generating_model(self.mus, self.sigmas,  sigma_taus=None,
                                               alpha_LKJ=None, lambda0=self.lambda0, 
                                   input_settings_global = self.input_settings_global,
                                   input_settings_learning = self.learning_settings, 
                                   input_settings_model = self.model_settings,
                                   model_type = "model")
        
        return d_samples

    def data_generating_model(self, mus, sigmas,  sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model, 
                              model_type):  
        # set seed
        #if input_settings_global["seed"] is not None:
        tf.random.set_seed(input_settings_global["seed"])
        
        # initialize variables
        if model_type == "expert":
            rep = input_settings_learning["rep_exp"]
            B = 1
        else:
            rep = input_settings_learning["rep_mod"]
            B = input_settings_learning["B"]
        
        # initialize hyperparameter for learning
        Nobs_cell = input_settings_model["model_specific"]["Nobs_cell"] 
        fct_a_lvl = input_settings_model["model_specific"]["fct_a_lvl"] 
        fct_b_lvl= input_settings_model["model_specific"]["fct_b_lvl"]
        
        # number of design cells
        no_cells = fct_a_lvl*fct_b_lvl
        
        # design matrix
        X_design = tf.constant(pa.dmatrix("a*b", pa.balanced(
            a=fct_a_lvl, b=fct_b_lvl, repeat=Nobs_cell)), dtype = tf.float32)
        
        model = tfd.JointDistributionNamed(dict(
            ## sample from priors
            beta_int = tfd.Normal(loc=mus[0], scale=tf.exp(sigmas[0])), 
            beta_a2 = tfd.Normal(loc=mus[1], scale=tf.exp(sigmas[1])), 
            beta_b2 = tfd.Normal(loc=mus[2], scale=tf.exp(sigmas[2])), 
            beta_b3 = tfd.Normal(loc=mus[3], scale=tf.exp(sigmas[3])), 
            beta_a2b2 = tfd.Normal(loc=mus[4], scale=tf.exp(sigmas[4])), 
            beta_a2b3 = tfd.Normal(loc=mus[5], scale=tf.exp(sigmas[5]))
            )).sample((B,rep))
        
        sigma = tfd.Gamma(concentration=rep, 
                          rate=(rep*tf.exp(lambda0))+0.0).sample((B,rep))
           
       
        # organize betas into vector
        betas = tf.stack([model["beta_int"],model["beta_a2"],model["beta_b2"],
                          model["beta_b3"],model["beta_a2b2"],model["beta_a2b3"]],
                          axis = -1)  
       
        # linear predictor
        mu = tf.squeeze(tf.matmul(X_design[None,None,:,:], 
                        tf.expand_dims(betas,2), 
                        transpose_b=True), axis = -1)
        
        # observations
        y_obs = tfd.Normal(mu, sigma).sample()
        
        ## expected data: joints
        # a1_b1, a1_b2, a1_b3, a2_b1, a2_b2, a2_b3
        obs_joints_ind = tf.stack([y_obs[:,:,i::no_cells] for i in range(no_cells)],-1)
        # avg. across individuals 
        obs_joints = tf.reduce_mean(obs_joints_ind, axis = 2)
        
        # grand mean
        gm = tf.reduce_mean(obs_joints, -1)
        ## marginals
        # marginal factor with 3 levels
        mb = obs_joints[:,:,0:3] + obs_joints[:,:,3:7]
        # marginal factor with 2 levels
        ma = tf.stack([tf.reduce_sum(obs_joints[:,:,0:3],-1),
                       tf.reduce_sum(obs_joints[:,:,3:7],-1)],-1)
        
        ## effects
        # effect of factor mb for each level of ma
        effects1 = tf.stack([obs_joints[:,:,i]-obs_joints[:,:,j] for i,j in zip(range(3,6),range(0,3))] ,-1)
        effects2 = tf.stack([obs_joints[:,:,i]-obs_joints[:,:,j] for i,j in zip([1,1,4,4],[0,2,3,5])] ,-1)
        ## R2
        R2 = tf.divide(tf.math.reduce_variance(mu,2), tf.math.reduce_variance(y_obs,2))
             
        return {"mb":mb, 
                "ma":ma,
                "gm":gm,
                "effects1":effects1, 
                "effects2":effects2,
                "model": model,
                "R2": R2,  
                "y_obs": y_obs, 
                "mu":mu, 
                "obs_joints":obs_joints,
                "betas": betas}

class Poisson_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning,
                 input_settings_global): 
        super(Poisson_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        self.mus = tf.Variable(initial_value=[0., 0., 0., 0.], 
                               trainable = True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([1., 1., 1., 1.]) , 
                                  trainable = True, name = "sigmas")
    
    def __call__(self):
        
        d_samples = self.data_generating_model(self.mus, self.sigmas, 
                                        sigma_taus=None,alpha_LKJ=None,lambda0 = None,
                                        input_settings_global = self.input_settings_global,
                                        input_settings_learning= self.input_settings_learning, 
                                        input_settings_model= self.input_settings_model,
                                        model_type = "model")
        
        return d_samples
        
    
    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model,
                              model_type):
        
         # set seed
         tf.random.set_seed(input_settings_global["seed"])
        
         # initialize variables
         if model_type == "expert":
            rep = input_settings_learning["rep_exp"]
            B = 1
         else:
            rep = input_settings_learning["rep_mod"]
            B = input_settings_learning["B"]
        
         X_design = input_settings_model["X"]
         idx = input_settings_model["X_idx"]
         max_number = input_settings_model["threshold_max"]
         temp = input_settings_learning["temp"]
         
         # sort by group and perc_urban in decreasing order
         df = pd.DataFrame(tf.squeeze(X_design)).sort_values(by=[2,3,1])
         # standardize metric predictor
         df[1] = (df[1] - df[1].mean())/df[1].std() 
         # reshape model matrix and create tensor
         X_model = tf.cast(tf.gather(df, idx), tf.float32)
         
         # sample from priors
         # intercept
         beta0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).sample((B,rep,1))
         # percent_urban (metric predictor)
         beta1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).sample((B,rep,1))
         # historical votings GOP vs. Dem
         beta2 = tfd.Normal(mus[2], tf.exp(sigmas[2])).sample((B,rep,1))
         # historical votings Swing vs. Dem
         beta3 = tfd.Normal(mus[3], tf.exp(sigmas[3])).sample((B,rep,1))
         
         # linear predictor
         betas = tf.stack([beta0, beta1, beta2, beta3], -1)
         mu = tf.exp(tf.matmul(X_model, betas, transpose_b=True))
         
         # compute N_obs
         N = len(idx)
         # constant outcome vector
         c = tf.ones((B,rep,1,max_number))*tf.cast(tf.range(0,max_number), tf.float32)
         # compute pmf value
         pi = tfd.Poisson(rate=mu).prob(c)
         # prevent zero value (causes inf for log)
         pi = tf.where(pi < 1.8*10**(-30), 1.8*10**(-30), pi)
         ## sample n-dimensional one-hot-like
         # Gumbel-Max Softmax trick
         # sample from uniform
         u = tfd.Uniform(0,1).sample((B,rep,N,max_number))
         # generate a gumbel sample
         g = -tf.math.log(-tf.math.log(u))
         # softmax trick
         w  = tf.nn.softmax(tf.math.divide(tf.math.add(tf.math.log(pi),g),temp))
         
         # apply transformation
         y_obs = tf.reduce_sum(tf.multiply(w,c), -1)    
         
         # select groups
         # shape = (B,N,n_gr,N_obs)
         y_obs_gr = tf.stack([y_obs[:,:,i:j] for i,j in zip([0,2,4], [2,4,6])], axis=2)
         
         # combine groups (avg. over N_obs)
         # shape: (B,rep)
         y_groups = tf.reduce_mean(y_obs_gr,-1)
        
         # R2
         R2 = tf.divide(tf.math.reduce_variance(tf.squeeze(mu, axis=-1),2),
                         tf.math.reduce_variance(y_obs,2))
         
         return {"y_groups": y_groups,
                 "y_obs": y_obs,
                 "R2": R2
                 }
     
class MLM_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning, 
                 input_settings_global): 
        super(MLM_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        # model coefficients
        self.mus = tf.Variable(initial_value=[240., 20.], trainable=True, name = "mus") #[200., 10.]
        self.sigmas = tf.Variable(initial_value=tf.math.log([10., 1.]), trainable=True, name = "sigmas")
        # random effects
        self.sigma_taus = tf.Variable(initial_value=tf.math.log([25., 15.]), trainable=True, name = "sigma_taus") 
        # param for correlation prior (between random effects)
        self.alpha_LKJ = tf.Variable(initial_value=1., trainable=False, name="alpha_LKJ")
        # param for random noise prior
        self.lambda0 = tf.Variable(initial_value=tf.math.log(0.1), trainable=True, name="lambda0") #0.028
        
    def __call__(self):
        
        d_samples =  self.data_generating_model(self.mus, self.sigmas, self.sigma_taus, 
                                                self.alpha_LKJ, self.lambda0, 
                                                input_settings_learning = self.input_settings_learning, 
                                                input_settings_model = self.input_settings_model, 
                                                input_settings_global = self.input_settings_global, 
                                                model_type = "model", 
                                                expert_R2_0= self.input_settings_model["model_specific"]["R2_0"], 
                                                expert_R2_1= self.input_settings_model["model_specific"]["R2_1"])
                
        return d_samples


    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_learning, input_settings_model,
                              input_settings_global, model_type, 
                              expert_R2_0 = None, expert_R2_1 = None):
        
        # set seed
        tf.random.set_seed(input_settings_global["seed"])
      
        # initialize variables
        if model_type == "expert":
          rep = input_settings_learning["rep_exp"]
          B = 1
          rep_mod = input_settings_learning["rep_mod"]
          N_subj = input_settings_model["model_specific"]["N_exp"]
        else:
          rep = input_settings_learning["rep_mod"]
          B = input_settings_learning["B"]
          N_subj = input_settings_model["model_specific"]["N_sim"]
        
        X_days = tf.cast(tf.tile(tf.range(0., 10, 1.), [N_subj]), tf.float32)
        idx = list(input_settings_model["X_idx"])
        N_days = input_settings_model["model_specific"]["N_days"]
        sd_x = tf.math.reduce_std(X_days)
        Z_days = (X_days)/sd_x
        
        # model
        model = tfd.JointDistributionNamed(dict(
            ## sample from priors
            # fixed effects: beta0, beta1
            beta0 = tfd.Sample(tfd.Normal(loc=mus[0], scale=tf.exp(sigmas[0])), rep),
            beta1 = tfd.Sample(tfd.Normal(loc=mus[1], scale=tf.exp(sigmas[1])), rep),
            # sd random effects: tau0, tau1
            tau0 = tfd.Sample(tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_taus[0]), low=0., high=500), (rep,rep)), 
            tau1 = tfd.Sample(tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_taus[1]), low=0., high=500), (rep,rep)),
            # LKJ prior; compute corr matrix: rho01
            corr_matrix = tfd.Sample(tfd.LKJ(2, alpha_LKJ),rep)
            )).sample(B)
        
        # sd random noise: sigma
        sigma = tfd.Gamma(concentration=rep, rate=rep*tf.exp(lambda0)).sample((B,rep,1))

        # broadcast sigma to the needed shape
        sigma_m = tf.squeeze(sigma, -1)
        sigma = tf.broadcast_to(sigma, (B,rep,N_subj*N_days))
        
        ## compute covariance matrix
        tau0_m = tf.reduce_mean(model["tau0"],(1,2))
        tau1_m = tf.reduce_mean(model["tau1"],(1,2))
        # SD matrix
        S = tf.linalg.diag(tf.stack([tau0_m, tau1_m], -1))
        # covariance matrix: Cov=S*R*S
        # for stability
        corr_mat = tf.linalg.diag(diagonal=(1.,1.), padding_value=tf.reduce_mean(model["corr_matrix"]))
        # compute cov mat
        model["cov_mx_subj"] = tf.matmul(tf.matmul(S,corr_mat),S)
        
        # generate by-subject random effects: T0s, T1s
        model["subj_rfx"] = tfd.Sample(tfd.MultivariateNormalTriL(loc= [0,0], 
                      scale_tril=tf.linalg.cholesky(model["cov_mx_subj"])), N_subj).sample()
        
        ## broadcast by-subject rfx and betas to needed shape
        model["T0s"] = tf.reshape(tf.broadcast_to(model["subj_rfx"][:,:,0,None], 
                                                  (B, N_subj, N_days)),  (B,N_subj*N_days))
        model["T1s"] = tf.reshape(tf.broadcast_to(model["subj_rfx"][:,:,1,None], 
                                                  (B,N_subj, N_days)),  (B,N_subj*N_days))
        model["beta0_b"] = tf.broadcast_to(model["beta0"][:,:,None], (B,rep,N_subj*N_days))
        model["beta1_b"] = tf.broadcast_to(model["beta1"][:,:,None], (B,rep,N_subj*N_days))
        
        ## compute mu_s
        # beta_days = beta_1 + T_1s 
        model["beta_days"] = tf.add(model["beta1_b"], tf.expand_dims(model["T1s"],1)) 
        # beta_intercept = beta_0 + T_0s 
        model["beta_intercept"] = tf.add(model["beta0_b"], tf.expand_dims(model["T0s"],1)) 
        
        # mu_s = beta_intercept + beta_days*Z_days
        model["mu"] = model["beta_intercept"] + model["beta_days"] * Z_days
        
        ## sample observed data
        # y_obs ~ Normal(mu, sigma)
        model["y_obs"] = tfd.Normal(model["mu"], sigma).sample()
                
        ## Transform observed data for training and mapping expert information to model 
        # reshape linear predictor from (B,rep,N_obs) to (B,rep,N_subj, N_days)
        mu_m = tf.stack([model["mu"][:,:,i::N_days] for i in range(N_days)], -1)
        # reshape observed data from (B,rep,N_obs) to (B,rep,N_subj, N_days)
        y_m = tf.stack([model["y_obs"][:,:,i::N_days] for i in range(N_days)], -1)
        
        # predictive dist days
        days = tf.reduce_mean(tf.gather(mu_m, indices=idx, axis=3), 2)
                
        # compute R2 for day0 and day9
        R2_0 = tf.divide((tf.math.reduce_variance(model["mu"][:,:,0::N_days],-1)), 
                         tf.math.reduce_variance(model["y_obs"][:,:,0::N_days],-1))
        
        R2_1 = tf.divide(tf.math.reduce_variance(model["mu"][:,:,9::N_days],-1), 
                         tf.math.reduce_variance(model["y_obs"][:,:,9::N_days],-1))
        
        if model_type == "expert":
            
            # broadcast to model tensor shape for input as argument
            R2_0_mod_input = tf.broadcast_to(R2_0[:,0:rep_mod], (B,rep_mod))
            R2_1_mod_input = tf.broadcast_to(R2_1[:,0:rep_mod], (B,rep_mod))
            
            #only to prevent error caused due to output
            expert_sdmu_day0 = tf.sqrt(tf.multiply(R2_0, tf.math.reduce_variance(model["y_obs"][:,:,0::N_days],-1)))
            actual_sdmu_day0 = tf.math.reduce_std(model["mu"][:,:,0::N_days],-1)
            actual_sdmu_day1 = tf.sqrt(tf.multiply(R2_1, tf.math.reduce_variance(model["y_obs"][:,:,9::N_days],-1)))
            expert_sdmu_day1 = tf.math.reduce_std(model["mu"][:,:,9::N_days],-1)
        
        if model_type == "model":
            ## day 0
            # compute sd(mu) = sqrt( R2_expert * var(y) ) ("expert part" in loss)
            expert_sdmu_day0 = tf.sqrt(tf.multiply(expert_R2_0, tf.math.reduce_variance(model["y_obs"][:,:,0::N_days],-1)))     
            # compute sd(mu) from current simulation ("model part" in loss)
            actual_sdmu_day0 = tf.math.reduce_std(model["mu"][:,:,0::N_days],-1)
            
            ## day9
            # compute sd(mu) = sqrt( R2_expert * var(y) ) ("expert part" in loss)
            expert_sdmu_day1 = tf.sqrt(tf.multiply(expert_R2_1, tf.math.reduce_variance(model["y_obs"][:,:,9::N_days],-1)))      
            # compute sd of linear predictor (tau0) ("model part" in loss)
            actual_sdmu_day1 = tf.math.reduce_std(model["mu"][:,:,9::N_days],-1)
            
            # only to prevent error caused due to output
            R2_0_mod_input = R2_1_mod_input = None
        
        return {"days":days, "days_sd": sigma_m, "R2_0": R2_0, "R2_1": R2_1,
                "R2_0_mod_input": R2_0_mod_input, "R2_1_mod_input": R2_1_mod_input,
                "target_sd_day0":expert_sdmu_day0, "sd_mu_day0":actual_sdmu_day0,
                "target_sd_day1":expert_sdmu_day1, "sd_mu_day1":actual_sdmu_day1,
                "tau0":tau0_m,"tau1":tau1_m,"model":model
                } 

class MLM_weibull_Model(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning, 
                 input_settings_global): 
        super(MLM_weibull_Model, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        # model coefficients
        self.mus = tf.Variable(initial_value=[4., 0.01], trainable=True, name = "mus") #[200., 10.]
        self.sigmas = tf.Variable(initial_value=tf.math.log([0.01, 0.01]), trainable=True, name = "sigmas")
        # random effects
        self.sigma_taus = tf.Variable(initial_value=tf.math.log([0.02, 0.02]), trainable=True, name = "sigma_taus") 
        # param for correlation prior (between random effects)
        self.alpha_LKJ = tf.Variable(initial_value=1., trainable=False, name="alpha_LKJ")
        # param for random noise prior
        self.lambda0 = tf.Variable(initial_value=tf.math.log(0.04), trainable=True, name="lambda0") #0.028
        
    def __call__(self):
        
        d_samples =  self.data_generating_model(self.mus, self.sigmas, self.sigma_taus, 
                                                self.alpha_LKJ, self.lambda0, 
                                                input_settings_learning = self.input_settings_learning, 
                                                input_settings_model = self.input_settings_model, 
                                                input_settings_global = self.input_settings_global, 
                                                model_type = "model", 
                                                expert_R2_0= self.input_settings_model["model_specific"]["R2_0"], 
                                                expert_R2_1= self.input_settings_model["model_specific"]["R2_1"]
                                                )
                
        return d_samples


    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_learning, input_settings_model,
                              input_settings_global, model_type, 
                              expert_R2_0 = None, expert_R2_1 = None):
        
        # set seed
        tf.random.set_seed(input_settings_global["seed"])
      
        # initialize variables
        if model_type == "expert":
          rep = input_settings_learning["rep_exp"]
          B = 1
          rep_mod = input_settings_learning["rep_mod"]
          N_subj = input_settings_model["model_specific"]["N_exp"]
        else:
          rep = input_settings_learning["rep_mod"]
          B = input_settings_learning["B"]
          N_subj = input_settings_model["model_specific"]["N_sim"]
        
        X_days = tf.cast(tf.tile(tf.range(0., 10, 1.), [N_subj]), tf.float32)
        idx = list(input_settings_model["X_idx"])
        N_days = input_settings_model["model_specific"]["N_days"]
        sd_x = tf.math.reduce_std(X_days)
        Z_days = (X_days)/sd_x
        
        # model
        model = tfd.JointDistributionNamed(dict(
            ## sample from priors
            # fixed effects: beta0, beta1
            beta0 = tfd.Sample(tfd.Normal(loc=mus[0], scale=tf.exp(sigmas[0])), rep),
            beta1 = tfd.Sample(tfd.Normal(loc=mus[1], scale=tf.exp(sigmas[1])), rep),
            # sd random effects: tau0, tau1
            tau0 = tfd.Sample(tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_taus[0]), low=0., high=500), (rep,rep)), 
            tau1 = tfd.Sample(tfd.TruncatedNormal(loc=0., scale=tf.exp(sigma_taus[1]), low=0., high=500), (rep,rep)),
            # LKJ prior; compute corr matrix: rho01
            corr_matrix = tfd.Sample(tfd.LKJ(2, alpha_LKJ),rep)
            )).sample(B)
        
        # sd random noise: sigma
        shape = tfd.Gamma(concentration=rep, rate=rep*tf.exp(lambda0)).sample((B,rep,1))
        shape_m = tf.reduce_mean(shape, -1)
        # broadcast sigma to the needed shape
        shape = tf.broadcast_to(shape, (B,rep,N_subj*N_days))
        
        ## compute covariance matrix
        tau0_m = tf.reduce_mean(model["tau0"],(1,2))
        tau1_m = tf.reduce_mean(model["tau1"],(1,2))
        # SD matrix
        S = tf.linalg.diag(tf.stack([tau0_m, tau1_m], -1))
        # covariance matrix: Cov=S*R*S
        # for stability
        corr_mat = tf.linalg.diag(diagonal=(1.,1.), padding_value=tf.reduce_mean(model["corr_matrix"]))
        # compute cov mat
        model["cov_mx_subj"] = tf.matmul(tf.matmul(S,corr_mat),S)
        
        # generate by-subject random effects: T0s, T1s
        model["subj_rfx"] = tfd.Sample(tfd.MultivariateNormalTriL(loc= [0,0], 
                      scale_tril=tf.linalg.cholesky(model["cov_mx_subj"])), N_subj).sample()
        
        ## broadcast by-subject rfx and betas to needed shape
        model["T0s"] = tf.reshape(tf.broadcast_to(model["subj_rfx"][:,:,0,None], 
                                                  (B, N_subj, N_days)),  (B,N_subj*N_days))
        model["T1s"] = tf.reshape(tf.broadcast_to(model["subj_rfx"][:,:,1,None], 
                                                  (B,N_subj, N_days)),  (B,N_subj*N_days))
        model["beta0_b"] = tf.broadcast_to(model["beta0"][:,:,None], (B,rep,N_subj*N_days))
        model["beta1_b"] = tf.broadcast_to(model["beta1"][:,:,None], (B,rep,N_subj*N_days))
        
        ## compute mu_s
        # beta_days = beta_1 + T_1s 
        model["beta_days"] = tf.add(model["beta1_b"], tf.expand_dims(model["T1s"],1)) 
        # beta_intercept = beta_0 + T_0s 
        model["beta_intercept"] = tf.add(model["beta0_b"], tf.expand_dims(model["T0s"],1)) 
        
        # mu_s = exp(beta_intercept + beta_days*Z_days)
        model["mu"] = tf.exp(model["beta_intercept"] + model["beta_days"] * Z_days)
        # scale = mu / Gamma(1+1/shape) = log(mu) - logGamma(1+1/shape)
        scale = model["mu"] - tf.math.lgamma(1+1/shape)
        # compute sigma for loss component
        sigma = tf.sqrt(tf.stop_gradient(scale**2) * (tf.exp(tf.math.lgamma(1+2/shape)) - (tf.exp(tf.math.lgamma(1+1/shape)))**2))
        sigma_m = tf.reduce_mean(sigma, -1)
        ## sample observed data
        # y_obs ~ Weibull(concentration, scale)
        model["y_obs"] = tfd.Weibull(concentration = shape, scale = scale).sample()
                
        ## Transform observed data for training and mapping expert information to model 
        # reshape linear predictor from (B,rep,N_obs) to (B,rep,N_subj, N_days)
        mu_m = tf.stack([model["mu"][:,:,i::N_days] for i in range(N_days)], -1)
        # reshape observed data from (B,rep,N_obs) to (B,rep,N_subj, N_days)
        y_m = tf.stack([model["y_obs"][:,:,i::N_days] for i in range(N_days)], -1)
        
        # predictive dist days
        days = tf.reduce_mean(tf.gather(mu_m, indices=idx, axis=3), 2)
                
        # compute R2 for day0 and day9
        R2_0 = tf.divide((tf.math.reduce_variance(model["mu"][:,:,0::N_days],-1)), 
                         tf.math.reduce_variance(model["y_obs"][:,:,0::N_days],-1))
        
        R2_1 = tf.divide(tf.math.reduce_variance(model["mu"][:,:,9::N_days],-1), 
                         tf.math.reduce_variance(model["y_obs"][:,:,9::N_days],-1))
        
        if model_type == "expert":
            
            # broadcast to model tensor shape for input as argument
            R2_0_mod_input = tf.broadcast_to(R2_0[:,0:rep_mod], (B,rep_mod))
            R2_1_mod_input = tf.broadcast_to(R2_1[:,0:rep_mod], (B,rep_mod))
            
            #only to prevent error caused due to output
            expert_sdmu_day0 = tf.sqrt(tf.multiply(R2_0, tf.math.reduce_variance(model["y_obs"][:,:,0::N_days],-1)))
            actual_sdmu_day0 = tf.math.reduce_std(model["mu"][:,:,0::N_days],-1)
            actual_sdmu_day1 = tf.sqrt(tf.multiply(R2_1, tf.math.reduce_variance(model["y_obs"][:,:,9::N_days],-1)))
            expert_sdmu_day1 = tf.math.reduce_std(model["mu"][:,:,9::N_days],-1)
            
        
        if model_type == "model": 
            ## day 0
            # compute sd(mu) = sqrt( R2_expert * var(y) ) ("expert part" in loss)
            expert_sdmu_day0 = tf.sqrt(tf.multiply(expert_R2_0, tf.math.reduce_variance(model["y_obs"][:,:,0::N_days],-1)))     
            # compute sd(mu) from current simulation ("model part" in loss)
            actual_sdmu_day0 = tf.math.reduce_std(model["mu"][:,:,0::N_days],-1)
            
            ## day9
            # compute sd(mu) = sqrt( R2_expert * var(y) ) ("expert part" in loss)
            expert_sdmu_day1 = tf.sqrt(tf.multiply(expert_R2_1, tf.math.reduce_variance(model["y_obs"][:,:,9::N_days],-1)))      
            # compute sd of linear predictor (tau0) ("model part" in loss)
            actual_sdmu_day1 = tf.math.reduce_std(model["mu"][:,:,9::N_days],-1)
            
            # only to prevent error caused due to output
            R2_0_mod_input = R2_1_mod_input = None
        
        return {"days":days, "days_sd": sigma_m, "R2_0": R2_0, "R2_1": R2_1,
                "R2_0_mod_input": R2_0_mod_input, "R2_1_mod_input": R2_1_mod_input,
                "target_sd_day0":expert_sdmu_day0, "sd_mu_day0":actual_sdmu_day0,
                "target_sd_day1":expert_sdmu_day1, "sd_mu_day1":actual_sdmu_day1,
                "tau0":tau0_m,"tau1":tau1_m,"model":model
                } 

class Poisson_Model_sensi(tf.Module):
    def __init__(self, input_settings_model, input_settings_learning,
                 input_settings_global): 
        super(Poisson_Model_sensi, self).__init__()
        tf.random.set_seed(input_settings_global["seed"])
        
        self.input_settings_global = input_settings_global
        self.input_settings_learning = input_settings_learning
        self.input_settings_model = input_settings_model
        
        self.mus = tf.Variable(initial_value=[0., 0., 0., 0.], 
                               trainable = True, name = "mus")
        self.sigmas = tf.Variable(initial_value=tf.math.log([1., 1., 1., 1.]) , 
                                  trainable = True, name = "sigmas")
    
    def __call__(self):
        
        d_samples = self.data_generating_model(self.mus, self.sigmas, 
                                        sigma_taus=None,alpha_LKJ=None,lambda0 = None,
                                        input_settings_global = self.input_settings_global,
                                        input_settings_learning= self.input_settings_learning, 
                                        input_settings_model= self.input_settings_model,
                                        model_type = "model")
        
        return d_samples
        
    
    def data_generating_model(self, mus, sigmas, sigma_taus, alpha_LKJ, lambda0, 
                              input_settings_global,
                              input_settings_learning, 
                              input_settings_model,
                              model_type):
        
         # set seed
         tf.random.set_seed(input_settings_global["seed"])
        
         # initialize variables
         if model_type == "expert":
            rep = input_settings_learning["rep_exp"]
            B = 1
         else:
            rep = input_settings_learning["rep_mod"]
            B = input_settings_learning["B"]
        
         X_design = input_settings_model["X"]
         idx = input_settings_model["X_idx"]
         max_number = input_settings_model["threshold_max"]
         temp = input_settings_learning["temp"]
         
         # sort by group and perc_urban in decreasing order
         df = pd.DataFrame(tf.squeeze(X_design)).sort_values(by=[2,3,1])
         # standardize metric predictor
         df[1] = (df[1] - df[1].mean())/df[1].std() 
         # reshape model matrix and create tensor
         X_model = tf.cast(df, tf.float32)
         
         # sample from priors
         # intercept
         beta0 = tfd.Normal(mus[0], tf.exp(sigmas[0])).sample((B,rep,1))
         # percent_urban (metric predictor)
         beta1 = tfd.Normal(mus[1], tf.exp(sigmas[1])).sample((B,rep,1))
         # historical votings GOP vs. Dem
         beta2 = tfd.Normal(mus[2], tf.exp(sigmas[2])).sample((B,rep,1))
         # historical votings Swing vs. Dem
         beta3 = tfd.Normal(mus[3], tf.exp(sigmas[3])).sample((B,rep,1))
         
         # linear predictor
         betas = tf.stack([beta0, beta1, beta2, beta3], -1)
         mu = tf.exp(tf.matmul(X_model, betas, transpose_b=True))
         
         # compute N_obs
         N = len(X_model)
         # constant outcome vector
         c = tf.ones((B,rep,1,max_number))*tf.cast(tf.range(0,max_number), tf.float32)
         # compute pmf value
         pi = tfd.Poisson(rate=mu).prob(c)
         # prevent zero value (causes inf for log)
         pi = tf.where(pi < 1.8*10**(-30), 1.8*10**(-30), pi)
         ## sample n-dimensional one-hot-like
         # Gumbel-Max Softmax trick
         # sample from uniform
         u = tfd.Uniform(0,1).sample((B,rep,N,max_number))
         # generate a gumbel sample
         g = -tf.math.log(-tf.math.log(u))
         # softmax trick
         w  = tf.nn.softmax(tf.math.divide(tf.math.add(tf.math.log(pi),g),temp))
         
         # apply transformation
         y_obs = tf.reduce_sum(tf.multiply(w,c), -1)    
         
         
         return {"y_obs": y_obs}

