# -*- coding: utf-8 -*-
"""
Plotting: Prior predictive, hyperparameter recovery, error analysis
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('stylesheet.mplstyle')
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from plot_global_settings import *
from helper_functions import plot_emp_prior, plot_true_prior, plot_titles, plot_handles, plot_alab


def plot_cs1(target_quant_exp, elicited_quant_exp, elicited_quant_sim, target_quant_sim):
    fig = plt.figure(constrained_layout=True, figsize=(6,3))
    subfigs = fig.subfigures(1, 4)
    axs0 = subfigs[0].subplots(3,1, sharex = True, sharey = True)
    axs1 = subfigs[1].subplots(2,1, sharex = True, sharey = True)
    axs2 = subfigs[3].subplots(2,1)
    axs3 = subfigs[2].subplots(3,1, sharex = True, sharey = True)
    
    # prior predictions: Encoding depth
    [density_exp_target_quant(target_quant_exp["mb"][0,:,i], axs0[i]) for i in range(3)]
    [hist_mod_target_quant(target_quant_sim["mb"][0,:,i], axs0[i]) for i in range(3)]
    for k in range(3):
        quantiles_mod(elicited_quant_exp["mb_loss"][:,:,k], y=[-0.5, 0.5], axes=axs0[k])
        quantiles_exp(elicited_quant_sim["mb_loss"][:,:,k], y=-1.5, axes=axs0[k])
    # prior predictions: Repetition
    for k in range(2):
        quantiles_mod(elicited_quant_exp["ma_loss"][:,:,k], y=[-0.2, 0.2], axes=axs1[k], zorder = 2)
        quantiles_exp(elicited_quant_sim["ma_loss"][:,:,k], y=-0.5, axes=axs1[k])
    legend_quantiles(axs1[0])
    [density_exp_target_quant(target_quant_exp["ma"][0,:,i], axs1[i], zorder=1) for i in range(2)]
    [hist_mod_target_quant(target_quant_sim["ma"][0,:,i], axs1[i]) for i in range(2)]
    # prior predictions: R2 and gm
    for k,i in zip(["R2", "gm"], range(2)):
        hist_target_elicited(target_quant_sim[k][0,:], axs2[i], type = "mod")
        hist_target_elicited(target_quant_exp[k][0,:], axs2[i], type = "exp")
    axs2[0].legend(["model", "expert"], loc = "upper right", fontsize="small")
    for k,i in zip(["R2", "gm"], range(2)): 
        density_exp_target_quant(target_quant_exp[k][0,:], axs2[i]) 
    # prior predictions: Effects
    [density_exp_target_quant(target_quant_exp["effects1"][0,:,i], axs3[i], zorder=1) for i in range(3)]
    [hist_mod_target_quant(target_quant_sim["effects1"][0,:,i], axs3[i]) for i in range(3)]
    for k in range (3):
        quantiles_mod(elicited_quant_exp["effects1_loss"][:,:,k], y=[-1, 1], axes=axs3[k])
        quantiles_exp(elicited_quant_sim["effects1_loss"][:,:,k], y=-2.5, axes=axs3[k])

    # titles for subplots
    fs = "small"
    [axs0[i].set_title(j, fontsize=fs) for i,j in zip(range(3),["deep", "standard", "shallow"])]
    [axs1[i].set_title(j, fontsize=fs) for i,j in zip(range(2),["new", "repeated"])]
    [axs2[i].set_title(j, fontsize=fs) for i,j in zip(range(2), ["R2", "grand mean"])]
    [axs3[i].set_title(j, fontsize=fs) for i,j in zip(range(3), ["deep","standard","shallow"])]
    # limits for x-axes
    [a[0].set_xlim(0,1) for a in [axs0, axs1, axs2]]
    axs2[1].set_xlim(0,0.5)
    [axs3[i].set_xlim(0,0.3) for i in range(3)]
    # labels for y-axes
    for a, j in zip([axs0,axs1,axs2, axs3],[3,2,2,3]): 
        [a[i].set_ylabel(None) for i in range(j)]
    # labels for x-axes
    fs = "small"
    [a[i].set_xlabel(r"$\textrm{PTJ}$", fontsize=fs) for a, i in zip([axs0, axs1, axs2], [2,1,1])]    
    [axs3[i].set_xlabel(r"$\Delta\textrm{PTJ}^\textrm{R-N}$", fontsize=fs) for i in range(3)]
    # title for subfigures
    [subfigs[i].suptitle(t) for i,t in zip(range(4), ["Encoding depth (EnC)", "Repetition (ReP)", "Effect of repetition", " "])]
    plt.savefig('../graphics/linreg_target_quants.png', dpi = 300)
 
def plot_results_binomial(input_settings_model, input_settings_learning, 
                          input_settings_global, var, elicited_quant_exp, 
                          elicited_quant_sim, target_quant_sim, target_quant_exp):
    # case study specific values
    xrange1=[-1.,1.]
    X = input_settings_model["X"]["no_axillary_nodes"]
    idx = input_settings_model["X_idx"]
    selected_designpoint = 2
    # hyperparamter
    true_mus = input_settings_model["hyperparameter"]["mus"]
    true_sigmas = tf.exp(input_settings_model["hyperparameter"]["sigmas"])
    # algorithm parameter
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    
    ################# data wrangling #################
    # prepare convergence plot
    xrge_mus = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    # compute final learned hyperparameters
    learned_mus = [final_hyperpar_avg(var[0], i, l_values, epochs, sigma = False) for i in range(2)]
    learned_sigmas = [tf.exp(final_hyperpar_avg(var[1], i, l_values, epochs, sigma = False)) for i in range(2)]
    # prepare prior distributions plot
    pdf_betas = [tfd.Normal(loc=learned_mus[i], scale=learned_sigmas[i]).prob(xrge_mus) for i in range(2)]
    pdf_betas_true = [tfd.Normal(loc=true_mus[i], scale=true_sigmas[i]).prob(xrge_mus) for i in range(2)]
    # prepare error plot
    mus_error = absolute_error(var[0], true_mus, epochs, log = False)
    sigmas_error = absolute_error(var[1], true_sigmas, epochs, log = True)
    # define labels for plotting
    n_mus = tex_label("mu", 2)
    n_sigmas = tex_label("sigma", 2)

    ################# plotting #################   
    ## set-up plot format
    fig = plt.figure(constrained_layout = True, figsize=(6,4.5)) 
    subfigs = fig.subfigures(2, 2, height_ratios = (1.,1), hspace = 0.1)
    axs0 = subfigs[0,0].subplots(2, 4)  
    axs1 = subfigs[1,0].subplots(1, 1)
    axs2 = subfigs[0,1].subplots(2, 1)
    axs3 = subfigs[1,1].subplots(1, 1)
    
    ## prior distribution plots
    [plot_prior_dist(axs3, xrge_mus, pdf_betas[i], pdf_betas_true[i], col_betas[i], 
                    label = rf"$\beta_{i} \sim N$({learned_mus[i]:.2f},{learned_sigmas[i]:.2f})") for i in range(2)]
    ## error plots
    plot_error_traj(axs2[0], mus_error, col_betas, n_mus, k = 2, sigma = False) 
    plot_error_traj(axs2[1], sigmas_error, col_betas, n_sigmas, k = 2, sigma = False)
    ## q-q plots
    [single_qq_plot(axs0[0,i], i, elicited_quant_exp["y_idx_loss"], elicited_quant_sim["y_idx_loss"], 
                    fr"$x_{i}={int(X[idx[i]])}$", d=0.05, Q = 9) for i in range(4)]
    [single_qq_plot(axs0[1,i], i+4, elicited_quant_exp["y_idx_loss"], elicited_quant_sim["y_idx_loss"], 
                    fr"$x_{i}={int(X[idx[i]])}$", d=0.05, Q = 9) for i in range(3)]
    axs0[1,3].axis('off') # remove 8th suplot
    for spine in axs0[0,2].spines.values(): spine.set_edgecolor("red") # highlight one subplot in red
    for spine in axs1.spines.values(): spine.set_edgecolor("red")
    ## histogram (target + elicited quantities for one selected design point)
    quantiles_mod(elicited_quant_exp["y_idx_loss"][:,:,selected_designpoint], y=[-0.02, 0.02], axes=axs1)
    quantiles_exp(elicited_quant_sim["y_idx_loss"][:,:,selected_designpoint], y=-0.05, axes=axs1)
    legend_quantiles(axs1)
    density_exp_target_quant(target_quant_exp["y_idx"][0,:,selected_designpoint], axs1) 
    hist_mod_target_quant(target_quant_sim["y_idx"][0,:,selected_designpoint], axs1) 

    ## adjust design elements   
    # title for subfigures and subplots
    axs0[0,2].set_title(r"$x_2 = 10$", color="red")
    axs2[0].set_title(" ")
    subfigs[1,1].suptitle("Learned prior distributions \n", ha="left", x=0.14)
    subfigs[0,0].suptitle("Prior predictions: Model-based vs. expert \n elicited quantiles", ha="left", x=0.05)
    subfigs[1,0].suptitle(f"Target quantity and elicited statistics \n"fr" for $x_{selected_designpoint} = {int(X[idx[selected_designpoint]])}$ axillary nodes", ha="left", x=0.07)
    subfigs[0,1].suptitle("Absolute error between true and learned \n hyperparameter", ha="left", x=0.14)
    # x-axes: labels
    axs0[1,2].set_xlabel("text", color = "white")
    [a.set_xlabel(t) for a,t in zip([axs2[1], axs3, axs1], ["epochs", r"model parameters $\beta_k$","\# detected axillary nodes"])]
    subfigs[0,0].text(0.5, 0.0, "model-based quantiles", ha="center", fontweight="bold",
                      fontsize = "small")
    # x-axes: ticks
    axs2[0].set_xticks([])
    # y-axes: labels
    axs0[1,0].set_ylabel("text", color = "white")
    subfigs[0,0].text(0.03, 0.2, "expert elicited quantiles", ha="center", 
                      fontweight="bold", rotation="vertical", fontsize="small")
    # y-axes: limits
    [axs2[i].set_ylim(r) for i,r in zip([0,1], [(-2,2.2), (-0.6,0.8)])]
    # y-axes: digit format
    [a.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for a in [axs3, axs2[0]]]
    # y-axes: ticks
    axs1.set_yticks([])
    [axs0[0,i].set_yticks([]) for i in range(4)]
    [axs0[1,i].set_yticks([]) for i in range(4)]
    # add legend
    [a.legend(loc=l) for a,l in zip([axs2[0],axs2[1], axs3],["upper right", "upper right", "upper left"])]
    [axs2[i].legend(ncol = 2) for i in range(2)]
    plt.savefig('../graphics/binom_summary_results.png', dpi = 300)
    
def plot_results_lm(input_settings_model, input_settings_learning, var, 
                    input_settings_global, target_quant_sim):
    # case study specific values
    xrange0=[0.0, 5.0]
    xrange1=[-0.4, 0.3]
    fct_b_lvl=3
    fct_a_lvl=2
    samples_d = target_quant_sim
    # define labels for plotting
    n_mus = tex_label("mu", 6)
    n_sigmas = tex_label("sigma", 6)
    # hyperparamter
    true_mus = input_settings_model["hyperparameter"]["mus"]
    true_sigmas = tf.exp(input_settings_model["hyperparameter"]["sigmas"])
    true_nu = tf.exp(input_settings_model["hyperparameter"]["lambda0"])
    # algorithm parameter
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    
    ################# data wrangling #################
    # prepare convergence plot
    xrge_nu = tf.cast(tf.range(xrange0[0],xrange0[1],0.001), tf.float32) 
    xrge_mus = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    # compute final learned hyperparameters
    learned_nu = tf.exp(final_hyperpar_avg(var[0],None, l_values, epochs, sigma = True))
    learned_mus = [final_hyperpar_avg(var[1], i, l_values, epochs, sigma = False) for i in range(6)]
    learned_sigmas = [tf.exp(final_hyperpar_avg(var[2], i, l_values, epochs, sigma = False)) for i in range(6)]
    # prepare prior distributions plot
    pdf_sigma = tfd.Exponential(learned_nu).prob(xrge_nu)
    pdf_sigma_true = tfd.Exponential(true_nu).prob(xrge_nu)
    pdf_sigma_m = tf.reduce_mean(tfd.Exponential(learned_nu).sample(1000))
    pdf_betas = [tfd.Normal(loc=learned_mus[i], scale=learned_sigmas[i]).prob(xrge_mus) for i in range(6)]
    pdf_betas_true = [tfd.Normal(loc=true_mus[i], scale=true_sigmas[i]).prob(xrge_mus) for i in range(6)]
    # prepare error plot
    nu_error = absolute_error(var[0], true_nu, epochs, log = True)
    mus_error = absolute_error(var[1], true_mus, epochs, log = False)
    sigmas_error = absolute_error(var[2], true_sigmas, epochs, log = True)

    ################# plotting #################   
    ## set-up plot format
    fig = plt.figure(constrained_layout = True, figsize=(6,4.2)) 
    subfigs = fig.subfigures(2, 1)
    axs1 = subfigs[0].subplots(1, 2, gridspec_kw=dict(width_ratios= [2.,1]))
    axs2 = subfigs[1].subplots(1, 3, sharex=True)
    
    ## prior distribution plots
    plot_prior_dist(axs1[1], xrge_nu, pdf_sigma, pdf_sigma_true, col_nu, rf"$s \sim Exp$({learned_nu:.2f})")
    [plot_prior_dist(axs1[0], xrge_mus, pdf_betas[i], pdf_betas_true[i], col_betas[i], 
                    label = rf"$\beta_{i} \sim N$({learned_mus[i]:.2f},{learned_sigmas[i]:.2f})") for i in range(6)]
    ## error plots
    plot_error_traj(axs2[0], nu_error[0,:], col_nu, None, k = None, sigma = True)
    plot_error_traj(axs2[1], mus_error, col_betas, n_mus, k = 6, sigma = False) 
    plot_error_traj(axs2[2], sigmas_error, col_betas, n_sigmas, k = 6, sigma = False)

    ## adjust design elements   
    # title for subfigures and subplots
    subfigs[0].suptitle("Learned prior distributions", x=0.21)
    subfigs[1].suptitle("Absolute error between true and learned hyperparameter", x=0.36)
    axs1[0].set_title("model parameters", fontsize = "medium")
    axs1[1].set_title(f"random noise (mean: {pdf_sigma_m:.2f})", fontsize = "medium")
    for i,n in enumerate([r"$\nu$", r"$\mu_k$", r"$\sigma_k$"]): axs2[i].set_title(n, fontsize = "medium")
    # x-axes: labels
    axs1[0].set_xlabel(r"$\beta_k$")
    axs1[1].set_xlabel(r"$s$", fontsize="large")
    subfigs[1].text(0.5,0.05,"epochs", va="center", fontsize = "small")
    # x-axes: ticks
    axs2[1].set_xticks([0,1500])
    # y-axes: digit-format 
    [axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for i in range(2)]
    [axs2[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(3)]
    # y-axes: limits
    [axs2[i].set_ylim(l) for i,l in zip(range(3),[(-2,2),(-0.3,0.3),(-0.4,0.4)])]
    plt.plot()
    # add legends
    [axs1[i].legend(ncol=1, loc=l) for i,l in zip([0,1], ["upper left", "upper right"])]
    [axs2[i].legend(ncol=2, loc="upper right") for i in [1,2]]
    plt.savefig('../graphics/linreg_summary_results.png', dpi = 300)
    
def plot_results_poisson(input_settings_model, input_settings_learning, 
                         input_settings_global, var, elicited_quant_exp, 
                         elicited_quant_sim, target_quant_sim, target_quant_exp):
    # case study specific values
    xrange1=[-3.,3.2]
    selected_designpoint = 1
    idx = input_settings_model["X_idx"]
    # hyperparamter
    true_mus = input_settings_model["hyperparameter"]["mus"]
    true_sigmas = tf.exp(input_settings_model["hyperparameter"]["sigmas"])
    # algorithm parameter
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]
    
    ################# data wrangling #################
    # prepare convergence plot
    xrge_mus = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    # compute final learned hyperparameters
    learned_mus = [final_hyperpar_avg(var[0], i, l_values, epochs, sigma = False) for i in range(4)]
    learned_sigmas = [tf.exp(final_hyperpar_avg(var[1], i, l_values, epochs, sigma = False)) for i in range(4)]
    # prepare prior distributions plot
    pdf_betas = [tfd.Normal(loc=learned_mus[i], scale=learned_sigmas[i]).prob(xrge_mus) for i in range(4)]
    pdf_betas_true = [tfd.Normal(loc=true_mus[i], scale=true_sigmas[i]).prob(xrge_mus) for i in range(4)]
    # prepare error plot
    mus_error = absolute_error(var[0], true_mus, epochs, log = False)
    sigmas_error = absolute_error(var[1], true_sigmas, epochs, log = True)
    # define labels for plotting
    n_mus = tex_label("mu", 4)
    n_sigmas = tex_label("sigma", 4)

    ################# plotting #################   
    ## set-up plot format
    fig = plt.figure(constrained_layout=True, figsize=(6,5.))
    subfigs = fig.subfigures(3, 2, height_ratios = [0.15, 1., 0.7] ) 
    axs00 = subfigs[0,0].subplots(1, 1)
    axs01 = subfigs[0,1].subplots(1, 1)
    axs0 = subfigs[1,0].subplots(3, 3, gridspec_kw = {"top": 7})
    axs1 = subfigs[2,0].subplots(1, 1)
    axs2 = subfigs[1,1].subplots(2, 1, sharex = True)
    axs3 = subfigs[2,1].subplots(1, 1)

    ## prior distribution plots
    [plot_prior_dist(axs3, xrge_mus, pdf_betas[i], pdf_betas_true[i], col_betas[i], 
                    label = rf"$\beta_{i} \sim N$({learned_mus[i]:.2f},{learned_sigmas[i]:.2f})") for i in range(4)]
    ## error plots
    plot_error_traj(axs2[0], mus_error, col_betas, n_mus, k = 4, sigma = False) 
    plot_error_traj(axs2[1], sigmas_error, col_betas, n_sigmas, k = 4, sigma = False)
    ## q-q plots
    [single_qq_plot(axs0[0,i], i, elicited_quant_exp["y_groups_loss"], elicited_quant_sim["y_groups_loss"], 
                    t, Q = 9, d = 1.5) for i,t in zip(range(3),["Democrats", "Swing", "Republican"])]
    for spine in axs0[0,1].spines.values(): spine.set_edgecolor("red") # highlight one subplot in red
    ## hist subplots (elicited quants)
    for quant,typ in zip([elicited_quant_sim,elicited_quant_exp], ["mod", "exp"]):
        [hist_target_elicited(quant["y_obs_loss"][0,:,j], axs0[k,a], type=typ) for k,j,a in zip(np.repeat([1,2],3),range(6),[0,1,2]*2)]
    ## hist plot (target + elicited quantities for one selected design point)
    quantiles_mod(elicited_quant_exp["y_groups_loss"][:,:,selected_designpoint], y=[-0.01, 0.01], axes=axs1)
    quantiles_exp(elicited_quant_sim["y_groups_loss"][:,:,selected_designpoint], y=-0.03, axes=axs1)
    legend_quantiles(axs1)
    density_exp_target_quant(target_quant_exp["y_groups"][0,:,selected_designpoint], axs1) 
    hist_mod_target_quant(target_quant_sim["y_groups"][0,:,selected_designpoint], axs1) 
    
    ## adjust design elements   
    # title for subfigures and subplots
    [axs0[j,i].set_title(t) for j,i,t in zip(np.repeat([1,2],3),[0,1,2]*2,[fr"$x_{{{k}}}$" for k in idx])] 
    axs0[0,1].set_title("Swing", color="red")
    axs2[0].set_title(" ") # maintain spaces between subplots
    axs3.set_title("Learned prior distributions \n", x=0.30)
    axs1.set_title("Target quantity and elicited statistics \n for Swing states", ha="left", x=-0.01)
    subfigs[0,1].suptitle("Absolute error between true and learned \n hyperparameter \n", ha="left", x=0.1)
    subfigs[0,0].suptitle("Prior predictions: Model-based vs. expert \n elicited statistics", ha="left", x=0.05)
    axs00.remove()
    axs01.remove()
    # x-axes: labels
    [axs0[i,1].set_xlabel(l, labelpad = 7.) for i,l in zip([0,2], ["model-based quantiles","\# LGBTQ+ anti-discrimination laws"])]
    axs2[1].set_xlabel("epochs")
    [a.set_xlabel(l) for a,l in zip([axs1, axs3],["\# LGBTQ+ anti-discrimination laws", r"model parameters $\beta_k$"])]
    # x-axes:ticks
    [axs0[1,i].set_xticks([]) for i in [0,1,2]]
    [axs0[2,i].set_xticks([]) for i in [0,1,2]]
    # y-axes: ticks
    [axs0[0,i].set_yticks([]) for i in range(3)]
    [axs0[1,i].set_yticks([]) for i in range(3)]
    [axs0[2,i].set_yticks([]) for i in range(3)]
    axs1.set_yticks([])
    # y-axes: labels
    [axs0[i,0].set_ylabel(l) for i,l in zip(range(3), [" \n expert \n"," \n density \n"," \n density \n"])]
    [axs0[j,i].set_ylabel(" ") for j,i in zip(np.repeat([1,2],2),[1,2]*2)]
    axs1.set_ylabel("density")
    # y-axes: digit format
    axs3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # y-axes: limits
    [axs2[i].set_ylim(r) for i,r in zip(range(2),[(-3,3),(-4,4)])]
    # add legends
    axs3.legend(loc = "upper left", fontsize = "small")
    [axs2[i].legend(loc = "upper right", ncol = 2) for i in range(2)]
    plt.savefig('../graphics/pois_summary_results.png', dpi = 300)

def plot_incon_info(input_settings_model, elicited_quant_exp, 
                    elicited_quant_sim, target_quant_sim, target_quant_exp):

    import pickle 
    
    open_file = open("../simulations/sim_mlm_incon1.pkl", "rb")
    (out, var, target_quant_sim1, elicited_quant_sim1,  elicited_quant_sim_ini, elicited_quant_exp1, target_quant_exp1, weights,  time_per_epoch, final_time, final_time, input_settings_model,  input_settings_learning, 
      input_settings_global,  input_settings_loss) = pickle.load(open_file)
    open_file.close()
    open_file = open("../simulations/sim_mlm_incon2.pkl", "rb")
    (out, var, target_quant_sim2, elicited_quant_sim2,  elicited_quant_sim_ini, elicited_quant_exp2, target_quant_exp2, weights,  time_per_epoch, final_time, final_time, input_settings_model,  input_settings_learning, 
      input_settings_global,  input_settings_loss) = pickle.load(open_file)
    open_file.close()
    
    elicited_quant_exp = [elicited_quant_exp1, elicited_quant_exp2]
    elicited_quant_sim = [elicited_quant_sim1, elicited_quant_sim2]
    target_quant_exp = [target_quant_exp1, target_quant_exp2]
    target_quant_sim = [target_quant_sim1, target_quant_sim2]
    
    idx = input_settings_model["X_idx"]
    
    fig = plt.figure(constrained_layout=True, figsize=(6,3.))
    subfigs = fig.subfigures(1, 3, width_ratios = (1,0.1,1.))
    axs0 = subfigs[0].subplots(3, 3)
    axs1 = subfigs[2].subplots(3, 3)
    axs2 = subfigs[1].subplots(3, 1)
    
    for a, s in zip([axs0,axs1], [0,1]):
        ## q-q plot
        [single_qq_plot(a[0,i], i, elicited_quant_exp[s]["days_loss"]/60, elicited_quant_sim[s]["days_loss"]/60, 
                        f"day {idx[i]}", d = 0.1, Q = 9) for i in range(3)]
        [single_qq_plot(a[1,i], i+3, elicited_quant_exp[s]["days_loss"]/60, elicited_quant_sim[s]["days_loss"]/60, 
                        f"day {idx[i+3]}", d = 0.3, Q = 9) for i in range(2)]
        ## hist subplots (elicited quants)
        [hist_target_elicited(quant[0,:], a[2,0], type=typ) for quant,typ in zip([target_quant_sim[s]["R2_0"],target_quant_exp[s]["R2_0_mod_input"]],["mod","exp"])]
        [hist_target_elicited(quant[0,:], a[2,1], type=typ) for quant,typ in zip([target_quant_sim[s]["R2_1"],target_quant_exp[s]["R2_1_mod_input"]],["mod","exp"])]
        ## mean-sd subplot
        mean_sd_plot(elicited_quant_sim[s]["days_sd_loss"], elicited_quant_exp[s]["days_sd_loss"], a[2,2],dev=20,i=0)
    
        ## adjust design elements   
        # title for subfigures and subplots
        [a[2,i].set_title(fr"$R^2$ (day {j})", fontsize = "small") for i,j in zip(range(2),[0,9])]
        a[2,2].set_title(rf"$s$ (mean, sd)", fontsize = "small")
        # x-axes: labels
        a[1,1].set_xlabel("model-implied quantiles")
        [a[2,i].set_xlabel(l) for l,i in zip([r"$R^2$",r"$R^2$", "model"], range(3))]
        # x-axes: limits
        [a[2,i].set_xlim(0,1) for i in range(2)]
        # y-axes: labels
        [a[i,0].set_ylabel("expert") for i in range(2)]
        [a[2,i].set_ylabel(l) for i,l in zip(range(3),["density", " ", "expert"])]
        # y-axes: ticks
        [a[0,i].set_yticks([]) for i in range(3)]
        [a[1,i].set_yticks([]) for i in range(3)]
        [a[2,i].set_yticks([]) for i in range(3)]
        a[1,2].remove()
    
    [axs2[i].set_visible(False) for i in range(3)]
    subfigs[0].suptitle(r"Scenario I: Increase $s$ by a factor of 2", 
                        x = 0.1, ha = "left", fontweight = "bold")
    subfigs[2].suptitle(r"Scenario II: Decrease $R^2$ by a factor of 2", 
                        x = 0.1, ha = "left", fontweight = "bold")
    
    plt.savefig('../graphics/mlm_compare_incon.png', dpi = 300)
    
def plot_results_mlm(var,input_settings_model, input_settings_global, input_settings_learning, 
                     elicited_quant_exp, elicited_quant_sim, target_quant_sim, target_quant_exp):
    matplotlib.use('ps')
    from matplotlib import rc
    rc('text.latex', preamble='\\usepackage{color}')

    # case study specific values
    xrange0 = [0, 200]
    xrange1 = [200, 300]
    xrange2 = [0, 60]
    selected_designpoint = 1
    idx = input_settings_model["X_idx"]
    # hyperparamter
    true_nu = tf.exp(input_settings_model["hyperparameter"]["lambda0"])
    true_mus = input_settings_model["hyperparameter"]["mus"]
    true_sigmas = tf.exp(input_settings_model["hyperparameter"]["sigmas"])
    true_sigma_taus = tf.exp(input_settings_model["hyperparameter"]["sigma_taus"])
    # algorithm parameter
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]

    ################# data wrangling #################
    # prepare convergence plot
    xrge_nu_taus =  tf.cast(tf.range(xrange0[0],xrange0[1],0.001), tf.float32)
    xrge_b0 = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    xrge_b1 = tf.cast(tf.range(xrange2[0],xrange2[1],0.001), tf.float32)
    xrge_mus = [xrge_b0, xrge_b1]
    # compute final learned hyperparameters
    learned_nu = tf.exp(final_hyperpar_avg(var[0], None, l_values, epochs, sigma = True)) 
    learned_mus = [final_hyperpar_avg(var[1], i, l_values, epochs, sigma = False) for i in range(2)]
    learned_sigmas = [tf.exp(final_hyperpar_avg(var[3], i, l_values, epochs, sigma = False)) for i in range(2)]
    learned_sigma_taus = [tf.exp(final_hyperpar_avg(var[2], i, l_values, epochs, sigma = False)) for i in range(2)]
    # prepare prior distributions plot
    pdf_sigma = tfd.Exponential(learned_nu).prob(xrge_nu_taus)
    pdf_sigma_true = tfd.Exponential(true_nu).prob(xrge_nu_taus)
    pdf_sigma_m = tf.reduce_mean(tfd.Exponential(learned_nu).sample(1000))
    pdf_betas = [tfd.Normal(loc=learned_mus[i], scale=learned_sigmas[i]).prob(xrge_mus[i]) for i in range(2)]
    pdf_betas_true = [tfd.Normal(loc=true_mus[i], scale=true_sigmas[i]).prob(xrge_mus[i]) for i in range(2)]
    pdf_taus = [tfd.TruncatedNormal(loc=0., scale=learned_sigma_taus[i], low=0., high=500).prob(xrge_nu_taus) for i in range(2)]
    pdf_taus_true = [tfd.TruncatedNormal(loc=0., scale=true_sigma_taus[i], low=0., high=500).prob(xrge_nu_taus) for i in range(2)]
    # prepare error plot
    nu_error = absolute_error(var[0], true_nu, epochs, log = True)
    mus_error = absolute_error(var[1], true_mus, epochs, log = False)
    sigmas_error = absolute_error(var[3], true_sigmas, epochs, log = True)
    sigma_taus_error = absolute_error(var[2], true_sigma_taus, epochs, log = True)
    # define labels for plotting
    n_mus = tex_label("mu", 2)
    n_sigmas = tex_label("sigma", 2)
    n_sigma_taus = tex_label("sigma_tau",2)

    ################# plotting #################   
    ## set-up plot format
    fig = plt.figure(constrained_layout=True, figsize=(6,5.))
    subfigs = fig.subfigures(2, 2, height_ratios = (1.5,1))
    axs0 = subfigs[0,0].subplots(3, 3)  
    axs1 = subfigs[1,0].subplots(1, 1)
    axs2 = subfigs[0,1].subplots(2, 1, sharex = True)
    axs3 = subfigs[1,1].subplots(1, 1)

    ## prior distribution plots
    [plot_prior_dist(axs1, xrge_mus[i], pdf_betas[i], pdf_betas_true[i], col_betas[i], 
                    label = rf"$\beta_{i} \sim N$({learned_mus[i]:.2f},{learned_sigmas[i]:.2f})") for i in range(2)]
    [plot_prior_dist(axs3, xrge_nu_taus, pdf_taus[i], pdf_taus_true[i], col_betas[i+2], 
                    label = rf"$\tau_{i} \sim N^+$(0, {learned_sigma_taus[i]:.2f})") for i in range(2)]
    plot_prior_dist(axs3, xrge_nu_taus, pdf_sigma, pdf_sigma_true, col_nu, 
                    label = fr"$s \sim Exp$({learned_nu:.2f})") 
    ## error plots
    plot_error_traj(axs2[0], mus_error, col_betas, n_mus, k = 2, sigma = False) 
    plot_error_traj(axs2[0], sigmas_error, col_betas[2:], n_sigmas, k = 2, sigma = False)
    plot_error_traj(axs2[1], sigma_taus_error, col_betas[2:], n_sigma_taus, k = 2, sigma = False)
    plot_error_traj(axs2[1], nu_error, col_nu, r"$\nu$", k = None, sigma = True)

    ## q-q plot
    [single_qq_plot(axs0[0,i], i, elicited_quant_exp["days_loss"]/60, elicited_quant_sim["days_loss"]/60, 
                    f"day {idx[i]}", d = 0.1, Q = 9) for i in range(3)]
    [single_qq_plot(axs0[1,i], i+3, elicited_quant_exp["days_loss"]/60, elicited_quant_sim["days_loss"]/60, 
                    f"day {idx[i+3]}", d = 0.3, Q = 9) for i in range(2)]
    ## hist subplots (elicited quants)
    [hist_target_elicited(quant["R2_0"][0,:], axs0[2,0], type=typ) for quant,typ in zip([target_quant_sim,target_quant_exp],["mod","exp"])]
    [hist_target_elicited(quant["R2_1"][0,:], axs0[2,1], type=typ) for quant,typ in zip([target_quant_sim,target_quant_exp],["mod","exp"])]
    ## mean-sd subplot
    mean_sd_plot(elicited_quant_sim["days_sd_loss"], elicited_quant_exp["days_sd_loss"], axs0[2,2],dev=20,i=0)

    ## adjust design elements   
    axs0[1,2].remove()
    # title for subfigures and subplots
    [axs0[2,i].set_title(fr"$R^2$ (day {j})") for i,j in zip(range(2),[0,9])]
    axs0[2,2].set_title(r"$s$ (\textcolor{red}{M}, \textcolor{blue}{S})")
    axs2[0].set_title(" ")
    subfigs[0,0].suptitle("Prior predictions: Model-based vs. expert \n elicited statistics", 
                          ha = "left", x = 0.14)
    subfigs[1,0].suptitle("Learned prior distributions", ha = "left", x = 0.14)
    subfigs[1,1].suptitle(" ") # maintain position between subplots
    subfigs[0,1].suptitle("Absolute error between true and learned \n hyperparameter", 
                          ha = "left", x = 0.11)
    # x-axes: labels
    axs0[1,1].set_xlabel("model-implied quantiles")
    [axs0[2,i].set_xlabel(l) for l,i in zip([r"$R^2$",r"$R^2$", "model"], range(3))]
    axs1.set_xlabel(fr"$\beta_k$")
    axs2[1].set_xlabel("epochs")
    axs3.set_xlabel(r"$\tau_k, s$")
    # x-axes: limits
    [axs0[2,i].set_xlim(0,1) for i in range(2)]
    # y-axes: labels
    axs1.set_ylabel("density", labelpad = 0)
    [axs0[i,0].set_ylabel(" \n expert \n") for i in range(2)]
    [axs0[2,i].set_ylabel(l) for i,l in zip(range(3),[" \n density \n", " ", "expert"])]
    # y-axes: ticks
    [axs0[0,i].set_yticks([]) for i in range(3)]
    [axs0[1,i].set_yticks([]) for i in range(3)]
    [axs0[2,i].set_yticks([]) for i in range(3)]
    # y-axes: digit format
    axs1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # y-axes: limits
    [axs1.set_ylim(0,tf.reduce_max(pdf_betas[i])+d) for i,d in zip(range(2),[0.03,0.05])]
    [axs2[i].set_ylim(r) for i,r in zip(range(2), [(-20,20),(-20,20)])]
    # add legend
    [a.legend(loc = "upper right") for a in [axs1, axs3]]
    [axs2[i].legend(ncol = 2, loc = "upper right") for i in range(2)]
    # save in postscript format in order to get red/blue text color
    figure = plt.gcf()
    plt.savefig('../graphics/mlm_summary_results.ps', dpi = 300)

def plot_results_weibull(var,input_settings_model, input_settings_global, input_settings_learning, 
                         elicited_quant_exp, elicited_quant_sim, target_quant_sim, target_quant_exp):
    matplotlib.use('ps')
    from matplotlib import rc
    rc('text.latex', preamble='\\usepackage{color}')

    # case study specific values
    xrange0 = [0, 50]
    xrange1 = [5.2, 6.1]
    xrange2 = [-0.05, 0.35]
    xrange3 = [0., 0.4]
    selected_designpoint = 1
    idx = input_settings_model["X_idx"]
    # hyperparamter
    true_nu = tf.exp(input_settings_model["hyperparameter"]["lambda0"])
    true_mus = input_settings_model["hyperparameter"]["mus"]
    true_sigmas = tf.exp(input_settings_model["hyperparameter"]["sigmas"])
    true_sigma_taus = tf.exp(input_settings_model["hyperparameter"]["sigma_taus"])
    # algorithm parameter
    epochs = input_settings_learning["epochs"]
    l_values = input_settings_global["l_values"]

    ################# data wrangling #################
    # prepare convergence plot
    xrge_nu =  tf.cast(tf.range(xrange0[0],xrange0[1],0.001), tf.float32)
    xrge_b0 = tf.cast(tf.range(xrange1[0],xrange1[1],0.001), tf.float32)
    xrge_b1 = tf.cast(tf.range(xrange2[0],xrange2[1],0.001), tf.float32)
    xrge_mus = [xrge_b0, xrge_b1]
    xrge_taus = tf.cast(tf.range(xrange3[0],xrange3[1],0.001), tf.float32)
    # compute final learned hyperparameters
    learned_nu = tf.exp(final_hyperpar_avg(var[0], None, l_values, epochs, sigma = True)) 
    learned_mus = [final_hyperpar_avg(var[1], i, l_values, epochs, sigma = False) for i in range(2)]
    learned_sigmas = [tf.exp(final_hyperpar_avg(var[3], i, l_values, epochs, sigma = False)) for i in range(2)]
    learned_sigma_taus = [tf.exp(final_hyperpar_avg(var[2], i, l_values, epochs, sigma = False)) for i in range(2)]
    # prepare prior distributions plot
    pdf_sigma = tfd.Exponential(learned_nu).prob(xrge_nu)
    pdf_sigma_true = tfd.Exponential(true_nu).prob(xrge_nu)
    pdf_sigma_m = tf.reduce_mean(tfd.Exponential(learned_nu).sample(1000))
    pdf_betas = [tfd.Normal(loc=learned_mus[i], scale=learned_sigmas[i]).prob(xrge_mus[i]) for i in range(2)]
    pdf_betas_true = [tfd.Normal(loc=true_mus[i], scale=true_sigmas[i]).prob(xrge_mus[i]) for i in range(2)]
    pdf_taus = [tfd.TruncatedNormal(loc=0., scale=learned_sigma_taus[i], low=0., high=500).prob(xrge_taus) for i in range(2)]
    pdf_taus_true = [tfd.TruncatedNormal(loc=0., scale=true_sigma_taus[i], low=0., high=500).prob(xrge_taus) for i in range(2)]
    # prepare error plot
    nu_error = absolute_error(var[0], true_nu, epochs, log = True)
    mus_error = absolute_error(var[1], true_mus, epochs, log = False)
    sigmas_error = absolute_error(var[3], true_sigmas, epochs, log = True)
    sigma_taus_error = absolute_error(var[2], true_sigma_taus, epochs, log = True)
    # define labels for plotting
    n_mus = tex_label("mu", 2)
    n_sigmas = tex_label("sigma", 2)
    n_sigma_taus = tex_label("sigma_tau",2)

    ################# plotting #################   
    ## set-up plot format
    fig = plt.figure(constrained_layout=True, figsize=(6,6.0))
    subfigs = fig.subfigures(2, 2)
    axs0 = subfigs[0,0].subplots(3, 3)  
    axs1 = subfigs[0,1].subplots(2, 1)
    axs2 = subfigs[1,0].subplots(2, 1, sharex = True)
    axs3 = subfigs[1,1].subplots(2, 1)

    ## prior distribution plots
    [plot_prior_dist(axs1[i], xrge_mus[i], pdf_betas[i], pdf_betas_true[i], col_betas[i], 
                    label = rf"$\beta_{i} \sim N$({learned_mus[i]:.3f},{learned_sigmas[i]:.3f})") for i in range(2)]
    [plot_prior_dist(axs3[0], xrge_taus, pdf_taus[i], pdf_taus_true[i], col_betas[i+2], 
                    label = rf"$\tau_{i} \sim N^+$(0, {learned_sigma_taus[i]:.3f})") for i in range(2)]
    plot_prior_dist(axs3[1], xrge_nu, pdf_sigma, pdf_sigma_true, col_nu, 
                    label = fr"$\alpha \sim Exp$({learned_nu:.3f})") 
    ## error plots
    plot_error_traj(axs2[0], mus_error, col_betas, n_mus, k = 2, sigma = False) 
    plot_error_traj(axs2[0], sigmas_error, col_betas[2:], n_sigmas, k = 2, sigma = False)
    plot_error_traj(axs2[1], sigma_taus_error, col_betas[2:], n_sigma_taus, k = 2, sigma = False)
    plot_error_traj(axs2[1], nu_error, col_nu, r"$\nu$", k = None, sigma = True)

    ## q-q plot
    [single_qq_plot(axs0[0,i], i, elicited_quant_exp["days_loss"]/60, elicited_quant_sim["days_loss"]/60, 
                    f"day {idx[i]}", d = 0.1, Q = 9) for i in range(3)]
    [single_qq_plot(axs0[1,i], i+3, elicited_quant_exp["days_loss"]/60, elicited_quant_sim["days_loss"]/60, 
                    f"day {idx[i+3]}", d = 0.3, Q = 9) for i in range(2)]
    ## hist subplots (elicited quants)
    [hist_target_elicited(quant["R2_0"][0,:], axs0[2,0], type=typ) for quant,typ in zip([target_quant_sim,target_quant_exp],["mod","exp"])]
    [hist_target_elicited(quant["R2_1"][0,:], axs0[2,1], type=typ) for quant,typ in zip([target_quant_sim,target_quant_exp],["mod","exp"])]
    ## mean-sd subplot
    mean_sd_plot(elicited_quant_sim["days_sd_loss"], elicited_quant_exp["days_sd_loss"], axs0[2,2],dev=20,i=0)

    ## adjust design elements   
    axs0[1,2].remove()
    # title for subfigures and subplots
    [axs0[2,i].set_title(fr"$R^2$ (day {j})") for i,j in zip(range(2),[0,9])]
    axs0[2,2].set_title(r"$s$ (\textcolor{red}{M}, \textcolor{blue}{S})")
    axs2[0].set_title(" ")
    subfigs[0,0].suptitle("Prior predictions: Model-based vs. expert \n elicited statistics", 
                          ha = "left", x = 0.14)
    subfigs[0,1].suptitle("Learned prior distributions", ha = "left", x = 0.14)
    #subfigs[1,1].suptitle(" \n ") # maintain position between subplots
    subfigs[1,0].suptitle("Absolute error between true and learned \n hyperparameter", 
                          ha = "left", x = 0.14)
    # x-axes: labels
    axs0[1,1].set_xlabel("model-implied quantiles")
    [axs0[2,i].set_xlabel(l) for l,i in zip([r"$R^2$",r"$R^2$", "model"], range(3))]
    [axs1[i].set_xlabel(fr"$\beta_{i}$") for i in range(2)] 
    [a[i].set_xlabel(l) for a,l,i in zip([axs3, axs3, axs2], [r"$\tau_0, \tau_1$", r"$\alpha$", "epochs"], [0,1,1])]
    # x-axes: limits
    [axs0[2,i].set_xlim(0,1) for i in range(2)]
    # y-axes: labels
    #axs1[0].set_ylabel("density", labelpad = 0)
    [axs0[i,0].set_ylabel(f" \n expert \n") for i in range(2)]
    [axs0[2,i].set_ylabel(l) for i,l in zip(range(3),[f" \n density \n", " ", "expert"])]
    # y-axes: digit format
    [axs1[a].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for a in range(2)]
    [axs3[a].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for a in range(2)]
    # y-axes: ticks
    [axs0[0,i].set_yticks([]) for i in range(3)]
    [axs0[1,i].set_yticks([]) for i in range(3)]
    [axs0[2,i].set_yticks([]) for i in range(3)]
    # y-axes: limits
    [axs1[i].set_ylim(0,tf.reduce_max(pdf_betas[i])+d) for i,d in zip(range(2),[0.1,0.1])]
    [axs2[i].set_ylim(r) for i,r in zip(range(2), [(-1,1),(-0.2,0.2)])]
    axs1[0].set_ylim(0,15.)
    axs1[1].set_ylim(0,25.)
    axs3[0].set_ylim(0,15.)
    # add legend
    [a[j].legend(loc = "upper right") for a,j in zip([axs1, axs1, axs3, axs3],[0,1]*2)]
    [axs2[i].legend(ncol = 2, loc = "upper right") for i in range(2)]
    # save in postscript format in order to get red/blue text color
    figure = plt.gcf()
    plt.savefig('../graphics/mlm2_summary_results.ps',bbox_inches='tight',pad_inches=0.0, dpi = 300)

def plot_sens_res_poisson():
    from loss_components import extract_loss_components
    from losses import energy_loss
    from trainer import trainer
    from trainer_step import trainer_step
    from configuration_global import settings
   
    # binom, linreg, pois, negbinom, mlm, weib
    selected_model = "pois_sensitivity"

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

    fig = plt.figure(figsize =(6,3.5))
    matplotlib.rcParams.update({"text.usetex": True,"font.family" : "serif", "font.serif" : ["Computer Modern Serif"]})
    axs = sns.kdeplot(tf.reshape(target_quant_exp["y_obs"], (1000*49)), color = "black", lw = 3, zorder=1)
    [sns.kdeplot(target_quant_exp["y_obs"][0,:,i], color = "#d88853", lw = 3, alpha = 0.3, ax = axs, zorder=0) for i in range(49)]
    for i in [5, 15, 30,55]:
        plt.axvline(i, linestyle="dotted", color = "black", lw = 2)
        plt.text(i+1,0.4, rf"$t^u = {i}$", fontsize="large", fontweight="bold")
    plt.xlabel("\# LGBTQ+ anti-discrimination laws", fontsize="large")
    plt.ylabel("density", fontsize="large")
    plt.xlim(0,70)
    plt.savefig('../graphics/pois_thresholds.png', dpi = 300)

def plot_thres_res_poisson():

    ########### data preparation #############
    error_mu = dict()
    error_sd = dict()
    dict_time = dict()

    for size in ["5", "15", "30", "55", "110", "210"]:
        open_file = open(f"../simulations/sim_pois_sensitivity_{size}.pkl", "rb")
        (out, var, target_quant_sim, elicited_quant_sim,  elicited_quant_sim_ini, elicited_quant_exp, target_quant_exp, weights,  time_per_epoch, final_time, final_time, input_settings_model,  input_settings_learning, 
        input_settings_global,  input_settings_loss) = pickle.load(open_file)
        open_file.close()
        
        l_values = 30
        epochs = 300
        # get learned mus and sigmas
        true_mus = input_settings_model["hyperparameter"]["mus"]
        true_sigmas = tf.exp(input_settings_model["hyperparameter"]["sigmas"])
        # compute learned mus and sigmas
        learned_mus = [final_hyperpar_avg(var[0], i, l_values, epochs, sigma = False) for i in range(4)]
        learned_sigmas = [tf.exp(final_hyperpar_avg(var[1], i, l_values, epochs, sigma = False)) for i in range(4)]
        # compute error between learned and true mus/sigmas
        mus_error = tf.abs(tf.math.subtract(learned_mus,true_mus))
        sigmas_error = tf.abs(tf.math.subtract(learned_sigmas,true_sigmas))
        # save results
        error_mu[size] = mus_error
        error_sd[size] = sigmas_error
        dict_time[size] = time_per_epoch

    ########### plotting #############
    _, axs = plt.subplots(1,3, figsize=(6,2), sharex = True, constrained_layout=True)
    for d,i in zip([error_mu, error_sd, dict_time], range(3)):
        sns.boxplot(pd.DataFrame(d), palette = q_cols_total[:5], ax = axs[i])
    
    [axs[i].set_title(t) for i,t in zip(range(3), [r"$\mu_k$",r"$\sigma_k$","time per epoch"])]
    [axs[i].set_ylim(0,0.4) for i in range(2)]
    [axs[i].set_ylabel(t) for i,t in zip([0,2],[r"abs. error: $\lambda^*$ vs. $\lambda$", "time in sec"])]
    axs[1].set_xlabel(r"upper threshold $t^u$")
    plt.savefig('../graphics/pois_thresholds2.png', dpi = 300)


