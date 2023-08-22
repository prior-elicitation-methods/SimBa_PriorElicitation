# -*- coding: utf-8 -*-
"""
Plotting: Convergence Diagnostics
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def plot_diagnostics_binomial(var, res, input_settings_learning):
    # define labels for plotting
    n_mus = [r"$\mu_0$",r"$\mu_1$"]
    n_sigmas = [r"$\sigma_0$",r"$\sigma_1$"]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad"]
    final_epoch = input_settings_learning["epochs"]
    
    # norm of gradients
    grad_mu = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][0])) for i in range(final_epoch)],0)
    grad_sigma = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1])) for i in range(final_epoch)],0) 
    grads = [grad_mu, grad_sigma]
    
    fig = plt.figure(figsize=(6,3.5), constrained_layout=True)
    matplotlib.rcParams.update({'text.usetex': True, "font.family" : "serif",
                        "font.serif" : ["Computer Modern Serif"]})
    subfigs = fig.subfigures(2, 1)
    axs0 = subfigs[0].subplots(1, 3, sharex = True)
    axs1 = subfigs[1].subplots(1, 3, sharex = True)
    
    # plot convergence
    for i in range(2):
       axs1[1].plot(tf.squeeze(tf.stack(var[0][1],-1))[i,:], color = col_betas[i], linewidth=2)
       axs1[2].plot(tf.exp(tf.squeeze(tf.stack(var[1][1],-1))[i,:]), color = col_betas[i], linewidth=2)

    # plot gradients   
    [sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grads[i-1], 
                   linewidth=0, alpha = 0.5, color="black", ax=axs0[i]) for i in [1,2]]
  
    # plot loss
    for a,n in zip([axs0, axs1],["loss","loss_tasks"]):
        a[0].plot([res[i][n] for i in range(final_epoch)], linewidth=2)  
    
    # add titles
    axs0[0].set_title("Total loss", x= 0.2)
    axs0[1].set_title(r"Gradients: $\mu_k$", x = 0.3)
    axs0[2].set_title(r"$\sigma_k$")
    axs1[1].set_title(r"Convergence: $\mu_k$", x =0.35)
    axs1[2].set_title(r"$\sigma_k$")
    axs1[0].set_title("Individual losses", x = 0.35)
    
    # add legends
    axs1[1].legend(n_mus, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
    axs1[2].legend(n_sigmas, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
    
    [axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(3)]
    [axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(3)]
    
    [axs1[i].set_xlabel("epochs") for i in range(3)]
    [axs0[i].set_xlabel("epochs", color = "white") for i in range(3)]
    plt.savefig('../graphics/binom_diagnostics.png', dpi = 300)

def plot_diagnostics_lm(var, res, input_settings_learning):
     # define labels for plotting
     n_mus = [fr"$\mu_{i}$" for i in range(6)]
     n_sigmas = [fr"$\sigma_{i}$" for i in range(6)]
     names_el = [n_mus, n_sigmas]
     # define color codes for plotting
     # betas 
     col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
     col_lambda = "#a6b7c6"
     final_epoch = input_settings_learning["epochs"]
     # norm of gradients
     grads = [tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][j])) for i in range(final_epoch)],0) for j in range(3)]
 
     fig = plt.figure(figsize=(6,3.5), constrained_layout=True)
     matplotlib.rcParams.update({'text.usetex': True, "font.family" : "serif",
                         "font.serif" : ["Computer Modern Serif"]})
     subfigs = fig.subfigures(2, 1)
     axs0 = subfigs[0].subplots(1, 4, sharex = True)
     axs1 = subfigs[1].subplots(1, 4, sharex = True)
     
     # plot convergence
     axs1[1].plot(tf.exp(var[0][1]),color=col_lambda, linewidth=2)
     for i in range(6):
        axs1[2].plot(tf.squeeze(tf.stack(var[1][1],-1))[i,:], color = col_betas[i], linewidth=2)
        axs1[3].plot(tf.exp(tf.squeeze(tf.stack(var[2][1],-1))[i,:]), color = col_betas[i], linewidth=2)

     # plot gradients   
     [sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grads[i-1], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[i]) for i in [1,2,3]]
   
     # plot loss
     for a,n in zip([axs0, axs1],["loss","loss_tasks"]):
         a[0].plot([res[i][n] for i in range(final_epoch)], linewidth=2)  
     
     # add titles
     axs0[0].set_title("Total loss", x= 0.25)
     axs0[1].set_title(r"Gradients: $\nu$", x = 0.35)
     [axs0[i].set_title(l) for i,l in zip([2,3], [r"$\mu_k$", r"$\sigma_k$"])]
     [axs1[i].set_title(l) for i,l in zip([2,3], [r"$\mu_k$", r"$\sigma_k$"])]
     axs1[1].set_title(r"Convergence: $\nu$", x =0.45)
     axs1[0].set_title("Individual losses", x = 0.45)
     
     # add legends
     axs1[1].legend([r"$\nu$"], loc="center right", handlelength=0.5, fontsize="small")
     axs1[2].legend(n_mus, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     axs1[3].legend(n_sigmas, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     
     [axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(4)]
     [axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(4)]
     
     [axs1[i].set_xlabel("epochs") for i in range(4)]
     [axs0[i].set_xlabel("epochs", color = "white") for i in range(4)]

     plt.savefig('../graphics/linreg_diagnostics.png', dpi = 300)
     
def plot_diagnostics_poisson(var, res, input_settings_learning):
    # define labels for plotting
    n_mus = [rf"$\mu_{i}$" for i in range(4)]
    n_sigmas = [rf"$\sigma_{i}$" for i in range(4)]
    
    # define color codes for plotting
    # betas 
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d"]
    final_epoch = input_settings_learning["epochs"]
    # norm of gradients
    grad_mu = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][0])) for i in range(final_epoch)],0)
    grad_sigma = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1])) for i in range(final_epoch)],0) 
    grads = [grad_mu, grad_sigma]

    fig = plt.figure(figsize=(6,3.5), constrained_layout=True)
    matplotlib.rcParams.update({'text.usetex': True, "font.family" : "serif",
                        "font.serif" : ["Computer Modern Serif"]})
    subfigs = fig.subfigures(2, 1)
    axs0 = subfigs[0].subplots(1, 3, sharex = True)
    axs1 = subfigs[1].subplots(1, 3, sharex = True)
    
    # plot convergence
    for i in range(4):
       axs1[1].plot(tf.squeeze(tf.stack(var[0][1],-1))[i,:], color = col_betas[i], linewidth=2)
       axs1[2].plot(tf.exp(tf.squeeze(tf.stack(var[1][1],-1))[i,:]), color = col_betas[i], linewidth=2)

    # plot gradients   
    [sns.scatterplot(x=tf.range(0,final_epoch,1).numpy(), y = grads[i-1], 
                   linewidth=0, alpha = 0.5, color="black", ax=axs0[i]) for i in [1,2]]
  
    # plot loss
    for a,n in zip([axs0, axs1],["loss","loss_tasks"]):
        a[0].plot([res[i][n] for i in range(final_epoch)], linewidth=2)  
    
    # add titles
    axs0[0].set_title("Total loss", x= 0.2)
    axs0[1].set_title(r"Gradients: $\mu_k$", x = 0.3)
    axs0[2].set_title(r"$\sigma_k$")
    axs1[1].set_title(r"Convergence: $\mu_k$", x =0.35)
    axs1[2].set_title(r"$\sigma_k$")
    axs1[0].set_title("Individual losses", x = 0.35)
    
    # add legends
    axs1[1].legend(n_mus, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
    axs1[2].legend(n_sigmas, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
    
    [axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(3)]
    [axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in range(3)]
    
    [axs1[i].set_xlabel("epochs") for i in range(3)]
    [axs0[i].set_xlabel("epochs", color = "white") for i in range(3)]
    plt.savefig('../graphics/pois_diagnostics.png', dpi = 300)
    
def plot_diagnostics_mlm(var, res, input_settings_learning):
     n_mus = [fr"$\mu_{i}$" for i in range(2)]
     n_sigmas = [fr"$\sigma_{i}$" for i in range(2)] 
     # names sigma tau
     name_stau = [fr"$\omega_{i}$" for i in range(2)] 
     # define color codes for plotting
     # betas 
     col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
     col_lambda = "#a6b7c6"
     final_epoch = input_settings_learning["epochs"]
     # norm of gradients
     grad_lambda = tf.stack([tf.squeeze(res[i]["gradients"][0]) for i in range(final_epoch)],0)
     grad_mu0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1][0])) for i in range(final_epoch)],0)
     grad_mu1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1][1])) for i in range(final_epoch)],0)
     grad_sigma0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][2][0])) for i in range(final_epoch)],0) 
     grad_sigma1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][2][1])) for i in range(final_epoch)],0) 
     grad_sigma_tau0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][3][0])) for i in range(final_epoch)],0) 
     grad_sigma_tau1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][3][1])) for i in range(final_epoch)],0) 
     
     grads = [tf.stack([grad_mu0,grad_mu1],-1), 
              tf.stack([grad_sigma0,grad_sigma1], -1), 
              tf.stack([grad_sigma_tau0,grad_sigma_tau1], -1)]
     
     fig = plt.figure(figsize=(6,4.5), constrained_layout=True)
     matplotlib.rcParams.update({'text.usetex': True, "font.family" : "serif",
                         "font.serif" : ["Computer Modern Serif"]})
     subfigs = fig.subfigures(3, 1)
     axs2 = subfigs[0].subplots(1, 2, sharex = True)
     axs0 = subfigs[1].subplots(1, 4, sharex = True)
     axs1 = subfigs[2].subplots(1, 4, sharex = True)
     
     # plot convergence
     axs1[0].plot(tf.exp(var[0][1]),color=col_lambda, linewidth=2)
     for i in range(2):
        axs1[1].plot(tf.squeeze(tf.stack(var[1][1],-1))[i,:], color = col_betas[i], linewidth=2)
        axs1[2].plot(tf.exp(tf.squeeze(tf.stack(var[3][1],-1))[i,:]), color = col_betas[i], linewidth=2)
        axs1[3].plot(tf.exp(tf.squeeze(tf.stack(var[2][1],-1))[i,:]), color = col_betas[i], linewidth=2)

     # plot gradients   
     sns.scatterplot(x=tf.range(0,final_epoch,1), y = grad_lambda, #grads[0][:,0], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[0])
     [sns.scatterplot(x=tf.range(0,final_epoch,1), y = grads[0][:,i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[1]) for i in range(2)]
     [sns.scatterplot(x=tf.range(0,final_epoch,1), y = grads[1][:,i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[2]) for i in range(2)]
     [sns.scatterplot(x=tf.range(0,final_epoch,1), y = grads[2][:,i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[3]) for i in range(2)]
   
     # plot loss
     for a,n in zip([0,1],["loss","loss_tasks"]):
         axs2[a].plot([res[i][n] for i in range(final_epoch)], linewidth=2)  
     
     # add titles
     axs2[0].set_title("Total loss", x= 0.12)
     axs0[0].set_title(r"Gradients: $\nu$", x = 0.35)
     [axs0[i].set_title(l) for i,l in zip([1,2,3], [r"$\mu_k$", r"$\sigma_k$", r"$\omega_k$"])]
     [axs1[i].set_title(l) for i,l in zip([1,2,3], [r"$\mu_k$", r"$\sigma_k$", r"$\omega_k$"])]
     axs1[0].set_title(r"Convergence: $\nu$", x =0.45)
     axs2[1].set_title("Individual losses", x = 0.2)
     
     # add legends
     axs1[0].legend([r"$\nu$"], loc="center right", handlelength=0.5, fontsize="small")
     axs1[1].legend(n_mus, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     axs1[2].legend(n_sigmas, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     axs1[3].legend(name_stau, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     
     [axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in [0,1,2,3]]
     [axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for i in [2,3]]
     axs1[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
     axs2[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
     axs2[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
     
     [axs1[i].set_xlabel("epochs") for i in range(4)]
     [axs0[i].set_xlabel("epochs", color = "white") for i in range(4)]

     plt.savefig('../graphics/mlm_diagnostics.png', dpi = 300)

def plot_diagnostics_weibull(var, res, input_settings_learning):
     n_mus = [fr"$\mu_{i}$" for i in range(2)]
     n_sigmas = [fr"$\sigma_{i}$" for i in range(2)] 
     name_stau = [fr"$\omega_{i}$" for i in range(2)] 
     
     # define color codes for plotting
     # betas 
     col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
     col_lambda = "#a6b7c6"
     final_epoch = input_settings_learning["epochs"]
     # norm of gradients
     grad_lambda = tf.stack([tf.squeeze(res[i]["gradients"][0]) for i in range(final_epoch)],0)
     grad_mu0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1][0])) for i in range(final_epoch)],0)
     grad_mu1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][1][1])) for i in range(final_epoch)],0)
     grad_sigma0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][2][0])) for i in range(final_epoch)],0) 
     grad_sigma1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][2][1])) for i in range(final_epoch)],0) 
     grad_sigma_tau0 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][3][0])) for i in range(final_epoch)],0) 
     grad_sigma_tau1 = tf.stack([tf.squeeze(tf.norm(res[i]["gradients"][3][1])) for i in range(final_epoch)],0) 
     
     grads = [tf.stack([grad_mu0,grad_mu1],-1), 
              tf.stack([grad_sigma0,grad_sigma1], -1), 
              tf.stack([grad_sigma_tau0,grad_sigma_tau1], -1)]
     
     fig = plt.figure(figsize=(6,4.5), constrained_layout=True)
     matplotlib.rcParams.update({'text.usetex': True, "font.family" : "serif",
                         "font.serif" : ["Computer Modern Serif"]})
     subfigs = fig.subfigures(3, 1)
     axs2 = subfigs[0].subplots(1, 2, sharex = True)
     axs0 = subfigs[1].subplots(1, 4, sharex = True)
     axs1 = subfigs[2].subplots(1, 4, sharex = True)
     
     # plot convergence
     axs1[0].plot(tf.exp(var[0][1]),color=col_lambda, linewidth=2)
     for i in range(2):
        axs1[1].plot(tf.squeeze(tf.stack(var[1][1],-1))[i,:], color = col_betas[i], linewidth=2)
        axs1[2].plot(tf.exp(tf.squeeze(tf.stack(var[3][1],-1))[i,:]), color = col_betas[i], linewidth=2)
        axs1[3].plot(tf.exp(tf.squeeze(tf.stack(var[2][1],-1))[i,:]), color = col_betas[i], linewidth=2)

     # plot gradients   
     sns.scatterplot(x=tf.range(0,final_epoch,1), y = grad_lambda, #grads[0][:,0], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[0])
     [sns.scatterplot(x=tf.range(0,final_epoch,1), y = grads[0][:,i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[1]) for i in range(2)]
     [sns.scatterplot(x=tf.range(0,final_epoch,1), y = grads[1][:,i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[2]) for i in range(2)]
     [sns.scatterplot(x=tf.range(0,final_epoch,1), y = grads[2][:,i], 
                    linewidth=0, alpha = 0.5, color="black", ax=axs0[3]) for i in range(2)]
   
     # plot loss
     for a,n in zip([0,1],["loss","loss_tasks"]):
         axs2[a].plot([res[i][n] for i in range(final_epoch)], linewidth=2)  
     
     # add titles
     axs2[0].set_title("Total loss", x= 0.12)
     axs0[0].set_title(r"Gradients: $\nu$", x = 0.35)
     [axs0[i].set_title(l) for i,l in zip([1,2,3], [r"$\mu_k$", r"$\sigma_k$", r"$\sigma_k$"])]
     [axs1[i].set_title(l) for i,l in zip([1,2,3], [r"$\mu_k$", name_stau[0], name_stau[1]])]
     axs1[0].set_title(r"Convergence: $\nu$", x =0.45)
     axs2[1].set_title("Individual losses", x = 0.2)
     
     # add legends
     axs1[0].legend([r"$\nu$"], loc="center right", handlelength=0.5, fontsize="small")
     axs1[1].legend(n_mus, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     axs1[2].legend(n_sigmas, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     axs1[3].legend(name_stau, ncol=2, labelspacing=0.2, columnspacing=0.4, handlelength=0.5, fontsize="small")
     
     [axs0[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) for i in [0,1,2,3]]
     [axs1[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f')) for i in [2,3]]
     axs1[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
     axs2[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
     axs2[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
     
     [axs1[i].set_xlabel("epochs") for i in range(4)]
     [axs0[i].set_xlabel("epochs", color = "white") for i in range(4)]

     plt.savefig('../graphics/weibull_diagnostics.png', dpi = 300)