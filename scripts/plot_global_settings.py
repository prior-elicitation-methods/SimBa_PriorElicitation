import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
plt.style.use('stylesheet.mplstyle')
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

############### global settings for plotting #############

### prior predictions
# for q-q-plots
q_cols_total = ["#49c1db", "#3faed9", "#499ad2", "#5c84c5", "#6f6db2", "#7c5697", "#833e78", "#822556", "#780d33"]
# blueish arrows
q_cols_exp = ["#49c1db", "#8fc2d5","#6596b6","#3c6a96","#123e76"]
# brownish lines
q_cols_mod = ["#d8a753", "#c97f44","#ad4b41","#983437","#780d33"]
# brown hist (model-implied)
hist_col_mod = "#d88853"
hist_col_exp = "#6596b6"
### Convergence plots
# beta coefficients
col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
# hyperpar of random noise
col_nu = "#a6b7c6"

def density_exp_target_quant(res, axes, **kwargs):
    return sns.kdeplot(res, linestyle = "--", color = "black", ax = axes, **kwargs)
# for single plot (with alpha)
def hist_mod_target_quant(res, axes, **kwargs):
    return sns.histplot(res, ax = axes, bins = 30, fill = True,  alpha = 0.3, color = hist_col_mod, stat = "density", **kwargs)
# for subplots (no alpha)
def hist_target_elicited(res, axes, type):
    if type == "exp": 
        c = hist_col_exp
    else: 
        c = hist_col_mod
    return sns.histplot(res, bins = 30, fill = True, color = c, stat = "density", ax = axes)
def quantiles_mod(res, y, axes, **kwargs):
    return [axes.vlines(tf.reduce_mean(res, 0)[i], ymin=y[0], ymax = y[1], lw = 3, color = q_cols_mod[j], **kwargs) for i,j in zip([0,2,4,6,8], range(5))]
def quantiles_exp(res, y, axes, **kwargs):
    return [axes.plot(tf.reduce_mean(res, 0)[i], y, marker = "^", ms = 5, color = q_cols_exp[j], **kwargs) for i,j in zip([0,2,4,6,8], range(5))]
def legend_quantiles(axes):
    return axes.legend([fr"$q_{{{i}}}$" for i in [0.1, 0.3, 0.5, 0.7, 0.9]], loc = "upper right")
def tex_label(var, k):
    if var == "mu":
        return [rf"$\mu_{i}$" for i in range(k)]
    if var == "sigma":
        return [rf"$\sigma_{i}$" for i in range(k)]
    if var == "sigma_tau":
        return [rf"$\omega_{i}$" for i in range(k)]


def final_hyperpar_avg(hyperpars, hyperpar_idx, l_values, epochs, sigma = False):
    if sigma: 
        param = tf.reduce_mean(hyperpars[1][epochs-l_values:epochs]).numpy()
    else:
        param = tf.reduce_mean([hyperpars[1][epochs-l_values:epochs][i][hyperpar_idx] for i in range(l_values)]).numpy()
    return param
def absolute_error(learned_hyp, true_hyp, epochs, log = False):
    if log:
        return tf.stack([tf.stack(tf.exp(learned_hyp[1][i])-tf.cast(true_hyp, dtype=tf.float32),0) for i in range(epochs)],-1)
    else: 
        return tf.stack([tf.stack(learned_hyp[1][i]-tf.cast(true_hyp, dtype=tf.float32),0) for i in range(epochs)],-1)
def plot_prior_dist(axes, x_range, pdf_learned, pdf_true, col, label):
    axes.plot(x_range, pdf_learned, color = col, linewidth = 3, label = label)
    axes.plot(x_range, pdf_true, linestyle="dotted",  color ="black", linewidth = 2)
def plot_error_traj(axes, error, col, label, k, sigma = False):
    if sigma: 
        axes.plot(error, color = col, linewidth = 3, label=label)
    else:
        [axes.plot(error[i,:], color = col[i], linewidth = 3, label=label[i]) for i in range(k)]
    axes.axhline(y = 0., color = 'black', linestyle = '--', linewidth=2)
def single_qq_plot(axes, i, elicited_quant_exp, elicited_quant_mod, sub_title, d, Q = 9):
    [axes.plot(tf.reduce_mean(elicited_quant_mod, 0)[j,i], 
               tf.reduce_mean(elicited_quant_exp, 0)[j,i], "o", ms=5, color = q_cols_total[j]) for j in range(Q)]
    axes.axline((0,0), slope=1, color = "black", linestyle = "dashed")
    axes.set_xlim(tf.math.reduce_min(tf.reduce_mean(elicited_quant_mod, 0)[:,i])-d,
                    tf.math.reduce_max(tf.reduce_mean(elicited_quant_mod, 0)[:,i])+d)
    axes.set_ylim(tf.math.reduce_min(tf.reduce_mean(elicited_quant_mod, 0)[:,i])-d,
                    tf.math.reduce_max(tf.reduce_mean(elicited_quant_mod, 0)[:,i])+d)
    axes.set_title(sub_title)  
    axes.set_xticks([]) 
    axes.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))   

def mean_sd_plot(elicited_sim, elicited_exp, axes, dev, i):
    for l,c in zip([f"mean_{i}", f"sd_{i}"], ["red", "blue"]): 
        axes.axline((0,0), slope=1, color = "black", linestyle = "dashed")
        axes.axvline(x = tf.reduce_mean(elicited_sim[l]), ymin = 0, 
                        ymax = tf.cast(tf.reduce_mean(elicited_exp[l]), tf.int32).numpy(),
                        linestyle = "dotted", color = "black", lw = 1)
        axes.axhline(y = tf.reduce_mean(elicited_exp[l]), xmin = 0, 
                        xmax = tf.cast(tf.reduce_mean(elicited_sim[l]), tf.int32).numpy(),
                        linestyle = "dotted", color = "black", lw = 1)
        axes.plot(tf.reduce_mean(elicited_sim[l]), 
                    tf.reduce_mean(elicited_exp[l]), "o", ms=8, color = c)
    axes.set_xlim(tf.math.reduce_min(tf.reduce_mean(elicited_sim[f"sd_{i}"], 0))-dev,
                        tf.math.reduce_max(tf.reduce_mean(elicited_exp[f"mean_{i}"], 0))+dev)
    axes.set_ylim(tf.math.reduce_min(tf.reduce_mean(elicited_sim[f"sd_{i}"], 0))-dev,
                        tf.math.reduce_max(tf.reduce_mean(elicited_exp[f"mean_{i}"], 0))+dev)


