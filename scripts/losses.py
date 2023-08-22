import tensorflow as tf

class energy_loss:
    """Class used to compute the MMD loss using an energy kernel"""
    def __init__(self):
        super(energy_loss, self).__init__()   
    
    def __call__(self, x, y, B, M, N, **kwargs):
        """
        Parameters
        ----------
        x : tensor, shape = (B,M)
            expert sample distribution
        y : tensor, shape = (B,N)
            model sample distribution
        B : int
            batch size
        M : int
            number of dimensions
        N : int
            number of dimensions
        """
        x = tf.expand_dims(tf.reshape(x, (B, M)), -1)
        y = tf.expand_dims(tf.reshape(y, (B, N)), -1)
        
        a = self.generate_weights(x, M)
        b = self.generate_weights(y, N)   
        
        return self.kernel_loss(x, y, a, b)

    def generate_weights(self, x, N):
        B, N, _ = x.shape
        return tf.divide(tf.ones(shape = (B, N), dtype = x.dtype), N)  
    
    # (x*x - 2xy + y*y)
    def squared_distances(self, x, y):
        # (B, N, 1)
        D_xx = tf.expand_dims(tf.math.reduce_sum(tf.multiply(x, x), axis = -1), axis = 2)
        # (B, 1, M)
        D_yy = tf.expand_dims(tf.math.reduce_sum(tf.multiply(y, y), axis = -1), axis = 1)
        # (B, N, M)
        D_xy = tf.matmul(x, tf.transpose(y, perm = (0, 2, 1)))
        return D_xx - 2*D_xy + D_yy
    
    # sqrt[(x*x - 2xy + y*y)]
    def distances(self, x, y):
        return tf.math.sqrt(tf.clip_by_value(self.squared_distances(x,y), 
                                             clip_value_min = 1e-8, 
                                             clip_value_max =  int(1e10)))   
    
    # - ||x-y|| = - sqrt[(x*x - 2xy + y*y)]
    def energy_kernel(self, x, y):
        """ Implements kernel norms between sampled measures.
        
        .. math::
            \\text{Loss}(\\alpha,\\beta) 
                ~&=~ \\text{Loss}\\big( \sum_{i=1}^N \\alpha_i \,\delta_{x_i} \,,\, \sum_{j=1}^M \\beta_j \,\delta_{y_j} \\big) 
                ~=~ \\tfrac{1}{2} \|\\alpha-\\beta\|_k^2 \\\\
                &=~ \\tfrac{1}{2} \langle \\alpha-\\beta \,,\, k\star (\\alpha - \\beta) \\rangle \\\\
                &=~ \\tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N  \\alpha_i \\alpha_j \cdot k(x_i,x_j) 
                  + \\tfrac{1}{2} \sum_{i=1}^M \sum_{j=1}^M  \\beta_i \\beta_j \cdot k(y_i,y_j) \\\\
                &-~\sum_{i=1}^N \sum_{j=1}^M  \\alpha_i \\beta_j \cdot k(x_i,y_j)
        where:
        .. math::
            k(x,y)~=~\\begin{cases}
                \exp( -\|x-y\|^2/2\sigma^2) & \\text{if loss = ``gaussian''} \\\\
                \exp( -\|x-y\|/\sigma) & \\text{if loss = ``laplacian''} \\\\
                -\|x-y\| & \\text{if loss = ``energy''} \\\\
            \\end{cases}
        """
        return -self.distances(x, y)    
    
    ## helper function
    def scal(self, a, f):
        B = a.shape[0]
        return tf.math.reduce_sum(tf.reshape(a, (B, -1)) * tf.reshape(f, (B, -1)), axis = 1)
      
    # k(x,y)=0.5*sum_i sum_j a_i a_j k(x_i,x_j) - sum_i sum_j a_i b_j k(x_i,y_j) + 0.5*sum_i sum_j b_i b_j k(y_i, y_j)
    def kernel_loss(self, x, y, a, b):
        B, N, _ = x.shape
        
        K_xx = self.energy_kernel(x,x) # (B,N,N)
        K_yy = self.energy_kernel(y,y) # (B,M,M)
        K_xy = self.energy_kernel(x,y) # (B,N,M)
        
        # (B,N)
        a_x = tf.squeeze(tf.matmul(K_xx, tf.expand_dims(a, axis = -1)))
        # (B,M)
        b_y = tf.squeeze(tf.matmul(K_yy, tf.expand_dims(b, axis = -1)))
        # (B,N)   
        b_x = tf.squeeze(tf.matmul(K_xy, tf.expand_dims(b, axis = -1)))
        
        K = 0.5*self.scal(a, a_x) + 0.5*self.scal(b, b_y) - self.scal(a, b_x)
        
        return tf.reduce_mean(K)
    