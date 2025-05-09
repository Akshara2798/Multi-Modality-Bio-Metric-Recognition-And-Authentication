import numpy as np
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
import random
import math
class Sim_GHO_Optimization(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                  amsgrad=False, weight_decay=None, clipnorm=None, clipvalue=None,
                  global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                  ema_overwrite_frequency=None, jit_compile=True, name="En_GHop_Optimization", **kwargs):
        super().__init__(name=name, weight_decay=weight_decay, clipnorm=clipnorm,
                          clipvalue=clipvalue, global_clipnorm=global_clipnorm, use_ema=use_ema,
                          ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency,
                          jit_compile=jit_compile, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        
    

    def Sim_GHO(self,gradient):
        
        def model_accuracy(x):
            accuracy = 0.9 + 0.1 * (sum(x) / len(x))  
            return accuracy
        def objective_function(x):
            accuracy = model_accuracy(x)
            return -accuracy
        
        # Grasshopper Optimization Algorithm
        def goa(objective_function, lb, ub, dim, n_grasshoppers, max_iterations):
            c_max = 1
            c_min = 0.00004
            T0 = 1000  # Initial temperature
            alpha = 0.9  # Cooling rate
            max_iterations_sa = 1000
            
            # Grasshopper Optimization Algorithm parameters
            n_grasshoppers = 30
            max_iterations_goa = 100
            lb = -10
            ub = 10
            dim = 5
            # Initialize the population
            grasshoppers = np.random.uniform(lb, ub, (n_grasshoppers, dim))
            grasshoppers_fitness = np.apply_along_axis(objective_function, 1, grasshoppers)
        
            for t in range(max_iterations):
                c = c_max - t * ((c_max - c_min) / max_iterations)
                for i in range(n_grasshoppers):
                    s_i = np.zeros(dim)
                    for j in range(n_grasshoppers):
                        if i != j:
                            distance = np.linalg.norm(grasshoppers[i] - grasshoppers[j])
                            r_ij = (grasshoppers[j] - grasshoppers[i]) / (distance + 1e-14)
                            s_ij = c * ((ub - lb) / 2) * np.exp(-distance / 2) * r_ij
                            s_i += s_ij
                    grasshoppers[i] = np.clip(grasshoppers[i] + s_i, lb, ub)
                
                grasshoppers_fitness = np.apply_along_axis(objective_function, 1, grasshoppers)
            
            best_grasshopper = grasshoppers[np.argmin(grasshoppers_fitness)]
            return best_grasshopper
        
        # Simulated Annealing
        def simulated_annealing(objective_function, x0, lb, ub, T0, alpha, max_iterations):
            max_iterations_sa = 1000
            T0 = 1000  # Initial temperature
            alpha = 0.9  # Cooling rate
           
            x = x0
            T = T0
            for i in range(max_iterations):
                x_new = x + np.random.uniform(-1, 1, x.shape)
                x_new = np.clip(x_new, lb, ub)
                delta_E = objective_function(x_new) - objective_function(x)
                if delta_E < 0 or random.uniform(0, 1) < math.exp(-delta_E / T):
                    x = x_new
                T = alpha * T
            return x
        
       
        def hybrid_gsa_sa(objective_function, lb, ub, dim, n_grasshoppers, max_iterations_goa, T0, alpha, max_iterations_sa):
            # Step 1: Optimize with GOA
            best_solution_goa = goa(objective_function, lb, ub, dim, n_grasshoppers, max_iterations_goa)
            
            
            best_solution_hybrid = simulated_annealing(objective_function, best_solution_goa, lb, ub, T0, alpha, max_iterations_sa)
            
            return best_solution_hybrid
     

        
    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        self.Sim_GHO(gradient)
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        if isinstance(gradient, tf.IndexedSlices):
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

 
 
