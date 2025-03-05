import optax
import jax.numpy as jnp
import jax
from customLayers.linears import MatrixVariateParameter


def discriminant(param):
    """ Discriminate between Bayesian parameters"""
    return hasattr(param, 'mu') and hasattr(param, 'sigma1') and hasattr(param, 'sigma2') and param.mu is not None and param.sigma1 is not None and param.sigma2 is not None

def solve_matrix_equation(v_mat, e_mat):
    """ 
    Solves the equation for X s.t. XX^T + VV^T E X^T - VV^T = 0
    """
    ve_product = jnp.dot(v_mat, e_mat)
    b_mat = v_mat + 0.25 * jnp.dot(ve_product, ve_product.T)
    left_mat, diag_mat, right_mat = jnp.linalg.svd(b_mat)
    assert jnp.min(diag_mat) > 0, "v_mat is singular!"
    l_mat = jnp.dot(jnp.dot(left_mat, jnp.diag(jnp.sqrt(diag_mat))), right_mat.T)
    inv_l_mat = jnp.dot(jnp.dot(right_mat, jnp.diag(1.0 / jnp.sqrt(diag_mat))), left_mat.T)
    s_mat, lambda_mat, w_mat = jnp.linalg.svd(jnp.dot(inv_l_mat, ve_product))
    q_mat = jnp.dot(s_mat, w_mat.T)
    x_mat = jnp.dot(l_mat, q_mat) - 0.5 * ve_product
    return x_mat
    
    

def foovbmatrixvariate(
        lr_mu: float = 1,
        lr_sigma: float = 1,
        clamp_grad: float = 0.) -> optax.GradientTransformation:
    """
    Optax gradient transformation for foovb-matrixvariate.

    Args:
        lr_mu: Learning rate for mu parameters.
        lr_sigma: Learning rate for sigma parameters.
        clamp_grad: Gradient clamping threshold.
    Returns:
        optax.GradientTransformation: The BGD update rule.
    """

    def init(params):
        # Check if all parameters are bayesian i.e that the path name contains mu or sigma
        return {
            'step': 0,
        }

    def update(gradients, state, params=None):
        state['step'] += 1
        # Update the parameters using sgd

        def update_matrixvariate(param, grad):
            """ Update the parameters based on the gradients and the prior"""
            # If clamp_grad > 0, then clamp gradients between -clamp_grad/sigma and clamp_grad/sigma
            if clamp_grad > 0:
                grad = jax.tree_util.tree_map(
                    lambda x: jnp.clip(x, -clamp_grad / param.sigma, clamp_grad / param.sigma), grad)
            
            mu = param.mu - lr_mu * ((param.sigma2 @ param.sigma2.T) @ grad.mu) @ (param.sigma1 @ param.sigma1.T) 
            sigma1 = - lr_sigma * solve_matrix_equation(param.sigma1, grad.sigma1)
            sigma2 = - lr_sigma * solve_matrix_equation(param.sigma2, grad.sigma2)
            return MatrixVariateParameter(mu, sigma1, sigma2)

        updates = map(
            update_matrixvariate,
            params,
            gradients,
            is_leaf=discriminant
        )
        return updates, state

    return optax.GradientTransformation(init, update)
