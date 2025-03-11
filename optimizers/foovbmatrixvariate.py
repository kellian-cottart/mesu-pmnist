import optax
import jax.numpy as jnp
from jax.tree import map
from customLayers.linears import MatrixVariateParameter
import jax.numpy as jnp
import jax

def discriminant(param):
    """ Discriminate between Bayesian parameters"""
    return hasattr(param, 'mu') and hasattr(param, 'sigma_1') and hasattr(param, 'sigma_2') and param.mu is not None and param.sigma_1 is not None and param.sigma_2 is not None


def solve_matrix_equation(v_mat, e_mat):
    """ 
    Solves the equation for X s.t. XX^T + VV^T E X^T - VV^T = 0
    """
    # B = V + 0.25 VE(VE)^T
    ve_product = v_mat @ e_mat
    b_mat = v_mat + 0.25 * (ve_product @ ve_product.T)
    left_mat, diag_mat, right_mat = jnp.linalg.svd(b_mat, full_matrices=False)
    # extract diagonal matrix
    # L = B^0.5
    temp_diag = jnp.diag(jnp.sqrt(diag_mat))
    l_mat = left_mat @ temp_diag @ right_mat.T
    inv_l_mat = right_mat @ jnp.diag(jnp.reciprocal(jnp.sqrt(diag_mat))) @ left_mat.T
    # S * LBDA * W = L^-1 * V * E
    s_mat, lambda_mat, w_mat = jnp.linalg.svd(
        inv_l_mat @ ve_product, full_matrices=False)
    # Q = SW^T
    q_mat = s_mat @ w_mat.T
    # X = LQ - 0.5 VE
    x_mat = l_mat @ q_mat - 0.5 * ve_product
    return x_mat


def foovbmatrixvariate(lr_mu: float = 1) -> optax.GradientTransformation:
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
            grad_mu = grad.mu
            grad_sigma_1 = grad.sigma_1
            grad_sigma_2 = grad.sigma_2
            
            if len(grad.mu.shape) == 1:
                grad_mu = jnp.expand_dims(grad_mu, axis=-1)
            large_sigma_1 = param.sigma_1 @ param.sigma_1.T
            large_sigma_2 = param.sigma_2 @ param.sigma_2.T
            update_mu = - lr_mu * large_sigma_2 @ grad_mu @ large_sigma_1
            update_sigma_1 = solve_matrix_equation(
                large_sigma_1, grad_sigma_1) - param.sigma_1
            update_sigma_2 = solve_matrix_equation(
                large_sigma_2, grad_sigma_2) - param.sigma_2            
            if len(grad.mu.shape) == 1:
                update_mu = jnp.squeeze(update_mu, axis=-1)
            return MatrixVariateParameter(update_mu, update_sigma_1, update_sigma_2)

        updates = map(
            update_matrixvariate,
            params,
            gradients,
            is_leaf=discriminant
        )
        return updates, state

    return optax.GradientTransformation(init, update)
