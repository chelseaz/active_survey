import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Dirichlet, MultivariateNormalTriL
from tensorflow.python.framework import dtypes

from common import CompletionResult, timing
from ordinal_logit import OrdinalLogit

def logit(p):
    return tf.log(p / (1-p))

# response_freq is K x (max # values) array of response counts
def get_cutpoints(response_freq):
    # Using Dirichlet prior to define cutpoints, per rstanarm:
    # https://cran.r-project.org/web/packages/rstanarm/vignettes/polr.html
    # Concentration parameters come from response counts
    pi = Dirichlet(tf.cast(response_freq, dtypes.float32))
    cumpi = tf.cumsum(pi, axis=1)
    cutpoints = logit(tf.clip_by_value(cumpi, 1e-6, 1-1e-6))[:,:-1]
    # exclude the final cumulative probability of 1, ensure numerical stability

    # In order to represent cutpoints for all questions in a single 2d array,
    # we are relying on cutpoint collapse for questions with fewer allowable response values
    return cutpoints


class OrdinalLogitPMF(object):
    """
    Probabilistic matrix factorization with ordinal logit likelihood,
    estimated with variational inference.
    """

    def __init__(self, rank, 
        override_preds=False  # whether to override predictions with given (true) values
    ):
        self.rank = rank
        self.override_preds = override_preds

    def set_response_freq(self, response_freq):
        self.response_freq = response_freq
    
    @timing
    def fit_transform(self, X):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        N, K = X.shape

        I_data = ~np.isnan(X)
        R_data = X[I_data].astype('int32')

        with tf.Session() as sess:
            # Define model in edward
            I = tf.placeholder(tf.float32, [N, K])
            U = Normal(loc=0.0, scale=1.0, sample_shape=[self.rank, N])
            V = Normal(loc=0.0, scale=1.0, sample_shape=[self.rank, K])
            logits = tf.matmul(tf.transpose(U), V)

            cutpoints = get_cutpoints(self.response_freq)

            R = OrdinalLogit(logits=logits, cutpoints=cutpoints, indicators=I)

            # Define variational parameters
            qU = Normal(loc=tf.get_variable("qU/loc", [self.rank, N]),
                        scale=tf.nn.softplus(
                            tf.get_variable("qU/scale", [self.rank, N])))
            qV = Normal(loc=tf.get_variable("qV/loc", [self.rank, K]),
                        scale=tf.nn.softplus(
                            tf.get_variable("qV/scale", [self.rank, K])))

            # Run variational inference
            inference = ed.KLqp({U: qU, V: qV}, data={R: R_data, I: I_data})
            inference.run()

            # Use posterior mean for prediction
            U_pred = qU.mean().eval()
            V_pred = qV.mean().eval()
            R_pred = OrdinalLogit(
                         # if we don't use qU.mean() and qV.mean(), edward will draw from qU and qV
                         logits=tf.matmul(tf.transpose(qU.mean()), qV.mean()),
                         cutpoints=cutpoints,
                         indicators=tf.ones(X.shape)
                     ).mean().eval().reshape(X.shape)

            if self.override_preds:
                R_pred[I_data] = R_data

        print("Resetting default graph")
        tf.reset_default_graph()

        return CompletionResult(
            X_filled=R_pred,
            U=U_pred.T,
            V=V_pred,
            S=None,
            rank=self.rank
        )



# Ordinal logit model for single user, up to k observations, latent dimension d
class OrdinalLogitRegression(object):
    def __init__(self, k, d, max_n_cutpoints):
        V = tf.placeholder(tf.float32, [k, d])
        prior_mean = tf.placeholder(tf.float32, [d])
        prior_cov = tf.placeholder(tf.float32, [d, d])
        u_i = MultivariateNormalTriL(loc=prior_mean, scale_tril=tf.cholesky(prior_cov))
        cutpoints = tf.placeholder(tf.float32, [k, max_n_cutpoints])
        I = tf.placeholder(tf.float32, [1, k])

        logits = tf.expand_dims(ed.dot(V, u_i), 0)

        R_i = OrdinalLogit(
            logits=logits, 
            cutpoints=cutpoints,
            indicators=I)

        def map_estimate(V_data, R_data, I_data, cutpoints_data, prior_mean_data, prior_prec_data, 
            n_print=10, n_iter=600):

            prior_cov_data = np.linalg.inv(prior_prec_data)

            qu_i = ed.models.PointMass(params=tf.Variable(np.zeros(d), dtype=dtypes.float32))
            inference = ed.MAP({u_i: qu_i}, data={
                V: V_data, R_i: R_data, I: I_data[np.newaxis,:],
                cutpoints: cutpoints_data, 
                prior_mean: prior_mean_data,
                prior_cov: prior_cov_data})
            inference.initialize(n_print=n_print, n_iter=n_iter)
            inference.run()
            return qu_i.eval()

        self.map_estimate = map_estimate



# Ordinal logit model for single user, single observation, latent dimension d
class OrdinalLogitSingleObs(object):
    def __init__(self, d, max_n_cutpoints):
        u_i = tf.placeholder(tf.float32, [d])  # fixed user i
        v_j = tf.placeholder(tf.float32, [1, d])  # choose question j
        cutpoints_j = tf.placeholder(tf.float32, [1, max_n_cutpoints])
        R_ij = OrdinalLogit(
            logits=tf.matmul(v_j, tf.expand_dims(u_i, -1)), 
            cutpoints=cutpoints_j,
            indicators=tf.ones((1, 1))  # single observation will be observed
        )
        R_ij_cumul_probs = R_ij._cumul_probs()  # create only once to avoid memory leak

        R_next = tf.Variable([1], dtype=dtypes.int32)
        R_next_hess = tf.hessians(R_ij.log_prob(R_next.value()), u_i)[0]

        # observed Fisher information, see https://en.wikipedia.org/wiki/Observed_information
        def new_point_info(v_new, u_map, R_new, cutpoints_new):
            return -R_next_hess.eval(
                feed_dict={v_j: v_new, u_i: u_map, cutpoints_j: cutpoints_new, R_next: R_new})

        # actual Fisher information (expectation taken over y_new)
        def new_point_info_expected(v_new, u_map, max_level, cutpoints_new):
            # integrate out R_next
            cumul_probs = R_ij_cumul_probs.eval(
                feed_dict={v_j: v_new, u_i: u_map, cutpoints_j: cutpoints_new})[0]
            probs = np.append(cumul_probs, 1) - np.append(0, cumul_probs)
            levels = np.arange(max_level+1)
            info_per_level = np.array([new_point_info(
                v_new, u_map, [level], cutpoints_new) for level in levels])
            return np.sum(probs[:(max_level+1),np.newaxis,np.newaxis] * info_per_level, axis=0)

        self.new_point_info = new_point_info
        self.new_point_info_expected = new_point_info_expected
