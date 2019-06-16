import tensorflow as tf

from edward.models import RandomVariable
from tensorflow.python.framework import dtypes
from tensorflow.contrib.distributions import Distribution, NOT_REPARAMETERIZED

# Inputs are matrices, but value of random variable is a vector 
# whose length is the number of revealed entries (data_size).
# This is to accommodate a different number of cutpoints per question.
# So far the only values supported are 0, 1, ..., n_cutpoints
# per component of the random variable.
class distributions_OrdinalLogit(Distribution):
    def __init__(self, logits, cutpoints, indicators, name="OrdinalLogit"):
        self._logits = logits  # N x K tensor
        self._cutpoints = cutpoints  # K x max(N_CUT_VEC) tensor
        self._indicators = indicators  # N x K tensor
        
        parameters = dict(locals())
        super(distributions_OrdinalLogit, self).__init__(
            dtype=dtypes.int32,
            reparameterization_type=NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True,
            parameters=parameters,
            graph_parents=[self._logits, self._cutpoints, self._indicators],
            name=name
        )

    def _cumul_probs(self):
        # form a N x K x max(N_CUT_VEC) tensor representing sum of logits and cutpoints
        # use broadcasting
        logits_with_cutpoints = tf.expand_dims(self._logits, -1) + tf.expand_dims(self._cutpoints, 0)
        # now subset to a data_size x max(N_CUT_VEC) array using indicators
        select_logits_with_cutpoints = tf.boolean_mask(logits_with_cutpoints, self._indicators)
        return tf.sigmoid(select_logits_with_cutpoints)  # data_size x max(N_CUT_VEC)
        
    def _log_prob(self, value):        
        cumul_probs = self._cumul_probs()
        data_size = tf.shape(cumul_probs)[0]
        level_probs = tf.concat([cumul_probs, tf.expand_dims(tf.ones([data_size]), -1)], axis=1) - \
                        tf.concat([tf.expand_dims(tf.zeros([data_size]), -1), cumul_probs], axis=1)
        levels = value
        indices_2d = tf.transpose(tf.stack([tf.range(data_size), levels]))  # n x 2
        selected_probs = tf.gather_nd(level_probs, indices_2d)
        
        return tf.reduce_sum(tf.log(selected_probs))
    
    def _sample_n(self, n, seed=None):
        cumul_probs = self._cumul_probs()
        data_size = tf.shape(cumul_probs)[0]
        new_shape = [n, data_size]
        sample_probs = tf.expand_dims(tf.random_uniform(new_shape), -1)
        booleans = tf.greater(sample_probs, cumul_probs)  # using broadcasting
        samples = tf.reduce_sum(tf.cast(booleans, dtype=dtypes.int32), axis=-1)
        return samples
    
    def mean(self):
        # Using the identity that the mean is \sum_{i=0}^k P(X > i)
        cumul_probs = self._cumul_probs()
        means = tf.reduce_sum(1-cumul_probs, axis=1)
        return means
        

def __init__(self, *args, **kwargs):
    RandomVariable.__init__(self, *args, **kwargs)

_name = 'OrdinalLogit'
_candidate = distributions_OrdinalLogit
__init__.__doc__ = _candidate.__init__.__doc__
_globals = globals()
_params = {'__doc__': _candidate.__doc__,
           '__init__': __init__,
           'support': 'countable'}
_globals[_name] = type(_name, (RandomVariable, _candidate), _params)