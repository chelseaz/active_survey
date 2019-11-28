library(dplyr)
library(ggplot2)
library(reshape2)
library(rstan)
# avoid recompilation of unchanged Stan programs
rstan_options(auto_write = TRUE)

dataset <- 'cces18'
rank <- 4

filepath <- '/Users/cyz/active_survey/'
responses <- read.csv(paste0(filepath, 'data/cces/', dataset, '_cs.csv'), stringsAsFactors = T)
responses <- responses %>% select(-X)

# Draw a sample of responses
n_sample <- 1000
n <- nrow(responses)
sample_idx <- sample(1:n, n_sample, replace = F)
sample_responses <- responses[sample_idx,]

R <- sample_responses
colnames(R) <- 1:ncol(R)
R_long <- R %>% mutate(user_idx = 1:n()) %>%
    melt(id.vars = 'user_idx', variable.name = 'item_idx', value.name = 'response') %>% 
    mutate(item_idx = as.integer(item_idx)) %>%
    na.omit()

data_list <- list(
    n = nrow(R),
    k = ncol(R),
    r = rank,
    N = nrow(R_long),
    user_idx = R_long$user_idx,
    item_idx = R_long$item_idx,
    R_obs = R_long$response
)

stan_model <- stan_model(file = paste0(filepath, 'stan/pmf-normal.stan'))
fit <- sampling(stan_model, data = data_list, cores = 4, chains = 4, iter = 2000)

summary(fit)
samples <- extract(fit, pars=c('sigma'), inc_warmup=F)

# Posterior of sigma (noise variance)
options(repr.plot.width=4, repr.plot.height=3)
ggplot(data.frame(sigma=samples[[1]]), aes(x=sigma)) +
    geom_histogram() +
    ggtitle("Posterior draws of noise variance")
ggsave(filename = paste0(filepath, 'stan/sigma_posterior_', dataset, '_rank', rank, '.png'),
       width=4, height=3, units='in')

# 95% credible interval for sigma
quantile(samples[[1]], probs=c(0.05,0.95))
