functions {
  // ... function declarations and definitions ...
}

data {
  int<lower=1> n;    // number of users
  int<lower=1> k;    // number of items
  int<lower=1,upper=min(n,k)> r;    // rank

  int<lower=1,upper=n*k> N;            // number of responses
  int<lower=1,upper=n> user_idx[N];    // i
  int<lower=1,upper=k> item_idx[N];    // j
  real R_obs[N];      // R_{ij}
}

transformed data {
   // ... declarations ... statements ...
}

parameters {
  vector[r] u[n];    // user factors
  vector[r] v[k];    // item factors
  real<lower=0> sigma;    // response noise

  vector[r] mu_U;
  vector[r] mu_V;
  cholesky_factor_corr[r] L_Omega_U;
  cholesky_factor_corr[r] L_Omega_V;
  vector<lower=0>[r] L_sigma_U;
  vector<lower=0>[r] L_sigma_V;
}

transformed parameters {
  matrix[r,r] L_Sigma_U;
  matrix[r,r] L_Sigma_V;
  real u_dot_v[N];

  L_Sigma_U = diag_pre_multiply(L_sigma_U, L_Omega_U);
  L_Sigma_V = diag_pre_multiply(L_sigma_V, L_Omega_V);

  for (i in 1:N)
    u_dot_v[i] = dot_product(u[user_idx[i]], v[item_idx[i]]);
}

model {
  mu_U ~ normal(0, 5);
  mu_V ~ normal(0, 5);
  L_sigma_U ~ cauchy(0, 2.5);
  L_sigma_V ~ cauchy(0, 2.5);
  L_Omega_U ~ lkj_corr_cholesky(2);
  L_Omega_V ~ lkj_corr_cholesky(2);

  u ~ multi_normal_cholesky(mu_U, L_Sigma_U);
  v ~ multi_normal_cholesky(mu_V, L_Sigma_V);

  R_obs ~ normal(u_dot_v, sigma);    
}

generated quantities {
}
