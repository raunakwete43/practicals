import numpy as np
from scipy.stats import norm


def em(X, num_clusters=2, max_iters=100, tol=1e-4):
    X = np.array(X)

    mu = np.random.choice(X, num_clusters)
    sigma = np.full(num_clusters, np.var(X))
    pi = np.full(num_clusters, 1 / num_clusters)

    for iter in range(max_iters):
        resp = np.zeros((len(X), num_clusters))
        for i in range(num_clusters):
            resp[:, i] = pi[i] * norm.pdf(X, mu[i], np.sqrt(sigma[i]))
        resp /= resp.sum(axis=1, keepdims=True)
        N_k = resp.sum(axis=0)
        new_mu = np.sum(resp * X[:, np.newaxis], axis=0) / N_k
        new_sigma = np.sum(resp * (X[:, np.newaxis] - new_mu) ** 2, axis=0) / N_k
        new_pi = N_k / len(X)

        # Convergence Check
        if np.allclose(mu, new_mu, atol=tol) and np.allclose(
            sigma, new_sigma, atol=tol
        ):
            break

        mu, sigma, pi = new_mu, new_sigma, new_pi  # Update parameters

    return mu, sigma, pi, resp


X = [2.0, 2.2, 1.8, 6.0, 5.8, 6.2]

# Run EM algorithm
mu_final, sigma_final, pi_final, responsibilities_final = em(X)

# Print results
print(f"Final Means: {mu_final}")
print(f"Final Variances: {sigma_final}")
print(f"Final Mixing Coefficients: {pi_final}")
print("Final Responsibilities:")
print(responsibilities_final)
