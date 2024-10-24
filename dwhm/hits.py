import numpy as np


def hits_algorithm(adj_matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-5):
    # Number of pages (nodes)
    n = adj_matrix.shape[0]

    # Initialize hub and authority scores to 1 for all pages
    hubs = np.ones(n)
    authorities = np.ones(n)

    for iteration in range(max_iter):
        # Previous hub and authority scores (to check for convergence)
        prev_hubs = hubs.copy()
        prev_authorities = authorities.copy()

        # Update authority scores: auth(i) = sum(hub(j)) for all pages j that link to i
        authorities = adj_matrix.T.dot(hubs)

        # Update hub scores: hub(i) = sum(auth(j)) for all pages i that link to j
        hubs = adj_matrix.dot(authorities)

        # Normalize the hub and authority scores to prevent overflow
        authorities /= np.linalg.norm(authorities, 2)
        hubs /= np.linalg.norm(hubs, 2)

        # Check for convergence (if the change in scores is below the tolerance)
        if np.all(np.abs(hubs - prev_hubs) < tol) and np.all(
            np.abs(authorities - prev_authorities) < tol
        ):
            print(f"Converged after {iteration+1} iterations.")
            break

    return hubs, authorities


# Eample usage
if __name__ == "__main__":
    # Adjacency matrix representing link structure between 4 pages
    adj_matrix = np.array(
        [
            [0, 1, 1, 0],  # Page 0 links to Page 1 and Page 2
            [0, 0, 1, 1],  # Page 1 links to Page 2 and Page 3
            [0, 0, 0, 1],  # Page 2 links to Page 3
            [0, 0, 0, 0],  # Page 3 has no outgoing links
        ]
    )

    hubs, authorities = hits_algorithm(adj_matrix)

    print("Hub scores: ", hubs)
    print("Authority scores: ", authorities)
