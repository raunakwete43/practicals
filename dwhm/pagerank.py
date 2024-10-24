import numpy as np


def pagerank(
    adj_matrix: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    # Number of pages (nodes)
    N = adj_matrix.shape[0]

    # Transition matrix (normalize each row to handle outgoing links)
    out_degree = np.sum(adj_matrix, axis=1)
    transition_matrix = adj_matrix / out_degree[:, None]

    # Handle dangling nodes (pages with no outgoing links)
    transition_matrix = np.nan_to_num(transition_matrix)

    # Initialize PageRank values equally
    pagerank_values = np.ones(N) / N

    # Iterative calculation of PageRank
    for iteration in range(max_iter):
        previous_pagerank = pagerank_values.copy()

        # Calculate new PageRank values based on the formula
        pagerank_values = (1 - damping) / N + damping * transition_matrix.T.dot(
            previous_pagerank
        )

        # Check for convergence
        if np.linalg.norm(pagerank_values - previous_pagerank, ord=1) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return pagerank_values


# Example usage
if __name__ == "__main__":
    # Adjacency matrix representing the link structure between 4 pages
    adj_matrix = np.array(
        [
            [0, 1, 1, 0],  # Page 0 links to Page 1 and Page 2
            [0, 0, 1, 1],  # Page 1 links to Page 2 and Page 3
            [0, 0, 0, 1],  # Page 2 links to Page 3
            [1, 0, 0, 0],  # Page 3 links to Page 0
        ]
    )

    pagerank_scores = pagerank(adj_matrix)

    print("PageRank scores: ", pagerank_scores)
