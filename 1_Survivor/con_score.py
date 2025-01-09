import pandas as pd
import numpy as np
import networkx as nx

def compute_con_u_v(adj_matrix, u, v):
    """Compute the number of common out-neighbors for nodes u and v."""
    return sum(min(adj_matrix[u, k], adj_matrix[v, k]) for k in range(len(adj_matrix)))

def compute_con_u(adj_matrix, u):
    """Compute the CON(u) for a node u."""
    return sum(compute_con_u_v(adj_matrix, u, v) for v in range(len(adj_matrix)) if u != v)

def compute_con_scores(graph):
    """Compute the CON score for all nodes in the graph."""
    adj_matrix = nx.to_numpy_array(graph, dtype=int)
    
    con_scores = {}
    for node in graph.nodes():
        u_index = list(graph.nodes()).index(node)
        con_scores[node] = compute_con_u(adj_matrix, u_index)
    return con_scores


def calculate_CON_multiorder(df):
    con_scores = []
    
    for season in df['season'].unique():
        df_season = df[df['season'] == season]
        
        for episode in df_season['episode'].unique():
            curr_epi_df = df_season[df_season['episode'] == episode]
            castaways = list(set(curr_epi_df['castaway_uid']) | set(curr_epi_df['vote_uid']))
            n = len(castaways)
            castaway_index = {castaway: idx for idx, castaway in enumerate(castaways)}

            adj_matrix = np.zeros((n, n), dtype=int)
            for _, row in curr_epi_df.iterrows():
                voter_idx = castaway_index[row['castaway_uid']]
                voted_for_idx = castaway_index[row['vote_uid']]
                adj_matrix[voter_idx, voted_for_idx] += 1

            adj_matrix_squared = np.dot(adj_matrix, adj_matrix)
            adj_mat_large = adj_matrix + adj_matrix_squared

            for u in range(n):
                con_1st_order = sum(min(adj_matrix[u, k], adj_matrix[v, k]) for v in range(n) for k in range(n) if u != v)
                con_2nd_order = sum(min(adj_matrix_squared[u, k], adj_matrix_squared[v, k]) for v in range(n) for k in range(n) if u != v)
                con_large = sum(min(adj_mat_large[u, k], adj_mat_large[v, k]) for v in range(n) for k in range(n) if u != v)

                castaway = castaways[u]

                con_scores.append({
                    'season': season,
                    'episode': episode,
                    'castaway_uid': castaway,
                    '1st_order_CON_score': con_1st_order,
                    '2nd_order_CON_score': con_2nd_order,
                    'full_CON_score': con_large,
                })

    return pd.DataFrame(con_scores)


def calculate_CON_multiorder1(df):
    con_scores = []
    
    for season in df['season'].unique():
        df_season = df[df['season'] == season]
        
        for episode in df_season['episode'].unique():
            curr_epi_df = df_season[df_season['episode'] == episode]
            castaways = list(set(curr_epi_df['castaway_uid']) | set(curr_epi_df['vote_uid']))
            n = len(castaways)
            castaway_index = {castaway: idx for idx, castaway in enumerate(castaways)}

            adj_matrix = np.zeros((n, n), dtype=int)
            for _, row in curr_epi_df.iterrows():
                voter_idx = castaway_index[row['castaway_uid']]
                voted_for_idx = castaway_index[row['vote_uid']]
                adj_matrix[voter_idx, voted_for_idx] += 1

            adj_matrix_squared = np.dot(adj_matrix, adj_matrix)
            adj_mat_large = adj_matrix + adj_matrix_squared

            # Create NetworkX directed graph from adjacency matrix
            G = nx.DiGraph(adj_matrix)
            
            # Calculate PageRank
            pagerank_scores = nx.pagerank(G)
            
            # Calculate closeness centrality
            closeness_scores = nx.closeness_centrality(G)
            
            # Calculate in-degree and out-degree
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())
            
            for u in range(n):
                # Calculate CON scores (1st, 2nd order, and combined)
                con_1st_order = sum(min(adj_matrix[u, k], adj_matrix[v, k]) for v in range(n) for k in range(n) if u != v)
                con_2nd_order = sum(min(adj_matrix_squared[u, k], adj_matrix_squared[v, k]) for v in range(n) for k in range(n) if u != v)
                con_large = sum(min(adj_mat_large[u, k], adj_mat_large[v, k]) for v in range(n) for k in range(n) if u != v)

                # Add centrality measures
                castaway = castaways[u]
                pagerank_score = pagerank_scores.get(u, 0)
                closeness_score = closeness_scores.get(u, 0)
                in_degree = in_degrees.get(u, 0)
                out_degree = out_degrees.get(u, 0)

                con_scores.append({
                    'season': season,
                    'episode': episode,
                    'castaway_uid': castaway,
                    '1st_order_CON_score': con_1st_order,
                    '2nd_order_CON_score': con_2nd_order,
                    'full_CON_score': con_large,
                    'pagerank': pagerank_score,
                    'closeness': closeness_score,
                    'in_degree': in_degree,
                    'out_degree': out_degree
                })

    return pd.DataFrame(con_scores)