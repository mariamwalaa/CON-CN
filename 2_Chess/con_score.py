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
    """
    For each game in the dataset, filter on players that are involved, create an adjacency matrix, square it, and calculate the CON score for each player.
    """
    con_scores = []
    for date in df['game_id'].unique():

        curr_game = df[df['game_id'] == date]
        players = list(set(curr_game['winner']) | set(curr_game['loser']))
        n = len(players)
        player_index = {player: idx for idx, player in enumerate(players)}
        
        adj_matrix = np.zeros((n, n), dtype=int)
        for _, row in curr_game.iterrows():
            winner_idx = player_index[row['winner']]
            loser_idx = player_index[row['loser']]
            adj_matrix[winner_idx, loser_idx] += 1
        
        adj_matrix_squared = np.dot(adj_matrix, adj_matrix)
        adj_mat_large = adj_matrix + adj_matrix_squared
    
        for u in range(n):
            con_1st_order = sum(min(adj_matrix[u, k], adj_matrix[v, k]) for v in range(n) for k in range(n) if u != v)
            con_2nd_order = sum(min(adj_matrix_squared[u, k], adj_matrix_squared[v, k]) for v in range(n) for k in range(n) if u != v)
            con_large = sum(min(adj_mat_large[u, k], adj_mat_large[v, k]) for v in range(n) for k in range(n) if u != v)
            con_scores.append({'date': date, 'player': players[u], '1st_order_CON_score': con_1st_order, '2nd_order_CON_score': con_2nd_order, 'full_CON_score': con_large})
    return pd.DataFrame(con_scores)

def calculate_CON_multiorder1(df):
    """
    For each game in the dataset, filter on players that are involved, create an adjacency matrix, square it, and calculate the CON score for each player.
    """
    con_scores = []

    for date in df['game_id'].unique():
        curr_game = df[df['game_id'] == date]
        
        players = list(set(curr_game['winner']) | set(curr_game['loser']))
        n = len(players)
        player_index = {player: idx for idx, player in enumerate(players)}
        
        adj_matrix = np.zeros((n, n), dtype=int)
        for _, row in curr_game.iterrows():
            
            winner_idx = player_index[row['winner']]
            loser_idx = player_index[row['loser']]
            
            adj_matrix[winner_idx, loser_idx] += 1
        
        adj_matrix_squared = np.dot(adj_matrix, adj_matrix)
        adj_mat_large = adj_matrix + adj_matrix_squared
    
        G = nx.DiGraph(adj_matrix)
        pagerank_scores = nx.pagerank(G)
        closeness_scores = nx.closeness_centrality(G)
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        betweenness_scores = nx.betweenness_centrality(G)
        
        for u in range(n):
            con_1st_order = sum(min(adj_matrix[u, k], adj_matrix[v, k]) for v in range(n) for k in range(n) if u != v)
            con_2nd_order = sum(min(adj_matrix_squared[u, k], adj_matrix_squared[v, k]) for v in range(n) for k in range(n) if u != v)
            con_large = sum(min(adj_mat_large[u, k], adj_mat_large[v, k]) for v in range(n) for k in range(n) if u != v)

            pagerank_score = pagerank_scores.get(u, 0)
            closeness_score = closeness_scores.get(u, 0)
            in_degree = in_degrees.get(u, 0)
            out_degree = out_degrees.get(u, 0)
            betweenness_score = betweenness_scores.get(u, 0)
        
            con_scores.append({'date': date, 'player': players[u], '1st_order_CON_score': con_1st_order, 
                               '2nd_order_CON_score': con_2nd_order, 'full_CON_score': con_large,
                                                   'pagerank': pagerank_score,
                    'closeness': closeness_score,
                    'in_degree': in_degree,
                    'out_degree': out_degree,
                    'betweenness': betweenness_score})

    return pd.DataFrame(con_scores)