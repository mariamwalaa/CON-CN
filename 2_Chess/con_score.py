import pandas as pd
import numpy as np

import networkx as nx
from scipy.sparse import csr_matrix

def compute_con_u_v(A, u, v):
    """Compute the number of common out-neighbors for nodes u and v."""
    return sum(min(A[u, k], A[v, k]) for k in range(len(A)))

def compute_con_u(A, u):
    """Compute the CON(u) for a node u."""
    return sum(compute_con_u_v(A, u, v) for v in range(len(A)) if u != v)

def compute_con_scores(graph):
    """Compute the CON score for all nodes in the graph."""
    A = nx.to_numpy_array(graph, dtype=int)
    
    con_scores = {}
    for node in graph.nodes():
        u_index = list(graph.nodes()).index(node)
        con_scores[node] = compute_con_u(A, u_index)
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
        
        A = csr_matrix((n, n), dtype=int)
        #A = np.zeros((n, n), dtype=int)
        for _, row in curr_game.iterrows():
            winner_idx = player_index[row['winner']]
            loser_idx = player_index[row['loser']]
            A[winner_idx, loser_idx] += 1
        
        A_squared = A.dot(A)
        A_large = A + A_squared
        #A_squared = np.dot(A, A)
        #A_large = A + A_squared
    
        for u in range(n):
            con_1st_order, con_2nd_order, con_large = 0, 0, 0

            #con_1st_order = sum(min(A[u, k], A[v, k]) for v in range(n) for k in range(n) if u != v)
            #con_2nd_order = sum(min(A_squared[u, k], A_squared[v, k]) for v in range(n) for k in range(n) if u != v)
            #con_large = sum(min(A_large[u, k], A_large[v, k]) for v in range(n) for k in range(n) if u != v)
            #con_scores.append({'date': date, 'player': players[u], '1st_order_CON_score': con_1st_order, '2nd_order_CON_score': con_2nd_order, 'full_CON_score': con_large})

            for v in range(n):
                if u != v:
                    con_1st_order += np.sum(np.minimum(A[u, :].toarray(), A[v, :].toarray()))
                    con_2nd_order += np.sum(np.minimum(A_squared[u, :].toarray(), A_squared[v, :].toarray()))
                    con_large += np.sum(np.minimum(A_large[u, :].toarray(), A_large[v, :].toarray()))
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
        
        #A = np.zeros((n, n), dtype=int)
        A = csr_matrix((n, n), dtype=int)
        for _, row in curr_game.iterrows():     
            winner_idx = player_index[row['winner']]
            loser_idx = player_index[row['loser']]
            A[winner_idx, loser_idx] += 1
        
        A_squared = A.dot(A)
        A_large = A + A_squared
        #A_squared = np.dot(A, A)
        #A_large = A + A_squared
    
        G = nx.DiGraph(A)
        pagerank_scores = nx.pagerank(G)
        closeness_scores = nx.closeness_centrality(G)
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        betweenness_scores = nx.betweenness_centrality(G)
        
        for u in range(n):
            
            #con_1st_order = sum(min(A[u, k], A[v, k]) for v in range(n) for k in range(n) if u != v)
            #con_2nd_order = sum(min(A_squared[u, k], A_squared[v, k]) for v in range(n) for k in range(n) if u != v)
            #con_large = sum(min(A_large[u, k], A_large[v, k]) for v in range(n) for k in range(n) if u != v)
            #con_scores.append({'date': date, 'player': players[u], '1st_order_CON_score': con_1st_order, '2nd_order_CON_score': con_2nd_order, 'full_CON_score': con_large})
            
            con_1st_order, con_2nd_order, con_large = 0, 0, 0
            for v in range(n):
                if u != v:
                    con_1st_order += np.sum(np.minimum(A[u, :].toarray(), A[v, :].toarray()))
                    con_2nd_order += np.sum(np.minimum(A_squared[u, :].toarray(), A_squared[v, :].toarray()))
                    con_large += np.sum(np.minimum(A_large[u, :].toarray(), A_large[v, :].toarray()))

            pagerank_score = pagerank_scores.get(u, 0)
            closeness_score = closeness_scores.get(u, 0)
            in_degree = in_degrees.get(u, 0)
            out_degree = out_degrees.get(u, 0)
            betweenness_score = betweenness_scores.get(u, 0)
        
            con_scores.append({'date': date, 'player': players[u], '1st_order_CON_score': con_1st_order, 
                               '2nd_order_CON_score': con_2nd_order, 'full_CON_score': con_large,
                               'pagerank': pagerank_score, 'closeness': closeness_score, 'in_degree': in_degree,
                               'out_degree': out_degree, 'betweenness': betweenness_score})

    return pd.DataFrame(con_scores)