def player(prev_play, opponent_history=[]):
    
    opponent_moves = {'R': 0, 'P': 0, 'S': 0}
    if prev_play:
        opponent_moves[prev_play] += 1 
    if opponent_history:  
        most_frequent_move = max(opponent_moves, key=opponent_moves.get)
        if most_frequent_move == 'R':
            return 'P'
        elif most_frequent_move == 'P':
            return 'S'
        else:
            return 'R'
    else:     
        import random
        return random.choice(['R', 'P', 'S'])
