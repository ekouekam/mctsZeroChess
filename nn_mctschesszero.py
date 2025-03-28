# -*- coding: utf-8 -*-
"""nn_mctsChessZero.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12oihA9oHFssV22YHe8un74SGuMc3rLES

#nn_mctsChessZero

## Chess AI with AlphaZero-inspired MCTS
Intègre un moteur IA de jeu basé sur l'apprentissage par renforcement et réseau de neurones inspiré d'AlphaZero avec l'algorithme de l'arbre de recherche de Monte Carlo(MCTS) et système d'entraînement par auto-jeu

 ## Explications des principales fonctionnalités :
 Board, Piece, Engine, et la classe Main
1. Board : Représente l'échiquier, initialise les pièces et gère les mouvements.
2. Piece : Représente une pièce individuelle sur l'échiquier.
3. ChessNET: Intégration Réseau de Neurones - PyTorch pour le réseau neuronal
4. Engine : Moteur de jeu de l'IA basé sur l'algorithme MCTS d'apprentissage par renforcement.
  
  -Chaque Node représente un état du jeu, et l'arbre MCTS explore les coups possibles pour choisir le meilleur coup.
  
  -Engine gère l'intégration de la logique de jeu de l'IA via l'algorithme de Monte Carlo en apprentissage par renforcement.
  
  -Engine MCTS avec Guide Neuronal optimisé avec UCB
5. ChessRL : Boucle d'Entraînement par Auto-Jeu - Système d'entraînement par renforcement
6. Main : La classe principale qui gère le déroulement du jeu, en alternant entre les joueurs(humain contre IA).
## Améliorations possibles :
Ce code présente une structure de base pour un moteur d'échecs utilisant MCTS pour l'apprentissage par renforcement, mais beaucoup d'aspects peuvent être enrichis pour un jeu complet et performant.
Cette implémentation se concentre sur la structure de base et sur l'architecture du moteur de jeu, mais elle peut nécessiter des améliorations pour une version plus avancée et réaliste.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
import random
from IPython.display import clear_output
from collections import deque

"""#Piece
Piece est responsable de la représentation d'une piece individuelle.



"""

from abc import ABC, abstractmethod

class Piece(ABC):
    def __init__(self, color, name):
        self.color = color
        self.name = name
        self.symbol = self._get_symbol()
        self.moved = False

    @abstractmethod
    def _get_symbol(self):
        pass

    @abstractmethod
    def get_legal_moves(self, position, board):
        pass

    def clone(self):
        return copy.deepcopy(self)

class King(Piece):
    def _get_symbol(self):
        return '♔' if self.color == 'white' else '♚'

    def get_legal_moves(self, position, board):
        x, y = position
        moves = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            if 0 <= x+dx < 8 and 0 <= y+dy < 8:
                target = board.get_piece((x+dx, y+dy))
                if not target or target.color != self.color:
                    moves.append((x+dx, y+dy))
        return moves

class Queen(Piece):
    def _get_symbol(self):
        return '♕' if self.color == 'white' else '♛'

    def get_legal_moves(self, position, board):
        return Rook(self.color, 'rook').get_legal_moves(position, board) + \
               Bishop(self.color, 'bishop').get_legal_moves(position, board)

class Rook(Piece):
    def _get_symbol(self):
        return '♖' if self.color == 'white' else '♜'

    def get_legal_moves(self, position, board):
        x, y = position
        moves = []
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                target = board.get_piece((nx, ny))
                if not target:
                    moves.append((nx, ny))
                else:
                    if target.color != self.color:
                        moves.append((nx, ny))
                    break
                nx += dx
                ny += dy
        return moves

class Bishop(Piece):
    def _get_symbol(self):
        return '♗' if self.color == 'white' else '♝'

    def get_legal_moves(self, position, board):
        x, y = position
        moves = []
        directions = [(-1,-1), (-1,1), (1,-1), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                target = board.get_piece((nx, ny))
                if not target:
                    moves.append((nx, ny))
                else:
                    if target.color != self.color:
                        moves.append((nx, ny))
                    break
                nx += dx
                ny += dy
        return moves

class Knight(Piece):
    def _get_symbol(self):
        return '♘' if self.color == 'white' else '♞'

    def get_legal_moves(self, position, board):
        x, y = position
        moves = []
        offsets = [(-2,-1), (-1,-2), (1,-2), (2,-1),
                   (2,1), (1,2), (-1,2), (-2,1)]
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                target = board.get_piece((nx, ny))
                if not target or target.color != self.color:
                    moves.append((nx, ny))
        return moves

class Pawn(Piece):
    def _get_symbol(self):
        return '♙' if self.color == 'white' else '♟'

    def get_legal_moves(self, position, board):
        x, y = position
        moves = []
        direction = -1 if self.color == 'white' else 1
        start_row = 6 if self.color == 'white' else 1

        # Forward moves
        if board.get_piece((x + direction, y)) is None:
            moves.append((x + direction, y))
            if x == start_row and board.get_piece((x + 2*direction, y)) is None:
                moves.append((x + 2*direction, y))

        # Captures
        for dy in [-1, 1]:
            if 0 <= y + dy < 8:
                target = board.get_piece((x + direction, y + dy))
                if target and target.color != self.color:
                    moves.append((x + direction, y + dy))

        # En passant
        if board.en_passant:
            ex, ey = board.en_passant
            if x == ex and abs(y - ey) == 1:
                moves.append((x + direction, ey))

        return moves

"""#Board
Board est responsable de la représentation de l'échiquier et des mouvements des pièces.

Gestion Avancée de l'État du Jeu
"""

class Board:
    def __init__(self):
        self.grid = np.empty((8,8), dtype=object)
        self.current_player = 'white'
        self.history = []
        self.en_passant = None
        self.castling_rights = {'white': {'kingside': True, 'queenside': True},
                               'black': {'kingside': True, 'queenside': True}}
        self._init_pieces()

    def _init_pieces(self):
        # Pawns
        for i in range(8):
            self.grid[1][i] = Pawn('black', 'pawn')
            self.grid[6][i] = Pawn('white', 'pawn')

        # Other pieces
        pieces = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        for i, piece in enumerate(pieces):
            self.grid[0][i] = piece('black', piece.__name__.lower())
            self.grid[7][i] = piece('white', piece.__name__.lower())

    def get_piece(self, position):
        x, y = position
        return self.grid[x][y]

    def apply_move(self, move):
        start, end = move
        piece = self.grid[start[0]][start[1]]

        # Save state
        self.history.append({
            'grid': copy.deepcopy(self.grid),
            'castling': copy.deepcopy(self.castling_rights),
            'en_passant': self.en_passant
        })

        # Handle special moves
        self._handle_castling(move, piece)
        self._handle_en_passant(move, piece)
        self._handle_promotion(end, piece)

        # Update castling rights
        if piece.name == 'king':
            self.castling_rights[piece.color] = {'kingside': False, 'queenside': False}
        elif piece.name == 'rook':
            if start == (7,0) or start == (0,0):
                self.castling_rights[piece.color]['queenside'] = False
            elif start == (7,7) or start == (0,7):
                self.castling_rights[piece.color]['kingside'] = False

        # Execute move
        self.grid[end[0]][end[1]] = piece
        self.grid[start[0]][start[1]] = None
        piece.moved = True

        # Update en passant
        self.en_passant = None
        if piece.name == 'pawn' and abs(start[0] - end[0]) == 2:
            self.en_passant = ((start[0] + end[0])//2, start[1])

        self.current_player = 'black' if self.current_player == 'white' else 'white'

    def _handle_castling(self, move, piece):
        # Implementation for castling moves
        pass

    def _handle_en_passant(self, move, piece):
        # Implementation for en passant
        pass

    def _handle_promotion(self, position, piece):
        # Handle pawn promotion
        pass

    def is_check(self, color):
        # Check detection logic
        pass

    def is_checkmate(self, color):
        # Checkmate detection
        pass

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        s = ''
        for row in self.grid:
            s += ' '.join([p.symbol if p else '·' for p in row]) + '\n'
        return s

"""# ChessNET

 Intégration Réseau de Neurones : PyTorch pour le réseau neuronal
"""

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(20, 256, 3, padding=1)
        self.resblocks = nn.ModuleList([ResBlock(256) for _ in range(19)])
        self.policy_head = PolicyHead(256)
        self.value_head = ValueHead(256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        for block in self.resblocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)

class PolicyHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 73, 1)

    def forward(self, x):
        batch_size = x.size(0)
        return self.conv(x).view(batch_size, -1)

class ValueHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.fc = nn.Linear(8*8, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.tanh(x)

"""#Engine
 Engine (Moteur de jeu de l'IA basé sur l'algorithme MCTS d'apprentissage par renforcement)

 Engine gère l'intégration de la logique de jeu de l'IA via l'algorithme de Monte Carlo en apprentissage par renforcement.

 Engine MCTS avec Guide Neuronal optimisé avec UCB
"""

class Engine:
    def __init__(self, model, simulations=800):
        self.model = model
        self.simulations = simulations

    def search(self, board):
        root = Node(board.copy())

        for _ in range(self.simulations):
            node = root
            state = board.copy()

            # Selection
            while not node.is_leaf():
                node = node.select_child()
                state.apply_move(node.move)

            # Expansion
            if not state.is_terminal():
                policy, value = self.model.predict(state.to_input())
                node.expand(state.get_legal_moves(), policy)

            # Backpropagation
            while node is not None:
                node.update(value)
                node = node.parent
                value = -value  # Switch perspective

        return root.best_child(0).move

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def select_child(self):
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            score = child.ucb_score()
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def ucb_score(self):
        if self.visits == 0:
            return float('inf')
        return (self.value_sum / self.visits) + \
               math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self, moves, policy):
        for move in moves:
            child_state = self.state.copy()
            child_state.apply_move(move)
            self.children.append(Node(child_state, self, move))

    def update(self, value):
        self.visits += 1
        self.value_sum += value

    def is_leaf(self):
        return len(self.children) == 0

"""#ChessRL

Boucle d'Entraînement par Auto-Jeu: Système d'entraînement par renforcement
"""

class ChessRL:
    def __init__(self, model_path=None):
        self.model = ChessNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.mcts = Engine(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)

    def get_move(self, board):
        return self.mcts.search(board)

    def train(self, epochs=10, batch_size=32):
        for epoch in range(epochs):
            batch = random.sample(self.memory, min(batch_size, len(self.memory)))
            states, policies, values = zip(*batch)

            # Convert to tensors
            states = torch.stack([self._board_to_tensor(s) for s in states])
            policies = torch.stack(policies)
            values = torch.stack(values)

            # Forward pass
            pred_policies, pred_values = self.model(states)

            # Calculate loss
            policy_loss = torch.mean(-torch.sum(policies * torch.log(pred_policies), dim=1))
            value_loss = torch.mean((values - pred_values.squeeze())**2)
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _board_to_tensor(self, board):
        # Convert board state to 20-channel tensor
        tensor = torch.zeros(20, 8, 8)
        # ... implementation of board state encoding ...
        return tensor

"""#Main
Main (exécution du jeu)

"""

class Main:
    def __init__(self, ai=None):
        self.board = Board()
        self.ai = ai or ChessRL()
        self.human_color = 'white'

    def play(self):
        while not self.board.is_checkmate():
            print(self.board)
            if self.board.current_player == self.human_color:
                move = self.get_human_move()
            else:
                move = self.ai.get_move(self.board)
            self.board.apply_move(move)
        print("Game Over!")

    def get_human_move(self):
        while True:
            try:
                move_str = input("Enter move (e.g. 'e2e4'): ")
                start = (int(move_str[1])-1, ord(move_str[0])-ord('a'))
                end = (int(move_str[3])-1, ord(move_str[2])-ord('a'))
                return (start, end)
            except:
                print("Invalid move format!")


# Execution
if __name__ == "__main__":
    game = Main()
    game.play()