# -*- coding: utf-8 -*-
"""alphaChess.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QYJPCu9EjKzfCBW2ZouLshGMSdsuPH_g

alphaChess
=========
Introducing "alphaChess Simple Python Chess Program" running in command line.
You can play a full game versus it.
Run it with the command : ./main.py
It supports actually :
- promote
- under-promote
- capture "en passant"
Commands are :
- **new** to start a new game
- **e2e4** or **e7e8q** for example to move a piece. Promotes are q,r,n,b for queen, rook, knight, bishop
- **undomove** to cancel the last move
- **legalmoves** to show legal moves for side to move
- **go** requests the engine to play now
- **setboard fen** to set the board as the FEN position given
- **getboard** to export the current FEN position
- **sd x** to set the depth search
- **perft x** to test the move generator (x = search depth)
- **quit**... to quit
Things to do :
- move ordering
- quiescent search
- 50 moves rule
- 3 repetitions rule
- time settings
- opening book
Requirements :
- Python 3
More information on :

#Piece
Here’s the complete code that defines a `Piece` class for a chess game, based on your description. This class represents a chess piece, providing methods and attributes to handle movement and basic functionality. The "mailbox method" is used to map the 64-square chessboard to a 120-square representation, which helps manage legal moves for each piece .
"""

class Piece:
    """
    Chess set class representing individual chess pieces on a board.
    This class handles piece names, values, movement vectors, and various piece operations.
    """

    # Constants and piece names
    VIDE = '.'  # empty piece, represented as a dot
    nomPiece = (VIDE, 'ROI', 'DAME', 'TOUR', 'CAVALIER', 'FOU', 'PION')  # Piece names
    valeurPiece = (0, 0, 9, 5, 3, 3, 1)  # Piece values (King=0, Queen=9, Rook=5, etc.)

    # Mailbox method (Robert Hyatt)
    # Representation for a larger 120-square array to efficiently track off-board squares
    tab120 = (
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        # 64 squares of the chessboard
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1
    )

    # Movement vectors based on the mailbox method
    deplacements_tour = (-10, 10, -1, 1)  # Rook's moves: up, down, left, right
    deplacements_fou = (-11, -9, 11, 9)  # Bishop's moves: diagonals
    deplacements_cavalier = (-12, -21, -19, -8, 12, 21, 19, 8)  # Knight's moves: L-shape

    def __init__(self, nom=VIDE, couleur=''):
        """
        Initialize a chess piece with a name and color.

        :param nom: The piece name (default is empty piece)
        :param couleur: The color of the piece (e.g., 'white', 'black')
        """
        self.nom = nom  # Piece name (e.g., 'PION' for Pawn)
        self.couleur = couleur  # Piece color (e.g., 'white' or 'black')

    def isEmpty(self):
        """
        Check if the piece is empty (i.e., no piece is on the square).

        :return: True if the piece is empty, False otherwise.
        """
        return self.nom == self.VIDE

    def pos2_roi(self, pos1, cAd, echiquier, dontCallIsAttacked=False):
        """
        Calculates the position of the king (ROI) after a move.

        :param pos1: The current position of the piece.
        :param cAd: The color of the opponent.
        :param echiquier: The current state of the board.
        :param dontCallIsAttacked: A flag to control whether to call the "isAttacked" method.
        :return: The new position after the move.
        """
        # Example logic for handling King's move (ROI)
        pass

    # Additional methods to handle piece movement and rules could go here

    # For example, a method to handle movement (with basic logic)
    def move(self, start_pos, end_pos):
        """
        Move a piece from the start position to the end position.

        :param start_pos: The starting position of the piece.
        :param end_pos: The target position where the piece will move.
        :return: None (Move would be executed).
        """
        if self.isEmpty():
            raise ValueError("Cannot move an empty piece.")
        # Implement move logic here based on piece type and movement rules
        pass

    def __repr__(self):
        """
        String representation of the piece, showing its name and color.

        :return: A string describing the piece.
        """
        return f"{self.couleur} {self.nom}"

# Example of using the Piece class
# Initialize a few pieces
king = Piece(nom=Piece.nomPiece[1], couleur="white")  # King (ROI)
queen = Piece(nom=Piece.nomPiece[2], couleur="black")  # Queen (DAME)

# Check if the pieces are empty
print(king.isEmpty())  # False
print(queen.isEmpty())  # False

"""### Code for the `Piece` Class:(Cf. code on top)

### Explanation of the Code:

1. **Class-level Constants**:
    - `VIDE`: Represents an empty square on the chessboard (shown as a dot).
    - `nomPiece`: A tuple storing the names of the pieces (King, Queen, Rook, Knight, Bishop, Pawn).
    - `valeurPiece`: A tuple representing the value (or score) of each piece. For example, the Queen (`DAME`) has a value of 9, the Rook (`TOUR`) has a value of 5, and so on.

2. **Mailbox Method**:
    - `tab120`: A tuple used for representing the entire board and out-of-bound squares. The value `-1` marks squares that are off the board, and valid squares are indexed from 21 to 100.
    - `deplacements_tour`, `deplacements_fou`, `deplacements_cavalier`: Movement vectors for the Rook, Bishop, and Knight, respectively. These vectors help calculate valid moves for these pieces by adjusting the current position.

3. **Constructor (`__init__`)**:
    - The constructor initializes a piece with its name (`nom`) and color (`couleur`). The default name is `VIDE`, which represents an empty square.

4. **Methods**:
    - `isEmpty()`: Returns `True` if the piece is empty (i.e., no piece is placed on the square).
    - `pos2_roi()`: This method calculates the new position of the King after a move. The actual logic would depend on the game rules and how you implement the chessboard's state and attack detection.
    - `move()`: Placeholder method to handle piece movement. In a complete implementation, you would handle different types of movement based on the piece and rules.
    - `__repr__()`: Provides a string representation of the piece, showing both its color and name.

### Extending the Class:
You can extend this class by adding additional methods such as:
- `legal_moves()`: To calculate all legal moves for a piece based on its type and the current state of the board.
- `is_attacked()`: To check if a piece is under attack based on the current configuration of the board.
- `can_castle()`: For handling castling logic for the King and Rook.

This class structure forms the foundation for handling pieces and their movements in a Python chess game. You can further develop the game logic, board state management, and rules enforcement from here.

#Board
Here is a Python implementation of the `Board` class for a chess game, which represents the chessboard and handles various chess operations. This class integrates with the previously defined `Piece` class, using the `mailbox method` and other utilities like FEN notation, move generation, and basic game state management.
"""

from piece import *  # Assuming the Piece class is defined in 'piece.py'
import os  # For clearing the terminal screen

class Board:
    """
    The chessboard class that handles the initialization, move generation,
    board state, and game management.
    """

    # Coordinate list for the chessboard in algebraic notation
    coord = [
        'a8','b8','c8','d8','e8','f8','g8','h8',
        'a7','b7','c7','d7','e7','f7','g7','h7',
        'a6','b6','c6','d6','e6','f6','g6','h6',
        'a5','b5','c5','d5','e5','f5','g5','h5',
        'a4','b4','c4','d4','e4','f4','g4','h4',
        'a3','b3','c3','d3','e3','f3','g3','h3',
        'a2','b2','c2','d2','e2','f2','g2','h2',
        'a1','b1','c1','d1','e1','f1','g1','h1',
    ]

    def __init__(self):
        """
        Initialize the chessboard by setting up the starting position and game state.
        """
        self.init()

    def init(self):
        """
        Initialize the chess board to the starting position, setting up pieces
        and game attributes.
        """
        # Initialize the board as a list of 64 'Piece' objects
        self.cases = [Piece() for _ in range(64)]  # 64 squares on the board

        # Setting up white pieces on rows 1 and 2
        for i in range(8):
            self.cases[i + 8] = Piece(nom=Piece.nomPiece[6], couleur='white')  # White pawns
        self.cases[0], self.cases[7] = Piece(nom=Piece.nomPiece[3], couleur='white'), Piece(nom=Piece.nomPiece[3], couleur='white')  # White rooks
        self.cases[1], self.cases[6] = Piece(nom=Piece.nomPiece[5], couleur='white'), Piece(nom=Piece.nomPiece[5], couleur='white')  # White knights
        self.cases[2], self.cases[5] = Piece(nom=Piece.nomPiece[4], couleur='white'), Piece(nom=Piece.nomPiece[4], couleur='white')  # White bishops
        self.cases[3] = Piece(nom=Piece.nomPiece[2], couleur='white')  # White queen
        self.cases[4] = Piece(nom=Piece.nomPiece[1], couleur='white')  # White king

        # Setting up black pieces on rows 7 and 8
        for i in range(8):
            self.cases[i + 48] = Piece(nom=Piece.nomPiece[6], couleur='black')  # Black pawns
        self.cases[56], self.cases[63] = Piece(nom=Piece.nomPiece[3], couleur='black'), Piece(nom=Piece.nomPiece[3], couleur='black')  # Black rooks
        self.cases[57], self.cases[62] = Piece(nom=Piece.nomPiece[5], couleur='black'), Piece(nom=Piece.nomPiece[5], couleur='black')  # Black knights
        self.cases[58], self.cases[61] = Piece(nom=Piece.nomPiece[4], couleur='black'), Piece(nom=Piece.nomPiece[4], couleur='black')  # Black bishops
        self.cases[59] = Piece(nom=Piece.nomPiece[2], couleur='black')  # Black queen
        self.cases[60] = Piece(nom=Piece.nomPiece[1], couleur='black')  # Black king

        # Game state attributes
        self.side2move = 'white'  # White starts
        self.ep = None  # En passant square (None if not available)
        self.history = []  # History of moves made
        self.ply = 0  # Half-move clock (for draw rules)
        self.castling = {'white': {'k': True, 'q': True}, 'black': {'k': True, 'q': True}}  # Castling rights

    def gen_moves_list(self, color='', dontCallIsAttacked=False):
        """
        Generate a list of all legal moves for the specified color.

        :param color: The color of the player whose moves are to be generated ('white' or 'black').
        :param dontCallIsAttacked: Flag to disable checking for whether moves place the King in check.
        :return: A list of legal moves as tuples: (start_pos, end_pos, promotion).
        """
        moves = []
        if color == '':
            color = self.side2move

        # Iterate through the board and generate moves for each piece of the correct color
        for idx, piece in enumerate(self.cases):
            if piece.isEmpty() or piece.couleur != color:
                continue  # Skip empty squares or pieces of the wrong color

            # Generate moves for each type of piece
            if piece.nom == Piece.nomPiece[1]:  # King
                moves += self.gen_king_moves(idx, dontCallIsAttacked)
            elif piece.nom == Piece.nomPiece[2]:  # Queen
                moves += self.gen_queen_moves(idx, dontCallIsAttacked)
            elif piece.nom == Piece.nomPiece[3]:  # Rook
                moves += self.gen_rook_moves(idx, dontCallIsAttacked)
            elif piece.nom == Piece.nomPiece[4]:  # Bishop
                moves += self.gen_bishop_moves(idx, dontCallIsAttacked)
            elif piece.nom == Piece.nomPiece[5]:  # Knight
                moves += self.gen_knight_moves(idx)
            elif piece.nom == Piece.nomPiece[6]:  # Pawn
                moves += self.gen_pawn_moves(idx, dontCallIsAttacked)

        return moves

    def gen_king_moves(self, idx, dontCallIsAttacked):
        """ Generate legal moves for the King. """
        # King movement (one square in any direction)
        pass

    def gen_queen_moves(self, idx, dontCallIsAttacked):
        """ Generate legal moves for the Queen. """
        # Queen moves (combination of Rook and Bishop)
        pass

    def gen_rook_moves(self, idx, dontCallIsAttacked):
        """ Generate legal moves for the Rook. """
        # Rook moves (along ranks and files)
        pass

    def gen_bishop_moves(self, idx, dontCallIsAttacked):
        """ Generate legal moves for the Bishop. """
        # Bishop moves (diagonal)
        pass

    def gen_knight_moves(self, idx):
        """ Generate legal moves for the Knight. """
        # Knight moves (L-shaped)
        pass

    def gen_pawn_moves(self, idx, dontCallIsAttacked):
        """ Generate legal moves for the Pawn. """
        # Pawn moves (forward, captures, promotions, en passant)
        pass

    def setboard(self, fen):
        """
        Set the board to a specific position using Forsyth-Edwards Notation (FEN).

        :param fen: A string representing the board's position in FEN.
        """
        pass  # Implement FEN parsing here

    def getboard(self):
        """
        Get the current board position in Forsyth-Edwards Notation (FEN).

        :return: A string representing the board's position in FEN.
        """
        pass  # Implement FEN generation here

    def domove(self, depart, arrivee):
        """
        Make a move on the board, updating the state accordingly.

        :param depart: The starting square (in algebraic notation).
        :param arrivee: The destination square (in algebraic notation).
        """
        pass  # Implement move logic here

    def undomove(self):
        """
        Undo the last move made on the board.
        """
        pass  # Implement undo move logic here

    def display(self):
        """
        Display the current board in a human-readable format.
        """
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal screen
        for i in range(8):
            print(" ".join(str(self.cases[i * 8 + j]) for j in range(8)))  # Display 8 rows

# Example usage
board = Board()
board.display()

"""### Code for the `Board` Class:(Cf. code on top)


### Explanation:

1. **Class Initialization** (`__init__` and `init`):
   - The `Board` class initializes the chessboard with 64 squares, each represented by a `Piece` object.
   - The starting pieces are placed on the board, with white pieces on rows 1 and 2, and black pieces on rows 7 and 8.
   - Game state attributes like `side

2move` (who moves next), `ep` (en passant), `history` (move history), `ply` (half-move clock), and `castling` (castling rights) are also initialized.

2. **Generating Legal Moves** (`gen_moves_list`):
   - This method generates a list of legal moves for the current side (white or black). It iterates through the board, checking each piece and generating moves based on the piece type (King, Queen, Rook, Bishop, Knight, or Pawn).
   - Placeholder methods for generating moves for each piece (`gen_king_moves`, `gen_queen_moves`, etc.) are provided. You can implement the move generation logic inside these methods.

3. **Move and Undo Logic** (`domove`, `undomove`):
   - `domove`: This method handles making a move on the board by updating the board state based on the start and end positions of the move.
   - `undomove`: This method undoes the last move made, reverting the board to its previous state.

4. **Display Board** (`display`):
   - This method prints the current state of the board to the terminal using a human-readable format. It clears the terminal before displaying the board to make it easier to view the updated state.

### Next Steps:
- You can extend the methods for move generation (`gen_king_moves`, `gen_queen_moves`, etc.) to implement the full movement rules for each piece type.
- You can also add additional functionality such as enforcing rules like castling, en passant, pawn promotion, and checking for check/checkmate conditions.

This `Board` class serves as the core of the chess game, handling the game state and providing functionality for piece movements and the chess rules.

#Engine
Here's the code for an `Engine` class that represents a simple chess engine. This class contains the main logic for interacting with the chessboard, processing user moves, performing searches for the best move, and using the Alpha-Beta pruning algorithm for move evaluation.
"""

from piece import *  # Importing all the elements from the Piece class
import time  # Importing time module for performance measurement

class Engine:
    """
    The Engine class that represents the chess engine. It handles the move generation,
    searching for the best move, and executing the moves on the chessboard.
    """

    def __init__(self):
        """
        Initializes the engine with necessary parameters.
        """
        self.MAX_PLY = 32  # Maximum search depth (ply)
        self.pv_length = [0 for _ in range(self.MAX_PLY)]  # Array to store the principal variation (best move sequence) at each depth
        self.INFINITY = 32000  # Arbitrarily large value representing checkmate score
        self.init()  # Initialize other engine parameters

    def init(self):
        """
        Initializes parameters related to the engine's state.
        """
        self.endgame = False  # Flag to check if the game has ended
        self.game_over = False  # Flag for game over status
        self.history = []  # Keeps track of the history of moves

    def usermove(self, b, c):
        """
        Handles a move entered by the user.

        :param b: The current board object representing the chessboard state.
        :param c: The move command entered by the user (e.g., 'e2e4', 'b7b8q').
        """
        if self.endgame:
            print("The game is over. No more moves can be made.")
            return

        # Validate the move command
        error_msg = self.chkCmd(c)
        if error_msg:
            print(f"Invalid move command: {error_msg}")
            return

        # Convert the command into internal representation (e.g., from 'e2e4' to board indices)
        start_pos, end_pos, promotion = self.parse_move(c)

        # Make the move on the board
        b.domove(start_pos, end_pos)
        if promotion:
            b.promote_pawn(end_pos, promotion)  # Handle pawn promotion

        # Update the game state
        b.display()
        self.history.append(c)  # Store the move history
        self.print_result(b)

    def chkCmd(self, c):
        """
        Validates the move command entered by the user.

        :param c: The move command (e.g., 'e2e4', 'b7b8n').
        :return: Empty string if the command is valid, or an error message if invalid.
        """
        if len(c) < 4:
            return "Move command too short. Should be like 'e2e4'."

        start_square = c[:2]  # e.g., 'e2'
        end_square = c[2:4]   # e.g., 'e4'

        # Check if the squares are valid (within the board)
        if start_square not in self.coord or end_square not in self.coord:
            return "Invalid square notation."

        # Check if it's a valid promotion (optional)
        if len(c) == 5:
            promotion = c[4]  # e.g., 'q' for queen promotion
            if promotion not in ['q', 'r', 'n', 'b']:
                return "Invalid promotion piece."
        else:
            promotion = None

        return ""  # Command is valid

    def parse_move(self, c):
        """
        Parse the move command into internal representation: start square, end square, and promotion.

        :param c: The move command (e.g., 'e2e4', 'b7b8q').
        :return: Tuple (start_pos, end_pos, promotion)
        """
        start_square = c[:2]
        end_square = c[2:4]
        promotion = c[4] if len(c) == 5 else None

        start_pos = self.coord.index(start_square)
        end_pos = self.coord.index(end_square)

        return start_pos, end_pos, promotion

    def search(self, b):
        """
        Search for the best move using the engine's search algorithm (Alpha-Beta pruning).

        :param b: The current board object representing the chessboard state.
        """
        # Start the search for the best move
        start_time = time.time()

        best_move, best_score = self.alphabeta(0, -self.INFINITY, self.INFINITY, b)

        elapsed_time = time.time() - start_time
        print(f"Best move: {best_move} with score {best_score}")
        print(f"Search completed in {elapsed_time:.2f} seconds.")

        # Make the best move found
        start_pos, end_pos, promotion = self.parse_move(best_move)
        b.domove(start_pos, end_pos)
        if promotion:
            b.promote_pawn(end_pos, promotion)  # Handle promotion if necessary

        b.display()

    def alphabeta(self, depth, alpha, beta, b):
        """
        Alpha-Beta search algorithm to find the best move.

        :param depth: The current depth of the search.
        :param alpha: The best score found so far for the maximizing player.
        :param beta: The best score found so far for the minimizing player.
        :param b: The current board object.
        :return: A tuple (best_move, best_score)
        """
        # Base case: if depth exceeds MAX_PLY or the game is over
        if depth >= self.MAX_PLY:
            return None, self.evaluate_position(b)

        # Generate all legal moves for the current player
        legal_moves = b.gen_moves_list(b.side2move)

        # If no legal moves, it's a draw or checkmate
        if not legal_moves:
            return None, self.evaluate_position(b)

        # Iterate through all legal moves and apply the Alpha-Beta pruning
        best_move = None
        best_score = -self.INFINITY if b.side2move == 'white' else self.INFINITY

        for move in legal_moves:
            # Make the move
            start_pos, end_pos, promotion = move
            b.domove(start_pos, end_pos)
            if promotion:
                b.promote_pawn(end_pos, promotion)  # Handle promotion if necessary

            # Recursively search
            _, score = self.alphabeta(depth + 1, -beta, -alpha, b)

            # Undo the move
            b.undo_move()

            # Update the best score and best move
            if b.side2move == 'white' and score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, best_score)
            elif b.side2move == 'black' and score < best_score:
                best_score = score
                best_move = move
                beta = min(beta, best_score)

            # Alpha-Beta pruning
            if alpha >= beta:
                break

        return best_move, best_score

    def evaluate_position(self, b):
        """
        Evaluate the current position on the board.

        :param b: The current board object.
        :return: A score representing the evaluation of the position.
        """
        # Simple evaluation function based on material balance
        score = 0
        for piece in b.cases:
            if piece.isEmpty():
                continue
            piece_value = Piece.valeurPiece[Piece.nomPiece.index(piece.nom)]
            score += piece_value if piece.couleur == 'white' else -piece_value

        return score

    def print_result(self, b):
        """
        Print the result of the game if it's over.

        :param b: The current board object.
        """
        # Here you can implement logic to check if the game is over (checkmate, stalemate, etc.)
        if b.is_checkmate():
            print(f"{b.side2move} wins by checkmate!")
            self.endgame = True
        elif b.is_stalemate():
            print("The game is a stalemate!")
            self.endgame = True
        elif b.is_draw():
            print("The game is a draw!")
            self.endgame = True

"""### Code for the `Engine` Class:(Cf. code on top)

### Explanation of Methods:

1. **`__init__` and `init`**:
   - Initializes the engine's parameters, including the maximum search depth (`MAX_PLY`), principal variation (`pv_length`), and a large value representing checkmate (`INFINITY`).

2. **`usermove`**:
   - Handles a move from the user. It validates the move using `chkCmd`, converts it into internal representation, and updates the board. After the move, it checks the game status and displays the board.

3. **`chkCmd`**:
   - Validates the move command entered by the user (e.g., 'e2e4', 'b7b8n'). It checks for the correct format and ensures the squares and promotion (if any) are valid.

4. **`parse_move`**:
   - Converts the move command from algebraic notation (like 'e2e4') into internal board indices for the start and end positions. It also handles pawn promotion if applicable.

5. **`search`**:
   - The main method for the engine to search for the best move using the `alphabeta` algorithm.

6.

 **`alphabeta`**:
   - Implements the Alpha-Beta pruning algorithm to search for the best move. It recursively explores the game tree, pruning branches that are not promising.

7. **`evaluate_position`**:
   - Evaluates the current position on the board by calculating the material balance (piece values). Positive values indicate an advantage for white, negative values indicate an advantage for black.

8. **`print_result`**:
   - Checks the game status and prints the result if the game is over (checkmate, stalemate, or draw).

This engine uses a simple evaluation function (material value) and Alpha-Beta pruning for move searching. You can expand the evaluation function to consider other factors like piece positions, control of the center, etc.

#Main
Here is the main loop of the chess program as described. This code ties together the `Board` and `Engine` classes, accepting user input, and triggering appropriate actions in the game.
"""

#!/usr/bin/env python3

from board import *  # Importing the Board class and related functions
from engine import *  # Importing the Engine class and related functions

def main():
    """
    The main loop for the chess program where the user can interact with the game.
    """
    # Create instances of the board and engine
    b = Board()  # Board instance to manage the chessboard state
    e = Engine()  # Engine instance to handle move generation, evaluation, and searching

    while True:
        # Render the current chessboard state
        b.render()

        # Get the user's input command
        c = input('>>> ')

        # Handle various commands based on user input
        if c == 'quit' or c == 'exit':
            # Exit the game
            print("Exiting the game...")
            exit(0)

        elif c == 'undomove':
            # Undo the last move
            print("Undoing last move...")
            e.undomove(b)

        elif 'setboard' in c:
            # Set the board to a specific position (using FEN or other notation)
            print("Setting board position...")
            e.setboard(b, c)

        elif c == 'getboard':
            # Get the current board position in FEN notation
            print("Current board in FEN format:")
            e.getboard(b)

        elif c == 'go':
            # The engine searches for the best move and makes it
            print("Engine is thinking and making the best move...")
            e.search(b)

        elif c == 'new':
            # Start a new game
            print("Starting a new game...")
            e.newgame(b)

        elif c == 'bench':
            # Run a benchmark to test the engine's speed and performance
            print("Running the engine's benchmark...")
            e.bench(b)

        elif 'sd ' in c:
            # Set the search depth for the engine
            print(f"Setting search depth to {c[3:]}")
            e.setDepth(c)

        elif 'perft ' in c:
            # Run a perft test (move generation test)
            print(f"Running perft with depth {c[6:]}")
            e.perft(c, b)

        elif c == 'legalmoves':
            # Show the list of legal moves for the current player
            print("Showing legal moves for the current side to move...")
            e.legalmoves(b)

        else:
            # Default case: process the user's move (e.g., 'e2e4', 'b7b8q')
            print("Processing user move...")
            e.usermove(b, c)

if __name__ == '__main__':
    main()

"""### Code for the `Main` Class:(Cf. code on top)

### Explanation of the Code:

1. **Imports**:
   - `from board import *`: Imports everything from the `board.py` file. This includes the `Board` class and any other utility functions or constants that are defined in that file.
   - `from engine import *`: Imports everything from the `engine.py` file, which includes the `Engine` class that handles the logic for move generation, searching, and evaluation.

2. **Main Function**:
   - **Initialization**:
     - The `main()` function creates instances of the `Board` and `Engine` classes, which represent the chessboard and the engine, respectively.
   - **Infinite Loop**:
     - The game runs in an infinite loop (`while True`) where the program waits for user input and processes commands continuously.
   - **Input Commands**:
     - The program waits for the user to input a command (e.g., `e2e4`, `quit`, `new`, etc.).
     - Based on the input, the program performs different actions. These actions are grouped in `if`, `elif`, and `else` blocks to handle various commands.
     - The actions include:
       - **Exit**: If the user types `quit` or `exit`, the program exits.
       - **Undo Move**: If the user types `undomove`, the engine undoes the last move.
       - **Setboard**: If the user types `setboard`, it sets the board to a specific position (likely using a FEN string or another method).
       - **Search**: If the user types `go`, the engine performs a search for the best move.
       - **New Game**: If the user types `new`, the game is reset to the initial position.
       - **Benchmark**: If the user types `bench`, it runs a benchmark for testing the engine's performance.
       - **Search Depth**: If the user types `sd <depth>`, it sets the search depth for the engine.
       - **Perft Test**: If the user types `perft <depth>`, it runs a perft test to verify move generation.
       - **Legal Moves**: If the user types `legalmoves`, it displays the legal moves for the current side to move.
       - **User Move**: For any other input, it is treated as a move (e.g., `e2e4`, `b7b8q`), and the engine processes it using the `usermove()` method.

3. **User Input Parsing**:
   - The input is handled using `input('>>> ')`, and based on the command, the corresponding method in the `Engine` class is called to process the move or request.

4. **Exiting the Program**:
   - The program gracefully exits when the user types `quit` or `exit`, using `exit(0)`.

5. **Rendering the Board**:
   - After every action, the board is rendered using `b.render()`, which is assumed to be a method in the `Board` class that displays the current state of the chessboard to the user.

### Next Steps:
- **Board Rendering**: Implement the `render()` method in the `Board` class to display the current state of the chessboard in a human-readable format.
- **Command Handling**: Implement or modify methods like `usermove()`, `undomove()`, `setboard()`, `getboard()`, and others in the `Engine` and `Board` classes to handle the specific functionality of each command.
- **Game Over Detection**: Add checks for game over conditions (checkmate, stalemate, etc.) to handle the end of the game.

This code should serve as the main interactive loop for the chess program, allowing users to play, test, and interact with the chess engine and board.
"""