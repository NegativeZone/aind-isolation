"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

# Knights in chess are best in the center and weakest on the edges and corners
# This heuristic is designed to look for moves that push the opponent to the edges while moving oneself to the center
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    h, w = game.height, game.width
    own_score = 0
    opp_score = 0
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    fac = 1

    # Now to account for the endgame, where being stuck in a corner is a lot worse than in the early game
    if len(game.get_blank_spaces()) < 0.3*h*w:
        fac = 2

    for move in own_moves:
        # Avoid the edge squares
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            own_score += 1*fac
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            own_score += 2*fac
        # Avoid the near-the-edge squares
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            own_score += 4*fac
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            own_score += 4*fac
        # The closer to the center the better
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            own_score += 6*fac
        else:
            own_score += 8*fac

    # But instead, push the enemy that way
    for move in opp_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            opp_score += 1*fac
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            opp_score += 2*fac
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            opp_score += 4*fac
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            opp_score += 4*fac
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            opp_score += 6*fac
        else:
            opp_score += 8*fac

    return float(own_score - opp_score)

# This is a simple aggression function that chases down the enemy
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    score = len(own_moves and opp_moves)
    return float(score)

# Improved Mobility: Weights the improved score with how many subsequent moves can be made from a given position
def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    # Check for next move
    own_mobility = float(sum([len(game.get_legal_moves(player)) for move in own_moves ]))
    opp_mobility = float(sum([len(game.get_legal_moves(game.get_opponent(player))) for move in opp_moves ]))
    return (len(own_moves) * own_mobility) - (len(opp_moves) * opp_mobility)



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Uses external function to make passing score of each move easier
        return self.minmax_helper(game, depth) [0]

    # Helper function that returns tuples in (move, score) format
    def minmax_helper(self, game, depth):

        # Timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If target depth is reached
        if depth == 0:
            return (game.get_player_location(self), self.score(game, self))

        # Set up default return value
        best_move = (-1, -1)

        # If active, maximize
        # Otherwise minimize
        if game.active_player == self:
            score = float("-inf")
            for move in game.get_legal_moves():
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                next_state = game.forecast_move(move)
                next_score = self.minmax_helper(next_state, depth-1)[1]
                if max(score, next_score) == next_score:
                    best_move = move
                    score = next_score
        else:
            score = float("inf")
            for move in game.get_legal_moves():
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                next_state = game.forecast_move(move)
                next_score = self.minmax_helper(next_state, depth-1)[1]
                if min(score, next_score) == next_score:
                    best_move = move
                    score = next_score

        # minimax function will only return best_move
        # score will be used in recursive reduction above
        return (best_move, score)



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Declare and initialize variable
        best_move = (-1, -1)

        # Iterative search until arbitrary large depth
        for i in range(1, 25):
            try:
                best_move = self.alphabeta(game, i)
            except SearchTimeout:
                break
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Similar approach, separating the helper function to ease coding
        return self.alphabeta_helper(game, depth)[0]

    def alphabeta_helper(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        # Timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If no more moves
        if not game.get_legal_moves():
            if game.active_player == self:
                return (-1, -1), float("-inf")
            else:
                return (-1, -1), float("inf")

        best_move = (-1, -1)
        lo_score= float("inf")
        hi_score = float("-inf")

        # For depth = 1
        if depth == 1:
            if game.active_player == self:
                for move in game.get_legal_moves():
                    if self.time_left() < self.TIMER_THRESHOLD:
                        raise SearchTimeout()
                    score = self.score(game.forecast_move(move), self)
                    if score > hi_score:
                        hi_score = score
                        best_move = move
                    if score >= beta:
                        return  move, score
                return best_move, hi_score
            else:
                for move in game.get_legal_moves():
                    if self.time_left() < self.TIMER_THRESHOLD:
                        raise SearchTimeout()
                    score = self.score(game.forecast_move(move), self)
                    if score < lo_score:
                        lo_score = score
                        best_move = move
                    if score <= alpha:
                        return move, score
                return best_move, lo_score

        # General execution
        if game.active_player == self:
            for move in game.get_legal_moves():
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                score = self.alphabeta_helper(game.forecast_move(move), depth-1, alpha, beta) [1]
                if score > hi_score:
                    hi_score = score
                    best_move = move
                if score >= beta:
                    return move, score
                alpha = max(alpha, hi_score)
            return best_move, hi_score
        else:
            for move in game.get_legal_moves():
                if self.time_left() < self.TIMER_THRESHOLD:
                    raise SearchTimeout()
                score = self.alphabeta_helper(game.forecast_move(move), depth-1, alpha, beta) [1]
                if score < lo_score:
                    lo_score = score
                    best_move = move
                if score <= alpha:
                    return move, score
                beta = min(beta, lo_score)
            return best_move, lo_score
