import random
import math

BOT_NAME = "BetaGo"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""

    def get_move(self, state, depth=None):
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""

    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state, depth=None):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state, depth)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state, depth):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board
            depth: for this agent, the depth argument should be ignored!

        Returns: the exact minimax utility value of the state
        """
        # Fill this in!
        # TODO:
        # if the state is the terminal state, then return the utility value (score)
        if state.is_full():
            return state.score()
        # if next player to play is Player1, then we want to choose the maximum minimax value of successors
        if state.next_player() == 1:
            max_minimax = float('-inf')
            for tup in state.successors():
                child_minimax = self.minimax(tup[1], depth)
                if max_minimax < child_minimax:
                    max_minimax = child_minimax
            return max_minimax
        else:  # else, next player is Player2, we want to choose the minimum minimax value of the successors
            min_minimax = float('inf')
            for tup in state.successors():
                child_minimax = self.minimax(tup[1], depth)
                if min_minimax > child_minimax:
                    min_minimax = child_minimax
            return min_minimax


def op_ed_streak2(lst):
    """Calculate the number of open-ended 2-in-a-row for player1 and player2 in the lst.

    Args:
        lst: list that represent a row/column/diagonal of the game board

    Returns: a tuple that contains the number of open-ended 2-in-a-row for player1 and player2 in the lst
    """
    # TODO:
    # if the length of the list have less than 3 elements, then we won't have open-ended 2-in-a-row
    if len(lst) < 3:
        result = (0, 0)
        return result

    # counters for p1 and p2
    p1_streaks = 0
    p2_streaks = 0
    # record previous ceil & current streak length
    prev = lst[0]
    curr_len = 1
    for curr in lst[1:]:
        if curr == 0:  # current cell is empty
            # update p1/p2 counters only if we have streak length > 1 and prev is not empty
            if curr_len > 1:
                if prev == 1:
                    p1_streaks += 1
                elif prev == -1:
                    p2_streaks += 1
        if curr == prev:  # current cell is the same as the previous one
            curr_len += 1
        else:
            prev = curr
            curr_len = 1
    result = (p1_streaks, p2_streaks)
    return result


def streaks(lst):
    """Return the lengths of all the streaks of the same element in a sequence."""
    rets = []  # list of (element, length) tuples
    prev = lst[0]
    curr_len = 1
    for curr in lst[1:]:
        if curr == prev:
            curr_len += 1
        else:
            rets.append((prev, curr_len))
            prev = curr
            curr_len = 1
    rets.append((prev, curr_len))
    return rets


class HeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def minimax(self, state, depth):
        return self.minimax_depth(state, depth)

    def minimax_depth(self, state, depth):
        """Determine the heuristically estimated minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board
            depth: the maximum depth of the game tree that minimax should traverse before
                estimating the utility using the evaluation() function.  If depth is 0, no
                traversal is performed, and minimax returns the results of a call to evaluation().
                If depth is None, the entire game tree is traversed.

        Returns: the minimax utility value of the state
        """
        # Fill this in!
        # TODO:
        # if the state is the terminal state, then return the utility value (score)
        if state.is_full():
            return state.score()
        # if the state reach the limit of depth
        if depth is not None and depth <= 0:
            return self.evaluation(state)
        # if next player to play is Player1, then we want to choose the maximum minimax value of successors
        if state.next_player() == 1:
            max_minimax = float('-inf')
            for tup in state.successors():
                if depth is None:
                    child_minimax = self.minimax_depth(tup[1], depth)
                else:
                    child_minimax = self.minimax_depth(tup[1], depth - 1)
                if max_minimax < child_minimax:
                    max_minimax = child_minimax
            return max_minimax
        else:  # else, next player is Player2, we want to choose the minimum minimax value of the successors
            min_minimax = float('inf')
            for tup in state.successors():
                if depth is None:
                    child_minimax = self.minimax_depth(tup[1], depth)
                else:
                    child_minimax = self.minimax_depth(tup[1], depth - 1)
                if min_minimax > child_minimax:
                    min_minimax = child_minimax
            return min_minimax

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in O(1) time!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        # TODO:
        # version 1:
        # p1_heuristic = 0
        # p2_heuristic = 0
        # rows = [r for r in range(state.num_rows)]
        # cols = [c for c in range(state.num_cols)]
        # next_row = False
        # for r in range(len(rows)):
        #     if next_row:
        #         next_row = False
        #         continue
        #     deleted = []
        #     for c in range(len(cols)):
        #         if state.board[rows[r]][cols[c]] == 0:
        #             for run in [state.get_row(rows[r])] + [state.get_col(cols[c])] + \
        #                        [state.get_diags(rows[r], cols[c])]:
        #                 h_pair = op_ed_streak2(run)
        #                 p1_heuristic += h_pair[0]
        #                 p2_heuristic += h_pair[1]
        #             deleted.append(c)
        #             next_row = True
        #     while len(deleted) > 0:
        #         d = deleted.pop()
        #         del cols[d]
        #
        # return 4 * (p1_heuristic - p2_heuristic) + state.score()

        # version 2:
        # TODO:
        p1_score = 0
        p2_score = 0
        prev_length = 0
        prev_elt = -999
        for run in state.get_all_rows() + state.get_all_cols() + state.get_all_diags():
            for elt, length in streaks(run):
                if (elt == 1) and (length >= 2):
                    if prev_elt == 0:  # check whether previous streak is 0 streak (case: 0xx)
                        p1_score += 3  # heuristic of open-ended 2-in-a-row
                    if length >= 3:  # calculate score
                        p1_score += length ** 2
                elif (elt == -1) and (length >= 2):
                    if prev_elt == 0:  # check whether previous streak is 0 streak (case: 0xx)
                        p2_score += 3  # heuristic of open-ended 2-in-a-row
                    if length >= 3:  # calculate score
                        p2_score += length ** 2
                else:  # 0 appears, check whether the length of streaks before 0 >= 2 (case: xx0)
                    if prev_length >= 2:
                        if prev_elt == 1:
                            p1_score += 3  # heuristic of open-ended 2-in-a-row
                        elif prev_elt == -1:
                            p2_score += 3  # heuristic of open-ended 2-in-a-row
                prev_length = length
                prev_elt = elt

        return p1_score - p2_score


class PruneAgent(HeuristicAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    def minimax(self, state, depth):
        return self.minimax_prune(state, depth)

    def minimax_prune(self, state, depth):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by ComputerAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the class variables
        GameState.p1_state_count and GameState.p2_state_count, which keep track of how many
        GameState objects were created over time.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: see ComputerDepthLimitAgent.minimax() above

        Returns: the minimax utility value of the state
        """
        #
        # Fill this in!
        # TODO:
        return self.alpha_beta(state, float('-inf'), float('inf'), depth)

    def alpha_beta(self, state, alpha, beta, depth):
        """
        Args:
            state: a connect383.GameState object representing the current board
            alpha: lower bound of predecessor state
            beta: upper bound of predecessor state
            depth: the maximum depth of the game tree that minimax should traverse before
                estimating the utility using the evaluation() function.  If depth is 0, no
                traversal is performed, and minimax returns the results of a call to evaluation().
                If depth is None, the entire game tree is traversed.

        Returns: the minimax utility value of the state
        """
        # Fill this in!
        # TODO:
        # if the state is the terminal state, then return the utility value (score)
        if state.is_full():
            return state.score()
        # if the state reach the limit of depth
        if depth is not None and depth <= 0:
            return self.evaluation(state)
        # if next player to play is Player1, then we want to choose the maximum minimax value of successors
        if state.next_player() == 1:
            max_minimax = float('-inf')
            for tup in state.successors():
                if depth is None:
                    child_minimax = self.alpha_beta(tup[1], alpha, beta, depth)
                else:
                    child_minimax = self.alpha_beta(tup[1], alpha, beta, depth - 1)
                if max_minimax < child_minimax:
                    max_minimax = child_minimax
                if alpha < max_minimax:
                    alpha = max_minimax
                if alpha >= beta:
                    break
            return max_minimax
        else:  # else, next player is Player2, we want to choose the minimum minimax value of the successors
            min_minimax = float('inf')
            for tup in state.successors():
                if depth is None:
                    child_minimax = self.alpha_beta(tup[1], alpha, beta, depth)
                else:
                    child_minimax = self.alpha_beta(tup[1], alpha, beta, depth - 1)
                if min_minimax > child_minimax:
                    min_minimax = child_minimax
                if beta > min_minimax:
                    beta = min_minimax
                if beta <= alpha:
                    break
            return min_minimax

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in O(1) time!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        # TODO:
        p1_score = 0
        p2_score = 0
        prev_length = 0
        prev_elt = -999
        for run in state.get_all_rows() + state.get_all_cols() + state.get_all_diags():
            for elt, length in streaks(run):
                if (elt == 1) and (length >= 2):
                    if prev_elt == 0:  # check whether previous streak is 0 streak (case: 0xx)
                        p1_score += 3  # heuristic of open-ended 2-in-a-row
                    if length >= 3:  # calculate score
                        p1_score += length ** 2
                elif (elt == -1) and (length >= 2):
                    if prev_elt == 0:  # check whether previous streak is 0 streak (case: 0xx)
                        p2_score += 3  # heuristic of open-ended 2-in-a-row
                    if length >= 3:  # calculate score
                        p2_score += length ** 2
                else:  # 0 appears, check whether the length of streaks before 0 >= 2 (case: xx0)
                    if prev_length >= 2:
                        if prev_elt == 1:
                            p1_score += 3  # heuristic of open-ended 2-in-a-row
                        elif prev_elt == -1:
                            p2_score += 3  # heuristic of open-ended 2-in-a-row
                prev_length = length
                prev_elt = elt

        return p1_score - p2_score
