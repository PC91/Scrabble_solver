#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:46:52 2021

@author: truongthai-chau
"""

import numpy as np
from queue import Queue

drow = np.array([-1, 0, 1, 0])
dcol = np.array([0, 1, 0, -1])
dfill_row = np.array([1, 0])
dfill_col = np.array([0, 1])
dfill_backward = {0:0, 1:3}
dfill_forward = {0:2, 1:1}

arr_fillable = np.empty(0)
arr_next_empty = np.empty(0)
arr_last_non_empty = np.empty(0)
board_letter = np.empty(0)
board_state = np.empty(0)
    
def init_board(board_letter):
    """Pre-calculate some variables that are used for the algorithm to find
    the word with highest score.
    
    Args:
        board_letter (array): the current Scrabble board.
    Raises:
        
    """
    height, width = np.shape(board_letter)
    
    # arr_next_empty[direction, row, col] stores the nearest empty cell when
    # going on each direction (0 for vertical, 1 for vertical)
    # from a given empty cell.
    # This is used to determine the next cell to fill.
    # arr_count_empty = np.array([[[0] * width] * height] * 2)
    
    # arr_count_empty[direction, row, col] stores the number of empty cells
    # when going on each direction (0 for vertical, 1 for vertical)
    # from a given empty cell.
    # This is used to determine if it is possible to fill n empty cells.
    arr_next_empty = np.array([[[-1] * width] * height] * 2)
    
    for col in range(width):
        last_empty_row = height - 1
        for row in reversed(range(height)):
            arr_next_empty[0, row, col] = last_empty_row
            if (board_letter[row, col] == " "):
                last_empty_row = row
                
        # count_empty_row = 0
        # for row in reversed(range(height)):
        #     if (board_letter[row, col] == " "):
        #         count_empty_row = count_empty_row + 1
        #         arr_count_empty[0, row, col] = count_empty_row
        
    for row in range(height):
        last_empty_col = width - 1
        for col in reversed(range(width)):
            arr_next_empty[1, row, col] = last_empty_col
            if (board_letter[row, col] == " "):
                last_empty_col = col
        
        # count_empty_col = 0
        # for col in reversed(range(width)):
        #     if (board_letter[row, col] == " "):
        #         count_empty_col = count_empty_col + 1
        #         arr_count_empty[1, row, col] = count_empty_col
                
            
    # arr_last_non_empty[direction, row, col] stores the last non-empty cell
    # in a range starting from the next cell when going on each of 4 directions
    # from a given cell. If there is no non-empty cell, it stores the values of
    # its row/column. This is used to determine newly formed words
    # when somes cells are filled. The 4 directions are determined as :
    #     0
    #   3 . 1
    #     2
    # Therefore, directions 0, 2 store row values and directions 1, 3 store columns
    # For example, given the following array
    #    0   1   2   3   4   5
    # 0 ' ' 'n' ' ' ' ' ' ' ' ' 
    # 1 'b' 'a' 'n' 'a' 'l' ' ' 
    # 2 ' ' 'v' ' ' ' ' 'e' ' ' 
    # 3 't' 'a' 'n' 'k' 's' ' '
    # 4 ' ' 'l' ' ' ' ' 's' ' '
    # 5 ' ' ' ' ' ' ' ' ' ' ' '
    # We have some values :
    # - arr_last_non_empty[0, 0, 0] = 0
    # - arr_last_non_empty[1, 0, 0] = 1
    # - arr_last_non_empty[2, 0, 0] = 1
    # - arr_last_non_empty[3, 0, 0] = 0
    # 
    # - arr_last_non_empty[0, 1, 2] = 1
    # - arr_last_non_empty[1, 1, 2] = 4
    # - arr_last_non_empty[2, 1, 2] = 1
    # - arr_last_non_empty[3, 1, 2] = 0
    # 
    # - arr_last_non_empty[0, 4, 2] = 3
    # - arr_last_non_empty[1, 4, 2] = 2
    # - arr_last_non_empty[2, 4, 2] = 4
    # - arr_last_non_empty[3, 4, 2] = 1
    arr_last_non_empty = np.array([[[-1] * width] * height] * 4)
    
    for col in range(width):
        last_nonempty_row = 0
        for row in range(height):
            if (board_letter[row, col] != " ") and \
                (
                    (row == 0) or \
                    (board_letter[row + drow[0], col + dcol[0]] == " ")
                ):
                last_nonempty_row = row
                
            if (board_letter[row, col] != " "):
                arr_last_non_empty[0, row, col] = last_nonempty_row
            else:
                if (row == 0) or \
                    (board_letter[row + drow[0], col + dcol[0]] == " "):
                    arr_last_non_empty[0, row, col] = row
                else:
                    arr_last_non_empty[0, row, col] = last_nonempty_row
                    
    for col in range(width):
        last_nonempty_row = height - 1
        for row in reversed(range(height)):
            if (board_letter[row, col] != " ") and \
                (
                    (row == height - 1) or \
                    (board_letter[row + drow[2], col + dcol[2]] == " ")
                ):
                last_nonempty_row = row
                
            if (board_letter[row, col] != " "):
                arr_last_non_empty[2, row, col] = last_nonempty_row
            else:
                if (row == height - 1) or \
                    (board_letter[row + drow[2], col + dcol[2]] == " "):
                    arr_last_non_empty[2, row, col] = row
                else:
                    arr_last_non_empty[2, row, col] = last_nonempty_row
    
    for row in range(height):
        last_nonempty_col = 0
        for col in range(width):
            if (board_letter[row, col] != " ") and \
                (
                    (col == 0) or \
                    (board_letter[row + drow[3], col + dcol[3]] == " ")
                ):
                last_nonempty_col = col
                
            if (board_letter[row, col] != " "):
                arr_last_non_empty[3, row, col] = last_nonempty_col
            else:
                if (col == 0) or \
                    (board_letter[row + drow[3], col + dcol[3]] == " "):
                    arr_last_non_empty[3, row, col] = col
                else:
                    arr_last_non_empty[3, row, col] = last_nonempty_col
                    
    for row in range(height):
        last_nonempty_col = width - 1
        for col in reversed(range(width)):
            if (board_letter[row, col] != " ") and \
                (
                    (col == width - 1) or \
                    (board_letter[row + drow[1], col + dcol[1]] == " ")
                ):
                last_nonempty_col = col
                
            if (board_letter[row, col] != " "):
                arr_last_non_empty[1, row, col] = last_nonempty_col
            else:
                if (col == width - 1) or \
                    (board_letter[row + drow[1], col + dcol[1]] == " "):
                    arr_last_non_empty[1, row, col] = col
                else:
                    arr_last_non_empty[1, row, col] = last_nonempty_col
    
    return arr_next_empty, arr_last_non_empty



class Node():
    def __init__(self, char):
        self.char = char
        self.children = dict()
        self.finished = False


class ScrabbleTrie():
    def __init__(self, letter_scores):
        self.root = Node('')
        self.letter_scores = letter_scores

    def add(self, word):
        """Adds a word to the trie"""
        node = self.root
        for char in word:
            if (char in node.children):
                node = node.children[char]
            else:
                new_node = Node(char)
                node.children[char] = new_node
                node = new_node
        node.finished = True

    def score(
        self,
        start_row, start_col,
        word, arr_joker,
        dfill_direction,
        start = False
    ):
        # print(word)
        total_score = 0
        factor = 1
        row = start_row
        col = start_col
        nb_filled_letters = 0
        crossed_ST = False
        for i in range(len(word)):
            
            new_score = self.letter_scores[word[i]] \
                        if (not arr_joker[row, col]) \
                        else 0
            
            crossed_ST = (board_state[row, col] == "ST")
            # print(" ".join([str(row), str(col), word[i]]))
            
            # Consider the special cells that double/triple words/letters
            if (arr_fillable[row, col]):
                
                nb_filled_letters += 1
                
                if (start) and (board_state[row, col] == "ST"):
                    factor *= 2
                    
                if (board_state[row, col] == "DL"):
                    new_score *= 2
                elif (board_state[row, col] == "TL"):
                    new_score *= 3
                
                if (board_state[row, col] == "DW"):
                    factor *= 2
                elif (board_state[row, col] == "TW"):
                    factor *= 3
                    
            total_score += new_score
            row += dfill_row[dfill_direction]
            col += dfill_col[dfill_direction]
            
        total_score *= factor
        if (nb_filled_letters == 7):
            total_score += 50
            
        return total_score
        
    def contains(self, arr_word):
        i = 0
        node = self.root
        while (i < len(arr_word)):
            if (arr_word[i] in node.children):
                node = node.children[arr_word[i]]
                i += 1
            else:
                return False
        
        return (node.finished)
    
    
    def check_word_intersection(
        self,
        start_pattern_pos, end_pos,
        fill_row, fill_col,
        dfill_direction,
        start = False
    ):
        """
            Test if the surrounding region of the newly formed words
            intersects with at leasts one existing character (a valid turn)
        """
        height, width = board_letter.shape

        # Get the region that contains the word and have 2 more rows/columns
        # to ensure that the newly formed word intersects with at least 1
        # filled letter.
        if (dfill_direction == 0):

            word_row_1 = start_pattern_pos
            word_col_1 = fill_col
            word_row_2 = end_pos
            word_col_2 = fill_col
            
            if (not start):
                small_bounding_row_1 = word_row_1
                small_bounding_col_1 = word_col_1
                small_bounding_row_2 = min(height - 1, word_row_2 + 1)
                small_bounding_col_2 = word_col_2
                
                big_bounding_row_1 = word_row_1
                big_bounding_col_1 = max(0, word_col_1 - 1)
                big_bounding_row_2 = word_row_2
                big_bounding_col_2 = min(width - 1, word_col_2 + 1)
            else:
                small_bounding_row_1 = word_row_1
                small_bounding_col_1 = word_col_1
                small_bounding_row_2 = word_row_2
                small_bounding_col_2 = word_col_2

        elif (dfill_direction == 1):

            word_row_1 = fill_row
            word_col_1 = start_pattern_pos
            word_row_2 = fill_row
            word_col_2 = end_pos

            if (not start):
                small_bounding_row_1 = word_row_1
                small_bounding_col_1 = word_col_1
                small_bounding_row_2 = word_row_2 
                small_bounding_col_2 = min(width - 1, word_col_2 + 1)
                
                big_bounding_row_1 = max(0, word_row_1 - 1)
                big_bounding_col_1 = word_col_1
                big_bounding_row_2 = min(width - 1, word_row_2 + 1)
                big_bounding_col_2 = word_col_2
            else:
                small_bounding_row_1 = word_row_1
                small_bounding_col_1 = word_col_1
                small_bounding_row_2 = word_row_2
                small_bounding_col_2 = word_col_2

        if (not start):
            nb_filled_letters = np.sum(
                board_letter[
                    word_row_1:(word_row_2+1),
                    word_col_1:(word_col_2+1)
                ] == " "
            )
            nb_exist_small_box = np.sum(
                board_letter[
                    small_bounding_row_1:(small_bounding_row_2+1),
                    small_bounding_col_1:(small_bounding_col_2+1)
                ] != " "
            )
            nb_exist_big_box = np.sum(
                board_letter[
                    big_bounding_row_1:(big_bounding_row_2+1),
                    big_bounding_col_1:(big_bounding_col_2+1)
                ] != " "
            )
            
            return (
                (nb_filled_letters > 0) and (
                    (nb_exist_small_box > 0) or \
                    (nb_exist_big_box > 0)
                )
            )
        else:
            return (
                np.sum(
                    board_state[
                        small_bounding_row_1:(small_bounding_row_2+1),
                        small_bounding_col_1:(small_bounding_col_2+1)
                    ] == "ST"
                ) > 0
            )
        
        
    def get_possible_words(
        self,
        lst_letter,
        dfill_direction,
        pattern, start_pattern_pos, end_pattern_pos,
        fill_row, fill_col,
        start = False
    ):
        """
        Generates all possible words that can be made in the trie

        letters: (list) 
            A list of letters
            * stands for a joker
        """
        height, width = board_state.shape

        i = 0
        arr_joker = np.array([[False] * width] * height)
        queue = Queue()
        queue.put((
            self.root, self.root.char, lst_letter,
            i, start_pattern_pos, arr_joker,
            0
        ))
        
        while queue.qsize() > 0:
            node, word, letters_left, i, \
            pos, arr_joker, total_score = queue.get()
            
            
            # If a cell is already filled in previous turns
            if (i < len(pattern)):
                if (pattern[i] != " "):
                    if (pattern[i] in node.children):
                        child = node.children[pattern[i]]
                        new_word = word + pattern[i]
                        # print(new_word)
                        queue.put((
                            child, new_word, letters_left,
                            i + 1, pos + 1,
                            np.copy(arr_joker),
                            total_score
                        ))
                        # Only consider a formed meaningful word in the filling direction
                        # when it touches the last non-empty cell in that direction
                        if (child.finished):
                            if ((dfill_direction == 0) and (pos == height - 1)) or \
                                ((dfill_direction == 1) and (pos == width - 1)) or \
                                (i == len(pattern) - 1) or \
                                (pattern[i + 1] == " "):
                                if (self.check_word_intersection(
                                    start_pattern_pos = start_pattern_pos,
                                    end_pos = pos,
                                    fill_row = fill_row,
                                    fill_col = fill_col,
                                    dfill_direction = dfill_direction,
                                    start = start
                                )):
                                    new_word_score = self.score(
                                        start_row = start_pattern_pos \
                                            if (dfill_direction == 0) \
                                            else fill_row,
                                        start_col = start_pattern_pos \
                                            if (dfill_direction == 1) \
                                            else fill_col,
                                        word = new_word,
                                        arr_joker = arr_joker,
                                        dfill_direction = dfill_direction,
                                        start = start
                                    )
                                    yield (
                                        new_word, fill_row, fill_col, \
                                        dfill_direction, \
                                        total_score + new_word_score \
                                    )
                else:
                    if (len(letters_left) > 0):
                        for char in node.children:
                            if (char in letters_left or "*" in letters_left):
                                # Test if the word formed in perpendicular
                                # direction exists in the dictionary
                                is_valid_perpend_word = False
                                new_perpend_word_score = 0

                                new_arr_joker = np.copy(arr_joker)
                                if (char not in letters_left):
                                    if (dfill_direction == 0):
                                        new_arr_joker[pos, fill_col] = True
                                    elif (dfill_direction == 1):
                                        new_arr_joker[fill_row, pos] = True

                                if (dfill_direction == 0):
                                    board_letter[pos, fill_col] = char
                                    start_pos = arr_last_non_empty[
                                        dfill_backward[1 - dfill_direction],
                                        pos,
                                        fill_col
                                    ]
                                    end_pos = arr_last_non_empty[
                                        dfill_forward[1 - dfill_direction], 
                                        pos,
                                        fill_col
                                    ]

                                    if (start_pos == end_pos):
                                        is_valid_perpend_word = True
                                    elif (self.contains(
                                        board_letter[
                                            pos, start_pos:(end_pos + 1)
                                        ]
                                    )):
                                        is_valid_perpend_word = True
                                        perpend_word = ''.join(
                                            board_letter[
                                                pos, start_pos:(end_pos + 1)
                                            ]
                                        )
                                        # print(perpend_word)
                                        new_perpend_word_score = self.score(
                                            start_row = pos,
                                            start_col = start_pos,
                                            word = perpend_word,
                                            arr_joker = new_arr_joker,
                                            dfill_direction = 1 - dfill_direction,
                                            start = start
                                        )
                                        # print(new_perpend_word_score)

                                    board_letter[pos, fill_col] = " "

                                elif (dfill_direction == 1):
                                    board_letter[fill_row, pos] = char
                                    start_pos = arr_last_non_empty[
                                        dfill_backward[1 - dfill_direction],
                                        fill_row,
                                        pos
                                    ]
                                    end_pos = arr_last_non_empty[
                                        dfill_forward[1 - dfill_direction],
                                        fill_row,
                                        pos
                                    ]                   

                                    if (start_pos == end_pos):
                                        is_valid_perpend_word = True
                                    elif (self.contains(
                                        board_letter[
                                            start_pos:(end_pos + 1), pos
                                        ]
                                    )):
                                        is_valid_perpend_word = True
                                        perpend_word = ''.join(
                                            board_letter[
                                                start_pos:(end_pos + 1), pos
                                            ]
                                        )
                                        # print(perpend_word)
                                        new_perpend_word_score = self.score(
                                            start_row = start_pos,
                                            start_col = pos,
                                            word = perpend_word,
                                            arr_joker = new_arr_joker,
                                            dfill_direction = 1 - dfill_direction,
                                            start = start
                                        )
                                        # print(new_perpend_word_score)

                                    board_letter[fill_row, pos] = " "
                                
                                if (start) or (is_valid_perpend_word):
                                    child = node.children[char]
                                    new_word = word + child.char
                                    # print(new_word)
                                    new_bag = letters_left[:]
                                    new_bag.remove(
                                        child.char \
                                        if child.char in letters_left \
                                        else "*"
                                    )

                                    queue.put((
                                        child, new_word, new_bag,
                                        i + 1, pos + 1,
                                        new_arr_joker,
                                        total_score + new_perpend_word_score
                                    ))

                                    # Only consider a formed meaningful word in the filling direction
                                    # when it touches the last non-empty cell in that direction
                                    if (child.finished):
                                        if ((dfill_direction == 0) and (pos == height - 1)) or \
                                            ((dfill_direction == 1) and (pos == width - 1)) or \
                                            (i == len(pattern) - 1) or \
                                            (pattern[i + 1] == " "):
                                            if (self.check_word_intersection(
                                                start_pattern_pos = start_pattern_pos,
                                                end_pos = pos,
                                                fill_row = fill_row,
                                                fill_col = fill_col,
                                                dfill_direction = dfill_direction,
                                                start = start
                                            )):
                                                new_word_score = self.score(
                                                    start_row = \
                                                        start_pattern_pos \
                                                        if (dfill_direction == 0) \
                                                        else fill_row,
                                                    start_col = \
                                                        start_pattern_pos \
                                                        if (dfill_direction == 1) \
                                                        else fill_col,
                                                    word = new_word,
                                                    arr_joker = new_arr_joker,
                                                    dfill_direction = dfill_direction,
                                                    start = start
                                                )
                                                    
                                                yield (
                                                    new_word,
                                                    fill_row, fill_col,
                                                    dfill_direction,
                                                    total_score + new_word_score + new_perpend_word_score
                                                )
    
    
def find_best_solutions(
    input_board_letter,
    input_board_state,
    lst_letter,
    trie,
    start = False
):
    """
    Consider every posibility that a player can fill and print out
    a new solution that has a better score.
    """
    
    global arr_fillable, arr_next_empty, arr_last_non_empty, board_letter, board_state

    board_letter = input_board_letter
    board_state = input_board_state
    arr_fillable = np.array(board_letter == " ")
    arr_next_empty, arr_last_non_empty = init_board(board_letter)

    if (start):
        start_row = np.where(board_state == "ST")[0][0]
        start_col = np.where(board_state == "ST")[1][0]
        
    height, width = board_state.shape
    max_score = -1
    for row in range(height):
        for col in range(width):
            if (board_letter[row, col] == " "):
                for dfill_direction in [0, 1]:
    
                    fillable = True
                    if (start):                    
                        if (
                            (1 < (start_row - row + 1) <= len(lst_letter)) and \
                            (start_col - col + 1 == 1) and \
                            (dfill_direction == 0)
                        ) or \
                        (
                            (1 < (start_col - col + 1) <= len(lst_letter)) and \
                            (start_row - row + 1 == 1) and \
                            (dfill_direction == 1)
                        ) or \
                        (
                            (start_col - col + 1 == 1) and \
                            (start_row - row + 1 == 1)
                        ):
                            fillable = True
                        else:
                            fillable = False
                        
                    if (not fillable):
                        continue
                    
                    row_1 = max(0, row - 1)
                    col_1 = max(0, col - 1)
                    row_2 = min(
                        height - 1,
                        row + len(lst_letter) * dfill_row[dfill_direction] + \
                            dfill_row[1-dfill_direction]
                    )
                    col_2 = min(
                        width - 1,
                        col + len(lst_letter) * dfill_col[dfill_direction] + \
                            dfill_col[1-dfill_direction]
                    )
                    
                    # A good position is when the bounding rectangle contains
                    # at least 1 filled letter or a starting fillable position
                    if (start) or (
                        np.sum(
                            board_letter[row_1:(row_2+1), col_1:(col_2+1)] == " "
                        ) != (row_2-row_1+1)*(col_2-col_1+1)
                    ):
                        # print("Found")
                        # count += 1
                        # print(str(row) + " " + str(col) + " " + str(dfill_direction))
                        # print(" ".join([str(row_1), str(col_1), str(row_2), str(col_2)]))
    
                        start_pattern_pos = arr_last_non_empty[
                            dfill_backward[dfill_direction], row, col
                        ]
                        end_pattern_pos = start_pattern_pos
                            
                        if (dfill_direction == 0):
                            
                            for i in range(len(lst_letter)):
                                end_pattern_pos = arr_next_empty[
                                    dfill_direction,
                                    end_pattern_pos,
                                    col
                                ]
                                
                            end_pattern_pos = arr_last_non_empty[
                                dfill_forward[dfill_direction],
                                end_pattern_pos,
                                col
                            ]
                            
                            pattern = board_letter[
                                start_pattern_pos:(end_pattern_pos+1),
                                col
                            ]
                            
                        elif (dfill_direction == 1):
                            
                            for i in range(len(lst_letter)):
                                end_pattern_pos = arr_next_empty[
                                    dfill_direction,
                                    row,
                                    end_pattern_pos
                                ]
                                
                            end_pattern_pos = arr_last_non_empty[
                                dfill_forward[dfill_direction],
                                row,
                                end_pattern_pos
                            ]
                            
                            pattern = board_letter[
                                row,
                                start_pattern_pos:(end_pattern_pos+1)
                            ]
                    
                        lst_possible_words = trie.get_possible_words(
                            lst_letter = lst_letter,
                            dfill_direction = dfill_direction,
                            pattern = pattern,
                            start_pattern_pos = start_pattern_pos,
                            end_pattern_pos = end_pattern_pos,
                            fill_row = row,
                            fill_col = col,
                            start = start
                        )
                        
                        if (lst_possible_words is not None):
                            for solution in enumerate(lst_possible_words):
                                word = solution[1][0]
                                score = solution[1][4]
                                
                                if (max_score < score):
                                    max_score = score
                                    
                                    fill_row = row
                                    fill_col = col
                                    
                                    if (dfill_direction == 0):
                                        fill_row = arr_last_non_empty[
                                            dfill_backward[dfill_direction],
                                            fill_row,
                                            fill_col
                                        ]
                                        
                                    elif (dfill_direction == 1):
                                        fill_col = arr_last_non_empty[
                                            dfill_backward[dfill_direction],
                                            fill_row,
                                            fill_col
                                        ]
                                        
                                    for i in range(len(word)):
                                        if (arr_fillable[fill_row, fill_col]):
                                            board_letter[fill_row, fill_col] = word[i]
                                        fill_row += dfill_row[dfill_direction]
                                        fill_col += dfill_col[dfill_direction]
    
                                    print(board_letter)
                                    print(score)
    
                                    fill_row = row
                                    fill_col = col
                                    if (dfill_direction == 0):
                                        fill_row = arr_last_non_empty[
                                            dfill_backward[dfill_direction],
                                            fill_row,
                                            fill_col
                                        ]
                                    elif (dfill_direction == 1):
                                        fill_col = arr_last_non_empty[
                                            dfill_backward[dfill_direction],
                                            fill_row,
                                            fill_col
                                        ]
                                    for i in range(len(word)):
                                        if (arr_fillable[fill_row, fill_col]):
                                            board_letter[fill_row, fill_col] = " "
                                        fill_row += dfill_row[dfill_direction]
                                        fill_col += dfill_col[dfill_direction]