import string
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from common import config


class Game:

    def __init__(self, config: Dict, interactive: bool):
        self.word_length = config['word_length']
        self.n_guesses = config['n_guesses']
        self.interactive = interactive
        self.current_turn = 0
        self.won = False
        self._set_alphabet()
        self._set_vocab()
        self._set_word()

    def _set_alphabet(self) -> List:
        self.available_letters = list(string.ascii_lowercase)
        self.state = np.array([1]*26 + [-1]*26)  # [Score of letter, index of letter]


    def _set_vocab(self, path: Path = config.WORD_FILE) -> Tuple[str, List]:
        all_words = pd.read_csv(path)
        vocab = all_words[all_words['words'].str.len() == self.word_length]
        self.vocab = vocab['words'].values.flatten()

        # Make sure letters don't appear twice in the word
        self.vocab = [word for word in self.vocab if len(set(word)) == self.word_length]

    def _set_word(self):
        self.target_word = np.random.choice(self.vocab, 1)[0]

    def validate_guess(self, guess: str) -> Optional[bool]:
        if guess == self.target_word:
            if self.interactive:
                print(f"Correct! {self.target_word}")
            self.won = True
            return True
        if len(guess) != self.word_length:
            if self.interactive:
                print(f"Word must be {self.word_length} letters! Try again.")
            return False
        if guess not in self.vocab:
            if self.interactive:
                print(f"Invalid word {guess}. Try again!")
            return False
        return True

    def _update_available_letters(self, guess: str, formatted_guess: List) -> None:
        letters = [letter for letter in self.available_letters if letter not in guess]
        letters = letters + [x for x in formatted_guess if x != '*' and x not in letters]
        self.available_letters = sorted(letters, key=str.casefold)        

    def _update_alphabet_scores(self, guess: str, formatted_guess: List):
        for idx, letter in enumerate(formatted_guess):
            if letter[0] == '*':
                pos_in_alphabet = self.available_letters.index(guess[idx])
                self.state[pos_in_alphabet] = letter[1]
            else:
                pos_in_alphabet = self.available_letters.index(letter[0].lower())
                self.state[pos_in_alphabet] = letter[1]
                self.state[pos_in_alphabet + 26] = idx

    def format_guess(self, guess: str) -> List:
        formatted_guess = []
        word_short = self.target_word

        for idx, letter in enumerate(guess):
            if letter == self.target_word[idx]:
                word_short = word_short[:idx] + word_short[idx+1:]
                if config.PLAY:
                    formatted_guess.append(letter.upper())
                else:
                    formatted_guess.append([letter.upper(), 20])
            elif letter in word_short:
                if config.PLAY:
                    formatted_guess.append(letter.lower())
                else:
                    formatted_guess.append([letter.lower(), 10])
            else:
                if config.PLAY:
                    formatted_guess.append('*')
                else:
                    formatted_guess.append(['*', 0])
        return formatted_guess

    def take_a_turn(self, guess):
            if not self.validate_guess(guess):
                return

            formatted_guess = self.format_guess(guess)
            if self.interactive:
                print(formatted_guess)
            if config.PLAY:
                self._update_available_letters(guess, formatted_guess)
            else:
                self._update_alphabet_scores(guess, formatted_guess)
            self.current_turn += 1

    def play(self, guess = None):
        while self.current_turn < self.n_guesses:
            if not guess:
                print('Available letters')
                print(self.available_letters)
                guess = input(f'Guess {self.current_turn}: ').lower()

            if not guess:
                raise ValueError("Please enter a guess in play() if not in interactive mode.")

            self.take_a_turn(guess)
