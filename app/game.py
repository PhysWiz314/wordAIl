import string
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

from common import config


class Game(py_environment.PyEnvironment):

    def __init__(self, config: Dict, interactive: bool):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.int32, minimum=0, maximum=25, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(26,), dtype=np.int32, minimum=0, maximum=1, name='observation'
        )
        # self._state = 0
        self._episode_ended = False

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
        self._state = np.zeros(26, dtype=np.int32)
        print(self._state)


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
                # self.state[pos_in_alphabet + 26] = idx

    def format_guess(self, guess: str) -> List:
        formatted_guess = []
        word_short = self.target_word

        for idx, letter in enumerate(guess):
            if letter == self.target_word[idx]:
                word_short = word_short[:idx] + word_short[idx+1:]
                if config.PLAY:
                    formatted_guess.append(letter.upper())
                else:
                    formatted_guess.append([letter.upper(), 50])
            elif letter in word_short:
                if config.PLAY:
                    formatted_guess.append(letter.lower())
                else:
                    formatted_guess.append([letter.lower(), 25])
            else:
                if config.PLAY:
                    formatted_guess.append('*')
                else:
                    formatted_guess.append(['*', 0])
        return formatted_guess

    def _step(self, action):
        # Turn the action into a word
        alphabet = list(string.ascii_lowercase)
        guess = ''.join([alphabet[i] for i in action])

        if self._episode_ended:
            return self.reset()

        print(self.current_turn)
        if self.current_turn >= self.n_guesses:
            self._episode_ended = True
        else:
            formatted_guess = self.format_guess(guess)
            if self.interactive:
                print(formatted_guess)
            if config.PLAY:
                self._update_available_letters(guess, formatted_guess)
            else:
                self._update_alphabet_scores(guess, formatted_guess)
            self._state = self.state

        print(self.state)
        if self._episode_ended:
            reward = sum(self._state)
            return ts.termination(self._state, reward)
        else:
            self.current_turn += 1
            return ts.transition(
                self._state, reward=0, discount=1.0
            )


    def _reset(self):
        # self._state = np.array([0]*26)
        self._episode_ended = False
        return ts.restart(self._state)

    def play(self, guess = None):
        while self.current_turn < self.n_guesses:
            if not guess:
                print('Available letters')
                print(self.available_letters)
                guess = input(f'Guess {self.current_turn}: ').lower()

            if not guess:
                raise ValueError("Please enter a guess in play() if not in interactive mode.")

            self._step(guess)


    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec