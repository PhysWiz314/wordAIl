from pathlib import Path
import string
from typing import Dict, List, Optional, Tuple
import pandas as pd
from numpy.random import choice
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

    def _set_vocab(self, path: Path = config.WORD_FILE) -> Tuple[str, List]:
        all_words = pd.read_csv(path)
        vocab = all_words[all_words['words'].str.len() == self.word_length]
        self.vocab = vocab['words'].values.flatten()

    def _set_word(self):
        self.target_word = choice(self.vocab, 1)[0]

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
                print("Invalid word. Try again!")
            return False
        return True

    def _update_available_letters(self, guess, formatted_guess):
        letters = [letter for letter in self.available_letters if letter not in guess]
        letters = letters + [x for x in formatted_guess if x != '*' and x not in letters]
        self.available_letters = sorted(letters, key=str.casefold)

    def format_guess(self, guess: str) -> List:
        formatted_guess = []
        word_short = self.target_word

        for idx, letter in enumerate(guess):
            if letter == self.target_word[idx]:
                word_short = word_short[:idx] + word_short[idx+1:]
                formatted_guess.append(letter.upper())
            elif letter in word_short:
                formatted_guess.append(letter.lower())
            else:
                formatted_guess.append('*')

        return formatted_guess

    def play(self, guess = None):
        while self.current_turn < self.n_guesses:
            if self.interactive:
                print('Available letters')
                print(self.available_letters)
                guess = input(f'Guess {self.current_turn}: ').lower()

            if not guess:
                raise ValueError("Please enter a guess in play() if not in interactive mode.")

            if not self.validate_guess(guess):
                continue
            elif self.won:
                break

            formatted_guess = self.format_guess(guess)
            if self.interactive:
                print(formatted_guess)

            self._update_available_letters(guess, formatted_guess)
            self.current_turn += 1

        if not self.won and self.interactive:
            print(f"Sorry, the word was {self.target_word}")


if __name__ == '__main__':
    game = Game(config.game_config, config.INTERACTIVE)
    game.play()