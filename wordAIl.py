from pathlib import Path
import string
from typing import List, Tuple
import pandas as pd
from numpy.random import choice
from common.config import WORD_FILE, N_GUESSES, WORD_LENGTH

class Game:

    def __init__(self, word_length: int = WORD_LENGTH, n_guesses: int = N_GUESSES):
        self.word_length = word_length
        self.n_guesses = n_guesses
        self.current_turn = 0
        self.won = False
        self._set_alphabet()
        self._set_vocab()
        self._set_word()

    def _set_alphabet(self) -> List:
        self.available_letters = list(string.ascii_lowercase)

    def _set_vocab(self, path: Path = WORD_FILE) -> Tuple[str, List]:
        all_words = pd.read_csv(path)
        vocab = all_words[all_words['words'].str.len() == self.word_length]
        self.vocab = vocab['words'].values.flatten()

    def _set_word(self):
        self.target_word = choice(self.vocab, 1)[0]
        print(self.target_word)


    def play(self):
        while self.current_turn < self.n_guesses:
            print('Available letters')
            print(self.available_letters)
            guess = input(f'Guess {self.current_turn}: ').lower()
            if guess == self.target_word:
                print(f"Correct! {self.target_word}")
                self.won = True
                break
            if len(guess) != self.word_length:
                print(f"Word must be {self.word_length} letters! Try again.")
                continue
            if guess not in self.vocab:
                print("Invalid word. Try again!")
                continue

            formatted_guess = []
            word_short = self.target_word
            for idx, letter in enumerate(guess):
                if letter == self.target_word[idx]:
                    word_short = word_short[:idx] + word_short[idx+1:]
                    formatted_guess.append(letter.capitalize())
                elif letter in word_short:
                    formatted_guess.append(letter.lower())
                else:
                    formatted_guess.append('*')

            self.available_letters = [letter for letter in self.available_letters if letter not in guess]
            self.available_letters = self.available_letters + [x for x in formatted_guess if x != '*']
            self.available_letters = sorted(self.available_letters, key=str.casefold)
            print(formatted_guess)
            self.current_turn += 1

        if not self.won:
            print(f"Sorry, the word was {self.target_word}")


if __name__ == '__main__':
    game = Game()
    game.play()