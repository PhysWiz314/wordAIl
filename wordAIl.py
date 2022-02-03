import pandas as pd
from common.config import WORD_FILE, N_GUESSES, WORD_LENGTH

all_words = pd.read_csv(WORD_FILE)
words = all_words[all_words['words'].str.len() == WORD_LENGTH]

word_of_the_day = words.sample(1)['words'].values[0]

i = 0
won = False
while i < N_GUESSES:
    guess = input(f'Guess {i}: ')
    if guess == word_of_the_day:
        print(f"Correct! {word_of_the_day}")
        won = True
        break
    if len(guess) != 5:
        print("Word must be 5 letters! Try again.")
        continue
    if guess not in words['words'].values.flatten():
        print("Invalid word. Try again!")
        continue
    good_letters = []
    right_letters = []
    formatted_guess = []
    word_short = word_of_the_day
    for idx, letter in enumerate(guess):
        if letter == word_of_the_day[idx]:
            word_short = word_short[:idx] + word_short[idx+1:]
            formatted_guess.append(letter.capitalize())
        elif letter in word_short:
            formatted_guess.append(letter.lower())
        else:
            formatted_guess.append('*')

    print(formatted_guess)
    i += 1

if not won:
    print(f"Sorry, the word was {word_of_the_day}")