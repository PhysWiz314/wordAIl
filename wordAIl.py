from common import config
import numpy as np
import string

from game import Game    


def generate_guess(state, word_length):
        # Initialize the alphabet
        alphabet = list(string.ascii_lowercase)
        # Initialize the guess
        guess = np.random.choice(alphabet, p=state[:26]/sum(state[:26]), size=word_length)
        guess = ''.join(guess)

        # Ensure that the guess is a real word
        attempt = 0
        while guess not in game.vocab:
            guess = np.random.choice(alphabet, p=state[:26]/sum(state[:26]), size=word_length, replace=False)
            guess = ''.join(guess)
            if attempt % 100 == 0:
                print(f'{sum(state[26:])}-{guess}: {attempt}', end='\r')
                # input()
            attempt += 1
            if attempt == 1_000_000:
                print("Can't find a word in over 1 million guesses.")
                return None
        print(guess, ' '*20)
        return guess


if __name__ == '__main__':
    if config.PLAY:
        while True:
            game = Game(config.game_config, config.INTERACTIVE)
            game.play()

            if not game.won and config.INTERACTIVE:
                print(f"Good try! The word was {game.target_word.upper()}.")
            
            if config.INTERACTIVE:
                again = input("Would you like to play again? (y/n): ")
                if again == "n":
                    break
    
    else:
        # state = np.array([1]*26 + [-1]*26)  # [Score of letter, index of letter]
        reward = 0  #  sum of letter scores 3 - right position, 2 - right letter, 1 - not guessed, 0 guessed
        game_n = 0

        won = []
        while game_n < 1000:
            game_n += 1
            game = Game(config.game_config, config.INTERACTIVE)
            print(f"Game: {game_n}")
            while game.current_turn < game.n_guesses:
                guess = generate_guess(game.state, game.word_length)
                if guess is None:
                    won.append(-1)
                    print(f"Sorry You Lose, the word was {game.target_word}")
                    break
                game.take_a_turn(guess)
                reward = sum(game.state[:26])
                if game.won:
                    print(f"You win! The word was {game.target_word}")
                    won.append(game.current_turn)
                    break

                if game.current_turn == game.n_guesses:
                    won.append(0)
                    print(f"Sorry You Lose, the word was {game.target_word}")
                # print(state[:26], state[26:], reward)
                # input()
        wins_vs_losses = [1 if x > 0 else 0 for x in won]
        turns = [x for x in won if x > 0]
        print(f'Stats: Win % - {sum(wins_vs_losses)/len(wins_vs_losses)}')
        print(f'Stats: Avg Turns - {sum(turns)/len(turns)}')
        print(f'Stats: >1M guesses - {sum([1 if x < 0 else 0 for x in won])}')