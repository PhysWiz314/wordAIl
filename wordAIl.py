from common import config
import numpy as np
import string

from game import Game    


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
        state = np.array([1]*26 + [-1]*26)  # [Score of letter, index of letter]
        reward = 0  #  sum of letter scores 3 - right position, 2 - right letter, 1 - not guessed, 0 guessed
        game_n = 0
        alphabet = list(string.ascii_lowercase)

        won = []
        while game_n < 100:
            game_n += 1
            game = Game(config.game_config, config.INTERACTIVE)
            while game.current_turn < game.n_guesses:
                # Initialize the guess
                guess = np.random.choice(alphabet, p=state[:26]/sum(state[:26]), size=game.word_length)
                guess = ''.join(guess)

                # Ensure that the guess is a real word
                while guess not in game.vocab:
                    guess = np.random.choice(alphabet, p=state[:26]/sum(state[:26]), size=game.word_length)
                    guess = ''.join(guess)
                    print(guess, end='\r')
                print(guess)
                game.take_a_turn(guess)
                state = game.state
                reward = sum(state[:26])
                if game.won:
                    print(f"You win! The word was {game.target_word}")
                    won.append(game.current_turn)
                    break

                if game.current_turn == game.n_guesses:
                    won.append(0)
                    print(f"Sorry You Lose, the word was {game.target_word}")
                # print(state[:26], state[26:], reward)
                # input()
        wins_vs_losses = [1 for x in won if x > 0]
        turns = [x for x in won if x > 0]
        print(f'Stats: Win % - {sum(wins_vs_losses)/len(wins_vs_losses)}')
        print(f'Stats: Avg Turns - {sum(turns)/len(turns)}')