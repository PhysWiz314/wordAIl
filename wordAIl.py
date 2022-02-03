from common import config

from game import Game


if __name__ == '__main__':
    while True:
        game = Game(config.game_config, config.INTERACTIVE)
        game.play()

        if not game.won and config.INTERACTIVE:
            print(f"Good try! The word was {game.target_word.upper()}.")
        
        if config.INTERACTIVE:
            again = input("Would you like to play again? (y/n): ")
            if again == "n":
                break