from common import config

from game import Game


if __name__ == '__main__':
    game = Game(config.game_config, config.INTERACTIVE)
    game.play()