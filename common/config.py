from pathlib import Path

DATA_PATH = Path('data')
WORD_FILE = DATA_PATH / 'words.csv'

game_config = {
    'word_length': 5,
    'n_guesses': 6,
}
