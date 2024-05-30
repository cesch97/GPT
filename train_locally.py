from main import main
import json
from argparse import Namespace


if __name__ == '__main__':

    with open('config/local_config.json') as f:
        config = json.load(f)
    config = Namespace(**config)
    main(config)
