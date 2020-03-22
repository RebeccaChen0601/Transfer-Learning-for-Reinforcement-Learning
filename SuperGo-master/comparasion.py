import multiprocessing
import time
import signal
import click
import os

from lib.evaluate import evaluate
from lib.train import train
from lib.play import play, self_play
from lib.process import MyPool
from lib.utils import load_player


@click.command()
@click.option("--folder", default=1584177784)
@click.option("--black", default=-1)
@click.option("--white", default=1)
def main(folder, black, white):

    player_black, checkpoint_black = load_player(str(folder), black)
    player_white, checkpoint_white = load_player(str(folder), white)

    print("Loaded Black player with iteration " + str(checkpoint_black['total_ite']))
    print("Loaded White player with iteration " + str(checkpoint_white['total_ite']))

    ## Start method for PyTorch
    multiprocessing.set_start_method('spawn')


    evaluate(player_black, player_white)




if __name__ == "__main__":
    main()


