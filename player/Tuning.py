import sys
sys.path.append("..")
import pandas as pd
from Game import Game
from Heuristic import Heuristic
from agent import Agent
from statistics import mean
from sklearn.model_selection import ParameterGrid
import os
import subprocess
import socket

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Tuning:

    def __init__(self, game: Game, heuristic: Heuristic):
        self._game = game
        self._heuristic = heuristic

    def __get_port_number(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        host, addr = s.getsockname()
        s.close()
        return addr

    def tune_parameters_bruteforce(self):
        """ Perform a grid search that finds the best possible parameter values for the Heuristic function """

        param_grid = {'alpha': [1, 5, 10], 'beta': [1, 5, 10], 'gamma': [1, 5, 10],
                      'delta': [1, 5, 10], 'win': [10, 100], 'lose': [-10, -100]}

        grid = ParameterGrid(param_grid)

        columns = ['alpha', 'beta', 'gamma', 'delta', 'win', 'lose', 'win_rate']
        df = pd.DataFrame(columns=columns)

        for params in grid:

            results = []
            # run 10 times and get average result
            for i in range(10):
                agent = Agent(self._game, self._heuristic)
                agent.set_heuristic_params(params['alpha'], params['beta'], params['gamma'], params['delta'],
                                           params['win'], params['lose'])

                port = self.__get_port_number()

                # run a game instance. We go second.
                subprocess.Popen(['../src/servt', '-p', str(port)])
                subprocess.Popen(['../src/lookt.mac', '-p', str(port)])
                result = agent.run(port=port)

                # set Heuristic to calculate per call instead of using hash
                results.append(result)

            average_result = mean(results)
            df.loc[len(df)] = [params['alpha'], params['beta'], params['gamma'], params['delta'], params['win'],
                               params['lose'], average_result]

        df.to_csv(os.path.join(SAVE_PATH, 'parameters.csv'))


if __name__ == "__main__":
    g = Game()
    h = Heuristic()
    g.load()
    h.load()

    t = Tuning(g, h)
    t.tune_parameters_bruteforce()
