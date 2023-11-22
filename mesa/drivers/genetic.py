import logging

from pandas import read_hdf

from mesa.drivers import Optimizer


class GeneticOptimizer(Optimizer):

    population: list
    generations: list
    mutation_rate = 0.1  # 10% mutation rate

    def __init__(
        self,
        params: dict,
        initial_sample_count=20,
        max_iterations=200,
        pop_size=8,
        tolerance=1e-8,
    ):
        super().__init__(
            params,
            initial_sample_count=initial_sample_count,
            max_iterations=max_iterations,
            concurrent_runs=1,
        )
        self.pop_size = pop_size
        self.current_generation = 0
        self.population = []
        self.generations = []
        self.opt_cols = ["logprob"]

    def breed(self, p1, p2):
        return

    def initialize(self, sim, objfn, trainingfile):
        """
        Run initial population randomly distributed
        """
        super().initialize(sim, objfn, trainingfile)
        for i in range(self.pop_size):
            individual = {}
            # select random value in bounds for free params
            for key in self.free_parameter_keys:
                bnds = self.optimization_bounds[key]
                individual[key] = (bnds[1] - bnds[0]) * self.rng.random() + bnds[0]
            # add the fixed values
            for key in self.fixed_parameter_keys:
                individual[key] = self.fixed_parameter_values[key]
            self.population.append(individual)
        self.generations.append(self.population)

        # run the simulations
        self.run(new_points=self.population, start_iter=True)

        return

    def get_next_points(self):
        """
        Get next generation
        """
        # load the training data
        df = read_hdf(self.training_file, "training")
        # break the loop if we've hit the max number of iterations
        if df.index[-1][0] >= self.max_iterations:
            logging.info(
                "[optimiser] Optimisation loop broken due to reaching the maximum allowed iterations"
            )
            return None

        # extract the training data
        # sort by figure of merit
        fom = df["logprob"].values[-(self.pop_size) :]
        lastpop = [self.population[ii] for ii in fom.argsort()]

        # take two best and pass them into the next population
        self.population[-1] = lastpop[0]
        self.population[-2] = lastpop[1]

        # remove all but top 50% and breed the remaining
        ntop = int(0.5 * self.pop_size)
        lastpop = lastpop[0:ntop]

        for i in range(self.pop_size - 2):
            # choose two random indices
            r1 = int(self.rng.random() * ntop)
            r2 = r1
            while r2 == r1:
                r2 = int(self.rng.random() * ntop)
            # average the two chosen, occasionally mutate a parameter
            for k in self.free_parameter_keys:
                self.population[i][k] = 0.5 * (lastpop[r1][k] + lastpop[r2][k])
                if self.rng.random() < self.mutation_rate:
                    bnds = self.optimization_bounds[k]
                    self.population[i][k] = (
                        bnds[1] - bnds[0]
                    ) * self.rng.random() + bnds[0]

        self.generations.append(self.population)

        logging.info("New population:")
        logging.info(
            [
                [param_dict[k] for k in self.free_parameter_keys]
                for param_dict in self.population
            ]
        )

        return self.generations[-1]
