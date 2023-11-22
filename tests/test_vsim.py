# ----------------------------------------------------------------------------
#   general settings
# ----------------------------------------------------------------------------
# directory where a reference SOLPS run is stored
ref_directory = '/pfs/work/g2hjame/solps-iter/runs/TCV_58196_small/ref_clean/'

# file name in which the training data will be stored
training_data_file = 'training_data.h5'

# ----------------------------------------------------------------------------
#   diagnostics settings
# ----------------------------------------------------------------------------
from numpy import load
from mesa.diagnostics import WeightedObjectiveFunction, Spectrum

Spec = Spectrum(
    frequency=1.0e9,
    goal="maximize"
)

dgns = [Spec]
wgts = [1.0]

objective_function = WeightedObjectiveFunction(dgns,wgts)

# ----------------------------------------------------------------------------
#   simulation settings
# ----------------------------------------------------------------------------
from mesa.simulations import VSim
simulation = VSim(
    exe='/Applications/VSim-12.0/VSimComposer.app/Contents/Resources/VSimComposer.sh',
    n_proc=6,
    timeout_hours=24
)

# ----------------------------------------------------------------------------
#   driver settings
# ----------------------------------------------------------------------------

from mesa.drivers import GeneticOptimizer

# parameters to fix or vary. To fix set as a single value. To vary set bounds as a tuple
params = {
    # testing
    "Parameter1" : (0.0, 1.0),
    "Parameter2" : (-10.0, 10.0)
}

driver = GeneticOptimizer(
    params,
    initial_sample_count = 0,
    max_iterations = 200,
    pop_size = 8
)