"""
Concrete query strategy classes.
"""
import logging
logger = logging.getLogger(__name__)

from .active_learning_by_learning import ActiveLearningByLearning
from .hintsvm import HintSVM
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .linear_ucb import LinUCB
from .exp3 import Exp3
from .linear_ucb_qs import LinUCBqs
from .random_mulqs  import RandMulQs
from .represent  import KMeansRepresent
from .boundary_uncertain  import BoundaryUncertain
from .model_change  import ModelChange
from .density_weighted_uncertainty_sampling import DWUS
