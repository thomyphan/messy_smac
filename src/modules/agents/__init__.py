REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .rep_rnn_agent import RepresentationRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent

REGISTRY["central_rnn"] = CentralRNNAgent

REGISTRY["rep_rnn_agent"] = RepresentationRNNAgent
