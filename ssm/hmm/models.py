from ssm.hmm.base import HMM



class GaussianHMM(HMM):
    def __init__(num_states,
                 initial_state_probs=None,
                 transition_matrix=None,
                 means=None,
                 covariances=None):

        transitions = ssm.hmm.transitions.StationaryTransitions(transition)
        emissions = ssm.hmm.emissions.GaussianEmissions(means=means, covariances=covariances)
        super(GaussianHMM, self).__init__(initia)