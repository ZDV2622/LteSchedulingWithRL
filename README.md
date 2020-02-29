# LteSchedulingWithRL

Description
-----------------------------------

This code can be used to estimate efficiency of resource scheduling  in LTE system.
It lets to compare three algorithms: RoundRobin, BestCQI and RLAgent.
This model lets to investigate, how diffrent RL technique(DQN, PG) can be applied to solve optimization problem.

Some information about resourse allocation in LTE can be found here:
[http://www.iosrjournals.org/iosr-jece/papers/Vol.%209%20Issue%206/Version-3/G09635053.pdf]
[https://www.researchgate.net/publication/263315028_Performance_comparaison_Of_scheduling_algorithms_For_downlink_LTE_system]
[https://www.researchgate.net/publication/316531160_Resource_scheduling_algorithms_for_LTE_using_weights]

Brief fiels description
-----------------------------------

PGAgent.py: it has class PGagent, wich contains all needed functionality for PG agent realization: _build_model, act,predict, train
Envirement.py: class Envirement lets to estimate reward and next_state from some action.

