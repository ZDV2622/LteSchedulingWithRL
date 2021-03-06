# Scheduling with RL in LTE network

Description
-----------------------------------

This code can be used to estimate efficiency of resource scheduling  in LTE system.
It lets to compare three algorithms: RoundRobin, BestCQI and RLAgent.
This model lets to investigate, how diffrent RL technique(DQN, PG) can be applied to solve optimization problem.

Very simple model is used for traffic generation. It doesn't take into account requests delay and UE prioryty, also it doesn't support buffer. In each subframe we generate new requests and if it is not sent, it will be lost. All these drawbacks will be eliminated in the next update. So the traffic model lets to investigate all cell throughput optimization but can't be applied for ue throughput optimization.

Some information about resourse allocation in LTE can be found here:
--------------------------------------------
http://www.iosrjournals.org/iosr-jece/papers/Vol.%209%20Issue%206/Version-3/G09635053.pdf

https://www.researchgate.net/publication/263315028_Performance_comparaison_Of_scheduling_algorithms_For_downlink_LTE_system

https://www.researchgate.net/publication/316531160_Resource_scheduling_algorithms_for_LTE_using_weights

Project structure
-----------------------------------
Parameters: input paremeters for traffic generation, agent and learning process

PGAgent.py: it has class PGagent, wich contains all needed functionality for PG agent realization: _build_model, act,predict, train

Envirement.py: class Envirement lets to estimate reward and next_state from some action.

PGLearnPredict.ipynb: use it to start and visualize learning process of PG agent

Example of results:
------------------
Here is example of PGagent learning.
Reward depends on the strategy

![Simulation](/img/example_7_3.png)




