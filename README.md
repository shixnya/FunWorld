# The first world
This is a readme for the first small world environment.

## World specifications
* The world is a 30 x 30 2D grid space.
* There is only one living entity (agent)
* The agent has 100 energy at the beginning.
* Resource (food) is distributed around.
* The agent can see food within 5 grid from itself (both horizontal or vertical)
* Time is descritized, and in every step, a food appears at a random location with 2% chance.
* In every time step, the agent can choose 5 choices. Move to each of 4 directions, or stay.
* The agent spend 1 energy if it moves, it spend 0.5 energy if it stays.
* If the agent use up energy, it dies and lose 3 reward points.
* The agent can replenish the energy by overlapping with food.
* In every time step, there is a 2% chance that a food appears in the field. The amount of food is random (max is 100).
* The agent is rewarded when it gets food (food / 100 is reward)


# The first training procedure

## Training of the agent
* Uses Keras-rl2
* Mostly reuse the Keras-rl2's example code that was used for Atari training.
* Changed the environment to mine, and use the food map as the input (that is returned form the
 environemnt)
* It is a convolutional neural network with the following specifications
  * Two convolution layers with 3x3x16 panels, with stride 1.
  * One fully connected layer with 32 neurons
  * Activation function is 'selu' 
* The training procedure is the following (pretty much what the given network was using)
  * Deep Q Network with epsilon greedy Q policy with linear annealing
  * Adam optimizer with learning rate = 0.001
  * 10000 warm up steps
  * Target model is updated every 10000 steps


# Sample image
![Sample Image] (images/SampleImage1.png?raw=true "Image")
