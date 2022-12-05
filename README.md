# Multi-Arm Bandit Problem in Reinforcement Learning & Implementation in Python

In this post, I am very excited to share the knowledge about the very basic problem solved by Reinforcement Learning i.e; Multi-Arm Bandit Problem (MABP)

![mabp](https://miro.medium.com/max/1400/1*5q0Mihf29fftuXpKWWX2uA.png)

A bandit is defined as someone who steals your money. A one-armed bandit is a simple slot machine wherein you insert a coin into the machine, pull a lever, and get an immediate reward. But why is it called a bandit? It turns out all casinos configure these slot machines in such a way that all gamblers end up losing money!

A multi-armed bandit is a complicated slot machine wherein instead of 1, there are several levers which a gambler can pull, with each lever giving a different return. The probability distribution for the reward corresponding to each lever is different and is unknown to the gambler.

The task is to identify which lever to pull in order to get maximum reward after a given set of trials. This problem statement is like a single step Markov decision process. Each arm chosen is equivalent to an action, which then leads to an immediate reward.

### Exploration Exploitation in the context of  Bernoulli MABP
The below table shows the sample results for a 5-armed Bernoulli bandit with arms labelled as 1, 2, 3, 4 and 5:

![image](https://user-images.githubusercontent.com/82467675/205731113-5269898f-9676-4820-92c8-89a0338e9df3.png)


This is called Bernoulli, as the reward returned is either 1 or 0. In this example, it looks like the arm number 3 gives the maximum return and hence one idea is to keep playing this arm in order to obtain the maximum reward (pure exploitation).

Just based on the knowledge from the given sample, 5 might look like a bad arm to play, but we need to keep in mind that we have played this arm only once and maybe we should play it a few more times (exploration) to be more confident. Only then should we decide which arm to play (exploitation).

### Uses Cases for MABP

- **Clinical Trials:** Here, exploration is equivalent to identifying the best treatment, and exploitation is treating patients as effectively as possible during the trial.
- **Network Routing**: Allocation of channels to the right users, such that the overall throughput is maximised, can be formulated as a MABP.
- **Online Advertising**: Similar to MABP, there is a trade-off between exploration, where the goal is to collect information on an ad’s performance using click-through rates, and exploitation, where we stick with the ad that has performed the best so far.
- **Game Designing**: Building a hit game is challenging. MABP can be used to test experimental changes in game play/interface and exploit the changes which show positive experiences for players.



## Solution Strategies



### Action-Value Function

The expected payoff or expected reward can also be called an action-value function. It is represented by q(a) and defines the average reward for each action at a time t.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_12.jpg" alt="img" style="zoom: 50%;" />

Suppose the reward probabilities for a K-armed bandit are given by {P1, P2, P3 …… Pk}. If the *i* th arm is selected at time t, then Qt(a) = Pi.

The question is, how do we decide whether a given strategy is better than the rest? One direct way is to compare the total or average reward which we get for each strategy after *n* trials. If we already know the best action for the given bandit problem, then an interesting way to look at this is the concept of regret.

### Regret

Let’s say that we are already aware of the best arm to pull for the given bandit problem. If we keep pulling this arm repeatedly, we will get a maximum expected reward which can be represented as a horizontal line (as shown in the figure below):

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_8.jpg" alt="img" style="zoom:50%;" />

But in a real problem statement, we need to make repeated trials by pulling different arms till we am approximately sure of the arm to pull for maximum average return at a time *t*. **The loss that we incur due to time/rounds spent due to the learning is called regret.** In other words, we want to maximise my reward even during the learning phase. Regret is very aptly named, as it quantifies exactly how much you regret not picking the optimal arm.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_9.jpg" alt="img" style="zoom:50%;" />

Now, one might be curious as to how does the regret change if we are following an approach that does not do enough exploration and ends exploiting a suboptimal arm. Initially there might be low regret but overall we are far lower than the maximum achievable reward for the given problem as shown by the green curve in the following figure.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_11.jpg" alt="img" style="zoom:50%;" />

Based on how exploration is done, there are several ways to solve the MABP.


### The Greedy Algorithm

Every time we plug into a socket we get a reward, in the form of an amount of charge, and every reward we get lets us calculate a more accurate estimate of a socket’s true output. If we then just choose the socket with the highest estimate hopefully this will be the best available socket.

The action values for each action will be stored at each timestep by the following function:

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_13.jpg)

When selecting the action with the highest value, the action chosen at time step ‘*t*’, can be expressed by the formula:

![img](https://miro.medium.com/max/229/1*wzdMxeBjRlu5_TFnBh6HNg.png)

However, for evaluating this expression at each time t, we will need to do calculations over the whole history of rewards. We can avoid this by doing a running sum. So, at each time t, the q-value for each action can be calculated using the reward:

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_15.jpg" alt="img" style="zoom: 80%;" />

The problem here is this approach only exploits, as it always picks the same action without worrying about exploring other actions that might return a better reward. Some exploration is necessary to actually find an optimal arm, otherwise we might end up pulling a sub optimal arm forever.

### Epsilon Greedy Approach

One potential solution could be to now, and we can then explore new actions so that we ensure we are not missing out on a better choice of arm. With epsilon probability, we will choose a random action (exploration) and choose an action with maximum qt(a) with probability 1-epsilon.

***With probability 1- epsilon – we choose action with maximum value (argmaxa Qt(a))***

***With probability epsilon – we randomly choose an action from a set of all actions A***

For example, if we have a problem with two actions – A and B, the epsilon greedy algorithm works as shown below:

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2018/09/im_16.jpg" alt="img" style="zoom: 50%;" />

This is much better than the greedy approach as we have an element of exploration here. However, if two actions have a very minute difference between their q values, then even this algorithm will choose only that action which has a probability higher than the others.



### Theoratical Explaination for MABP and code

Below is the handwritten notebook explaining solution to MABP using action-value function.

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bf686780-b5d6-429f-8d5c-e6caf5e486bb/EADC9B79-FB80-44B5-A28E-87EEC41E29D2.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192213Z&X-Amz-Expires=86400&X-Amz-Signature=cdd8ca97f4c240c3e8d9d8b246d3f0873ded2091cadd5d3c1c50a5a664dd4825&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22EADC9B79-FB80-44B5-A28E-87EEC41E29D2.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eae19d22-3164-4ca1-8267-4461c1402c21/6F921F3E-A609-4971-BEB4-83439F4ABE66.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192646Z&X-Amz-Expires=86400&X-Amz-Signature=864cc42e25204f69809b273abeb2f4f9a707d5ae8768e6c108a0398a94b65726&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%226F921F3E-A609-4971-BEB4-83439F4ABE66.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d31cbd7b-b75f-4fa4-81fd-29f4e319a40b/1459FC59-7411-42CD-AC56-06C3124A343A.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192716Z&X-Amz-Expires=86400&X-Amz-Signature=05c34ee9d15e56d5002fe929534fdd5f89ca8614f764c4705087b9a6872528c6&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%221459FC59-7411-42CD-AC56-06C3124A343A.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/20e1d56e-2b02-4b84-9633-5d5654854c0c/732261DD-47E3-407F-8042-99BE3660124A.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192728Z&X-Amz-Expires=86400&X-Amz-Signature=abfac9f52b84060aabc5c20ba550fdca46b1ae5fec7beeafb79418902ff2236f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22732261DD-47E3-407F-8042-99BE3660124A.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/244ba253-da5f-4492-81d7-e01a0a11dad2/528DA07C-D1EA-4B7C-AAC9-4F1C5F79C07E.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192744Z&X-Amz-Expires=86400&X-Amz-Signature=cd52c51a19d0c84555e734ac71a1d9092d0ba80d209424c332e3a276700e1980&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22528DA07C-D1EA-4B7C-AAC9-4F1C5F79C07E.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/07bc766f-9fca-4b04-8a96-ec97b022ae9d/99699790-BB20-45D6-864D-9F5E4EA164FE.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192756Z&X-Amz-Expires=86400&X-Amz-Signature=9fe70101ed07f6a0ed6f24e60929307541f386e70868b0ad20cb813ae6a11a25&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%2299699790-BB20-45D6-864D-9F5E4EA164FE.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/729f6db7-4de6-4930-b09c-d6aa0f2e6f8d/88890BF9-7862-4A38-B345-7BFDA25CDDAF.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192821Z&X-Amz-Expires=86400&X-Amz-Signature=95ec0c66848e0862725b7dbc99ee14b134caa00d6a1c045bf1870f9f6f0344da&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%2288890BF9-7862-4A38-B345-7BFDA25CDDAF.png%22&x-id=GetObject)

![img](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/89c9ae4e-432d-4a96-960c-56015f3e6537/8DB73A44-49D2-4685-9421-AFEFB7BC9D4A.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221205T192832Z&X-Amz-Expires=86400&X-Amz-Signature=534a3e0d895eade9543f22cc45f4348a7a2522dff88fb17c4cea5e5e05a9d2de&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%228DB73A44-49D2-4685-9421-AFEFB7BC9D4A.png%22&x-id=GetObject)



This approach can be implemented as follows:

1. Select a real number Epsilon large than 0 and smaller than 1.
2. Draw a random value *p* from the uniform distribution on the interval 0 to 1.
3. If *p > epsilon*, then select the actions by maximising *Amax* Equation.
4. If *p<= epsilon*, then randomly select an action. That is randomly pick any *a(j)* from the set of all possible actions *{a1, a2, a3, ..., an}*, and apply them to the system.

That is basically it. This was a brief description of the solution of the multi-arm bandit problem.


~~~python
## writing the base classes for the bandit problem -> banditClass, select actions and greedy-actions
import numpy as np

class BanditProblem(object):
    # trueActionValues- its the mean value of the distribution of reward generated when taking an action.
    # the number of arms is equal to the number of entries in tree
    # epsilon - probability value for selecting non-greedy action
    # totalSteps - number of total steps used to simulate the problem
    
    def __init__(self, trueActionValues, epsilon, totalSteps) -> None:
        
        #number of arms
        self.armNumber= np.size(trueActionValues)

        #prob. of ignoring the greedy action and selecting an arm by random
        self.epsilon = epsilon

        #current step
        self.currentStep = 0

        #this step tracks the number of times an arm is pulled
        self.howManyTimesParticularArmIsSelected = np.zeros(self.armNumber)

        #total steps
        self.totalSteps = totalSteps

        #true action values - that are expectations of reward for arms
        self.trueActionValues = trueActionValues

        #vector that stores mean reward of every arm
        self.armMeanRewards = np.zeros(self.armNumber)

        #variable that stores the current value of reward
        self.currentReward=0

        # mean reward
        self.meanReward=np.zeros(totalSteps+1)

    # select actions according to the epsilon-greedy approach
    def selectActions(self):
        # draw a real number from uniform distribution on [0,1]
        # this number is our probability of performing greedy action
        # if this prob. is larger than epsilon, we perform greedy actions
        # otherwise, we randomly select an arm (Exploration policy)
        
        probabilityDraw = np.random.rand()

        # in the initial step, we select a random arm since all the mean rewards are zero
        # we also select a random arm if the prob. is smaller than epsilon
        if (self.currentStep==0) or (probabilityDraw<=self.epsilon):
            selectedArmIndex = np.random.choice(self.armNumber)
        
        # we select the arm that has the largest past mean reward
        if (probabilityDraw>self.epsilon):
            selectedArmIndex=np.argmax(self.armMeanRewards)
        
        #increase the step value
        self.currentStep=self.currentStep+1

        # take a record that the particular arm is selected
        self.howManyTimesParticularArmIsSelected[selectedArmIndex]=self.howManyTimesParticularArmIsSelected[selectedArmIndex]+1

        # draw from the prob dist. of the selected arm the reward
        self.currentReward= np.random.normal(self.trueActionValues[selectedArmIndex], 2)

        # update the estimate of the mean reward
        self.meanReward[self.currentStep]= self.meanReward[self.currentStep-1]+(1/(self.currentStep))*(self.currentReward-self.meanReward[self.currentStep-1])
        
        # update the estimate of the mean reward for the selected arm
        self.armMeanRewards[selectedArmIndex]=self.armMeanRewards[selectedArmIndex]+(1/(self.howManyTimesParticularArmIsSelected[selectedArmIndex]))*(self.currentReward-self.armMeanRewards[selectedArmIndex])

    def playGame(self):
        for i in range(self.totalSteps):
            self.selectActions()
    
    #reset all the variables to the original state
    def clearAll(self):
        # current step
        self.currentStep= 0

        # this variable tracks for the history of how many times an armed is being pulled 
        self.howManyTimesParticularArmIsSelected=np.zeros(self.armNumber)

        # vector that stores mean rewards of every arm
        self.armMeanRewards=np.zeros(self.armNumber)

        # variable that stores the current value of reward
        self.currentReward=0;

        # mean reward
        self.meanReward=np.zeros(self.totalSteps+1)
        
# Next we will be writing a driver code for the above methods defined, to run a simulation for our bandit problem.
~~~



~~~python
# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from banditproblem import BanditProblem
 
# these are the means of the action values that are used to simulate the multi-armed bandit problem
actionValues=np.array([1,4,2,0,7,1,-1])
 
# epsilon values to investigate the performance of the method
epsilon1=0
epsilon2=0.1
epsilon3=0.2
epsilon4=0.3
 
# total number of simulation steps 
totalSteps=100000
 
# create four different bandit problems and simulate the method performance
Bandit1=BanditProblem(actionValues, epsilon1, totalSteps)
Bandit1.playGame()
epsilon1MeanReward=Bandit1.meanReward
Bandit2=BanditProblem(actionValues, epsilon2, totalSteps)
Bandit2.playGame()
epsilon2MeanReward=Bandit2.meanReward
Bandit3=BanditProblem(actionValues, epsilon3, totalSteps)
Bandit3.playGame()
epsilon3MeanReward=Bandit3.meanReward
Bandit4=BanditProblem(actionValues, epsilon4, totalSteps)
Bandit4.playGame()
epsilon4MeanReward=Bandit4.meanReward
 
#plot the results
plt.plot(np.arange(totalSteps+1),epsilon1MeanReward,linewidth=2, color='r', label='epsilon =0')
plt.plot(np.arange(totalSteps+1),epsilon2MeanReward,linewidth=2, color='k', label='epsilon =0.1')
plt.plot(np.arange(totalSteps+1),epsilon3MeanReward,linewidth=2, color='m', label='epsilon =0.2')
plt.plot(np.arange(totalSteps+1),epsilon4MeanReward,linewidth=2, color='b', label='epsilon =0.3')
plt.xscale("log")
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('results.png',dpi=300)
plt.show()
~~~

The code is self-explanatory. We basically create 4 problems for 4 different values of epsilon and we simulate them, and plot the results. 
The Results for the above experiments are shown in the figure below:
![image](https://user-images.githubusercontent.com/82467675/205731392-cfe20f34-0c48-4ce4-9a21-59abba3284e6.png)


The above code simulates the MABP solution using the action-value function on a sample (custom) dataset ***actionValues=np.array([1,4,2,0,7,1,-1])***

Here, it should be emphasized that we have tested the solution approach by only drawing a single realization of action values (code line 16). This is done for brevity in the post. In a more detailed analysis, we need to draw true action values (code line 16) from some random distribution and run the approach many times, and then we need to average the results, to get a better insight into the algorithm performance. The pure greedy approach (red line) does not produce the best results. The average reward converges to ![1](https://aleksandarhaber.com/wp-content/ql-cache/quicklatex.com-4868771cbc422b5818f85500909ce433_l3.svg) and that is one of the true action values that are entries of the vector of true action values defined on the code line 16. On the other hand, we can observe that for epsilon=0.1, we obtain the best results. The average reward approaches to **6.5**, and that is very close to **7** which is one of the true action values for the action **5**. In fact, action **5** should produce the best performance since it has the highest action value. We can see that the algorithm is able to find the most appropriate action value.

We can also track how many times a particular action is being selected by inspecting “howManyTimesParticularArmIsSelected “, for epsilon=0.1, we obtain:

~~~python
print(Bandit2.howManyTimesParticularArmIsSelected)
Output: [ 1421.  1473.  1393.  1458. 91463.  1445.  1347.]
~~~

**Clearly, the action number 5 is selected most of the times, so we are very close to the optimal performance!**

Some form of cross-validation or grid search can be used to find the most optimal value of the epsilon parameter. Of course, there are many other approaches for solving the multi-arm bandit problem. However, since this is an introductory tutorial, I did not cover all the approaches.
