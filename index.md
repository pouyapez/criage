---
layout: default
---
<p align="justify">
Representing entities and relations in an embedding space is a well-studied approach for machine learning on relational data. Existing approaches, however, primarily focus on improving accuracy and overlook other aspects such as robustness and interpretability. In this paper, we propose adversarial modifications for link prediction models: identifying the fact to add into or remove from the knowledge graph that changes the prediction for a target fact after the model is retrained. Using these single modifications of the graph, we identify the most influential fact for a predicted link and evaluate the sensitivity of the model to the addition of fake facts. We introduce an efficient approach to estimate the effect of such modifications by approximating the change in the embeddings when the knowledge graph changes. To avoid the combinatorial search over all possible facts, we train a network to decode embeddings to their corresponding graph components, allowing the use of gradient-based optimization to identify the adversarial modification. We use these techniques to evaluate the robustness of link prediction models (by measuring sensitivity to additional facts), study interpretability through the facts most responsible for predictions (by identifying the most influential neighbors), and detect incorrect facts in the knowledge base.
</p>

# Criage

* * *

![Branching](/images/criage.png)

<p align="justify">
For adversarial modifications on KGs, we first define the space of possible modifications. 
For a target triple <s, r, o>, we constrain the possible triples that we can remove (or inject) to be in the form of <s', r', o> , i.e, s' and r' may be different from the target, but the object is not. 
</p>

### Removing a fact (CRIAGE-Remove)

<p align="justify">
For explaining a target prediction, we are interested in identifying the observed fact that has the most influence (according to the model) on the prediction.
We define influence of an observed fact on the prediction as the change in the prediction score if the observed fact was not present when the embeddings were learned. Previous work have used this concept of influence similarly for several different tasks.

Formally, for the target triple <s,r,o> and observed graph G, we want to identify a neighboring triple <s',r',o> in G such that the score ψ(s,r,o) when trained on G and the score ψ'(s,r,o) when trained on G-<s',r',o> are maximally different, i.e.
</p>

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Coperatorname*%7Bargmax%7D_%7B%28s%27%2C%20r%27%29%5Cin%20%5Ctext%7BNei%7D%28o%29%7D%20%5CDelta_%7B%28s%27%2Cr%27%29%7D%28s%2Cr%2Co%29%5Cnonumber%20%5Cend%7Balign%7D)

Where ∆(s,r,o) =ψ(s,r,o)−ψ'(s,r,o).

### Adding a new fact (CRIAGE-Add)

<p align="justify">
We are also interested in investigating the robustness of models, i.e., how sensitive are the predictions to small additions to the knowledge graph. Specifically, for a target prediction <s,r,o>, we are interested in identifying a single fake fact <s',r',o> that, when added to the knowledge graph G, changes the prediction score ψ(s,r,o) the most.
Using ψ'(s,r,o) as the score after training on G ∪ <s',r',o>, we define the adversary as:
</p>

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Coperatorname*%7Bargmax%7D_%7B%28s%27%2C%20r%27%29%7D%20%5CDelta_%7B%28s%27%2Cr%27%29%7D%28s%2Cr%2Co%29%5Cnonumber%20%5Cend%7Balign%7D)

Where ∆(s,r,o) =ψ(s,r,o)−ψ'(s,r,o).

# Efficiently Identifying the Modification

* * *

In this section, we propose algorithms to addressmentioned challenges by (1) approximating the effect of changing the graph on a target prediction,and (2) using continuous optimization for the discrete search over potential modifications.

### First-order Approximation of Influence

<p align="justify">
We first study the addition of a fact to the graph, and  then  extend  it  to  cover  removal  as  well. To capture the effect of an adversarial modification on the score of a target triple, we need to study the effect of the change on the vector representations of the target triple. As a result, using the Taylor expansion on the changes of the loss after conducting the attack, we approximate ψ(s,r,o)−ψ'(s,r,o) as:
 </p>

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%26%5Coverline%7B%5Cpsi%7D%7B%28s%2Cr%2Co%29%7D-%5Cpsi%28s%2C%20r%2C%20o%29%3D%20%5Cmathbf%7Bz%7D_%7Bs%2Cr%7D%20%28%5Coverline%7B%5Cmathbf%7Be%7D_o%7D-%5Cmathbf%7Be%7D_o%29%20%26%5Clabel%7Beq%3Aapprox%3Aadd%7D%5Cnonumber%5C%5C%20%26%3D%20%5Cmathbf%7Bz%7D_%7Bs%2Cr%7D%20%28%281-%5Cvarphi%29%20%28H_o%20&plus;%20%5Cvarphi%20%281-%5Cvarphi%29%20%5Cmathbf%7Bz%7D_%7Bs%27%2Cr%27%7D%5E%5Cintercal%20%5Cmathbf%7Bz%7D_%7Bs%27%2Cr%27%7D%29%5E%7B-1%7D%20%5Cmathbf%7Bz%7D_%7Bs%27%2Cr%27%7D%5E%5Cintercal%20%29.%26%5Cnonumber%20%5Cend%7Balign%7D)

### Continuous Optimization for Search

![Branching](/images/autoencoder.png)

<p align="justify">
Using the approximations provided in the previous section, we can use brute force enumeration to find the adversary〈s′,r′,o. This approach is feasible when removing an observed triple since the search space of such modifications is usually small; it is the number of observed facts that share the object with the target. On the other hand, finding the most influential unobserved facts to add requires search over a much larger space of all possible unobserved facts (that share the object). Instead, we identify the most influential unobserved fact〈s′,r′,o〉by using a gradient-based algorithm on vector Z(s',r') in the embedding space. After identifying the optimal Z(s′,r′), we still need to generate the pair (s′,r′). We design a network, shown in above figure, that maps the vector Z(s′,r′) to the entity-relation space, i.e., translating it into (s′,r′).
</p>

# Experiments

* * *

<p align="justify">
We evaluate CRIAGE by, 1) comparing CRIAGE estimate with the actual effect of the attacks, 2) studying the effect of adversarial attacks on evaluation metrics, 3) exploring its application to the interpretability of KG representations, and 4) detecting incorrect triples.
</p>

### Influence Function vs CRIAGE

![Branching](/images/IF.pdf)

<p align="justify">
We show the time to compute a single adversary by IF (influence function) compared to CRIAGE, as we steadily grow the number of entities (randomly chosen subgraphs), averaged over 10 random triples. As it shows, CRIAGE is mostly unaffected by the number of entities while IF increases quadratically. Considering that real-world KGs have tens of thousands of times more entities, making IF unfeasible for them.
</p>

### Robustness of Link Prediction Models

![Branching](/images/robustness.png)

<p align="justify">
Now we evaluate the effectiveness of CRIAGE to successfully attack link prediction by adding false facts. The goal here is to identify the attacks for triples in the test data, and measuring their effect on MRR and Hits@ metrics (ranking evaluations) after conducting the attack and retraining the model. Since this is the first work on adversarial attacks for link prediction, we introduce several baselines to compare against our method. For finding the adversarial fact to add for the target triple〈s,r,o〉,we consider two baselines: 1) choosing a random fake fact〈s′,r′,o〉(Random Attack); 2) finding (s′,r′) by first calculating f(e_s,e_r) and then feeding −f(e_s,e_r) to the decoder of the inverter function (Opposite Attack).  In addition to CRIAGE-Add, we introduce two other alternatives of our method:  (1) CRIAGE-FT, that uses CRIAGE to increase the score of fake fact over a test triple, i.e., we find the fake fact the model ranks second after the test triple, and identify the adversary for them, and (2) CRIAGE-Best that selects between CRIAGE-Add and CRIAGE-FT attacks based on which has a higher estimated change in score. 
 
All-Test: The result of the attack on all test factsas targets is provided in the Table 4. CRIAGE-Add outperforms the baselines, demonstrating its ability to effectively attack the KG representations. It seems DistMult is more robust against random attacks, while ConvE is more robust against designed attacks. CRIAGE-FT is more effective than CRIAGE-Add since changing the score of a fake fact is easier than of actual facts; there is no existing evidence to support fake facts. We also see that YAGO3-10 models are more robust than those for WN18. 

Uncertain-Test: To better understand the effect of attacks,  we consider a subset of test triples that 1) the model predicts correctly, 2) difference between their scores and the negative sample with the highest score is minimum. This “Uncertain-Test” subset contains 100 triples from each of the original test sets, and we provide results of attacks on this data in Table 4. The attacks are much more effective in this scenario, causing a considerable drop in the metrics. Further, in addition to CRIAGE significantly outperforming other baselines, they indicate that ConvE’s confidence is much more robust.
</p>

###  Interpretability of Models

![Octocat](/images/int.png)

<p align="justify">
To be able to understand and interpret why a link is predicted using the opaque, dense embeddings, we need to find out which part of the graph was most influential on the prediction. To provide such explanations for each predictions, we identify the most influential fact using CRIAGE-Remove. Instead of focusing on individual predictions, we aggregate the explanations over the whole dataset for each relation using a simple rule extraction technique: we find simple patterns on subgraphs that surround the target triple and the removed fact from CRIAGE-Remove, and appear more than 90% of the time. We only focus on extracting length-2 horn rules, i.e., R1(a,c)∧R2(c,b)⇒R(a,b), where R(a,b) is the target and R2(c,b) is the removed fact. Table 5 shows extracted YAGO3-10 rules that are common to both models, and ones that are not. The rules show several interesting inferences, such that "hasChildis" often inferred via married parents, and "isLocatedIn" via transitivity. There are several differences in how the models reason as well; DistMult often uses the "hasCapitalas" an intermediate step for "isLocatedIn", while ConvE incorrectly uses "isNeighbor".  We also compare against rules extracted in DistMult paper for YAGO3-10 that utilizes the structure of DistMult: they require domain knowledge on types and cannot be applied to ConvE. Interestingly, the extracted rules contain all the rules provided by CRIAGE, demonstrating that CRIAGE can be used to accurately interpret models, including ones that are not interpretable, such as ConvE. These are preliminary steps toward interpretability of link prediction models, and we leave more analysis of interpretability to future work
</p>
