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
For a target triple <s, r, o>, we constrain the possible triples that we can remove (or inject) to be in the form of <s', r', o> i.e s' and r' may be different from the target, but the object is not. 
  
</p>

### Removing a fact (CRIAGE-Remove)

<p align="justify">
For explaining a target prediction, we are interested in identifying the observed fact that has the most influence (according to the model) on the prediction.
We define influence of an observed fact on the prediction as the change in the prediction score if the observed fact was not present when the embeddings were learned. Previous work have used this concept of influence similarly for several different tasks.

Formally, for the target triple <s,r,o> and observed graph G, we want to identify a neighboring triple <s',r',o> in G such that the score \psi(s,r,o) when trained on G and the score \overline{\psi}(s,r,o) when trained on G-triple{s',r',o} are maximally different, i.e.
</p>

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Coperatorname*%7Bargmax%7D_%7B%28s%27%2C%20r%27%29%5Cin%20%5Ctext%7BNei%7D%28o%29%7D%20%5CDelta_%7B%28s%27%2Cr%27%29%7D%28s%2Cr%2Co%29%5Cnonumber%20%5Cend%7Balign%7D)

### Adding a new fact (CRIAGE-Add)

<p align="justify">
We are also interested in investigating the robustness of models, i.e., how sensitive are the predictions to small additions to the knowledge graph. Specifically, for a target prediction \triple{s,r,o}, we are interested in identifying a single fake fact \triple{s',r',o} that, when added to the knowledge graph G, changes the prediction score \psi(s,r,o) the most.
Using \overline{\psi}(s,r,o) as the score after training on G\cup\{\triple{s',r',o}, we define the adversary as:
</p>

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%5Coperatorname*%7Bargmax%7D_%7B%28s%27%2C%20r%27%29%7D%20%5CDelta_%7B%28s%27%2Cr%27%29%7D%28s%2Cr%2Co%29%5Cnonumber%20%5Cend%7Balign%7D)

# Efficiently Identifying the Modification

* * *

In this section, we propose algorithms to addressmentioned challenges by (1) approximating the ef-fect of changing the graph on a target prediction,and (2) using continuous optimization for the dis-crete search over potential modifications.

### First-order Approximation of Influence

<p align="justify">
We first study the addition of a fact to the graph,and  then  extend  it  to  cover  removal  as  well.To  capture  the  effect  of  an  adversarial  modifi-cation  on  the  score  of  a  target  triple,  we  needto  study  the  effect  of  the  change  on  the  vectorrepresentations  of  the  target  triple.   We  usees,er,  andeoto  denote  the  embeddings  ofs,r,oat  the  solution  ofargminL(G),  and  when  con-sidering  the  adversarial  triple〈s′,r′,o〉,  we  usees,er, andeofor the new embeddings ofs,r,o,respectively.Thuses,er,eois  a  solution  toargminL(G∪ {〈s′,r′,o〉}),  which  can  also  bewritten asargminL(G) +L(〈s′,r′,o〉). Similarly,f(es,er)changes tof(es,er)after retraining.Since we only consider adversaries in the formof〈s′,r′,o〉, we only consider the effect of the at-tack oneoand neglect its effect onesander. Thisassumption is reasonable since the adversary is con-nected withoand directly affects its embeddingwhen added, but it will only have a secondary, neg-ligible effect onesander, in comparison to its
 effect oneo. Further, calculating the effect of theattack onesanderrequires a third order derivativeof the loss, which is not practical (O(n3)in thenumber of parameters). In other words, we assumethates'esander'er. As a result, to calculatethe effect of the attack,ψ(s,r,o)−ψ(s,r,o), weneed to computeeo−eo, followed by:
 </p>

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%20%26%5Coverline%7B%5Cpsi%7D%7B%28s%2Cr%2Co%29%7D-%5Cpsi%28s%2C%20r%2C%20o%29%3D%20%5Cmathbf%7Bz%7D_%7Bs%2Cr%7D%20%28%5Coverline%7B%5Cmathbf%7Be%7D_o%7D-%5Cmathbf%7Be%7D_o%29%20%26%5Clabel%7Beq%3Aapprox%3Aadd%7D%5Cnonumber%5C%5C%20%26%3D%20%5Cmathbf%7Bz%7D_%7Bs%2Cr%7D%20%28%281-%5Cvarphi%29%20%28H_o%20&plus;%20%5Cvarphi%20%281-%5Cvarphi%29%20%5Cmathbf%7Bz%7D_%7Bs%27%2Cr%27%7D%5E%5Cintercal%20%5Cmathbf%7Bz%7D_%7Bs%27%2Cr%27%7D%29%5E%7B-1%7D%20%5Cmathbf%7Bz%7D_%7Bs%27%2Cr%27%7D%5E%5Cintercal%20%29.%26%5Cnonumber%20%5Cend%7Balign%7D)

### Continuous Optimization for Search

![Branching](/images/autoencoder.png)

<p align="justify">
Using the approximations provided in the previoussection, Eq.(7)and(4.1), we can use brute forceenumeration to find the adversary〈s′,r′,o〉. Thisapproach is feasible when removing an observedtriple since the search space of such modificationsis usually small; it is the number of observed factsthat share the object with the target. On the otherhand, finding the most influential unobserved factsesrerf(es,er)(Fixed)zs,rInverterNetwork ̃s ̃es ̃r ̃erFigure 2:Inverter NetworkThe architecture of our in-verter function that translatezs,rto its respective( ̃s, ̃r).The encoder component is fixed to be the encoder net-work of DistMult and ConvE respectively.to add requires search over a much larger space ofall possible unobserved facts (that share the object).Instead, we identify the most influential unobservedfact〈s′,r′,o〉by using a gradient-based algorithmon vectorzs′,r′in the embedding space (reminder,zs′,r′=f(e′s,e′r)), solving the following continu-ous optimization problem inRd:
<\p>

# Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
