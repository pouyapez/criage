---
layout: default
---
<p align="justify">
Representing entities and relations in an embedding space is a well-studied approach for machine learning on relational data. Existing approaches, however, primarily focus on improving accuracy and overlook other aspects such as robustness and interpretability. In this paper, we propose adversarial modifications for link prediction models: identifying the fact to add into or remove from the knowledge graph that changes the prediction for a target fact after the model is retrained. Using these single modifications of the graph, we identify the most influential fact for a predicted link and evaluate the sensitivity of the model to the addition of fake facts. We introduce an efficient approach to estimate the effect of such modifications by approximating the change in the embeddings when the knowledge graph changes. To avoid the combinatorial search over all possible facts, we train a network to decode embeddings to their corresponding graph components, allowing the use of gradient-based optimization to identify the adversarial modification. We use these techniques to evaluate the robustness of link prediction models (by measuring sensitivity to additional facts), study interpretability through the facts most responsible for predictions (by identifying the most influential neighbors), and detect incorrect facts in the knowledge base.
</p>

# Criage

* * *

![Branching](https://github.com/pouyapez/criage/blob/gh-pages/images/criage.png)

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

![\Large \begin{align}
   \operatorname*{argmax}_{(s', r')} \Delta_{(s',r')}(s,r,o)  %\in \nu} 
\end{align}](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)

# Efficiently Identifying the Modification

* * *

In this section, we propose algorithms to addressmentioned challenges by (1) approximating the ef-fect of changing the graph on a target prediction,and (2) using continuous optimization for the dis-crete search over potential modifications.

### First-order Approximation of Influence


### Continuous Optimization for Search


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
