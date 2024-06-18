MAWT: Multi-Attention-Weight Transformers
by
Deniz Askin
A thesis submitted to the Faculty of Graduate and Postdoctoral Affairs in partial fulfillment of
the requirements for the degree of
PhD of Arts and Social Sciences
in
Cognitive Science
Carleton University
Ottawa, Ontario
© 2023, Deniz Askin 
ii
Abstract
Transformers are machine learning models designed to learn and predict sequential and
structured data, which are crucial to tasks such as neural machine translation and semantic
parsing. They have become state-of-the-art engines for both of these tasks and much research in
natural language processing is devoted to increasing their performance by introducing
modifications to their architectures.
In light of this trend, this thesis introduces a new Transformer architecture called
MAWT: Multi-Attention-Weight Transformers in an attempt to increase the accuracy and variety
of the acceptable predictions of a Transformer. It attempts to achieve this by training multiple
weights per each Transformer attention head, which then are used to test the accuracy of the
engine. This creates a new architecture under which the system produces a candidate set of
outputs (instead of a singly output), along with a method for selecting from the candidate set. My
proposal rests on the assumption -- motivated by statistical considerations -- that having a
candidate set increases the probability of finding an exact match within the set.
Upon testing, I observed that my system outperforms the regular transformer on 5/6
benchmark neural machine translation and semantic parsing datasets, where engine performance
is measured by exact match accuracy. Exact match accuracy demands syntactic identity between
the output and the target. In order to investigate how well my new architecture generalizes to
measures of semantic equivalence that don't also demand syntactic identity, I also recorded the
BLEU scores on these datasets. The BLEU score is a measure of performance based on n-grams
rather than exact symbolic match (i.e., how many contiguous sequence of n-many strings from
the predicted output match the desired output). The results I report on the BLEU scores are more 
iii
mixed, raising important questions that I highlight about the role of syntax in measures of
semantic equivalence.
Keywords: Transformers, neural machine translation, semantic parsing
iv
Table of Contents
Abstract............................................................................................................................... ii
 Acknowledgements............................................................................................................ vi
 List of Tables..................................................................................................................... vii
 List of Illustrations............................................................................................................. ix
Chapter 1: Introduction to Neural Machine Translation and Semantic Parsing ................. 1
1.1 Brief Preview ............................................................................................................ 1
1.2 Neural Machine Translation...................................................................................... 2
1.3 A Primer on Neural Networks................................................................................... 3
1.4 Back to Neural Machine Translation ........................................................................ 7
1.5 Semantic Parsing....................................................................................................... 7
1.6 Logical Forms..........................................................................................................11
1.7 Neural Semantic Parsing......................................................................................... 13
1.8 Different Metrics for Performance.......................................................................... 14
Chapter 2: Introduction to Transformers........................................................................... 19
2.1 Encoder Self-Attention ........................................................................................... 27
2.2 Decoder Self-Attention ........................................................................................... 29
2.3 Encoder-Decoder Self-Attention............................................................................. 30
2.4 Multi-Head Attention .............................................................................................. 30
2.5 Testing..................................................................................................................... 31
v
Chapter 3: MAWT: Multi-Attention-Weight Transformers.............................................. 33
3.1 Methodology ........................................................................................................... 35
3.2 Datasets............................................................................................................. 42
3.3 Results............................................................................................................... 43
3.3.1 Exact Match Accuracies for Sentences............................................................ 43
3.3.2 BLEU Accuracies for Sentences...................................................................... 46
3.3.3 Exact Match Accuracies for Word/Token Translations.................................... 49
3.3.4 Dataset Dependence of the Performance for Neural Machine Translation...... 54
Chapter 4: Concluding Remarks....................................................................................... 58
4.1 Summary ................................................................................................................. 58
4.2 Limitations.............................................................................................................. 59
4.3 Future Work ............................................................................................................ 60
4 References................................................................................................................. 65
vi
Acknowledgements
I would like to thank my thesis supervisor and mentor Professor Raj Singh for his
invaluable help and guidance through this research. I would also like to thank my family and
friends for their continuous support during this extremely informative time.
I would also like to thank Dr. Peter Dobias from the Department of National Defence for
his support, mentorship and advocating diligently for me. I would like to thank Payment
Evolution for their continual support and collaboration.
Finally, I would like to thank all of my committee members for their careful reading of
my thesis and for their very helpful suggestions: Dr. Ida Toivonen, Dr. Diana Inkpen, Dr. Alan
Bale and Dr. Karen Jesney.
vii
List of Tables
Table 1. State of the Art Results of Transformer Based Models on Benchmark Neural Machine
Translation Datasets...................................................................................................................... 19
Table 2. State of the Art Results of Transformer Based Models on Benchmark Semantic Parsing
Datasets......................................................................................................................................... 20
Table 3. Exact Match Accuracies for Regular versus Our Transformer for Semantic Parsing
Datasets......................................................................................................................................... 43
Table 4. Exact Match Accuracies for Regular versus Our Transformer for Neural Machine
Translation Datasets...................................................................................................................... 44
Table 5. Average BLEU scores for Regular versus Our Transformer for Semantic Parsing
Datasets......................................................................................................................................... 46
Table 6. Average BLEU scores for Regular versus Our Transformer for Neural Machine
Translation Datasets...................................................................................................................... 46
Table 7. Predictions and correct output for sentences from the English-German dataset............ 48
Table 8. Exact Match Accuracies for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets. ...................................................................................................... 49
Table 9. Average BLEU Scores for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets. ...................................................................................................... 50
Table 10. Exact Match Accuracies for Regular versus Our Algorithm versus Our New Hybrid
Algorithm...................................................................................................................................... 52
Table 11. Exact Match Accuracies for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets. ...................................................................................................... 52
viii
Table 12. Average BLEU Scores for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets. ...................................................................................................... 53
Table 13. Complexity Measures................................................................................................... 56
ix
List of Illustrations
Figure 1. A four-node neural network............................................................................................. 4
Figure 2. A recurrent neural network .............................................................................................. 6
Figure 3. Example of BLEU scoring. ........................................................................................... 21
Figure 4. The Encoder and Decoder layers of a Transformer....................................................... 21
Figure 5. The positional embedding equation (for even indices) ................................................. 25
Figure 6. The positional embedding equation (for odd indices).. ................................................. 26
Figure 7. Using column vectors of constants (e.g. 0,1,2,3) as position encodings....................... 26
Figure 8. The layers of the Encoder and the Decoder................................................................... 27
Figure 9. Sequential predictions of a Transformer during testing. ............................................... 32
Chapter 1: Introduction to Neural Machine Translation and Semantic Parsing
1.1 Brief Preview
In this thesis, I propose a Transformer architecture which has multiple attention weights
per each attention head instead of one (the definition of the technical terms will follow in
Sections 3 and 4). I do this for two reasons:
1. To have a Transformer that can produce multiple candidate outputs instead of just one.
2. To observe if I can use a function to select from multiple candidate outputs to construct a
new attention weight that can produce a translation/parse that increases the performance
of the engine with respect to exact match accuracy and BLEU score as compared to the
regular Transformer.
In this work, I will be only reporting on my findings with respect to 2, and I hope in future
work to perform the additional analyses required for 1, some of which will likely involve
hand-checking. This will be left for future work.
Therefore, for this thesis, my aim is to have multiple trained weights to pick from in hope
of getting higher performance than a regular Transformer. I will be reporting on my findings
about the performance with respect to exact match accuracy and BLEU score.
Exact match is a binary score: 1 if every string in the prediction matches every string in
the true output, 0 otherwise (No alternative translations/parses are allowed).
BLEU score is a score between 0 and 1: How close is the output to the true output. How
many contiguous strings of the prediction match contiguous string of the true output.
My reasoning behind constructing an engine that can choose from different outputs is purely
statistical: The more you roll a dice, the higher the likelihood of getting any side as an outcome. 
2
In other words, by increasing the number of possible different ways that the words of the
attention layers could correlate to each other, we are allowing the engine to try more possible
ways of getting to the right answer than the regular Transformer does. As a result, we are
increasing the likelihood of receiving the right translation/parse as an output (the measurement of
“right” here being exact match accuracy. Findings on BLEU scores will also be reported.
Neural machine translation refers to the task of translating a natural language sentence to
another natural language sentence using neural network-based code (ex. translating from French
to English). Semantic parsing refers to the task of translating a natural language sentence to a
computer-readable code that captures its meaning. The reason I study these tasks together is
because they are both string prediction tasks for which the input and output are a sequence of
strings, hence they are similar challenges1
. Therefore, our research will develop a formalism that
will be used for both tasks. Therefore, before introducing our research, we will first explore the
detailed definition of both tasks in the following chapters. I will first explain what Neural
Machine Translation and Semantic Parsing are and provide a succinct history of their
development, and then will give concrete examples of how Transformers are used to achieve
both tasks.
1.2 Neural Machine Translation
Neural machine translation is the task of using a neural network-based system to translate a
natural language to another. One major issue that needed to be addressed in developing neural
network-based translation engines was the sequential nature of language. Words and their
1 Of course semantic parsing aims to capture the proper structure inherent in a string. However, the models
consideration here only work with strings, and the task we face with such models is to "mimic" structure with strings
alone.
3
meaning as well as semantic role in natural languages are sometimes position specific. For
example, the meaning of the sentence “The man looked at the screen” is different than the
sentence “The screen looked at the man” because the order of the words man and dog are
different in the two sentences. An engine that is not able to treat words in a position-specific way
cannot register this difference.
Neural machine translation started with the use of recurrent neural networks in 2014
(Bahdanau, D., Cho, K., & Bengio, Y., 2014), which are neural networks that form a loop
between their inputs and outputs during training. This allows them to transfer knowledge from
previous learning stages to the next learning stage, hence sequential learning. Before going
further, I will briefly give a primer on neural networks.
1.3 A Primer on Neural Networks
Neural networks are computational structures that have proven extremely useful in the
tasks of classification and regression. Classification refers to the task of placing data into a given
discrete category; e.g. given the following data on a patient, which disease do they have.
Regression refers to the task of placing data within a continuous range: e.g. given the following
data on weather, what is the likelihood of it raining tomorrow.
Neural networks are composed of nodes and connections between them called weights
(see Figure 1). They are used for two procedures: training and testing. During training, one
randomly generates numbers between zero to one (to represent probabilities) and assigns them to
each of the weights. Then, one uses a dataset which contains a number for each input node and
passes these numbers to the following layers. The passing is achieved by each layer accepting the
sum of the product of the value of the nodes they are connected to and the weight that connects 
4
them as input (detailed explanation below). This sum is called a weighted sum, and the passing of
the values is called feedforwarding.
Figure 1. A four-node neural network. The dots in each layer are nodes and the lines between
them are weights2
. The Figure is from Matt Mazur, (https://mattmazur.com/2015/03/17/a-stepby-step-backpropagation-example/)
Let us make this notion step of feedforwarding more concrete with an example. In Figure
1, i1 and i2 are the first and second input nodes, and h1 and h2 are the first and second nodes of
the next layer. h1 is connected to i1 via weight w1, and to i2 via weight w3. The input of h1 is the
sum of the product of the weights and input nodes that it is connected to. In this example, this
would equal:
�1 ∙ �1 + �2 ∙ �2 = .15 ∙ .05 + .25 ∙ .10
2 The example outlined here is a fully-connected network, which means that each node of a layer is
connected to every node of the following layer. A network need not be fully-connected (meaning not all of the nodes
of a previous layer have to be connected to the nodes of the following layer via weights), but since Transformers use
fully-connected networks, we use them as an example. 
5
where ∙ represents a product. This is the weighted sum and this is how the input of all nodes
(except the input nodes, which have fixed values provided by the dataset) are calculated.
For each node that receive a weighted sum as input, their inputs are then fed into a
function which ensures that they return a value between zero and one; namely, a probability. This
measure serves as the output of that node, which then is feedforwarded to the next layers using
the same methodology.
During training, once feedforwarding is applied until the final layers, a process called
backpropagation is applied which changes the values of each weight in the neural network such
that the final values of the output nodes are closer to their optimal value. If their optimal value is
provided in the dataset, this training is called supervised. If it is not, then it is unsupervised.
Since all the datasets we will be dealing with in this prospectus are supervised, I will not explain
unsupervised learning.
The process of training is applied multiple times to get to the optimal weight values. Each
training run is called an epoch.
Upon completion of training, testing is applied by using a test set which must be different
from the train set. The testing stage contains only feedforwarding and assessment of the
performance of the engine.
What makes us regard weights us probabilities is the fact that, during testing, once data is
feedforwarded to the system, the engine will pay the most attention to the output nodes with the
highest-valued weights. This calculation is very much like a probabilistic calculation which
addresses the question: given an input, which of the possible outputs is the most likely outcome?
6
These processes outlined are constant across every neural-network model, however, how
the connections (weights) between nodes are structured can differ greatly. This brings us back to
recurrent neural networks we mentioned in the previous section.
Unlike the neural network displayed in Figure 1, in recurrent neural networks, the
weights of nodes of the same layer are functions of the weights of the nodes preceding them (see
Figure 2).
Figure 2. A recurrent neural network. U’s represent the values of the nodes representing the
words and W’s are the weights. Notice that the weight of the last word movie is dependent on the
weights of the previous words. The figure is from Towards Data Science,
(https://towardsdatascience.com/recurrent-neural-networks-rnn-explained-the-eli5-way3956887e8b75)
Notice that in Figure 1, the value of i1 has no influence on the weights of i2, since they
are not even connected. None of the nodes that are in the same layer are connected via weights.
In recurrent neural networks however, the input of nodes of the same layer are impacted by the
values and weights of the nodes preceding them. This means that information from previous
nodes are carried onto the following nodes. Therefore, recurrent networks have an implicit
memory whereby the weights of the later nodes of a layer are informed by the weights and 
7
values of the previous nodes. This comes with a cost for long inputs which we will explore in the
following section.
1.4 Back to Neural Machine Translation
As mentioned before, neural machine translation initially depended on recurrent neural
networks. Although this approach proved useful in getting state of the art results for neural
machine translation, the recurrence required in the training of the neural networks proved
problematic for long sentences, since the time required to process recurrent neural network units
increases significantly with respect to the number of tokens of the sentence they learn. This
problem is explained in (Cheng, J., Dong, L. & Lapata, M., 2016): “RNNs treat each sentence as
a sequence of words and recursively compose each word with its previous memory, until the
meaning of the whole sentence has been derived. In practice, however, sequence-level networks
are met with…challenges...sufficiently large memory capacity is required to store past
information. As a result, the network generalizes poorly to long sequences while wasting
memory on shorter ones.”
The challenge that this problem poses was addressed by Transformers which were
introduced in 2017. Before introducing Transformers and how they address this challenge, I will
explain semantic parsing and we will see that a similar challenge exists there as well.
1.5 Semantic Parsing
Semantic parsing is the task of automating methods for translating natural language
sentences (strings of words) to a structured language that captures the meaning of the sentence,
which then can be processed by a computer. This translation allows one, for example, to query 
8
databases and command virtual assistants using natural language. It also captures computational
aspects of how meanings in natural language may be derived as well as acquired from data.
Semantic parsing therefore sits at the intersection of theoretical linguistics (natural language syntax
and semantics) as well as artificial intelligence (machine learning and natural language parsing).
My thesis aims to improve on state-of-the-art semantic parsing, and to use my investigations to
shed computational insights into natural language as well as to incorporate linguistic insights into
current NLP models (especially neural network models). The computational insights into natural
language would be achieved by observing our system endowed with the ability to produce multiple
candidate parses can in fact lead to a higher exact match accuracy and BLEU score. Incorporation
of linguistic insights into current NLP models would be achieved by the linguistic insight of the
fact that a sentence can be parsed in multiple ways influencing our work to create an engine that
is capable of producing multiple candidate parses and selecting from them.
Traditionally, natural language semantics aims to give a compositional account of how the
meaning of a sentence is derived (e.g., Gamut, 1991; Carpenter, 1997; Heim & Kratzer, 1998;
Kearns, 2000). Given meanings for lexical items, as well as the form of a given sentence, the
meaning of the whole is derived using a small set of basic composition principles that apply to
sub-parts of the sentence. For example, consider a basic sentence like Arkansas borders Texas,
and suppose we assume a Combinatory Categorial Grammar (CCG) framework (e.g., Steedman,
2000) for linguistic analysis. This is commonly assumed in the semantic parsing literature, so I
will use it as my starting point although I will eventually present a different semantic
representation. In the CCG framework, there are no real phrase-structure grammar rules. The
lexicon stores all the relevant syntactic and semantic information, and there are functional
application and composition principles that apply to adjacent words in the sentence that do 
9
syntactic and semantic work. I will say more about my assumptions about syntax and semantics a
bit later in the document, but let me illustrate here with a somewhat simplified presentation that
glosses over some details.
Suppose that Arkansas is listed in the lexicon as a noun phrase (NP) and as semantically
denoting the element Arkansas (say, in some database we might have an object a which we
understand as `Arkansas’). Suppose something similar for Texas (syntactically a noun phrase,
semantically denoting t). We want the sentence as a whole to syntactically be a sentence (S), and
to denote the logical form borders(a,t), which in predicate logic would be understood as being
true if and only if Arkansas borders Texas. In a database, for example, we might have an
exhaustive listing of all states and what the bordering relation is between them, and we would
produce `true’ if Arkansas and Texas are in the relation and false otherwise. We get to the
desired syntactic category and semantic logical form by assuming that borders denotes functions
in both its syntax and semantics, and that these functions apply to the other elements until an S is
derived in the syntax and borders(a,t) is derived in the semantics. Syntactically, suppose the
borders is a transitive verb with category (S\NP)/NP. This means that it first looks for an NP to
the right, yielding a syntactic category (S\NP); this then looks for an NP to its left, and yields
category S (see the section on categorial grammar for more details). In our case, the object Texas
would be the first argument, yielding the string borders Texas with category (S\NP); then this
phrase would take Arkansas as its next argument, yielding the string Arkansas borders Texas
with category S. Semantically, we assume (as standard) that borders is represented as a function
in the lambda-calculus: �x �y. borders(y,x); this takes an argument for x, here t, yielding a new
function �y. borders(y,t); this then applies to a, yielding borders(a,t).
10
Given the large amount of memory available in the human mind and modern computer
systems, one might wonder why we can’t simply store sentences with their logical form meanings
to save on all the computational work. That work is required because no amount of memory is
enough to account for the fact that there are infinitely many sentences that have meaning, and for
the fact that we can understand sentences that we’ve never before heard. Hence, it is common to
assume that there is a finite set of primitives, along with projection principles that allow you to
combine those primitives to create (in principle) an unbounded number of sentences together with
their meanings. Formal syntax and semantics aims to discover those primitives and principles.
Semantic parsing adds two computational layers to this: (i) Parsing: It aims to provide (ideally
efficient) parsing algorithms for finding the right syntactic derivation and logical form for a
sentence (or the set of right derivations and logical forms in case of ambiguity), and (ii) Learning:
It aims to learn from some finite data set what the primitives and projection principles are. For
example, given some initial lexicon, as well as some annotated examples like the one above that
pair a sentence with its logical form or even its derivation, a semantic parser would aim to discover
generalizations such that it could apply to (efficiently) parse any sentence even (and in particular)
ones it hasn’t yet encountered. Our focus here is on the semantics and their interfaces with
databases, so we will mostly be talking about logical forms, The goal is to give our machine
examples of sentences with their logical forms, and to have it discover the translation between
sentence and logical form such that it can then scale and do this for new sentences without a person
having to sit down and write down all the grammatical rules for the machine. The semantic parsing
challenge raises many questions: what is the right or optimal semantic representation? Does it
depend on the task? What is a good learning mechanism? What are the tradeoffs between
complexity, accuracy, generalizability, context-dependence, and faithfulness to human 
11
competence that we need to make in formulating our model? These questions get sharpened by
trying to formulate a learning model that can acquire a semantic parser that can find the right
logical forms on various standard datasets.
1.6 Logical Forms
It is commonly assumed that the language to be processed by a computer is represented by
logical forms, which are logical expressions containing at least the following: logical operators,
predicates, relations and constants. Because the semantics and algorithmic properties of such
logics are formalized and well-understood from mathematical logic, the translation of natural
language into such a logical syntax simultaneously captures the meaning and facilitates
computational reasoning and analysis. Although there are various parsing languages used by
computers, all of them utilize at least these four elements to represent information. Here I briefly
outline two common meaning representation formulations: Lambda calculus and Prolog.
• Lambda Calculus
One very common representation of logical forms uses the lambda calculus (e.g., Gamut, 1991;
Carpenter, 1997; Heim & Kratzer, 1998; Kearns, 2000, among others).
“The lambda calculus is, at heart, a simple notation for functions and application. The main
idea...is forming functions by abstraction.” (Alama, J. & Johannes K., 2021). Abstraction (forming
of functions) is achieved by binding variables using the lambda-operator.
Let us look at a concrete example. Suppose, using lambda calculus, we would like to bind a
variable x which will be the argument of a function, say the State function. This function will
accept the variable x as an argument, and will return true if and only if a constant given as the input
of x is in fact a state, and will return false otherwise. In lambda calculus, the resulting logical form
will be:
12
�x. State(x)
A computer reading this abstraction can interpret it as the command: Retrieve all the constants in
your database which are states.
Lambda calculus commonly uses two binary operators (like other logical systems, it can
choose to use all sorts of operators that suit the purposes the system is used for): And (&) and Or
(V), and a unary operator: Negation (¬). Let us consider an example that uses one of these
operators. Take the following question:
Which states border Texas?
The part of the question before the word border are identical to the above example: �x.
State(x). The rest of the question confines the search space to only those states that border Texas.
To achieve this in a logical parse, we need to conjunct State(x) with a function that gives true if a
constant borders Texas and false otherwise. This is achieved by the following parse:
�x. State(x)& border(x, Texas)
which a computer reading would interpret it as the command: Retrieve all the constants in your
database which are states AND which border Texas.
• Prolog
Another common representation language of logical forms is Prolog, which is a programming
language used for building, querying and reasoning with respect to datasets. What makes Prolog
appealing is that it comes with a reasoning engine, capable of performing deduction with if-else
statements.
Another is that it does not require the quantification of variables, which comes in handy when
one has many variables to keep track of, since languages that require the use of quantifiers require
their presence in any formula that uses quantified variables.
13
The question we used above, Which states border Texas?, would be represented in Prolog via
the following logical form:
state(X), borders(X, texas)
where , represents the logical operator AND.
Prolog, by default, will print only one constant at a time for every query. Therefore, if we
were to pass the above query, we would get only one state that borders Texas, even though there
are more than one state that borders Texas.
In order to print all the outputs of a query, and to format them in any desired way,
programmers often define a function that takes a query as an input, and returns all of its outputs.
Let us define such a function and call it answer. Then, our logical form would read:
answer(X, state(X), borders(X, texas))
Benchmark datasets in semantic parsing like Geo880 (Zelle, J. M., & Mooney, R. J., 1996),
and Jobs640 (Mooney, R., 1999) use Prolog as the logical form representation language. They
contain pairs of queries in natural language (e.g. Which states border Texas?) and the
corresponding logical forms in Prolog (e.g. answer(X, state(X), borders(X, texas)))).
The task of any semantic parser using these datasets is to translate as many of the queries
in natural language to their correct semantic parse.
1.7 Neural Semantic Parsing
As soon as the success of recurrent neural-networks in natural language translation became
evident, their use in semantic parsing became popularized and gave state-of-the-art results on some
benchmark semantic parsing datasets (Liu & Lane, 2016; Jia & Liang, 2016; Rabinovich & Klein,
2017; Ling et al., 2016). This marked the definition of neural semantic parsing.
14
Visiting back the challenge recurrent neural-network based neural machine translators had
with long sentences due to the large memory required to retain past information, we can see that
they will face the same problem while parsing sentences, since the format of the information passed
to the engine is exactly the same: strings. In the case of neural machine translation, its solely strings
representing natural language and in semantic parsing it is both strings representing natural
language and a logical form. However, this doesn’t change the fact that a recurrent neural network
storing previous strings in memory will face the same challenge of parsing longer natural
language-logical form pairs.
1.8 Different Metrics for Performance
Automated verification of an engine’s ability to perform the tasks of neural machine
translation and semantic parsing is particularly difficult, since both of these tasks involve
transforming a sentence (a sequence of symbols) into some equivalent but different sequence of
symbols, which brings about the following natural challenge: How do you know that the
transformation is in fact equivalent? And how do you automate this equivalence detection?
For example, take natural language translation: Translating a sentence A (string of words)
from one language into a sentence B (string of words) from another language. How do you know
that a sentence has been properly translated from one language to another, since there is more
than one way of translating any sentence? For example, Pat called Kim and Kim was called by
Pat are arguably truth-conditionally equivalent ways to express the Swedish sentence Pat ringde
Kim. If presented with the passive translation, should that be marked as correct? Is the active
translation more correct? These are questions that come up when one attempts to automate the
correctness of a translation.
15
As another example, take semantic parsing: Associating a sentence A (string of words) in
natural language with a semantic representation, M(A), in some formal language that captures its
meaning (say, in the lambda calculus, or first-order logic, or prolog, or whatever). How do you
know that the lambda calculus formula associated with a sentence properly captures its truthconditions? Once again, answering this question in an automated manner is not easy since there
are multiple ways of representing semantically equivalent formulas. For example: "book(a)" and
"[lambda x. book(x)](a)" are semantically equivalent representations of a is a book.
Verification of both a successful natural language translation and semantic parsing’s
performance often require human judges, since there are no automated algorithms that can judge
if the output of a translator or parser semantically captures the meaning of what it was asked to.
As a metric of this verification, one can imagine two methods:
(i) Exact-match: demand symbolic equivalence as well as semantic equivalence,
(ii) Meaning-only equivalence: demand semantic equivalence while allowing
symbolic variation
(ii) is more challenging, because it often demands human expertise. For example, are
active and passive variants of a sentence even equivalent? Are different candidate LFs
equivalent? Note that the logical form equivalence problem has been argued to be AI-complete,
and is computationally hard in any event (Shieber, S. M., 1993).
Meaning-only equivalence is more challenging because it often demands human expertise
and has been argued to be computationally too complex to expect any general algorithmic
solution to be available. There are questions here at the cognitive level as well as at the
algorithmic level. For example, active and passive variants of a sentence are arguably truthconditionally equivalent but they may have differences that are relevant in some sense to 
16
meaning. For example, there might be contexts in which only one can be used as an answer to
the question under discussion. In response to the question who did Pat call?, for example, the
active Pat called Kim is a better answer than Kim was called by Pat. Furthermore, contextual
features (like the question under discussion) may affect focus placement, which in turn may
affect interpretation. The sentence Pat called KIM, with focus on Kim, implies or even entails
that Pat didn't call anyone else, whereas Kim was called by PAT implies or even entails that no
one else called Kim. Similarly, the disjunction Sam or Kim called Pat and Sam or Kim or both
called Pat are arguably equivalent in denoting the inclusive disjunction as their basic meaning,
but only the first can sometimes convey that the conjunction Sam and Kim called Pat is false
(and that the speaker knows this) and only the second will typically convey as a matter of course
that the speaker is ignorant about whether the conjunction is true or false. There are important
questions here about the extent to which these and other considerations are a matter of sentence
meaning or speaker meaning (in the Gricean sense; see Grice, 1989), and these in turn have
consequences for how we ought to score performance in automated tasks that in part consider
sameness of meaning as part of their metric.
There are also non-trivial algorithmic questions here. Suppose that we accept that there
can be differences in form that do not affect meaning in some relevant sense. For example,
suppose we accept that active and passive are equivalent and that p or q and p or q or both are
equivalent, and more generally that there can be multiple logical forms that differ in structure but
not in whatever aspect of meaning that happens to concern us. Is it feasible to construct
algorithms to solve the problem of detecting whether two logical forms that differ in structure
nevertheless are semantically equivalent? It has been argued that this logical form equivalence
problem is AI-complete, meaning roughly that it would require machines to be as intelligent as 
17
humans for this problem to have an automated solution (Shieber, 1993). Furthermore, the
problem is also "hard" to solve algorithmically: under various formulations of the problem, either
there is no algorithm for solving it or (if the problem is restricted in some natural way) all known
algorithms are computationally intractable in that their run time grows too fast for them to be
used in any practical way (see e.g., van Noord 1993 for important discussion).
In light of this considerations, it would seem that exact-match equivalence is the better
candidate performance metric to use for problems like the ones discussed here, even though it
requires incorporating symbolic considerations in our notion of equivalence (deeper question: are
differences in form always reflecting of some difference in meaning?). Thus, my goal in this
dissertation is primarily to improve performance on exact-match accuracy in the tasks of neural
machine translation and semantic parsing.
However, I also examine how our proposal fares on performance metrics that do not
demand symbolic equivalence for the whole predicted sentence, such as BLEU. The BLEU is a
scoring mechanism that is based on how many n-grams of the predictions and desired output match
(i.e. how many 1-words (unigrams) of the prediction match 1-words of the desired output, how
many 2-word sequences (bigrams) of the prediction match 2-word sequences of the desired output,
etc.). This relaxes the role of syntax but doesn't undo it. Instead, it replaces a requirement of
syntactic identity with syntactic proximity. With the BLEU scores, I observed more mixed results
on the performance of my algorithm, which is to be expected, given how little is understood about
how to automate the task of determining whether two symbolic representations are equivalent (and,
recall, there are arguments that this may be inherently outside the scope of automation).
Nevertheless, I hope that my work shines light on precisely where the challenges lie, and (ideally)
which subparts of the more general problem we might hope to make some progress on in the future
18
Having explained what neural machine translation and neural semantic parsing are and
the challenges they entail, I will now explain what Transformers are and how they address the
memory challenge of both tasks.
19
Chapter 2: Introduction to Transformers
Transformers (Vaswani et al., 2017) are neural-network based models that have proven
extremely useful in natural language processing related tasks. In particular with respect to neural
machine translation and semantic parsing, they give state of the art results (see Table 1 and Table
2).
Table 1. State of the Art Results of Transformer Based Models on Benchmark Neural Machine
Translation Datasets
Dataset BLEU Score %
WMT2014
(EnglishGerman)
35.14
WMT2014
(English-French)
46.4
IWSLT2014
(GermanEnglish)
38.61
WMT2016
(EnglishRomanian)
34.7
IWSLT2015
(GermanEnglish)
36.2
WMT2016
(RomanianEnglish)
40.3
Note: Source is paperswithcode.com. For details on BLEU score, refer to:
https://cloud.google.com/translate/automl/docs/evaluate#bleu
20
Table 2. State of the Art Results of Transformer Based Models on Benchmark Semantic Parsing
Datasets
Dataset Exact Match
Accuracy %
Geo880 88,2
ATIS 90.4
spider 71.9
WebQuestionsSP 74.3
WikiTableQuestions 51.8
WikiSQL 89
Note: Source is paperswithcode.com
The BLEU scores displayed in Table 1 refer to scores calculated using the Bilingual
Evaluation Understudy, an algorithm used “to judge the quality of a machine translation”
(Papineni et al., 2002), which “measures the translation’s closeness to one or more reference
human translations according to a numerical metric” (Papineni et al., 2002). Instead of requiring
an input text to be translated to a specific output text, the BLEU score allows an engine to
translate an input text in different ways and scores the outputs that are the most accurate
translations higher.
21
Figure 3. Example of BLEU scoring. The figure is from Kirti Vashee
(https://www.rws.com/blog/understanding-mt-quality-bleu-scores/)
The phrase “exact match accuracy” in Table 2 refers to the number of correctly translated
sentences, where “correct” translation refers to translating exactly to a form specified in the
supervised dataset. This definition does not accept variant phrasings/representations of a
sentence and only considers one translation per sentence. This metric is primarily used for
semantic parsing, since verification of the semantic equivalence of different machine-readable
translations (codes) is difficult to verify.
A Transformer consists of two neural-network based structures called Encoders and
Decoders. During training, the Encoder accepts an input sentence (the sentence to be translated)
and the Decoder accepts the target sentence (the sentence to be translated to).
Figure 4. The Encoder and Decoder layers of a Transformer
22
Prior to being inputted, both sentences are tokenized. This refers to the process of
splitting the words in the sentences to word vectors called word embeddings. These word
embeddings are column vectors of dimension 512x1 where the value of each entry is randomly
generated. They represent a collection of numbers to be trained to capture qualities of tokens:
position, semantic similarity to other words, likelihood of occurring before/after another token,
etc. Before explaining the different ways of splitting words to word vectors, let us first explain
the motivation behind representing words as vectors.
Since neural networks are ultimately mathematical structures, the motivation behind
representing inputs via numbers is self-evident. However, a very valid concern is how to
represent inputs (in our example, words) as numbers.
An intuitive proposal would be to simply represent each word as a unique number, say a
positive integer, and pass them as values of the input nodes. However, an immediate problem
comes to mind with such an approach; since there are many words in any given dataset, we will
eventually end up with large numbers. Although as we mentioned in Section 1.2 outputs of nodes
are calculated by inputting them to functions that return values between zeros and ones, if the
value of the inputs are high, the output value will not be numerically very different (e.g.
0.9999999 vs 0. 9999998). Therefore, the neural network will not be able to distinguish between
many words. One can see that the same problem of the smallness of the difference between the
numbers will arise if one uses fractional numbers instead of integers too.
In order to circumvent this problem, a common approach has been to represent inputs as
vectors: column matrices containing many numbers. The simplest example of such word vectors
are called one-hot vectors which are column vectors with all elements equal zero except one,
which equals one. The size of these vectors is decided by the size of the vocabulary. If the dataset 
23
contains 1,000 words, then each word would be represented as a column vector of dimension
1,000x1, and each word vector would have a unique element that equals one.
One obvious problem with this approach is that for any decent sized dataset, the
dimension of the word vectors would have to be huge, since the vocabulary would be large; and
larger the size of the word vectors, the longer it takes to compute operations on them (matrix
product, etc.). Another problem is that simply using ones as the elements of each word vector is
not informative enough for engines to learn about certain details about the word (say, its position
in the sentence or closeness in meaning to another word).
This is why the usage of words vectors of fixed dimension and multiple real-valued
elements called word embeddings have been popularized. The initial values of the entries in the
word embeddings are randomly generated, and their dimensions are decided after a process
called hyperparameter tuning whereby the engine is trained and tested on the same dataset for
different word embedding dimensions, and the dimension of the run that gives the best
performance is selected. Upon doing this, the creators of the Transformer found 512x1 to be the
optimal dimension for the embeddings.
Having explained the motivation behind the usage of word embeddings, we now go back
to the previously raised question of how Transformers split words to word vectors.
There are different ways of achieving this split. The simplest is called white space
tokenization where each word separated by a space are treated as word vectors. Another is
subword tokenization, which only treats the most frequently occurring words as word vectors and
separates all other words into constituent parts. In other words, it takes frequently occurring
words and separates all other words these high frequency words are part of into constituent parts.
For example, if the word high occurs a lot in the dataset inputted, it will be treated as a word 
24
vector. However, if the adverb highly occurs significantly less, then a subword tokenizer will
treat high and ly as two separate word vectors. Doing this informs the Transformer that the words
high and highly have the same root, since it does not treat the two as separate word vectors, but
instead highly as the word vector high followed by another word vector, the suffix ly. This is the
tokenizer we use for our thesis (both for the regular Transformer as well as our algorithm).
Once tokens, each of which are column vectors of size 512x1, are generated, they are
passed into the Encoder and the Decoder simultaneously during training. That is, tokens are not
inputted to the Encoder and the Decoder one by one, but rather all at the same time. This
property of passing all tokens of the input and target sentence into the Transformer during
training is called bidirectionality.
An important question to raise here is: if all tokens are passed simultaneously, does the
Transformer have any idea on which token precedes the other? That is, are the word vectors
generated position-sensitive?
The answer is yes, and this is due to a process called position encoding. Prior to word
vectors being inputted to the Encoder and the Decoder of the Transformer, a position encoding
are added to them to inform the Transformer of their position. A position encoding is also a
512x1 column vector whose entries are decided by the two equations displayed in Figure 6 and
7.
25
Figure 5. The positional embedding equation (for even indices). Figure is from Visual Guide to
Transformer Neural Networks (https://www.youtube.com/watch?v=dichIcUZfOw)
In Figure 5, pos is a positive integer informing the position of the token (1 for the first
token, 2 for the next, etc.). d is the dimension of the token at that position (always 512). i is
number of the matrix element (1 for the first element of the 512x1 matrix, 2 for the second, etc.).
This equation always gives different values for different embeddings, and different values for
different entries of the same embedding. The same token’s position embedding values for each
of its 512 entries will be different, just as they will be different from each the entries of all other
word vectors.
The equation in Figure 5 is used to calculate the values of only the even valued indices of
the position encoding. For the odd, the same equation, but instead, using a cosine function is
used (see Figure 6).
26
Figure 6. The positional embedding equation (for odd indices). Same source as Figure 4.
Adding these encodings to the word vectors informs the system of their position in the
sentences inputted to the Encoder and the Decoder. However, an obvious question for the reader
could be: why go through these lengths and not just use constants for the position encodings?
Why not just use a word vector with all entries equal one as the position encoding for the first
word, a word vector with all entries equal to two for the second word, etc. (see Figure 7).
Figure 7. Using column vectors of constants (e.g. 0,1,2,3) as position encodings. Same source
as Figure 4 and 5.
The answer is that the value of these position encodings grow very quickly, and adding
them to the real values of the word embeddings will greatly distort their value. Another problem
is that adding a constant value to all of the elements of the word embeddings decrease the
informativeness of each value. It is important to recall that each of the 512 values of the word
embeddings represent an input node of the Transformer neural network, therefore each value is
of significance in the final output. For these two reasons, the creators of the Transformer created
a much more innovative approach of keeping the values of the position encodings bound (since 
27
they are outputs of sine and cosine functions) and their values unique per each word embedding
element.
Upon this process is completed, the input and the target sentence are passed onto the first
layer of the Encoders and Decoders: the self-attention layer.
Figure 8. The layers of the Encoder and the Decoder.
The Self-Attention layers of the Encoder and Decoder differ slightly. Using the sentence
outlined in Figure 4 as an example, I will now explain both in detail.
2.1 Encoder Self-Attention
The Encoder Self-Attention consists of mapping each token to every other token (during
training) in a sentence to learn semantic and positional relation between the tokens. For example,
going back to our sentence in Figure 4, the word Je would most likely be followed by the word
suis instead of étudiant (since otherwise the sentence would be ungrammatical: Je étudiant).
During training, the encoder self-attention mechanism would allow the Transformer to learn
this).
How this mechanism learns about the semantic relation between the tokens is a bit more
abstract, however we know that this process of mapping proves to be extremely useful in
calculating semantic similarity between words, part of speech tagging, named entity recognition 
28
and many other natural language processing related tasks as illustrated by Encoder-only
architectures like BERT (Devlin, J., Chang, M. W., Lee, K., & Toutanova, K., 2018).
The mapping is achieved by representing the inputs via 3 column vectors called key, value
and query. For the Encoder, all three are vectors with shape (number of sentences, sentence
length, 512), meaning they are lists containing input sentences, each of which contain a number
of tokens; and each token is a column vector of dimension 512.
So, going back to our example in Figure 4, the sentence Je suis étudiant will have key, value
and query vectors of shape (1, 3, 512), where the number of sentence is 1, the number of tokens
in the sentence (je, suis and étudiant) is 3 and each token is a column vector of dimension 512.
Self-attention layer initially passes all three vectors through three different neural networks
and gets new key (k), value (v) and query (q) vectors as outputs.
� = [
��
����
�����𝑛�
] � = [
��
����
�����𝑛�
] � = [
��
����
�����𝑛�
]
Each of these words are column vectors of dimension 512.
In order to perform a mapping between them, the self-attention layer will perform the
following computation:
�����𝑥(� ∙ �
�
) ∙ �
where ∙ represents matrix multiplication, �
�
represents the key vector transposed and softmax is a
function that returns values between (0,1), i.e. probabilities. For our example sentence, this
operation can be visualized as:
�����𝑥 ([
��
����
é����𝑛�
] ∙ [�� ���� é����𝑛�]) ∙ [
��
����
é����𝑛�
]
which outputs the following equation:
29
[
(��|��) (��|����) (��|é����𝑛�)
(����|��) (����|����) (����|é����𝑛�)
(é����𝑛�|��) (é����𝑛�|����) (é����𝑛�|é����𝑛�)
] ∙ [
��
����
é����𝑛�
]
where (�|�) represents the probability of y given x. The matrix containing the (�|�) terms is
called the attention weights. The values of these conditional probabilities are initially randomly
generated and then learned during the training.
The result of this matrix product gives us a new collection of word vectors of the same
dimension as our k,t and q vectors:
(number of sentences, sentence length, 512) = (1, 3, 512)
2.2 Decoder Self-Attention
The same process of mapping is repeated by the Decoder for the target sequence but with
a slight difference: In the probability calculations for a token, only tokens preceding it are taken
into consideration. For example, for our example sentence “I am a student”, the outcome of
�����𝑥(� ∙ �
�
) ∙ �
would be
[
(�|�) 0 0 0
(𝑚|�) (𝑚|𝑚) 0 0
(�|�) (�|𝑚) (�|�) 0
(�������|�) (�������|𝑚) (�������|�) (�������|�������)
] ∙ [
�
𝑚
�
�������]
This is done because we want the engine to learn the target sequence sequentially. As we
will see, during testing, the engine will generate a translated sentence sequentially, where each
predictive step predicts what the next target token will be based on a previously predicted token
or set of tokens.
30
2.3 Encoder-Decoder Self-Attention
As can be seen in Figure 8, the Decoder has an additional self-attention layer called the
encoder-decoder self-attention. This is where the translation of input tokens to target tokens are
achieved by using a self-attention mapping. However, unlike the previous two examples we saw,
this time our k,t and v vectors represent different languages. k and v represent our input
language, whereas q represents the target language. Therefore, for our example, the mapping will
take the following form:
�����𝑥 ([
�
𝑚
�
�������] ∙ [�� ���� é����𝑛�]) ∙ [
��
����
é����𝑛�
]
which equals
[



(�|��) (�|����) (�|é����𝑛�)
(𝑚|��) (𝑚|����) (𝑚|é����𝑛�)
(�|��) (�|����) (�|é����𝑛�)
(�������|��) (�������|����) (�������|é����𝑛�)]



∙ [
��
����
é����𝑛�
]
The output of this matrix product will produce a matrix of the same dimension as that of
the input of the Decoder layer.
As can be seen in Figure 8, the output of the self-attention layer of the Encoder and the
output of the self-attention layer of the Decoder followed by the output of the encoder-decoder
attention layer are both passed through feed-forward neural networks, which further advance the
learning of the input and target tokens.
2.4 Multi-Head Attention
 An important advancement made by Transformers is the ability to train multiple attention
weights concurrently. This is called multi-head attention. This is achieved by reducing the 
31
dimensions of the k,t and q vectors to smaller dimensions. Each smaller dimensioned k,t and q’s
are used to calculated self-attention mappings exactly as outlined above. Each of these selfattention mappings are called attention heads.
 For example, for word vectors of dimension 512 if we had 8 attention heads, this would mean
that our k,t and q vectors would be of dimension 512/8=64. That would mean, for each selfattention layer, we would have 8 outputs of dimension 64 instead of 1 of dimension 512.
 After calculating �����𝑥(� ∙ �
�
) ∙ � for all the attention heads, in order to get back to
dimension 512, the outputs of the multi-attention heads are concatenated.
2.5 Testing
Unlike the training, during testing, Transformers predict a target sentence sequentially. At
each time step, the Transformer produces a new token until it finishes its prediction. It achieves
this by feeding the complete input sentence to the decoder at each time step to get a new token as
the output of the Decoder. Then, at the next time step, it feeds all the tokens previously predicted
by the Decoder as inputs to the Decoder. It continues this process until the complete sentence is
predicted.
 This sequential process of testing can be seen in Figure 9 below.
32
Figure 9. Sequential predictions of a Transformer during testing.
This innovative approach replaces the need of using recurrent neural networks to translate
data sequentially. The sequential aspect of the engine is achieved through the simple masking of
previous tokens in the attention weights of the Decoder. As a result, the Transformer can process
much longer texts than a recurrent neural network-based model without having to pay the same
computational price.
33
Chapter 3: MAWT: Multi-Attention-Weight Transformers
As we saw in the Transformer Architecture section, Transformers have attention weights
which are probability measures of informing the engine on which tokens are most related to each
other. We have one attention weight for each of the attention heads.
Similar to having multiple attention heads instead of one, I propose an architecture which
has multiple attention weights per each attention head. I do this in order to have multiple trained
weights to pick from, which, I conjecture, will give me more than one translation/parse per
sentence.
For example, revisiting the sentence outlined above:
how high is alaska ( elevation:i alaska )
our formulation will train n-many attention weights:
�����𝑥1 ([
(
����𝑡���: �
𝑙𝑠��
)
] ∙ [ℎ�� ℎ��ℎ �� 𝑙𝑠��])
�����𝑥2 ([
(
����𝑡���:�
𝑙𝑠��
)
] ∙ [ℎ�� ℎ��ℎ �� 𝑙𝑠��])
⋮
�����𝑥� ([
(
����𝑡���: �
𝑙𝑠��
)
] ∙ [ℎ�� ℎ��ℎ �� 𝑙𝑠��])
During testing, the engine can select any of the weights and perform the final matrix product to
get a prediction:
�����𝑥 ([
(
����𝑡���:�
𝑙𝑠��
)
] ∙ [ℎ�� ℎ��ℎ �� 𝑙𝑠��]) ∙ [
ℎ��
ℎ��ℎ
��
𝑙𝑠��]
34
The motivation behind my formulation is twofold:
1. Seeing if the exact match accuracy and the BLEU scores of benchmark neural machine
translation and semantic parsing datasets can be increased by using this many-weight
approach.
2. Seeing if semantically equivalent variants of correct translations and parses can be
obtained using the same engine.
As mentioned in the Introduction section, this thesis will report only on our findings with
respect to 1.
The motivation for 1. is self-evident; increasing the performance of traditional
Transformers. The reason why increasing the dimensions of the attention weights could increase
performance is that with higher number of attention weights, one will have more than one
attention weight to select from per each attention head, and each selection will yield a different
prediction. And allowing the system to make multiple predictions could lead to higher the
likelihood of getting the correct prediction as the outcome, since there will be more options to
select from.
The motivation for 2. needs to be elaborated on.
There are different ways of translating a sentence from one language to another, which is
why we have the BLEU score in the first place. Equivalently, there are different ways of parsing
the same sentence to a logical form. As mentioned in section 1.7, book(a) and [lambda x.
book(x)](a) are semantically equivalent representations of a is a book. In fact, in tests I ran on
the Geo880 dataset, I observed such semantically equivalent but structurally different
representations being produced by an engine I trained.
35
With our proposed engine, using different weights, my intention is to be able to receive
semantically equivalent but structurally different translations and parses. For neural machine
translation, this will allow us to check which of the two translations yield a higher BLEU score
and select it as the output accordingly. For semantic parsing, this will allow us to check which of
the two parses matches the desired output exactly (i.e. string for string) and select that parse as
the output.
Apart from possibly increasing performance, using different attention weights to output
semantically equivalent translations/parses which are structurally different can also allow the
fine-tuning of our engine for lexicon generation. Just because we get correct translations/parses
does not mean that the Transformer can correctly translate/parse each worded contained in the
sentence. For example, the engine can correctly translate which states border Texas? to a foreign
language, but when fed border (a word in the sentence), it can mistranslate it. The same problem
occurs in semantic parsing.
By building an engine that can output multiple semantically equivalent but structurally
different translations/parses, we will have different translations/parses of words that occur in
those sentences. And since we will have more than one representation of each word, we might be
able to leverage our engine to deduce the correct translation/parses of words contained in the
sentences of the dataset; hence generate a lexicon.
3.1 Methodology
The way in which I increase the dimensions of the attention weights is through the
following operations:
The regular transformer attention weight has dimensions:
36
(batch size, # attention heads, sequence length of q, sequence length of k)
where batch size refers to the number of sentences we feed into the transformer during
training, # attention heads refers to the number of attention heads, and sequence length of q and
sequence length of k refer to the number of tokens contained in the sentence represented by the q
vectors and the number of tokens in the sentence represented by the k vector respectively.
For example, at the self-attention layer of the encoder of the transformer we used as an
example in Figure 4, the sequence of the q and k vectors would be 3, since the sequence they
represent is Je suis étudiant and each word is treated as a token. If we considered the number of
attention heads to be 8, the dimension proposed by (Vaswani et al., 2017), then the attention
weight of the encoder layer would be dimension:
(1, 8, 3, 3)
where batch size is one in this example since we are focusing on one sentence.
The way in which this attention weight is calculated is by taking the matrix product of the
q and k vectors by doing the following:
The q vector has dimensions
(batch size, # attention heads, sequence length of q, depth)
where, as we saw on page 28: depth = �𝑚���𝑜� �� ���� �������
# 𝑡����𝑜� ℎ�𝑑�
.
Similarly, for the k vector, its dimensions is:
(batch size, # attention heads, sequence length of k, depth)
In order to calculate the attention weights, the shape of the k vector is changed by an
operation called transposing (represented by a superscripted T), which gives us the transposed k
vector: �
�which has dimensions:
(batch size, # attention heads, depth, sequence length of k)
37
Notice that the depth and sequence length of k has switched places.
Revisiting linear algebra, let’s remember that matrices are represented by the number of
their columns and rows. For example, an m by n matrix, where m represents the number of rows
and n represents the number of columns has dimensions (m,n).
Another important point to remember is that the matrix product of two matrices are made
possible by the number of columns of the one of the matrices being equal to the number of rows
of the other. For example, an m by n matrix with shape (m,n) can be multiplied with a matrix of n
number of rows and any arbitrary number of columns (which, let’s represent by k). Therefore,
the following shaped matrices can be multiplied:
(m,n)∙(n,k)
and the resultant matrix will have dimensions:
(m,k)
Having revisited this fact, now we can observe how the matrix multiplication of the q and
the �
�vectors yield the attention weight matrix with the dimensions we outlined.
Notice that the order of the last two dimensions (sequence length and depth) of both of
the vectors are reversed. The last two dimension of � is like a matrix of dimensions:
(sequence length of q, depth), whereas for �
�
, this is (depth, sequence length of k). Notice that
very much like the matrix multiplication of matrices: (m,n)∙(n,k), the dimensions of the column
of q and the dimensions of the row of �
�
agree. Therefore, their matrix multiplication will give:
(batch size, # attention heads, sequence length of q, depth)
∙
(batch size, # attention heads, depth, sequence length of k)
=
38
(batch size, # attention heads, sequence length of q, sequence length of k)
As detailed through pages 24-27, this attention weight is then fed to a softmax function
and its matrix multiplication with the value vector is what produces the predictions of the engine.
I modify this calculation of the attention weight in order to have multiple attention
weights rather than one for reasons I outlined on page 31. In order to have multiple attention
weights that can be fed into a neural network for being trained, I do the following:
First, I transpose the q and k vectors to have the following dimensions respectively:
(batch size, # attention heads, depth, sequence length of q)
(batch size, # attention heads, depth, sequence length of k)
I do this so that the first three dimensions of the vectors are the same which, as we will
see, will allow me to perform a matrix multiplication to increase the dimension of the attention
weights.
Second, I increase the dimensions of the q and k vectors by one. For the q vector I change
its shape to:
(batch size, # attention heads, depth, sequence length of q, 1)
and for the k vector, I change the shape as follows:
(batch size, # attention heads, depth, 1, sequence length of k)
Now the last two dimensions are compatible for performing a matrix product, because
they are of shape:
(sequence length of q, 1) ∙ (1, sequence length of k)
This matrix multiplication of my transposed q and k vectors produces the following
shape:
(batch size, # attention heads, depth, sequence length of q, sequence length of k)
39
I lastly transpose this vector to change the position of depth to create the following
attention weight:
(batch size, # attention heads, sequence length of q, sequence length of k, depth)
Notice that this is different by one dimension from the regular transformer attention
weight, which lacks the last depth dimension. Having the depth dimension allows me to feed it
into a neural network and train depth many attention weights which I can choose from during
training and testing.
The scoring mechanism use is so select the highest scoring value of depth, after which I
retrieve the regular transformer attention weight dimension of
(batch size, # attention heads, sequence length of q, sequence length of k)
and everything after this operation is identical to that of a regular transformer.
This can be visualized as follows:
Revisiting our example attention weight in Section 3.3, let our multiple attention weights
at the encoder-decoder self-attention layer be represented as follows:
[



(�|��) (�|����) (�|é����𝑛�)
(𝑚|��) (𝑚|����) (𝑚|é����𝑛�)
(�|��) (�|����) (�|é����𝑛�)
(�������|��) (�������|����) (�������|é����𝑛�)]



1
[



(�|��) (�|����) (�|é����𝑛�)
(𝑚|��) (𝑚|����) (𝑚|é����𝑛�)
(�|��) (�|����) (�|é����𝑛�)
(�������|��) (�������|����) (�������|é����𝑛�)]



2
⋮
[



(�|��) (�|����) (�|é����𝑛�)
(𝑚|��) (𝑚|����) (𝑚|é����𝑛�)
(�|��) (�|����) (�|é����𝑛�)
(�������|��) (�������|����) (�������|é����𝑛�)]



����ℎ
40
where the subscripts of the attention weights denote their index. In our formulation we have depthmany attention weights per attention weight instead of one.
The max function we utilize will select one number from each of the depth-many entries
of our attention weights and select that number as the entry for a new attention weight. Therefore,
upon applying a max function to this list of attention weights, our output which will be used for
predict could look like:
[



(�|��)7
(�|����)1
(�|é����𝑛�)3
(𝑚|��)8
(𝑚|����)2
(𝑚|é����𝑛�)6
(�|��)1
(�|����)4
(�|é����𝑛�)4
(�������|��)9
(�������|����)11 (�������|é����𝑛�)1
]



where the subscripted indices of the entries represented which attention weight the number was
selected from. For example, (�|��)7 would mean that for all attention weights that our algorithm
produced, it was the 7th attention weight that gave the highest score for the calculation (�|��),
which is a calculation that gives the probability of Je translating to I.
In order to observe if my algorithm increases exact match accuracy of the natural language
translation and semantic parsing outputs of the transformer, I tested hyperparameter fine-tuned
regular transformers and my algorithm on six benchmark datasets: three for semantic parsing and
three for natural language translation.
I did not apply my algorithm on the masked decoder self-attention layer, since this would
not preserve the upper triangular matrix that is used for the masked decoder self-attention weights.
And upper triangular matrix is one which has all entries above the diagonal set equal to zero. For
the French to English translation example we used previously, this attention weight would look
like:
41
[
(�|�) 0 0 0
(�|𝑚) (𝑚|𝑚) 0 0
(�|�) (𝑚|�) (�|�) 0
(�|�������) (𝑚|�������) (�|�������) (�������|�������)
]
Feeding this to a neural network would mean multiplying this matrix with a matrix with all
entries (representing weights) equal non-zero, which would give us a new attention weight that is
no longer upper triangular. This would have a corrosive effect on the auto-regressive word
predictions needed during the testing stage.
I could have chosen to use my algorithm and then apply a mask to make the resultant
attention weight upper triangular, but I chose not to do this in order to observe if I can get good
performance without adding another layer of computation (which would result in higher run time)
into the engine.
Hyperparameters were fine-tuned by observing training and validation loss during training
to ensure underfitting or overfitting didn’t occur. Results for 1 and 2 layers, and 1024 and 2048
hidden layer units were observed. The maximum number of epochs I used was 500 due to my
limited computational resources. All tests for the regular Transformer as well as my algorithm used
8 attention heads.
I trained and tested on 6 datasets (3 translation, 3 parsing) using Google Colab GPU. Since the
training and testing was very time consuming I used a maximum of 500 epochs.
3 of the datasets did not underfit using 1 layer, therefore I did not have to run more tests by
increasing the number of layers. Therefore, I ran:
3 (datasets) x 2 (one for the regular Transformer and one for my algorithm) = 6 tests for these
datasets.
3 of the dataset (two of them being largest datasets) underfit with 1 layer, so I used 2 layers.
42
• 2 (tests) x 2 (1 test for 1 layer + 1 test for 2 layers) x 3 (datasets) = 12 tests.
Therefore, in total I ran 18 tests.
Generating multiple attention weights and training them comes with an additional
runtime. In order to match the runtime of our algorithm to that of the regular Transformer, I
reduced the dimension of my word vectors from 512 to 128, which gave us a depth of 128
8
= 16
where 128 is the dimension of the word vectors and 8 is the number of attention heads we used.
3.2 Datasets
We used 6 datasets in total for our tests; 3 for semantic parsing and 3 for natural language
processing. We selected the datasets for each task such that they would have increasing dataset
sizes and different output languages. This was done in order to observe if the performance of our
Transformer depends on the size of our datasets and to observe if there would be a considerable
difference in the performance across translating to different parsing languages as well as
translating to different natural languages.
We wanted to observe if our algorithm’s performance is dependent on the size of the dataset
as well as the output logical form/language to see if our results are generalizable across different
datasets which would be a measure of statistical significance. For example, if we saw a drastic
change in the performance of our algorithm as compared to the regular Transformer as we
increased the size of the datasets (both with respect to exact match accuracy and BLEU scores),
this would suggest that our contributions are only applicable for very specific datasets, which
would render our work less significant than if it seems to hold its performance across datasets of
increasing size as well as varying output languages.
Each of the datasets consist of single sentences as input and single sentences as outputs.
3.3 Results
3.3.1 Exact Match Accuracies for Sentences
Except for one dataset (DJANGO), our algorithm produces better exact match accuracy on
natural language translation and semantic parsing datasets than regular transformers.
Table 3. Exact Match Accuracies for Regular versus Our Transformer for Semantic Parsing
Datasets
Dataset Regular
Transformer
Our
Transformer
Geo880 57.9% 63%
ATIS 78% 81%
DJANGO 34.15% 32.37%
Note: Geo880 (Train dataset size: 600, Test dataset size: 280)
ATIS (Train dataset size: 4435, Test dataset size: 448)
DJANGO (Train dataset size: 15967, Test dataset size: 1801)
Hyperparameters for all datasets: Word vector dimensions: 512, # Layers: 1, # Hidden
units: 2048
44
Table 4. Exact Match Accuracies for Regular versus Our Transformer for Neural Machine
Translation Datasets
Dataset Regular
Transformer
Our
Transformer
Tatoeba EnglishGerman
20.44% 21.6%
IWSLT'15 EnglishVietnamese
0.24% 0.79%
Tatoeba EnglishGreek
10.83% 11.49%
Note: Tatoeba English-German (Train dataset size: 179536, Test dataset size: 44884)
English-Vietnamese (Train dataset size: 133166, Test dataset size: 1590)
English-Greek (Train dataset size: 13955, Test dataset size: 3489)
Hyperparameters for all datasets: Word vector dimensions: 512, # Layers: 2, # Hidden
units: 2048
Our results demand the question: why should it be that for 5/6 of our datasets, the exact
match accuracy of sentence translations and parses are superior to that of the regular
transformer? In order to investigate this, we follow a deductive process of first reminding
ourselves of the difference between our algorithm versus the regular transformer, then speculate
what contribution the difference could have made to the exact match accuracy of the translation
and parse of sentences.
The only difference between our algorithm and the regular transformer is that our
algorithm uses our proposed MAWT self-attention mechanism in which there are multiple selfattention matrices instead of one. As we explained in detail on Section 2, self-attention matrices 
45
are matrices containing probabilities of words/tokens of either of them translating to each other
(at the second-attention layer of the decoder), or their semantic correlation (at the self-attention
layer of the encoder). At the encoder self-attention layer, the Transformer learns how each word
of a sentence from a given language relate to each other word in the same sentence of the same
language. As we have explained, this methodology has proven enormously useful in tasks that
require the deduction of the meaning of words from context (that is, their place within the
sentence) and is a state-of-the-art method in tasks such as named entity recognition, part of
speech tagging, semantic similarity and other inter-language tasks (that is, tasks that are executed
using one and only one language).
At the decoder, the self-attention weights consist of a matrix containing numbers
representing the probability of words/tokens from the input language translating to words/tokens
from the output language.
In our methodology which we called MAWT, we simply raised the dimension of both the
encoder and decoder self-attention matrices in order to have a list of matrices rather than one
matrix for each and have selected the highest scoring matrices during training and testing to get
our translations. Why would this increase the Transformer’s exact match accuracy in translation
and parsing of sentences?
One assumption that comes to mind is purely statistical; the more you roll a dice, the
higher the likelihood of getting any side as an outcome. In other words, by increasing the number
of possible different ways that the words of the encoder could correlate to each other and
increasing the number of possible ways that the words of the decoder could translate to each
other, and finally picking the result that gives the highest score, we are allowing the engine to try
more possible ways of getting to the right answer than the regular Transformer does, as a result 
46
of which we are increasing the likelihood of receiving the right translation/parse as an output (the
measurement of “right” here being exact match accuracy.
3.3.2 BLEU Accuracies for Sentences
For BLEU scores, we observe that our algorithm gives higher BLEU scores compared to
the regular Transformer on 2/6 of the datasets.
Table 5. Average BLEU scores for Regular versus Our Transformer for Semantic Parsing
Datasets
Dataset Regular
Transformer
Our
Transformer
Geo880 90.9% 90.4%
ATIS 89.9% 90.2%
DJANGO 30.96% 28.24%
Note: Average BLEU score: summation of all BLEU scores divided by the size of the test set.
Table 6. Average BLEU scores for Regular versus Our Transformer for Neural Machine
Translation Datasets
Dataset Regular
Transformer
Our
Transformer
Tatoeba EnglishGerman
28.38% 28.2%
IWSLT'15 EnglishVietnamese
15.59% 15.59%
Tatoeba EnglishGreek
14.13% 14.7%
47
Our results show that, compared to the regular Transformer, our algorithm does better
with respect to the exact match accuracy, but not with respect to the BLEU score, since only 2/6
of our algorithm’s sentence translation and parses give superior BLEU score relative to that of
the regular Transformer’s. This suggests that although our algorithm gets more translations and
parses right in terms of exact match accuracy (which would mean a 100% BLEU score), when
they get a translation/parse wrong, they produce strings that take more n-gram edits to transform
into the right translation/parse as compared to the regular Transformer. In other words, our
algorithm produces higher number of correct translations/parses, but the ones that it gets wrong
would take more n-gram edits to correct into the right translation/parse, as compared to the
regular Transformers. Given this observation, we think that a more comprehensive answer to our
initial question as to why our algorithm gives higher exact match accuracy in comparison to the
regular Transformer is needed. In other words, why is it not only giving better exact match
accuracy 5/6 of the time but giving superior BLEU score only 2/6 of the time?
An important question that arises as a result of finding a more comprehensive answer as
to why our algorithm gives superior exact match accuracy but non-superior BLEU score results
to sentence translation/parses is if our engine, which has the capability of producing multiple
candidate outputs and select one, produces outputs that might capture the meaning of the true
output sentence but by using different words and perhaps longer sentences, which would give a
lower n-gram score for the BLEU score calculations. In other words, is the reason why we are
observing lower BLEU scores because, for predictions for which exact match accuracy is zero
the prediction indeed captures the meaning of the desired output sentence (at least sometimes),
but does so using different words and longer sentences than those of the desired output sentence
which would have a corrosive effect on the BLEU score? To investigate this, I checked some 
48
predictions of our algorithm (for the German-English dataset) for which the exact match
accuracy is 0, and compared them to the regular transformer’s outputs to see if this could be the
case. I specifically picked sentences that contain words whose translation could be ambivalent
(for example sentences containing the word “you” which could be translated to “du” or “Sie”, the
polite form of “you” in German).
Table 7. Predictions and correct output for sentences from the English-German dataset.
English Sentence Correct Output Regular Transformer Our Transformer
Do you have a
computer?
Besitzt du einen
Computer?
Hast du einen
Computer?
Habt ihr einen
Computer?.
Do you know that? Wissen Sie das? Kennen Sie das? Weißt du das?
You want me to go,
don't you?
Du willst, dass ich
gehe, oder?
Du willst, dass ich
gehe, oder?
Sie wollen, dass ich
gehe, oder?
You're fashionable. Sie sind
modebewusst.
Sie sind
modebewusst.
Du bist
modebewusst.
For all the sentences outlined here, the BLEU score for the regular transformer will score
higher than our transformer, however they are all sentences that contain the word “you” in
English which could translate to the polite form or the informal form in German; the translation
is ambivalent. Therefore, I assume that our algorithm’s ability to capture different meanings 
49
through using multiple self-attention endows it with a deeper understanding of how many
different words a word can map, which, in the ambivalent cases, sometimes gives us a lower
BLEU score even if the translation is accurate.
Here I analyzed an example of how ambivalent expressions can impact the BLEU score
of our engine. However, for future work, I hope to hand check numerous predictions of our
engine to see which other factors played into its reduced BLEU score compared to the regular
transformer.
3.3.3 Exact Match Accuracies for Word/Token Translations
In order to observe our trained Transformers’ ability to deduce the translation of individual
words from the translation of sentences they have been trained on, we ran translation tests for 500
most frequently occurring words in each of the neural machine translation datasets we used and
compared the exact match accuracy and BLEU score results of our and the regular Transformer3
.
Table 8. Exact Match Accuracies for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets.
Dataset Regular
Transformer
Our
Transformer
Tatoeba EnglishGerman
12.4% 5.2%
IWSLT'15 EnglishVietnamese
30.8% 19.2%
Tatoeba EnglishGreek
17.6% 16.4%
3 We didn’t perform this task for semantic parsing, because there is no engine like Google Translate we
could have used for parsing words into a logical form of a given language in an automated way.
50
Note: Accuracy of word translations were achieved by comparing them to Google Translate
translations.
Table 9. Average BLEU Scores for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets.
Dataset Regular
Transformer
Our
Transformer
Tatoeba EnglishGerman
21.9% 13.25%
IWSLT'15 EnglishVietnamese
47.4% 34.36%
Tatoeba EnglishGreek
39.19% 33.5%
We observe that for all datasets, regular transformers produce better exact match accuracy and
BLEU score performance on the translation of the 500 most frequently occurring words on our
neural machine translation datasets.
Given the black box nature of these algorithms it is difficult to pinpoint what specifically
causes our algorithm to give poorer word/token translations. However, the fact that regular
transformers give better token translations is revealing in that; the only difference between the
regular transformer and our algorithm is the number of attention weights. Since attention weights
are simply lists that contain the numerical probability of words translating to other words, it seems
that, although having the system choose from a list of weights benefits whole sentence translations,
it confuses the system when it comes to word/token translation by providing it with too many
options to choose from. In other words, we conjecture that the reason why words and tokens are 
51
translated wrongly by our algorithm is because it chooses the wrong translation from the option of
translations it has learned.
Using this as an ansatz, I considered what would happen if I had a two-layer transformer
where the first layer is a regular transformer and the second layer is our algorithm. This would
introduce a structure where the first layer would have the regular self-attention mechanism,
whereas the second layer would have our self-attention mechanism. The motivation in introducing
such a structure is to have a transformer which, at first, learns the translation of word/tokens in the
first layer, then learn the most appropriate translation of the word/tokens in a sentence in the second
layer. I consider using a regular transformer for the first layer because my tests have shown that it
outperforms my algorithm in word/token translation. And I consider using my algorithm for the
second layer, because my tests have shown that it outperforms the regular transformer in sentence
translation/parsing. I will call this structure our hybrid algorithm.
I implemented this two-layer hybrid algorithm and ran a preliminary test on the GEO880
dataset (since it’s the smallest set) and observed a considerable increase in the exact match
accuracy of the sentence parses, both compared to the regular transformer and our algorithm.
52
Table 10. Exact Match Accuracies for Regular versus Our Algorithm versus Our New Hybrid
Algorithm
Dataset Regular
Transformer
Our
Transformer
Hybrid
Transformer
Geo880 57.9% 63% 66.79%
Note: Geo880 (Train dataset size: 600, Test dataset size: 280)
Hyperparameters:
For Regular and Our Transformer: Epochs: 500, # Hidden Layers: 2048, Layers: 1.
For the Hybrid Transformer: Layers: 2. All other hyperparameters are the same as
above.
Having observed a significant increase in the exact match accuracy on this dataset using
our hybrid transformer, my last was to run an exact match accuracy and BLEU score test for the
English to Greek dataset for both sentence and word/token translations to observe if I see an
increase in performance.
Table 11. Exact Match Accuracies for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets.
Dataset Regular
Transformer
Our
Transformer
Hybrid
Transformer
Tatoeba EnglishGreek
17.4% 16.4% 11.21%
Hyperparameters: Epochs: 500, Word vector dimensions: 128, # Hidden Layers: 2048, Layers: 2.
(The word vector dimension for Regular Transformer only is 512).
53
Table 12. Average BLEU Scores for Translations of 500 Most Frequent Words Used in Neural
Machine Translation Datasets.
Dataset Regular
Transformer
Our
Transformer
Hybrid
Transformer
Tatoeba EnglishGreek
38.82% 33.5% 22.75%
As can be seen from Table 11 and 12, the hybrid transformer underperformed on word
translations for the English-Greek dataset, relative to the regular transformer and our algorithm.
This suggests that more work is that our ansatz did not lead to a satisfactory result, which
would have been an increase in the performance of sentence and word/token translations.
Therefore, merely combining our algorithm with the regular transformer by creating a twolayered hybrid model as we did does not suffice to create an algorithm that improves sentential
as well as word/token understanding of the engine.
So, to recap our observations, both the exact match accuracy and BLEU score of the
word/token translations of our algorithm performed poorer than that of the regular Transformer.
Somehow, although our methodology leads to superior sentence translation and parses (with
respect to exact match accuracy), it does a worse job of translating individual words and tokens
as compared to the regular Transformer; and creating a hybrid model does not seem to solve this
problem. Therefore, the question remains: Why does our algorithm underperform with respect to
word/token translations?
We suspect that this is because dimension of the word vectors for our algorithm is 128
whereas for the regular Transformer, they are 512. As mentioned before, given the reason why
we lowered the word vector dimensions was to achieve the same run-time as a regular
Transformer, and we had to lower the dimension of hyperparameters in order to achieve this 
54
because we increased the dimension of our attention-weights, which increased run-time. A
higher dimensional word vector allows more ability to capture the meaning of the word by
having more numbers to represent different aspects of a word (position, relation to other words,
etc.). Since we wanted to create a formalism that performs just as well in exact match accuracy
and BLEU score of sentence translations and parses with equivalent run-time, we decreased our
word vector dimensions which allowed our algorithm to run at the same pace as the regular
Transformer. However, we suspect that the trade-off was that this reduction in the dimension of
the word vectors is responsible for poorer word/token translations.
3.3.4 Dataset Dependence of the Performance for Neural Machine Translation
Observing that there are significant differences in the exact match accuracy and BLEU
scores of both the regular transformer and our algorithm for the task of neural machine
translation, it is important to address why in general the performances for neural machine
translation are significantly worse than for those of semantic parsing, and why the performance
differs greatly from one dataset to another.
The reason why we get poorer exact match accuracy and BLEU score results for neural
machine translation datasets is due to the fact that neural machine translation datasets are
significantly larger in size than more semantic parsing datasets, and the maximum computational
complexity I had access to during my work was Google Colab’s GPU’s. Using it, even for the
smallest neural machine translation dataset we reported on, the training for the hyperparameters
that we reported on took at least two days. As a result, the performance of the neural machine
translation tests were significantly impacted by the limited computational resources we had
access to. 
55
The second question I raised above was, why is the performance of the regular
transformer as well as our algorithm so dataset dependent for exact match accuracy and BLEU
scores? There are significant performance differences across datasets. In order to investigate this,
I calculated the vocabulary size (number of words in the input language), the average sentence
length (the sum of the length of all sentences divided by the number sentences in the dataset) for
the train and test set of each dataset. My objective in doing this was to observe these as measures
of complexity, since, for any statistical engine like the Transformer, it is more difficult a task to
learn to translate a big vocabulary language with long/nested sentences as opposed to translating
from a dataset with a small vocabulary and shorter sentences. Of course, one can circumvent the
problem of having a dataset with a big vocabulary and long sentences by increasing the number
of the size of the dataset, therefore another variable I will take into account in measuring
complexity is the size of the dataset.
Also, it is important to compare the difference in the average length of the train set and
the test set of a dataset, because if in fact somehow the average length of the test set is larger than
that of the train set, this would add to the complexity of the task of the Transformer, since it
would have to deduce forming longer sentences from having been trained on shorter ones.
Therefore, my reason in calculating these complexity measures was conjecturing that: if
there is a significant difference in the size of the vocabulary of the datasets compared to the size
of their vocabulary, and if there is a significant difference between the average length of the train
and test set of a dataset, this would explain why we got such differing performances across
datasets.
56
Table 13. Complexity Measures
Dataset Vocabulary Average
sentence
length of the
train set
Average
sentence
length of the
test set
Tatoeba EnglishGerman
29818 6.34 6.34
IWSLT'15
EnglishVietnamese
54170 20.3 21.08
Tatoeba
EnglishGreek
 5430 4.89 4.90
Notes. Tatoeba English-German (Train dataset size: 179536, Test dataset size: 44884)
English-Vietnamese (Train dataset size: 133166, Test dataset size: 1268)
English-Greek (Train dataset size: 13955, Test dataset size: 3489)
Observing my findings on Table 13, firstly, it is evident that the difference in the three
variables differ greatly from dataset to dataset and this could be the reason why the regular
Transformer as well as our engine’s performance was greatly dataset dependent for these three
datasets.
Secondly, we can see that the English-Vietnamese dataset has the largest vocabulary size and
the largest gap between the average length of the train and test datasets, therefore it is the most
complicated dataset for a Transformer to process. This explains why this dataset gave us the
lowest exact match accuracy score.
The second lowest performing dataset was English-Greek. We observe that it this dataset has
the second largest gap between the average length of the train and test dataset, therefore this 
57
could explain as to why it is the second lowest performing dataset for our and the regular
Transformer.
58
Chapter 4: Concluding Remarks
4.1 Summary
In this thesis, I introduced a new Transformer architecture called MAWT: Multi-Attention
Weight Transformers in an attempt to increase the accuracy and variety of the acceptable
predictions of a Transformer, as well as attempting to increase their performance in translating
and parsing individual words/tokens in its vocabulary.
I attempted to achieve this by training multiple weights per each Transformer attention
head, and selecting the weight that produces the highest score for the engine. I then used those
recorded weights to test the accuracy of the engine. During testing I wanted to answer two
questions: can selecting between different weights during testing increase the accuracy of the
engine, and can it increase the performance of the translation/parse of the words/tokens in the
vocabulary of the dataset that the Transformer has been trained upon.
I trained and ran tests on 6 datasets supervised: three of which were for semantic parsing,
and the other three for natural language translation. I observed that my proposed algorithm
produced higher exact match accuracy and BLEU scores for sentence translation/parses for all 6
of the datasets, while producing lower exact match accuracy and BLEU scores for word/token
translation for all 6 datasets.
This clearly shows that although our algorithm is better at translating and parsing
sentences compared to the regular transformer architecture, more work is needed to make it
better at translating and parsing individual words/tokens. In an attempt to make this possible, I
lastly implemented what I called a hybrid model, which consisted of a two-layer Transformer;
first layer being a regular Transformer whereas the second implementing my algorithm. Having
tested this hybrid model on the Geo880 dataset I saw significant increase in the exact match 
59
accuracy of the sentence parses, however, after testing it on the Tatoeba English-Greek dataset,
observed that it gave lower sentence and word/token translation relative to both the regular
transformer and my algorithm.
My future task will be to theorize and implement different methods that can combine the
superior performance of my algorithm compared to the regular transformer with respect to
sentence translation/parses and the superior performance of the regular transformer with respect
to word/token parses, in order to create an engine that outperforms the regular transformer with
respect to both tasks.
4.2 Limitations
We used Google Colab’s GPU’s to perform our tests, therefore this was a very time
consuming process, especially for large datasets like the German-English dataset. When we have
access to TPU’s we would like to run more tests on larger datasets to observe the performance of
our algorithm as the datasets we use gets larger.
Another limitation of our results is that we used only 6 datasets majority of which would be
considered small especially relative to other natural language processing datasets in the
literature.
Also, our datasets have only 6 different output languages. Ideally, we would like to see if the
performance of our engine is generalizable across translation to different natural languages as
well as parsing to different logical forms. For example, we observed that the exact match
accuracy of our engine was lower for only the DJANGO dataset for semantic parsing. There are
widely used SQL based datasets that are much larger relative to the semantic parsing datasets we 
60
used like WikiSQL which is “a dataset of 80654 hand-annotated examples of questions and SQL
queries distributed across 24241 tables from Wikipedia” (Zhong et al., 2017).
Given that we only had access to CPU’s our tests were limited in number of datasets as well
as sizes of the datasets. This must be addressed by running more tests on more datasets of larger
sizes as well as different parsing and natural languages in order to have a firmer understanding of
the generalizability of the performance of our algorithm.
4.3 Future Work
In the future, I will like to continue my research by working on multiple implementations.
Firstly, I will test our algorithm on different trained attention weights (as opposed to the one that
scores the highest) to see if some word/token translations can be improved. If so, I will analyze if
there is a correlation between the index of the weight that gives the best performance and the
given word/token. In other words, I will investigate if the engine give a good performance for all
500 word translations if I use just one weight with a given index, or does the performance
depend on which weight I utilize during testing for which specific word.
Second, I will observe the different parses and translations each of the list of weights in
our algorithm produces, and see if majority of them are sensible and perhaps even more accurate
than the one that our algorithm selects (i.e. the one that scores the highest).
Currently, for very few sentences that I ran tests on, I observe that merely selecting the
attention weight by index (say, using attention weight #3 for prediction) gives non-sensical
outputs. For example, for the German-English dataset, when I outputted all 16 outputs that our
engine can generate, I got (outputs in chronological order with respect to the attention weight
index used):
61
I believe you. Sie arbeiten mit dem Käfi.
I believe you. den den den den…
I believe you. den den den…
I believe you. den den den den den den den…
I believe you. den den den den…
I believe you. jaja?ja???ja??????ja?ja?ja?...
I believe you. den den…
I believe you. den den den den…
I believe you. besten den Menschen unterschdie den behaltenden behaltenden du du
bistdu bistdu bistdu bistdu bistdu…
I believe you. alles den alles den alles den alles den alles den den den…
I believe you. den den den den den den den den…
I believe you. den den den den machstden den den…
I believe you. voller voller voller voller voller…
I believe you. Mit dem Kämpfen hat sich der Welt gestritten.
I believe you. ganz jede jede jede…
I believe you. In den machenden alles extrweitermachstden machtden alles
weiterextrjeden…
where … represents the iteration of the same text. The outputs are either non-sensical or
completely wrong translations. This is because our engine was trained on selecting candidate
translations using a max-function. Therefore, selecting attention weights simply by index does
not give us an attention weight that produces reasonable outputs. We need to use other functions 
62
like the max function in order to obtain candidate translations that are hopefully structural
variants of the desired translation.
The work I have pursued here also raises questions about the role of structure in
capturing semantic equivalence. It is noteworthy that each of the two measures I used to test
performance (exact match and BLEU) aims to measure a semantic relation -- equivalence -- with
structure alone. Exact match accuracy demands syntactic identity and BLEU scores demand
syntactic closeness (based on edit distance). This of course is natural if the goal is to automate
performance measures, but it also raises cognitive questions about whether syntactic identity or
closeness are crucial in characterizing human judgments about semantic equivalence. Let’s
revisit our example in Section 1.7 and assume for current purposes that the active Pat called Kim
and the passive Kim was called by Pat are semantically equivalent in all relevant respects (say,
truth-conditions). Then, given the active Swedish sentence Pat ringde Kim, would human judges
accept the active and passive as equally good translations? It seems plausible to expect that
active and passive would be judged equally good or that the active would be judged better, but it
seems unlikely (according to my judgment) that the passive would be judged better. Similarly,
given the Swedish sentence Pat eller Sam ringde Kim, Pat or Sam called Kim appears to be a
better translation than Pat or Sam called Kim (thanks to Professor Ida Toivonen for this
judgment). Is this because Pat or Sam is more syntactically similar to the Swedish input than Pat
or Sam or both, or because (as we discussed earlier) Pat or Sam has semantic/pragmatic
properties are more similar to the Swedish input than Pat or Sam or both, such as differences
concerning what is concluded about the conjunction Pat and Sam. There are ways we might
control for this. For example, if we embed these structures in downward-entailing environments
like questions or the antecedent of a conditional, the inferential differences between p or q and p 
63
or q or both tend to disappear, leaving the sentences only with their basic inclusive disjunction
meaning (see e.g., Chierchia et al., 2012). When we do this, we see that Pat or Sam still seems to
be a better translation than Pat or Sam or both. For example, given a Swedish conditional om Pat
eller Sam ringde Kim, ... the English If Pat or Sam called Kim, ... is a better translation than If
Pat or Sam or both called Kim, ... for all consequents "...". The only relevant difference here is a
syntactic one if we're right that we've controlled for all relevant meaning differences between p
or q and p or q or both. If this is correct, then it would appear that syntactic considerations are
not merely useful for automation but may also be cognitively relevant as well -- a happy
coincidence between scientific correctness and engineering efficiency.
I also hope to expand my work to consider the role of context in determining semantic
equivalence. For example, the sentences only Sam called and Sam called are not truthconditionally equivalent, but in response to the question Which of Sam, Kim, and Pat called they
both arguably entail that Sam called and Kim and Pat didn't, and hence are contextually
equivalent if they're not semantically equivalent. Incorporating context is thus crucial for
capturing equivalence in actual language use, where sentences occur not in isolation but in some
particular context. I hope to return to this challenge in future work.
Another important work that needs to be implemented is to measure the statistical
significance of our work; in other words, how generalizable is it across different datasets and
also different shuffles of the datasets we used. For example, if we were to shuffle all the datasets
we ran tests on to change the ordering of their train and test sentences, would this have a
negative impact on the exact match accuracy and BLEU scores of our tests? A very reliable way
of checking this is k-fold cross-validation which means running tests on each datasets k-many
times (k is a hyperparameter that needs to be fine-tuned) where each run performs the test for 
64
different shuffles of the dataset. If the performance of the engine differs greatly across different
shuffles, this would render our work less significant since it would be proof that it only performs
well for specific arrangements of train and test sentences.
65
4 References
Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to
align and translate. arXiv preprint arXiv:1409.0473.
Carpenter, B. (1998). Type-logical semantics. MIT.
Cheng, J., Dong, L. & Lapata, M. (2016). Long short-term memory-networks for machine
reading. Proceedings of the 2016 Conference on Empirical Methods in Natural
Language Processing. https://doi.org/10.18653/v1/d16-1053
Chierchia, Gennaro, Danny Fox, and Benjamin Spector. "Scalar implicature as a grammatical
phenomenon." Handbücher zur Sprach-und Kommunikationswissenschaft/Handbooks of
Linguistics and Communication Science Semantics Volume 3. de Gruyter, 2012.
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
Gamut, L. T. (2005). Logic, language, and meaning. Univ. of Chicago Pr.
Grice, Paul. Studies in the Way of Words. Harvard University Press, 1989.
Heim, I., & Kratzer, A. (2012). Semantics in generative grammar. Blackwell.
Jia, R., & Liang, P. (2016). Data recombination for neural semantic parsing. arXiv preprint
arXiv:1606.03622.
Kearns, K. (2000). Semantics. Macmillan.
Ling, W., Grefenstette, E., Hermann, K. M., Kočiský, T., Senior, A., Wang, F., & Blunsom, P.
(2016). Latent predictor networks for code generation. arXiv preprint arXiv:1603.06744.
Liu, B., & Lane, I. (2016). Attention-based recurrent neural network models for joint intent
detection and slot filling. arXiv preprint arXiv:1609.01454.
Machine Translation. Papers With Code. (n.d.). Retrieved November 22, 2021, from
66
https://paperswithcode.com/task/machine-translation.
Mooney, R. (1999, July). Relational learning of pattern-match rules for information extraction.
In Proceedings of the sixteenth national conference on artificial intelligence (Vol. 328, p.
334).
Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). Bleu: a method for automatic
evaluation of machine translation. In Proceedings of the 40th annual meeting of the
Association for Computational Linguistics (pp. 311-318).
Rabinovich, M., Stern, M., & Klein, D. (2017). Abstract syntax networks for code generation and
semantic parsing. arXiv preprint arXiv:1704.07535.
Semantic parsing. Papers With Code. (n.d.). Retrieved November 22, 2021, from
https://paperswithcode.com/task/semantic-parsing.
Shieber, Stuart M. "The problem of logical form equivalence." Computational Linguistics 19.1
(1993): 179-190.
Understanding the BLEU Score. Google. (n.d.). Retrieved November 29, 2022, from
https://cloud.google.com/translate/automl/docs/evaluate#bleu.
Van Noord, Gerardus Johannes Maria. Reversibility in natural language processing.
Rijksuniversiteit Utrecht, 1993.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin,
I.(2017). Attention is all you need. In Advances in neural information processing systems
(pp. 5998-6008).
Wikimedia Foundation. (2021, November 10). Bleu. Wikipedia. Retrieved November 22, 2021,
from https://en.wikipedia.org/wiki/BLEU. 
67
Zelle, J. M., & Mooney, R. J. (1996, August). Learning to parse database queries using inductive
logic programming. In Proceedings of the national conference on artificial
intelligence (pp. 1050-1055).
Zhong, Victor, Caiming Xiong, and Richard Socher. "Seq2sql: Generating structured
queries from natural language using reinforcement learning." arXiv preprint
arXiv:1709.00103 (2017).
