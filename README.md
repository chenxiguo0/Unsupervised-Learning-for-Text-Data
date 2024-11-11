# HW-4: Unsupervised Learning for Text Data
DSAN-5000 Staff, Georgetown University, Fall 2024

## Overview

The term “Exploratory Data Analysis” came into common usage among
scientists and engineers <a
href="https://books.google.com/ngrams/graph?content=Exploratory+Data+Analysis&amp;year_start=1800&amp;year_end=2022&amp;corpus=en&amp;smoothing=2&amp;case_insensitive=false"
target="_blank">starting in the early 1970s</a>, then reached an initial
peak in usage around 1985, bolstered especially by discussions around
the principles of John Tukey’s 1977 *Exploratory Data Analysis* book.

Though the plot linked above also shows a modern resurgence in its
usage, with the yearly growth rate from 2016 to the present now
paralleling that of the 1970s, the earlier rise-in-usage $\rightarrow$
popular-work-on-the-term $\rightarrow$ peak-in-usage $\rightarrow$
decline-in-usage pattern is a common phenomenon when new terms are
coined, which historians call a period of <a
href="https://press.princeton.edu/books/paperback/9780691022239/the-terms-of-political-discourse"
target="_blank">“essential contestation”</a>. Basically, this is the
period during which the term’s meaning is “hammered out”, transforming
it from a vague neologism into a term with a definition written down in
a bunch of dictionaries.

The relevance of all that, for our purposes here, is that it means the
term’s definition was cemented in essentially a pre-text-as-data era,
**before** the advent of neural NLP in the 1990s, and especially before
the creation of the Large Language Models that have become so prominent
today! For example, a quick search for web resources about EDA will
mostly return examples based on “born-numeric” (and typically tabular)
data, rather than textual data, using visualization approaches
popularized by Tukey’s book:

<img src="images/eda_search.jpg" data-fig-align="center" width="500" />

So, whereas in class we’ve gone over many of these key EDA steps and
visualization techniques for different types of numeric and/or tabular
data, in this homework we will expand your EDA toolkit by introducing
you to two modern NLP-rooted techniques, both of which are often used as
initial EDA steps for **text corpora** specifically:

- Topic Models
- Word Embeddings

Our hope is that working through the basics of these two methods can
serve as a “bridge” connecting the previous Data Cleaning and EDA
material with the more advanced Unsupervised Learning and Clustering
material that we are exploring in the coming weeks.

## Parts 1 and 2: Topic Models for EDA at the Corpus Level

As you’ll start to see when we look at Unsupervised Learning techniques
in the coming weeks, a helpful approach to learning these methods when
first starting out is to perform a kind of “calibration exercise”:

- Take a dataset or corpus with an **already established** set of
  clusters, but then
- Run an **unsupervised** learning algorithm on the dataset or corpus,
  so that you can
- **Compare and contrast** the patterns discovered by the unsupervised
  algorithm with your prior expectations.

On top of being helpful for learning how exactly to implement a given
unsupervised algorithm, this approach can also be a fruitful EDA step.
This is especially \[and somewhat paradoxically\] the case when the
discovered patterns **don’t** match what you expected, since this can
help with challenging your prior beliefs around the data, raising
questions like:

- Why didn’t it detect the pattern that seemed “obvious” to me?
- Is it detecting some even-stronger signal within the data that I
  hadn’t recognized?
- If so, what exactly is this stronger signal? How can it be
  characterized? Does it conform to existing theories within my field?

So, to this end, your task will be to perform EDA on a
<a href="https://open.nytimes.com/fatten-up-your-corpus-e72b22272222"
target="_blank">corpus of <span
class="math inline"><em>N</em> = 1529</span> newspaper articles</a>,
written between January 1 and January 31 of 2007 and published in the
following $K = 5$ sections of the *New York Times*:

- US News
- World News
- Arts
- Sports
- Real Estate

Using a simple **Topic Model** of the corpus, your goal will be to carry
out the type of “calibration exercise” described above: to see whether
or not this **unsupervised model** “detects” the original sections that
the articles are drawn from, or comes up with some other way of
classifying the contents of the articles into $K = 5$ topics.

> [!NOTE]
>
> ### HW4.1 and HW4.2 Deliverables
>
> Please follow the instructions provided in the Jupyter notebooks
>
> - `dsan5000_hw4-1.ipynb` and
> - `dsan5000_hw4-2.ipynb`,
>
> and include both files within the root directory of your GitHub
> Classroom repository for the assignment.

## Parts 3 and 4: Word Embeddings for EDA at the Token Level

Returning to our discussion of the history of EDA in the Overview
section above, linguistic Topic Models like the model whose parameters
you estimated in HW4.1 and HW4.2 were created well after the “cementing”
of the term Exploratory Data Analysis. Though several were introduced in
the 1990s, the most widely-used approach (called Latent Dirichlet
Allocation) was
<a href="https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf"
target="_blank">introduced in 2003</a> by **statistical** NLP
researchers David Blei, Andrew Ng, and Michael I. Jordan.

We emphasize the **statistical** in “statistical NLP” here because,
about a decade later
<a href="https://arxiv.org/abs/1301.3781" target="_blank">in 2012</a>,
another unsupervised approach to analyzing text termed “Word Embeddings”
was developed (and quickly become even more popular than LDA), which
uses a **neural** rather than statistical NLP approach to the
unsupervised learning of linguistic patterns. In broad strokes, the
difference between these two approaches is that:

- The statistical-NLP-based Latent Dirichlet Allocation algorithm
  estimates a set of **explicitly-defined** parameters from a
  written-out \[generative\] statistical model of word usage (written in
  the above-linked paper as a <a
  href="https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/"
  target="_blank"><em>Probabilistic Graphical Model</em></a>), while
- The neural-NLP-based Word Embedding algorithm learns linguistic
  structure **implicitly** through optimization with respect to a
  linguistic **prediction** task: the task of solving
  “fill-in-the-blanks” questions across a large corpus of
  natural-language sentences.

One result of this difference is that, while fitting a **statistical**
NLP model will produce explicit probabilistic estimates of
**linguistic** parameters (e.g., explicit values of the joint
probability $\Pr(W_i = w_i, W_j = w_j)$ that a word $w_j$ will be
followed by a word $w_i$ within a given sentence), fitting a **neural**
NLP model will produce a neural network optimized for the task of
**predicting** unknown words in a sentence on the basis of known words
in the same sentence.

With this difference in mind, let’s see what we can learn from a set of
pre-computed word vectors for each (sufficiently-common) word used in
the *New York Times* articles!

> [!NOTE]
>
> ### HW4.3 and HW4.4 Deliverables
>
> Please follow the instructions provided in the Jupyter notebooks
>
> - `dsan5000_hw4-3.ipynb` and
> - `dsan5000_hw4-4.ipynb`,
>
> and include both files within the root directory of your GitHub
> Classroom repository for the assignment.
