# Neural Network Language Model

## General Concepts

In a neural network, there are 3 main parts:

* Your input (in our case, a list of preceding words)
* The nuerons
* Your ourput (in our case, the next word)

## Weights

Weights connect the neurons to the input and the output.

Think of these as connections or phone lines.

Let's say a neuron has learned to recognize a pattern in words
for when something is on fire.

We want this neuron to activate reliably! 
Think of the lives it could save.

However, let's say some of our output words are things like:

  * stay-calm
  * dance
  * sleep

We want these outputs NOT to activate when this nueron activates.

The way we do this is by lowering the `weight` from the example `output`s 
to this example neuron.

Think of weight as attention. It comes from `weigh`ing votes.

This nueron is voting, and saying, to every output at the same time,
I'm quite certain this passage is about something being on fire.

Words like "help" and "do-something" will want to pay close attention 
to this signal / neuron. 

So they will give the connection to this neuron a high `weight`.

## Biases

A bias, in our example, is how `bias` an output (an individual word)
is to vote for itself to be the next word.

Word like "the", "be", and "to"  might want to have a high bias.  They're
the most commonly used words in English.

There's a decent chance given any pattern they might be the next word to occur.

Words like "taradiddle" and "borborygmus" might want to have really low biases.

Even if they see a pattern where they've been used before, the odds aren't in
their favor that they'll be the next word to follow.

It would be much more likely for "lie" or "grumble" to occur instead.

## Hidden weights and biases

I described the *OUTPUT* weights and baises, because they are easier to understand.

But individual neurons have their own biases.

Let's imagine one of our neurons, through random luck, starts with a high bias.

This means it'll activate frequently, telling all output words often, "HEY, LISTEN TO ME!"

In the beginning, before the nueral network is trained, nothing has meaning.

But as the neural network trains and learns, this could eventually become a useful signal for
common words to activate.

Maybe this signal ends up meaning, "I think a common word should follow".

Words like "the" and "be" will pay close attention to this signal (giving it a high weight).

Words like "taradiddle" and "borborygus" will be annoyed that they won't get the time to shine.

However, if this signal is ever going to be useful, it can't just activate *ALL* the time.

Through training (backpropagation), it will adjust it's weight to inputs which include words 
that *should* have common words follow, maybe: "of", "in", "to", and "on".

The weights neurons give to inputs is called `Hidden Weight`, and the bias a neuron 
has to activate is called `Hidden Bias`.
