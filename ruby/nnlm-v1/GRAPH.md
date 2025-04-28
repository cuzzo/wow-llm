```
 ~~~ ~~~~~~~ ~~~ 
 ~~~ FORWARD ~~~ 
 ~~~ ~~~~~~~ ~~~ 
    EMBEDDINGS            IL                 HW                  RNI       HB         HI             HA                OW              RNO       OB         RAW              PRED  
                          0.01     [ -0.02,  0.03,  0.02 ]      -0.00     -0.02      -0.02          -0.02    N1 [ -0.04, -0.02 ]                                                   
xx [  0.01,  0.03 ]  =>   0.03  x  [ -0.05,  0.00,  0.02 ]  =>   0.00  +  -0.04  =>  -0.03  ~tanh~  -0.03  x N2 [ -0.05,  0.02 ]  =>   0.00  +   0.03  =>   0.03  ~softmax~   0.51 
-- [  0.00,  0.04 ]  =>   0.01  x  [  0.04, -0.02, -0.04 ]  =>   0.00      0.03       0.03           0.03    N3 [ -0.01, -0.04 ]      -0.00     -0.00      -0.01              0.49 
                          0.03     [ -0.02,  0.02, -0.01 ]                                                                                                                       
                                      N1      N2     N3                                               |          
                                                                                                      |
                                                                                                      |
                                                                                                      |
 ~~~ ~~~~~~~~ ~~~                                                                                     |
 ~~~ BACKWARD ~~~                                                                                     |
 ~~~ ~~~~~~~~ ~~~                                                                                     |
                                                                                                      |
The Output Weight and Hidden Weight matrixes must be transposed, because we are                       |
working backward (in reverse). To get to the error signal, we travel from:                            |
                                                                                                      |
 1. Input Layer                                                                                       |
 2. Hidden Weights (connections from neurons to the input layer)                                      |
 3. Neurons (do their voting)                                                                         |
 3. Output Weights (connections from the neurons to the output layer)                                 |
 4. Output Layer (make the prediction)                                                                |
                                                                                                      |
To propagate the gradients / deltas all the way back, we need to travel in reverse.                   |
                                                                                                      |
  1. Output Layer (with the exact error signal)                                                       |
  2. Output Weights (TRANSPOSED)                                                                      |
  3. Hidden Weights (TRANSPOSED)                                                                      |
  4. Input Layer                                                                                      |
                                                                                                      |
                                                                                                      |
                                                                                                      |
 -> GRAD OUTPUT BIAS (Error Signal / ErrS)                                                            |
                                                                                                      |
 dBO                                                                                                  |
                                                                                                      |   
 -0.49                                                                                                |
  0.49                                                                                                |
                                                                                                      |
 -> GRAD OUTPUT WEIGHTS    +----------------------------------- +-------------------------------------+
                                                                |
  ErrS                     HA                  dWO              |
                         -0.02      [  0.00,  0.00, -0.00 ]     |
 -0.49  ~outer product~  -0.03  =>  [  0.00,  0.00, -0.00 ]     |
  0.49                    0.03  =>  [  0.00,  0.00, -0.00 ]     |
                                    [  0.00,  0.00, -0.00 ]     |
                                                                |
                                                                |
                                                                |
                                                                |
 -> GRAD HIDDEN WEIGHTS (dWH)                       +---------- +
                                                    |

  ErrS             TRANS(OH)             dHIS     dT(HA)       dHI                      IL                  dWH           
                                         0.01      1.00       0.01                     0.01      [  0.00,  0.00, -0.00 ]
 -0.49  x  [ -0.04, -0.05, -0.01 ]  =>   0.03  x   1.00  =>   0.03   ~outer product~   0.03  =>  [  0.00,  0.00, -0.00 ]
  0.49  x  [ -0.02,  0.02, -0.04 ]  =>  -0.02  x   1.00  =>  -0.02                     0.01  =>  [  0.00,  0.00, -0.00 ]
               N1     N2     N3                                                        0.03      [  0.00,  0.00, -0.00 ]
  
                                                                |
                                                                |
 -> GRAD HIDDEN BIAS (dHI)                                      |
                                                                |
  dBH  ---------------------------------------------------------+
  0.01                                                          | 
  0.03                                                          | 
 -0.02                                                          |
                                                                |
                                                                | 
 -> GRAD EMBEDDINGS                                             |
                                                                |
    +-----------------------------------------------------------+

   dHI                TRANS(WH)                    dIL  
   0.01    N1 [ -0.02, -0.05,  0.04, -0.02 ]       0.00 
   0.03  x N2 [  0.03,  0.00, -0.02,  0.02 ]  =>  -0.00 
  -0.02  x N3 [  0.02,  0.02, -0.04, -0.01 ]  =>   0.00 
                                                   0.00 

 --> FOR EACH EMBEDDING IN CONTEXT
 -----> APPLY ERROR (SCALED FOR LEARNING RATE)

     EMBEDDINGS

xx [  0.01,  0.03 ] 
-- [  0.00,  0.04 ]

Since the first embedding  [ 0.01, 0.03 ] appeared TWICE in the context,
We will update it with each part of the dIL:

  dIL
 0.00 * lR => Applied to [ 0.01 ] Dimension 1, Input 0 "False"
-0.00 * lR => Applied to [ 0.03 ] Dimension 2, Input 0 "False"
 0.00 * lR => Applied to [ 0.01 ] Dimension 1, Input 0 "False"
 0.00 * lR => Applied to [ 0.04 ] Dimension 2, Input 0 "False"

At the end of Back Propagation, these embeddings will be updated.

Additionally, the gradients / deltas will be applied for:

  * GRAD HIDDEN BIAS / dBH => Applied to W_b bias vector (the hidden bias vector)
  * GRAD HIDDEN WEIGHTS / dHW => Applied to W_h connections matrix (the hidden weights matrix)
  * GRAD OUTPUT BIAS / dBO => Applied to the b_o vector (the output bias vector)
  * GRAD OUTPUT WEIGHTS / dWO => Applied to the w_o connections matrix (the output weights matrix)

```

#### How is the Error Signal calculated?

In this example, 0 represents `false`.

`false & false` = `false`

So, if the Network sees the context for `[ 0, 0 ]`, it should activate `false`.

Instead, the network activated `true`.

```

        PRED     REAL      ErrS

FALSE   0.51  -  0.00  =>  0.51
TRUE    0.49     1.00  => -0.51

```

*What does this mean?*

The Neural Network is currently saying:

> I think 51% of the time I see `false & false` I should respond with `true`.

We are saying:

> No! 100% of the time you should respond with true.
> 0% of the the time you should respond with false.

If the network got this correct, the Error Signal would look different,
let's say for `[ 0, 1 ]` or `false & true` it correctly gave `false` the
highest probability:

```

        PRED     REAL       ErrS 

FALSE   0.51  -  1.00  =>  -0.49
TRUE    0.49     0.00       0.49

```

In the section below, you'll see how this signal, when fed `backward`
through the network will correct each stage by the ideal amount.


### Why is each delta / gradient what it is?

This graph seems complicated.

But, in essence, it's quite simple.

Each `layer` of the network (Hidden Weights, Hidden Bias, Output Weights, Output Bias) has 
*ONE* input - which is some transformation on the original input.

In the example above, think of the input as `[ 0, 0 ]` or `[ false, false ]`.

 * The Hidden Weights are the first to transform this input into something different.
 * Then the Hidden Baises take that output and transform it again.
 * Then the Output Weights transform it again.
 * Finally, the Output Weights make the last transformation.

To determine the gradients, we use two things at each stage:

  * The original input to that specific layer
  * The final Error Signal (which we learned to calculate above)

#### Output Bias Gradient (dBO)

Let's start with `dBO`, the gradient (or change) to the Output Bias.

An output `votes for itself` when:

 * It takes the input from the previous layer (Ouput Weights)
 * Applies its bias
 * Ends up with the highest score

If an output votes for itself when it shouldn't, correcting *itself* is simple.

It just changes the bias it gives itself to vote.

*INTUITIVELY*, think of it like this:

A nueral network is in the early stages of training.

It starts with the phrase "I want to" and it's trying to predict the next word.

It comes up with these scores:

```
        PRED
 the    0.4
 go     0.3
 dance  0.2
 fly    0.05
 ...
```

We know the actual next word is "go", so the *REAL* probability distribution
for this *EXACT* example is:

```
        PRED   REAL PROBABILITY
 the    0.4        0.0
 go     0.3        1.0
 dance  0.2        0.0
 fly    0.05       0.0
 ...
```

That is, everything besides "go" did not occur (in this example), and "go" did occur.

So we update each bias by the *INVERSE* of this:

```

        PRED   REAL PROBABILITY        GRADIENT
 the    0.4        0.0              0.4 - (0.0 * lR)
 go     0.3        1.0              0.3 - (1.0 * lR)
 dance  0.2        0.0              0.2 - (0.0 * lR)
 fly    0.05       0.0              0.05 - (0.0 * lR)
 ...

 * lR = learning rate
```

For simplicity, let's imagine the learning rate is 0.5.
The gradient (or change) we would apply to the output baises is:

```
        GRADIENT
 the      0.2
 go      -0.35
 dance    0.1
 fly      0.025
 ...
```

*What does this mean?*

After the gradients are applied (subtraction), the `-0.35` applied to go
will give it a higher bias, and the positve values assigned to all other biases,
will give them lower biases.

If the original biases looked like:

```
         BIAS       GRADIENT
 the     0.42         0.2
 go      0.15        -0.35
 dance   0.1          0.1
 fly     0.01         0.025
 ...
```

Then the updated biases will look like:

```
             UPDATED BIAS
 the     0.42 -  0.2   =  0.22
 go      0.15 - -0.35  =  0.5
 dance   0.1  -  0.1   =  0.0
 fly     0.01 -  0.025 = -0.015
 ...
```

*Remember* the word "the" might have a high bias to vote for itself.

Every time it is wrong, it will make small adjustments to its bias.

Each time we make an adjustment, we want to do so specifically where 
it will have the maximum impact / the maximum *change*.

To do that, we need to look at the rate of *change* (the derivative).

#### Output Weights Gradient (dWO)

Remember, this represents how much an `output` pays attention to a neuron.

The output is sent signals directly from neurons. An output can only be *wrong*
if it gets the bad inputs from its connections to the neurons.

These connections are the `weights`, and so the output must adjust the `weight` 
that it gives to neuron's vote whenever the output makes the wrong prediction.

Imagine a neuron that learned to pick up on the word "fire" being in the context. 
The word "dance" may have learned to weight signals from this neuron very low. 
But what if the full context is "We just put out the fire" ?
It might be time to dance! 

To correct this, the "dance" output will adjust its weight to the "fire" neuron
to increase its attention / the *weight* it gives this neuron's votes -
Remember, it *should* have been activated in this context, but it wasn't.

*In simple terms:*

  * If an output neuron was wrong, the weights leading into it need to be corrected.
  * How much correction each weight needs depends on:
    * How wrong the output neuron was (Error Signal)
    * How active the input neuron was (Hidden Activation)

This is calculated via the `OUTER PRODUCT` of the Error Signal and Hidden Activation.

*Why Outer Prouct for Weights but simple Subtraction for Biases?*

Weights connect many inputs to many outputs — they form a matrix. 
To know how much each individual connection needs to change, you need a 2D operation — the outer product.

Biases are added after all that complexity — one bias per output neuron — 
so you can adjust them directly with simple subtraction.

Why doesn't dBO (the Output Bias Gradient) need to calculate a derivative,
while other parts (like weights) do?

The reason is simple but crucial:

Biases are added *after* any `activation` or transformation.

When a bias is applied, it's just added — nothing fancy: No squashing, no scaling, 
no `activation` like `sigmoid`, `ReLU`, or `softmax` is happening between the 
bias and the final score (sometimes called the `logit`).

Weights do a transformation, which cannot be undone / reversed / changed so simply.

*Why these values?*

We use the Error Signal in some part to make all changes (simple subtraction for biases, outer products for weights).

Hidden Activation is relevant to the Output Weights, because this is the signal sent *TO*
the Output Weights.

To calculate the original RAW Neural Output, we did:

```
  Hidden Activation              Output Weights       Raw Neural Output
      -0.02                  N1 [ -0.04, -0.02 ]           
      -0.03            x     N2 [ -0.05,  0.02 ]  =>       0.00
       0.03                  N3 [ -0.01, -0.04 ]          -0.00

```

To calculate the gradients / changes to MINIMIZE the Error Signal, we'll need to do:

```

     ErrS          Outer Product         Hidden Activation                      dWO
                                                                        N1       N2       N3
O1 [ -0.49 ]                           N1   [-0.02 ]               [  0.0098   0.0147  -0.0147 ] O1
02 [  0.49 ]             ⊗             N2   [-0.03 ]       =       [ -0.0098  -0.0147   0.0147 ] O2
                                       N3   [ 0.03 ]

```

This is somewhat easy to follow in our Truth Table example.

If O1 (output `false`) is wrong, O2 (output `true`) must have been right. 
In this example, we are doing two things:

 * Telling O1 to make small adjustments to every connection to the neurons 
    * to pay attention less to N1 and N2, and more to N3.
 * We are making the exact opposite small adjustments to the connections to O2 
    * pay more attention to N1 and N2, less to N3.

We use the Outer Product because we have 2 vectors. 
Remeber, we only have two things to calculate gradients / changes to weights
at each stage.

 * The Error Signal 
 * The input to that layer

The input to the matrix layers (the connections to and from neurons) is a vector.
This vector `passes` through the matrix (via vector matrix multiplication) to produce
another vector.

We have two vectors (`Error Signal` & `Hidden Activation`) and, from these, we need
to product a matrix of adustments to make to weights (a matrix).

*OUTER PRODUCT* takes in two vectors (what we have) and gives a matrix (what we want).
This matrix will tell us how much to adjust each weight.

*Technically*, at every stage, we take the derivative 
of the Error Signal with respect to the input at the stage:

```
   -              -
  / Error Signal   \
 d| -------------- |
  \     Input      /
   -              -
```

Or, for the Output Weights Gradient Specifically:

```
   -                     -
  /     Error Signal      \
 d| ----------------------|
  \   Hidden Activation   /
   -                     -
```

Again, to calculate the original RAW Neural Output, we did:

```
  Hidden Activation              Output Weights       Raw Neural Output
      -0.02                  N1 [ -0.04, -0.02 ]           
      -0.03            x     N2 [ -0.05,  0.02 ]  =>       0.00
       0.03                  N3 [ -0.01, -0.04 ]          -0.00

```

So mathematically, again, we are trying to calculate where the 
Rate of Change (derivative) for the Error Signal with respect to
the original input for this calculation (the Hidden Activation) is close to zero.

```
   -                     -
  /     Error Signal      \
 d| ----------------------|
  \   Hidden Activation   /
   -                     -
```

This matches the intuition seen above:

```

     ErrS          Outer Product         Hidden Activation                      dWO
                                                                        N1       N2       N3
O1 [ -0.49 ]                           N1   [-0.02 ]               [  0.0098   0.0147  -0.0147 ] O1
02 [  0.49 ]             ⊗             N2   [-0.03 ]       =       [ -0.0098  -0.0147   0.0147 ] O2
                                       N3   [ 0.03 ]

```

This might seem like a giant leap of faith, if you have limited knowledge of
calculus or linear algebra.

The derivative of vector matrix multiplication is *OUTER PRODUCT*.

However, even with rudimentary knowledge of pre-calculus, it shouldn't be *too*
surprising it ends up being somewhat simple.

Let's examine:

```
 d(x^3+2x^2)   = 3x^2 + 4x
 d(3x^2 + 4x)  = 6x + 4
 d(6x + 4)     = 6 + 0
 d(6)          = 0
```

Each time we take a derivative, we are removing a layer of complexity.

Visualize this like this:

```
y
^
|                            *
|                         *     *
|                      *           *
|                   *                 *        (3x^2, curve)
|                *                       *       (x & y, change)
|             *                             *      (as does the slope)
|          *                                   *
|       *                                         *
|    *                                               *
| *-----------------*-----------------*-------------------> x
|                 /      (6x x & y change)
|               /          (but the slope does not)
|             /
|           /
|          /
|        /    
|      /        
|    /            
| *-----------------*-----------------*-------------------> x
|                            (6)
|~~~~~~~~~~~~~~~~~~~~(constant line - only x changes)~~~~~~~
```

See this [Wolfram Alpha Graph](https://www.wolframalpha.com/input?i=plot+of+y%3Dx%5E3+%2B2x%5E2+and+y%3D3x%5E2+%2B+4x+and+y%3D6x+%2B+4++for+x%3D2+to+-1+).

Remember, by adding `tanh`, we gained a layer of complexity.

We took something linear and made it non-linear.

To get Hidden Activation initially, we performed vector matrix multiplication
which *CAN* (and in this case did) change dimensionality.

It is not surprising that *OUTER PRODUCT* is the derivative, given that it
*CAN* (and in this case does) change dimensionality in the exact opposite direction
(which conveniently is exactly what we need to do).

*Let's take a simple example:*

To calculate Raw Neural Output, we did:

```
  Hidden Activation              Output Weights       Raw Neural Output
    [ -0.02 ]                N1 [ -0.04, -0.02 ]           
    [ -0.03 ]          x     N2 [ -0.05,  0.02 ]  =>     [  0.00 ] O1
    [  0.03 ]                N3 [ -0.01, -0.04 ]         [ -0.00 ] O2

```

To calculate an individual cell in Raw Neural Output `RNO[0]` we did:


```
             HA[0]     OW[0, 0]     HA[1]      OW[1, 0]     HA[2]     OW[2, 0]
RNO[O1] =   -0.02   *   -0.04    +  -0.03   *   -0.05    +  0.03    *  -0.01
```

Now, replace row 0 with `i` to generlaize, and the hardcoded `[0, 1, 2]` with `k`,
We get the vector matrix formula for a cell:

```
RNO[i] = sum_k( HA[k] x OW[k, i] )
```

All terms where `k` is not the Error Signal we are differentiating with respect to become zero.


*What does this mean?*

We want to adjust the `WEIGHT` for each neuron `k` with respect to each output `i`.
To do this, we use the Error Signal for neuron `i`.

*What does THAT mean?*

To figure out how to adjust the weight for neuron `2` to output `1`:

```
   -              -
  /    ErrS[O1]    \
 d| -------------- |
  \     HA[N2]     /
   -              -
```


*Which means:*

Multiply the error signal at output 1 (ErrS[O1]) by the hidden activation HA[2].

This gives the gradient for the weight OW[2, 0].

There's an operation in linear algebra which does this. It's the *OUTER PRODUCT*.

