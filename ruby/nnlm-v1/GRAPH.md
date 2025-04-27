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
 ErrS                     HA                  dWO               |
                         -0.02      [  0.00,  0.00, -0.00 ]     |
 -0.49  ~outer product~  -0.03  =>  [  0.00,  0.00, -0.00 ]     |
  0.49                    0.03  =>  [  0.00,  0.00, -0.00 ]     |
                                    [  0.00,  0.00, -0.00 ]     |
                                                                |
                                                                |
                                                                |
                                                                |
 -> GRAD EMBEDDINGS                                 +---------- +
                                                    |

 ErrS             TRANS(OH)             dHIS     dT(ha)       dHI                TRANS(WH)                    dIL  
                                         0.01      1.00       0.01    N1 [ -0.02, -0.05,  0.04, -0.02 ]       0.00 
 -0.49  x  [ -0.04, -0.05, -0.01 ]  =>   0.03  x   1.00  =>   0.03  x N2 [  0.03,  0.00, -0.02,  0.02 ]  =>  -0.00 
  0.49  x  [ -0.02,  0.02, -0.04 ]  =>  -0.02  x   1.00  =>  -0.02  x N3 [  0.02,  0.02, -0.04, -0.01 ]  =>   0.00 
               N1     N2     N3                                                                               0.00 
  
                                                                |
                                                                |
                                                                |
                                                                |
 -> GRAD HIDDEN WEIGHTS    +------------------------------------+
                                                                |
  IL                      dHI                 dWH               |
  0.01                    0.01      [  0.00,  0.00, -0.00 ]     |
  0.03  ~outer product~   0.03  =>  [  0.00,  0.00, -0.00 ]     |
  0.01                   -0.02  =>  [  0.00,  0.00, -0.00 ]     |
  0.03                              [  0.00,  0.00, -0.00 ]     |
                                                                |
                                                                |
                                                                |
 -> GRAD HIDDEN BIAS (dHI)                                      |
                                                                |
  dBH  ---------------------------------------------------------+
  0.01 
  0.03 
 -0.02 
     

 -> GRAD EMBEDDINGS
 --> FOR EACH EMBEDDING IN CONTEXT
 -----> APPLY ERROR (SCALED FOR LEARNING RATE)

     EMBEDDINGS

xx [  0.01,  0.03 ] 
-- [  0.00,  0.04 ]

Since the first embedding  [ 0.01, 0.03 ] appeared TWICE in the context,
We will update it with each part of the dIL:

  dIL
 0.00 * lR => Applied to [ 0.01 ]
-0.00 * lR => Applied to [ 0.03 ]
 0.00 * lR => Applied to [ 0.01 ]
 0.00 * lR => Applied to [ 0.04 ]

At the end of Back Propagation, these embeddings will be updated.

Additionally, the gradients / deltas will be applied for:

  * GRAD HIDDEN BIAS / dBH => Applied to W_b bias vector (the hidden bias vector)
  * GRAD HIDDEN WEIGHTS / dHW => Applied to W_h connections matrix (the hidden weights matrix)
  * GRAD OUTPUT BIAS / dBO => Applied to the b_o vector (the output bias vector)
  * GRAD OUTPUT WEIGHTS / dWO => Applied to the w_o connections matrix (the output weights matrix)

### Why is each delta / gradient what it is?

#### Output Bias Gradient (dBO)

Let's start with `dBO`, the output bias.

If an output votes for itself (given the signals it received from the network),
correcting *itself* is simple.

It just changes the bias it gives itself to vote.

Remember the word "the" might have a high bias to vote for itself.

Every time it is wrong, it will make small adjustments to its signal.

#### Output Bias Weights (dWO)

Remember, this represents how much an `output` pays attention to a neuron.

The output is sent signals directly from neurons. If a neuron sends the wrong
signal to an output, the output can just directly adjust its `weight` that
that it gives to that neurons signals.

Imagine the neuron that learned to pick up on fire. 
The word "dance" may have learned to weight signals from this neuron very low. 
But what if, in the context is "We just put out the fire"?
It might be time to dance! 

To correct this, the output will adjust its weight to that neuron. 
And it will increase weights to other neurons that perhaps send signals that
preceding words negated the meaning of the most recent word.

This is calculated via the OUTER PRODUCT of the Error Signal and Hidden Activation.

Why these values?

We use the Error Signal in some part to make all changes.

Hidden Activation is relevant to the Output Weights, because this is the signal sent *TO*
the Output Weights.


     ErrS         Outer Product                HA                           dWO
                                                                        N1       N2       N3
O1 [ -0.49 ]                           N1   [-0.02 ]               [  0.0098   0.0147  -0.0147 ] O1
02 [  0.49 ]             ⊗             N2   [-0.03 ]       =       [ -0.0098  -0.0147   0.0147 ] O2
                                       N3   [ 0.03 ]

This is somewhat easy to follow in our Truth Table example.

If O1 is wrong, O2 must have been right. In this example, we are doing two things:

 * Telling O1 to make small adjustments to every connection to the neurons 
    * to pay attention less to N1 and N2, and more to N3.
 * We are making the exact opposite small adjustments to the connections to O2 
    * pay more attention to N1 and N2, less to N3.


### How is the Error Signal calculated?

In this example, 0 represents false.

`false & false` = false

So, if the Network sees the context for [0,0], it should activate `false`.

Instead, the network activated `true`.


  PRED     GRAD      ErrS

  0.51  -  1.00  => -0.49
  0.49     0.00      0.49

If the network got this correct, the gradient would be [ 0.00, 0.00 ].
Nothing would change.

  PRED     GRAD      ErrS

  0.51  -  0.00  =>  0.51
  0.49     0.00      0.49

This would still be fed backward through the neural network to 
strengthen connections rather than weaken them.

```
