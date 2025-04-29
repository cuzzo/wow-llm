#! /usr/bin/env ruby
# frozen_string_literal: true

require "set"
require "cmath" # Using CMath just for tanh convenience, could implement manually
require "msgpack"

DASHES = [150, 151].map(&:chr) # em & en dash
PARAGRAPH = "[PARAGRAPH]"
MODEL_FILE = "model.msgpack"
TOKEN_FILE = "tokens.json"

# Helper functions for basic vector/matrix operations on Ruby Arrays
module BasicLinAlg
  def dot_product(vec1, vec2)
    vec1.zip(vec2).sum { |a, b| a * b }
  end

  # vector * matrix
  def multiply_vec_mat(vec, mat)
    raise ArgumentError, "Matrix cannot be empty" if mat.empty?
    # CORRECTED CHECK: Compare vector size to matrix ROWS
    raise ArgumentError, "Vector size #{vec.size} != Matrix rows #{mat.size}" if vec.size != mat.size

    num_cols_out = mat[0].size
    result = Array.new(num_cols_out, 0.0)
    num_cols_out.times do |j|
      sum = 0.0
      vec.size.times do |i|
        sum += vec[i] * mat[i][j]
      end
      result[j] = sum
    end
    result
  end

  # matrix * vector (column vector assumed)
  def multiply_mat_vec(mat, vec)
    raise ArgumentError, "Matrix columns #{mat[0].size} != Vector size #{vec.size}" if mat.empty? || mat[0].size != vec.size
    result = Array.new(mat.size, 0.0)
    mat.size.times do |i|
      result[i] = dot_product(mat[i], vec)
    end
    result
  end

  def mat_addition(mat1, mat2)
    mat1.map.with_index do |row, r_idx|
      add_vectors(row, mat2[r_idx])
    end
  end

  # outer product: vec1 (col) * vec2 (row) -> matrix
  def outer_product(vec1, vec2)
    vec1.map do |v1_elem|
      vec2.map { |v2_elem| v1_elem * v2_elem }
    end
  end

  def transpose(mat)
    return [] if mat.empty?
    num_rows = mat.size
    num_cols = mat[0].size
    Array.new(num_cols) { |j| Array.new(num_rows) { |i| mat[i][j] } }
  end

  def add_vectors(vec1, vec2)
    raise ArgumentError, "Vectors must be the same size #{vec1.size} != #{vec2.size}" if vec1.size != vec2.size
    vec1.zip(vec2).map { |a, b| a + b }
  end

  def subtract_vectors(vec1, vec2)
    raise ArgumentError, "Vectors must be the same size #{vec1.size} != #{vec2.size}" if vec1.size != vec2.size
    vec1.zip(vec2).map { |a, b| a - b }
  end

  def multiply_elementwise(vec1, vec2)
    vec1.zip(vec2).map { |a, b| a * b }
  end

  def scalar_multiply(scalar, vec)
    vec.map { |x| scalar * x }
  end

  def tanh(vec)
    vec.map { |x| CMath.tanh(x).real } # Use CMath for tanh, take real part
  end

  # Derivative of tanh: 1 - tanh(x)^2
  def dtanh(tanh_output_vec)
    tanh_output_vec.map { |y| 1.0 - (y**2) }
  end

  def softmax(vec)
    # Subtract max for numerical stability
    max_val = vec.max || 0.0
    exps = vec.map { |x| Math.exp(x - max_val) }
    sum_exps = exps.sum
    return vec.map { |_| 1.0 / vec.size } if sum_exps == 0 # Handle edge case
    exps.map { |e| e / sum_exps }
  end

  # Get the derivative of softmax (for the attention layer)
  #
  # Calculates the gradient of the Loss w.r.t. the *input* of the softmax function (scores/logits)
  # given the gradient w.r.t. the *output* (dL/dAlpha) and the output itself (Alpha).
  # Uses the Jacobian formulation: dL/dScore_k = sum_j (dL/dAlpha_j * Alpha_j * (delta_jk - Alpha_k))
  #
  # Args:
  #   d_output: Gradient of the Loss w.r.t. the softmax output vector (dL/dAlpha). Array[Float].
  #   output:   The output vector of the softmax function (Alpha). Array[Float].
  #
  # Returns:
  #   Gradient of the Loss w.r.t. the softmax input vector (dL/dScores). Array[Float].
  def dsoftmax(d_output, output)
    raise ArgumentError, "d_output size != output size" if d_output.size != output.size
    num_classes = output.size
    d_input = Array.new(num_classes, 0.0)

    num_classes.times do |k|
      sum_k = 0.0
      d_output.zip(output).each_with_index do |(d_alpha_j, alpha_j), j|
        delta_jk = (j == k) ? 1.0 : 0.0
        sum_k += d_alpha_j * alpha_j * (delta_jk - output[k]) # output[k] is alpha_k
      end
      d_input[k] = sum_k
    end

    d_input
  end

  # score_i = vec2^T * vec1_i
  def scalar_field(mat, vec)
    mat.map { |vi| dot_product(vec, vi) }
  end

  def weighted_sum(mat, vec)
    raise ArgumentError, "Vectors must be the same size #{mat.size} != #{vec.size}" if mat.size != vec.size
    raise ArgumentError, "Matrix must not be empty" if mat.empty?
    resp = Array.new(mat.first.size, 0.0)
    mat.zip(vec).each do |row, weight|
      scaled_row = scalar_multiply(weight, row)
      resp = add_vectors(resp, scaled_row)
    end
    resp
  end

  # NEW LIN ALG

  # Standard Matrix Multiplication: C = A * B
  # A: m x n, B: n x p => C: m x p
  def multiply_mat_mat(mat_a, mat_b)
    m = mat_a.size
    n_a = mat_a[0].size
    n_b = mat_b.size
    p = mat_b[0].size
    raise ArgumentError, "Matrix dimensions mismatch for multiplication" if n_a != n_b

    result = Array.new(m) { Array.new(p, 0.0) }
    m.times do |i|
      p.times do |j|
        sum = 0.0
        n_a.times do |k|
          sum += mat_a[i][k] * mat_b[k][j]
        end
        result[i][j] = sum
      end
    end
    result
  end

  # Multiply matrix by scalar
  def matrix_scalar_multiply(mat, scalar)
    mat.map { |row| scalar_multiply(scalar, row) }
  end

  # Add bias vector to each row of a matrix
  def add_bias_to_matrix(mat, bias_vec)
    raise ArgumentError, "Matrix columns must match bias vector size" if !mat.empty? && mat[0].size != bias_vec.size
    mat.map { |row| add_vectors(row, bias_vec) }
  end

  # Apply ReLU element-wise to a vector
  def relu(vec)
    vec.map { |x| [0.0, x].max }
  end

  # Apply ReLU element-wise to each row of a matrix
  def matrix_relu(mat)
    mat.map { |row| relu(row) }
  end

  # Apply Softmax row-wise to a matrix
  def matrix_softmax(mat)
    mat.map { |row| softmax(row) }
  end

  # Add two matrices element-wise (already exists as mat_addition)
  # def add_matrices(mat1, mat2) ...
  #
  # Layer Normalization (operates on rows independently)
  # Input: Matrix (e.g., [seq_len, features])
  # Gamma, Beta: Vectors of size 'features'
  def layer_norm(mat, gamma, beta, epsilon = 1e-5)
    raise ArgumentError, "Gamma/Beta size must match matrix columns" if !mat.empty? && mat[0].size != gamma.size
    mat.map do |row|
      # Calculate mean and variance across the feature dimension (the row itself)
      mean = row.sum / row.size.to_f
      variance = row.sum { |x| (x - mean)**2 } / row.size.to_f
      std_dev = Math.sqrt(variance + epsilon)

      # Normalize the row
      normalized_row = row.map { |x| (x - mean) / std_dev }

      # Scale and shift
      add_vectors(multiply_elementwise(normalized_row, gamma), beta)
    end
  end

  # Mean Pooling: Average vectors across rows of a matrix
  def mean_pool(mat)
    return [] if mat.empty?
    num_rows = mat.size
    num_cols = mat[0].size
    pooled = Array.new(num_cols, 0.0)
    mat.each do |row|
        pooled = add_vectors(pooled, row)
    end
    scalar_multiply(1.0 / num_rows, pooled)
  end
end


class NNLM
  include BasicLinAlg

  attr_reader :word_to_ix, :ix_to_word, :vocab_size

  def initialize(embedding_dim:, context_size:, hidden_size:, learning_rate: 0.01, transform_ff_dim: nil)
    @embedding_dim = embedding_dim
    @context_size = context_size # Number of preceding words (n-1 grams for predicting nth)
    @hidden_size = hidden_size

    @transformer_ff_dim = transformer_ff_dim || @hidden_size * 4 # Common heuristic

    # Gradient Stability: Ensure the learning rate isn't too high.
    # Complex interactions can lead to exploding/vanishing gradients.
    @learning_rate = learning_rate

    # Placeholders - vocabulary needs to be built first
    @vocab_size = 0
    @word_to_ix = {}
    @ix_to_word = []

    # Parameters - will be initialized after vocab is built
    @embeddings = nil # Hash { word_ix => Array[Float] }
    @W_h = nil # Hidden layer weights: attn_hidden_dim x hidden_size
    @b_h = nil # Hidden layer biases: hidden_size
    @W_o = nil # Output layer weights: hidden_size x vocab_size
    @b_o = nil # Output layer biases: vocab_size

    # Transformer Parameters (will be initialized later)
    # Self-Attention (Single Head for simplicity)
    @W_Q = nil # Query weights: embedding_dim x embedding_dim
    @W_K = nil # Key weights:   embedding_dim x embedding_dim
    @W_V = nil # Value weights: embedding_dim x embedding_dim
    # Layer Normalization 1
    @ln1_gamma = nil # Scale: embedding_dim
    @ln1_beta = nil  # Shift: embedding_dim
    # Feed-Forward Network
    @W_ff1 = nil # embedding_dim x transformer_ff_dim
    @b_ff1 = nil # transformer_ff_dim
    @W_ff2 = nil # transformer_ff_dim x embedding_dim
    @b_ff2 = nil # embedding_dim
    # Layer Normalization 2
    @ln2_gamma = nil # Scale: embedding_dim
    @ln2_beta = nil  # Shift: embedding_dim
  end

  def build_vocabulary(training_dir)
    tokens = get_input(training_dir).flatten.uniq

    puts "Building vocabulary..."
    @ix_to_word = tokens
    @word_to_ix = @ix_to_word.each_with_index.to_h
    @vocab_size = @ix_to_word.size

    puts "Vocabulary size: #{@vocab_size}"
    _initialize_parameters()
  end

  def _initialize_parameters
    puts "Initializing parameters..."
    input_concat_size = @context_size * @embedding_dim

    # Embedding Matrix C (represented as a Hash lookup)
    @embeddings = Hash.new do |h, k|
      # Default init for unknown words encountered later (should ideally not happen if vocab is fixed)
      h[k] = Array.new(@embedding_dim) { (rand * 0.1) - 0.05 }
    end
    @vocab_size.times do |i|
      @embeddings[i] = Array.new(@embedding_dim) { (rand * 0.1) - 0.05 }
    end
    # Ensure PAD embedding is zero? Often helpful.
    @embeddings[@word_to_ix["[PAD]"]] = Array.new(@embedding_dim, 0.0)

    # Hidden Layer Weights/Biases
    @W_h = Array.new(input_concat_size) { Array.new(@hidden_size) { (rand * 0.1) - 0.05 } }
    @b_h = Array.new(@hidden_size) { (rand * 0.1) - 0.05 }

    # Output Layer Weights/Biases
    @W_o = Array.new(@hidden_size) { Array.new(@vocab_size) { (rand * 0.1) - 0.05 } }
    @b_o = Array.new(@vocab_size) { (rand * 0.1) - 0.05 }

      grad_W_attn: @nnlm.zeros_matrix(@attn_hidden_dim, @embedding_dim),
      grad_b_attn: @nnlm.zeros_vector(@attn_hidden_dim),
      grad_v_attn: @nnlm.zeros_vector(@attn_hidden_dim),

    # Xavier/Glorot initialization is common, but using simple random for now
    init_scale = 0.05 # Keep consistent with others for now

    # Self-Attention Weights (emb_dim x emb_dim)
    @W_Q = Array.new(@embedding_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }
    @W_K = Array.new(@embedding_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }
    @W_V = Array.new(@embedding_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }

    # Layer Normalization Parameters (often initialized to gamma=1, beta=0)
    @ln1_gamma = Array.new(@embedding_dim, 1.0)
    @ln1_beta = Array.new(@embedding_dim, 0.0)
    @ln2_gamma = Array.new(@embedding_dim, 1.0)
    @ln2_beta = Array.new(@embedding_dim, 0.0)

    # Feed-Forward Network Weights/Biases
    # Layer 1: emb_dim x ff_dim
    @W_ff1 = Array.new(@embedding_dim) { Array.new(@transformer_ff_dim) { (rand * 2 * init_scale) - init_scale } }
    @b_ff1 = Array.new(@transformer_ff_dim, 0.0) # Initialize biases to zero
    # Layer 2: ff_dim x emb_dim
    @W_ff2 = Array.new(@transformer_ff_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }
    @b_ff2 = Array.new(@embedding_dim, 0.0) # Initialize biases to zero

    puts "Parameter initialization complete."
  end


  # --- Forward Pass ---
  # O(C*E*H + H*V) => ContextSize * EmbeddingDim * HiddenSize + HiddenSize * VocabSize
  def forward(context_indices)
    # Get the inputs represented by the words in our context
    context_embeddings = context_embeddings(context_indices)

    # 2. Apply Attention
    transformer_output = transformer_layer(context_embeddings) # Shape: [context_size, embedding_dim]

    # Aggregate the sequence output (e.g., mean pooling)
    aggregated_context = mean_pool(transformer_output)

    # 3. Hidden Layer (operates on the single context_vector)
    # hidden_input now takes context_vector (size embedding_dim)
    # W_h must be shape: embedding_dim x hidden_size
    #
    # Get the raw nueral network signals for the input
    # Then, apply tanh to keep values between -1 and 1
    # Tanh results have other desirable features we use later in `backward`
    hidden_activation = activate(weigh_and_bias(aggregated_context, @W_h, @b_h))

    # 4. Output Layer
    # Apply Softmax / Probability Calculation: Convert raw scores to proper probabilities
    # Transform the scores into proper probabilities that sum to 1
    probabilities = softmax(weigh_and_bias(hidden_activation, @W_o, @b_o)) 

    # Softmax does three things:
    # - Makes all values positive (using exponential function)
    # - Amplifies differences (higher scores become much more likely)
    # - Normalizes everything to sum to 1 (creates a valid probability distribution)
    # Result: [0.01, 0.02, 0.8, 0.15, 0.02] = 80% chance the 3rd word comes next

    # Return values needed for backpropagation
    {
      probabilities: probabilities,
      hidden_activation: hidden_activation,
      aggregated_context: aggregated_context,
      context_embeddings: context_embeddings
      # TODO: transformer intermediate values needed for backprop here...
      # e.g., transformer_q: q, transformer_k: k, transformer_v: v, transformer_attn_weights: attention_weights, ...
    }
  end

  # 1. Projection Layer: Look up and concatenate embeddings
  # Think of embeddings as a dictionary where each word has a unique "meaning vector"
  #
  # Example: If context_indices = [42, 15] (representing "the cat")
  # And embeddings = { 42 => [0.1, 0.2], 15 => [0.3, 0.4] }
  # Then input_layer = [0.1, 0.2, 0.3, 0.4]
  #
  # The `inputs` are the way a series of words can be expressed to the Neural Network
  #
  # In some networks, the dimensions for each word are caculated carefully, and not updated.
  # In this network, the dimensions are initalized randomly, and mutated as the network learns.
  #
  # If you think of the MNIST Handwriting example, each hand-written character is
  # represented by 28x28 pixel images (784 inputs).
  # 
  # Here, we are doing something very similar. Think of each word as a row.
  # Each word has dimenssions (embedding_dim).
  # We end up with context_size x embedding_dims pieces of data.
  # If we had 28 words in the context, and each word had 28 dimensions,
  # It would be similar to feeding 28x28 pixel images into a neural network.
  def context_embeddings(context_indices)
    context_indices
      .map { |ix| @embeddings[ix] }
  end

  # Forward pass for a single, simple Transformer block
  def transformer_layer(input_embeddings)
    # input_embeddings shape: [context_size, embedding_dim]
    seq_len = input_embeddings.size
    emb_dim = @embedding_dim # Assuming input_embeddings has correct dimension

    # --- 1. Self-Attention ---
    # Project inputs to Q, K, V
    # Input (seq_len x emb_dim) * Weights (emb_dim x emb_dim) -> Output (seq_len x emb_dim)
    q = multiply_mat_mat(input_embeddings, @W_Q)
    k = multiply_mat_mat(input_embeddings, @W_K)
    v = multiply_mat_mat(input_embeddings, @W_V)

    # Calculate Attention Scores: Q * K^T
    # Q (seq_len x emb_dim) * K^T (emb_dim x seq_len) -> Scores (seq_len x seq_len)
    scores = multiply_mat_mat(q, transpose(k))

    # Scale scores
    scale_factor = 1.0 / Math.sqrt(emb_dim)
    scaled_scores = matrix_scalar_multiply(scores, scale_factor)

    # Apply Softmax row-wise to get attention weights
    attention_weights = matrix_softmax(scaled_scores) # Shape: [seq_len, seq_len]

    # Calculate weighted sum of Values: AttnWeights * V
    # AttnWeights (seq_len x seq_len) * V (seq_len x emb_dim) -> AttnOutput (seq_len x emb_dim)
    attn_output = multiply_mat_mat(attention_weights, v)

    # --- 2. Add & Norm 1 ---
    # Residual connection
    residual1_input = add_matrices(input_embeddings, attn_output)
    # Layer Normalization
    norm1_output = layer_norm(residual1_input, @ln1_gamma, @ln1_beta)

    # --- 3. Feed-Forward Network ---
    # First linear layer + ReLU
    # Input (seq_len x emb_dim) * W_ff1 (emb_dim x ff_dim) -> (seq_len x ff_dim)
    ff1_output = multiply_mat_mat(norm1_output, @W_ff1)
    ff1_biased = add_bias_to_matrix(ff1_output, @b_ff1)
    ff1_activated = matrix_relu(ff1_biased) # Shape: [seq_len, ff_dim]

    # Second linear layer
    # Input (seq_len x ff_dim) * W_ff2 (ff_dim x emb_dim) -> (seq_len x emb_dim)
    ff2_output = multiply_mat_mat(ff1_activated, @W_ff2)
    ff_output = add_bias_to_matrix(ff2_output, @b_ff2) # Shape: [seq_len, emb_dim]

    # --- 4. Add & Norm 2 ---
    # Residual connection (add input of FFN, which is norm1_output)
    residual2_input = add_matrices(norm1_output, ff_output)
    # Layer Normalization
    transformer_block_output = layer_norm(residual2_input, @ln2_gamma, @ln2_beta)

    # TODO: Store intermediate values needed for backpropagation if implementing it:
    # q, k, v, attention_weights, norm1_output, ff1_activated, etc.
    # For now, just return the final output for the forward pass integration.
    transformer_block_output # Shape: [context_size, embedding_dim]
  end

  # input: vector: shape => n
  # weights: matrix: shape => n x m
  # biases: vector: shape => m
  def weigh_and_bias(input_vector, weight_matrix, bias_vector)
    add_vectors(
      multiply_vec_mat(input_vector, weight_matrix), # Multiply -> Transform: Apply weights to meaningfully amplify input to each neuron
      bias_vector) # Add -> Adjust for the baseline preference of each neuron
  end

  def reverse_weigh(vector, weights)
    multiply_vec_mat(vector, transpose(weights))
  end

  # The tanh function adds non-linearity, allowing the network to learn complex patterns
  # Without this, we'd only capture simple linear relationships between words
  #
  # The hidden input is a simple linear combination
  # Imagine trying to predict whether someone will like a movie
  # The `features` of the movie could be percentage Action, and percentage Romance:
  # (Linear) Prediction = (0.7 × Action) + (0.3 × Romance)
  #
  # Here, 0.7 and 0.3 represent the hidden weights
  # If a movie is 50% Action and 50% Romance, with a simple linear combination,
  # we would get:
  #
  # Prediction = (0.7 x 0.5) + (0.3 x 0.5) = 0.35 + 0.15 = 0.5
  #
  # But perhaps things aren't this simple / linear. We need to curve the scores with tanh.
  # Movies that are 50/50 may not be liked, but movies that are closer to 100 Romance
  # or 100 Action may be liked.
  #
  # tanh(10,0) = 0.96, (pure action)
  # tanh(0,10) = -0.96, (pure romance)
  # tanh(5,5) = 0, (50/50 action/romance)
  def activate(context_indices)
    tanh(context_indices)
  end

  # To deactivate a signal, we need to get the derivative of the input
  # WRT the loss function.
  #
  # The `input` here was what originally was `activated` / the signal
  # that went into dtanh.
  #
  # By definition, the derivative of `tanh` is `dtanh`.
  def deactivate(d_vec, o_vec)
    multiply_elementwise(d_vec, dtanh(o_vec))
  end

  # --- Backward Pass (Backpropagation) ---
  # O(C*E*H + H*V)
  def backward(context_indices, target_index, forward_pass_data)
    probabilities = forward_pass_data[:probabilities]
    hidden_activation = forward_pass_data[:hidden_activation]
    context_vector = forward_pass_data[:context_vector]
    attention_weights = forward_pass_data[:attention_weights]
    projected_embeddings = forward_pass_data[:projected_embeddings]
    context_embeddings = forward_pass_data[:context_embeddings] # List of vectors

    # Initialize gradients (matching parameter structures)
    grad_embeddings = Hash.new { |h, k| h[k] = Array.new(@embedding_dim, 0.0) }
    Array.new(@W_h.size) { Array.new(@hidden_size, 0.0) }
    Array.new(@hidden_size, 0.0)
    Array.new(@hidden_size) { Array.new(@vocab_size, 0.0) }
    Array.new(@vocab_size, 0.0)

    # W_attn is attn_hidden_dim x embedding_dim
    grad_W_attn = Array.new(@attn_hidden_dim) { Array.new(@embedding_dim, 0.0) }
    grad_b_attn = Array.new(@attn_hidden_dim, 0.0)
    grad_v_attn = Array.new(@attn_hidden_dim, 0.0)

    # 1. Calculate the main error signal: "How wrong was our prediction?"
    # This is remarkably simple: subtract 1 from the probability of the correct word
    # Example: If we predicted [0.1, 0.2, 0.7] but target_index was 0
    # Then error is [0.1-1, 0.2, 0.7] = [-0.9, 0.2, 0.7]
    # This means: "Increase probability of word 0, decrease probability of words 1 and 2"
    d_output_scores = probabilities.dup
    d_output_scores[target_index] -= 1.0

    # 2. Calculate how to adjust output layer weights
    # For each connection between hidden layer and output layer:
    # - If hidden value was strong AND error was large, make a big adjustment
    # - If either was small, make a smaller adjustment
    grad_W_o = outer_product(hidden_activation, d_output_scores)
    grad_b_o = d_output_scores # Bias gradient is just the error signal

    # 3. Send the error signal backwards to the hidden layer
    # "How much did each hidden neuron contribute to our mistakes?"
    # We multiply the error by the output weights to find out
    d_hidden_input_signal = reverse_weigh(d_output_scores, @W_o) # Calculate dL/dHiddenActivation * dHiddenActivation/dHiddenInput part 1

    # 4. Account for the tanh function we used
    # Since tanh squished values, we need to "unsquish" the error signal
    # This uses the derivative of tanh: 1 - (activation)²
    # Example: If activation was 0.8, derivative is 1-(0.8)² = 0.36
    # This means neurons closer to 0 can change more than those near -1 or 1
    d_hidden_input = deactivate(d_hidden_input_signal, hidden_activation) 

    # 5. Calculate how to adjust hidden layer weights
    # Similar to step 2, but for the connections between hidden and attention layers
    # Input was context_vector, W_h is emb_dim x hidden_size
    grad_W_h = outer_product(context_vector, d_hidden_input) # context_vector is row, d_hidden_input is col -> ok
    grad_b_h = d_hidden_input

    # 6. Backprop error to the context vector (dL/dContextVector)
    # dL/dContextVector = dL/dHiddenInput * dHiddenInput/dContextVector
    # dHiddenInput/dContextVector = W_h^T
    # Need W_h^T which is hidden_size x embedding_dim
    d_context_vector = reverse_weigh(d_hidden_input, @W_h) # Result size: embedding_dim

    attn_gradients = backward_attention(context_embeddings, projected_embeddings, d_context_vector, attention_weights)
    d_embeddings = attn_gradients[:d_embeddings]
    attn_gradients.delete(:d_embeddings)

    # 7. Accumulate gradients for the original embeddings
    context_indices.each_with_index do |word_ix, i|
      grad_embeddings[word_ix] = add_vectors(grad_embeddings[word_ix], d_embeddings[i])
    end

    # Return all gradients
    attn_gradients.merge({
      grad_embeddings: grad_embeddings,
      grad_W_h: grad_W_h, grad_b_h: grad_b_h,
      grad_W_o: grad_W_o, grad_b_o: grad_b_o
    })
  end

  # --- TRANSFORMER BACKPROPAGATION ---
  def backward_transformer()
    {}
  end

  # --- Parameter Update ---
  def update_parameters(gradients)
    # Update Embeddings
    gradients[:grad_embeddings].each do |word_ix, grad|
      @embeddings[word_ix] = subtract_vectors(@embeddings[word_ix], scalar_multiply(@learning_rate, grad))
    end

    lr = @learning_rate
    @W_Q = @W_Q.map.with_index { |r, i| subtract_vectors(r, scalar_multiply(lr, gradients[:grad_W_Q][i])) }
    @W_K = @W_K.map.with_index { |r, i| subtract_vectors(r, scalar_multiply(lr, gradients[:grad_W_K][i])) }
    @W_V = @W_V.map.with_index { |r, i| subtract_vectors(r, scalar_multiply(lr, gradients[:grad_W_V][i])) }

    @ln1_gamma = subtract_vectors(@ln1_gamma, scalar_multiply(lr, gradients[:grad_ln1_gamma]))
    @ln1_beta = subtract_vectors(@ln1_beta, scalar_multiply(lr, gradients[:grad_ln1_beta]))

    @W_ff1 = @W_ff1.map.with_index { |r, i| subtract_vectors(r, scalar_multiply(lr, gradients[:grad_W_ff1][i])) }
    @b_ff1 = subtract_vectors(@b_ff1, scalar_multiply(lr, gradients[:grad_b_ff1]))
    @W_ff2 = @W_ff2.map.with_index { |r, i| subtract_vectors(r, scalar_multiply(lr, gradients[:grad_W_ff2][i])) }
    @b_ff2 = subtract_vectors(@b_ff2, scalar_multiply(lr, gradients[:grad_b_ff2]))

    @ln2_gamma = subtract_vectors(@ln2_gamma, scalar_multiply(lr, gradients[:grad_ln2_gamma]))
    @ln2_beta = subtract_vectors(@ln2_beta, scalar_multiply(lr, gradients[:grad_ln2_beta]))

    # Update Hidden Layer
    @W_h = @W_h.map.with_index do |row, i|
      subtract_vectors(row, scalar_multiply(@learning_rate, gradients[:grad_W_h][i]))
    end
    @b_h = subtract_vectors(@b_h, scalar_multiply(@learning_rate, gradients[:grad_b_h]))

    # Update Output Layer
    @W_o = @W_o.map.with_index do |row, i|
      subtract_vectors(row, scalar_multiply(@learning_rate, gradients[:grad_W_o][i]))
    end
    @b_o = subtract_vectors(@b_o, scalar_multiply(@learning_rate, gradients[:grad_b_o]))
  end

  # w -> for each word
  # O(3*C*E*H + 3*H*V) => O(3*5*10*20 + 3*20*4096) => O(3000 + 245760)
  # embedding_dim: 10, # Small embedding size
  # context_size: 5,   # Use 2 preceding words (trigrams)
  # hidden_size: 20,   # Small hidden layer
  def process_context(input, i)
    context_indices = input[i...(i + @context_size)]
    target_index = input[i + @context_size]

    # Forward pass
    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    # Calculate Loss (Cross-Entropy) - optional for training but good for monitoring
    loss = -Math.log(probabilities[target_index] + 1e-9) # Add epsilon for numerical stability

    # Backward pass
    gradients = backward(context_indices, target_index, forward_data)

    # Update parameters
    update_parameters(gradients)

    loss
  end

  # --- Training Loop ---
  def train(training_dir, epochs: 10)
    raise "Vocabulary not built!" unless @vocab_size > 0

    padding_ix = @word_to_ix["[PAD]"]
    sentences = get_input(training_dir)

    puts "\nStarting training..."
    epochs.times do |epoch|
      total_loss = 0.0
      example_count = 0

      sentences.each_with_index do |sentence, _s_id|
        # Create context windows and targets
        padded_sentence = Array.new(@context_size, padding_ix) + encode(sentence)
        (padded_sentence.size - @context_size).times do |i|
          total_loss += process_context(padded_sentence, i)
          example_count += 1
        end
      end
      avg_loss = example_count > 0 ? total_loss / example_count : 0
      puts "Epoch #{epoch + 1}/#{epochs}, Average Loss: #{avg_loss.round(4)}, Perplexity: #{(Math::E**avg_loss).round(4)}"
    end
    puts "Training finished."
  end

  # --- Prediction ---
  def predict_next_word(prompt)
    raise "Vocabulary not built!" unless @vocab_size > 0

    context_indices = tokenize(prompt)
    puts "CONTEXT INDICES: #{context_indices}"
    raise ArgumentError, "Context size mismatch" unless context_indices.size == @context_size

    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    # Find the index with the highest probability
    predicted_index = probabilities.each_with_index.max_by { |prob, _ix| prob }[1]

    @ix_to_word[predicted_index]
  end

  def save_model(filepath)
    puts "Saving model to #{filepath}..."
    model_data = {
      # Hyperparameters
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      attn_hidden_dim: @attn_hidden_dim,
      vocab_size: @vocab_size,

      # Vocabulary
      word_to_ix: @word_to_ix,
      ix_to_word: @ix_to_word,

      # Parameters
      embeddings: @embeddings,
      W_h: @W_h,
      b_h: @b_h,
      W_o: @W_o,
      b_o: @b_o,
      # TODO: SAVE TRANSFORMER 
    }

    begin
      File.open(filepath, "wb") do |file|
        MessagePack.pack(model_data, file)
      end
      puts "Model saved successfully."
    rescue => e
      puts "Error saving model: #{e.message}"
    end
  end

  def self.load_model(filepath)
    puts "Loading model from #{filepath}..."
    begin
      packed_data = File.binread(filepath)
      model_data = MessagePack.unpack(packed_data)

      # 1. Create a new instance with saved hyperparameters
      loaded_model = NNLM.new(
        embedding_dim: model_data["embedding_dim"],
        context_size: model_data["context_size"],
        hidden_size: model_data["hidden_size"],
        attn_hidden_dim: model_data["attn_hidden_dim"]
        # learning_rate is not needed for loading/inference, can use default
      )

      # 2. Load the state into the new instance
      loaded_model.instance_variable_set(:@vocab_size, model_data["vocab_size"])
      loaded_model.instance_variable_set(:@word_to_ix, model_data["word_to_ix"])
      loaded_model.instance_variable_set(:@ix_to_word, model_data["ix_to_word"])
      loaded_model.instance_variable_set(:@embeddings, model_data["embeddings"])
      loaded_model.instance_variable_set(:@W_h, model_data["W_h"])
      loaded_model.instance_variable_set(:@b_h, model_data["b_h"])
      loaded_model.instance_variable_set(:@W_o, model_data["W_o"])
      loaded_model.instance_variable_set(:@b_o, model_data["b_o"])
      # TODO: LOAD TRANSFORMER

      puts "Model loaded successfully."
      loaded_model # Return the rehydrated model object
    rescue => e
      puts "Error loading model: #{e.message}"
      nil
    end
  end

  def get_files(training_dir)
    Dir
      .foreach(training_dir)
      .to_a
      .reject { |p| File.basename(p).start_with?(".") }
      .map { |p| File.join(training_dir, p) }
  end


  def get_input(training_dir)
    files = get_files(training_dir)
    puts "TRAINING ON THESE FILES: #{files}"

    files
      .map do |f|
        if f == "." || f == ".."
          next acc
        end
        tokenize(File.read(f))
      end
  end

  def tokenize(str)
    str
      .downcase
      .chars
      .map do |c|
        if c == "“" || c == "”"
          c = "\""
        elsif c == "’" || c == "‘"
          c = "'"
        elsif DASHES.include?(c)
          c = "-"
        elsif c == "—"
          c = "-"
        elsif c == "{" || c == "["
          c = "("
        elsif c == "}" || c == "]"
          c = ")"
        elsif c == "€"
          c = "$"
        end
        c
      end
      .join("")
      .gsub(/([a-z])'([a-z])/, '\1 \' \2') # handle contractions
      .gsub(/([.,!?;:()\[\]{}"'…`_-])/, ' \1 ') # handle punctation
      .gsub(/(\d) \. (\d)/, '\1.\2') # handle numbers
      .gsub(/\s*\n+\s*/, " #{PARAGRAPH} ")
      .split(/\s+/)[0...500]
  end

  def encode(tokens)
    tokens.map { |t| @word_to_ix[t] }
  end
end
