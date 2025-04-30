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
end


class NNLM
  include BasicLinAlg

  attr_reader :word_to_ix, :ix_to_word, :vocab_size

  def initialize(embedding_dim:, context_size:, hidden_size:, learning_rate: 0.01, attn_hidden_dim: nil)
    @embedding_dim = embedding_dim
    @context_size = context_size # Number of preceding words (n-1 grams for predicting nth)
    @hidden_size = hidden_size

    # By default embedding_dim * context_size, can be smaller or larger.
    # Smaller = contexts will be compressed into smaller representations before being passed to the context layer.
    # Larger = contexts can be expanded before being passed into the context layer.
    @attn_hidden_dim = attn_hidden_dim.nil? ? embedding_dim * context_size : attn_hidden_dim

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

    # Note: The dimensions need care. 
    # If @W_attn is attn_hidden_dim x embedding_dim, multiply_mat_vec(@W_attn, emb) works. 
    # @v_attn must be attn_hidden_dim. 
    # The final context_vector will be embedding_dim.
    @W_attn = nil # Attention layer weights (connections to the attention layer / matrix). Size: embedding_dim x attn_hidden_dim
    @b_attn = nil # Attention layer biases. Size: attn_hidden_dim
    @v_attn = nil # An attention "context" or "query" vector used to score the projected embeddings. Size: attn_hidden_dim
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

    # Initialize attention layer
    @W_attn = Array.new(@attn_hidden_dim) { Array.new(@embedding_dim) { (rand * 0.1) - 0.5 } }
    @b_attn = Array.new(@attn_hidden_dim) { (rand * 0.1) - 0.05 }
    @v_attn = Array.new(@attn_hidden_dim) { (rand * 0.1) - 0.05 }

    # Hidden Layer Weights/Biases
    @W_h = Array.new(input_concat_size) { Array.new(@hidden_size) { (rand * 0.1) - 0.05 } }
    @b_h = Array.new(@hidden_size) { (rand * 0.1) - 0.05 }

    # Output Layer Weights/Biases
    @W_o = Array.new(@hidden_size) { Array.new(@vocab_size) { (rand * 0.1) - 0.05 } }
    @b_o = Array.new(@vocab_size) { (rand * 0.1) - 0.05 }
    puts "Parameter initialization complete."
  end


  # --- Forward Pass ---
  # O(C*E*H + H*V) => ContextSize * EmbeddingDim * HiddenSize + HiddenSize * VocabSize
  def forward(context_indices)
    # Get the inputs represented by the words in our context
    context_embeddings = context_embeddings(context_indices)

    # 2. Apply Attention
    attention_data = attention_layer(context_embeddings)
    context_vector = attention_data[:context_vector]

    # 3. Hidden Layer (operates on the single context_vector)
    # hidden_input now takes context_vector (size embedding_dim)
    # W_h must be shape: embedding_dim x hidden_size
    #
    # Get the raw nueral network signals for the input
    # Then, apply tanh to keep values between -1 and 1
    # Tanh results have other desirable features we use later in `backward`
    hidden_activation = activate(weigh_and_bias(context_vector, @W_h, @b_h))

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
      context_vector: context_vector,
      attention_weights: attention_data[:attention_weights],
      projected_embeddings: attention_data[:projected_embeddings],
      context_embeddings: context_embeddings
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

  def attention_layer(context_embeddings)
    # 1. Project each embedding and apply tanh activation
    projected_embeddings = context_embeddings.map do |emb|
      # hidden = W_attn^T * emb + b_attn (using multiply_mat_vec as W is emb_dim x attn_hidden)
      # We need W_attn to be attn_hidden x emb_dim for mat_vec or transpose it.
      # Let's define W_attn as attn_hidden_dim x embedding_dim
      hidden_attn_input = add_vectors(multiply_mat_vec(@W_attn, emb), @b_attn)
      tanh(hidden_attn_input) # Apply non-linearity
    end
    # projected_embeddings is now [[proj1], [proj2], ...]

    # 2. Calculate attention scores using the context vector v_attn
    attn_scores = scalar_field(projected_embeddings, @v_attn)
    # attn_scores is now [score1, score2, ...]

    # 3. Calculate attention weights using softmax
    attn_weights = softmax(attn_scores)
    # attn_weights is now [alpha1, alpha2, ...]

    # 4. Calculate the context vector (weighted sum of original embeddings)
    context_vector = weighted_sum(context_embeddings, attn_weights)

    # Return context vector and weights (weights needed for backprop)
    { 
      context_vector: context_vector, 
      attention_weights: attn_weights,
      projected_embeddings: projected_embeddings
    }
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

  # --- ATTENTION BACKPROPAGATION ---
  # Need gradients dL/dW_attn, dL/db_attn, dL/dv_attn
  # And need to propagate dL/dContextVector back to dL/dEmbedding_i
  #
  # Let C = context_vector = sum_i(alpha_i * emb_i)
  # Let alpha = softmax(scores)
  # Let scores_i = v_attn^T * projected_emb_i
  # Let projected_emb_i = tanh(W_attn * emb_i + b_attn)
  def backward_attention(context_embeddings, projected_embeddings, d_context_vector, attention_weights)
    # Initialize gradients passed back to each embedding
    d_embeddings = Array.new(context_embeddings.size) { Array.new(@embedding_dim, 0.0) }
    # Initialize gradients for attention parameters for this step
    current_grad_W_attn = Array.new(@attn_hidden_dim) { Array.new(@embedding_dim, 0.0) }
    current_grad_b_attn = Array.new(@attn_hidden_dim, 0.0)
    current_grad_v_attn = Array.new(@attn_hidden_dim, 0.0)

    # Backprop through context_vector calculation: C = sum(alpha_i * emb_i)
    # dL/dalpha_i = dL/dC * dC/dalpha_i = d_context_vector^T * emb_i
    d_alphas = context_embeddings.map { |emb| dot_product(d_context_vector, emb) } # size: context_size

    # dL/demb_i (direct path) = dL/dC * dC/demb_i = d_context_vector * alpha_i
    context_embeddings.each_with_index do |emb, i|
      d_embeddings[i] = add_vectors(d_embeddings[i], scalar_multiply(attention_weights[i], d_context_vector))
    end

    # Backprop through Softmax: alpha = softmax(scores)
    # This requires the softmax Jacobian, but a common pattern is:
    # dL/dscore_k = sum_j (dL/dalpha_j * dalpha_j/dscore_k)
    # dL/dscores = (dL/dalpha - sum(dL/dalpha * alpha)) * alpha (element-wise)
    # Let's compute dL/dscores = d_scores (size: context_size)
    d_scores = dsoftmax(d_alphas, attention_weights)

    # Accumulate attention gradients
    #
    # Backprop through Scores: scores_i = v_attn^T * projected_emb_i
    context_embeddings.each_with_index do |emb, i|
      d_score_i = d_scores[i]
      projected_emb_i = projected_embeddings[i]

      # dL/dv_attn += dL/dscore_i * dscore_i/dv_attn = d_score_i * projected_emb_i
      current_grad_v_attn = add_vectors(current_grad_v_attn, scalar_multiply(d_score_i, projected_emb_i))

      # dL/dprojected_emb_i = dL/dscore_i * dscore_i/dprojected_emb_i = d_score_i * v_attn
      d_projected_emb = scalar_multiply(d_score_i, @v_attn) # size: attn_hidden_dim

      # Backprop through Tanh: projected_emb_i = tanh(attn_hidden_input_i)
      # dL/dattn_hidden_input_i = dL/dprojected_emb_i * dprojected_emb_i/dtanh_input
      d_tanh_input = deactivate(d_projected_emb, projected_emb_i) # size: attn_hidden_dim

      # Backprop through Affine Transform: attn_hidden_input_i = W_attn * emb_i + b_attn
      # dL/db_attn += dL/dattn_hidden_input_i * d(...)/db_attn = d_tanh_input * 1
      current_grad_b_attn = add_vectors(current_grad_b_attn, d_tanh_input)

      # dL/dW_attn += dL/dattn_hidden_input_i * d(...)/dW_attn = d_tanh_input (col) * emb_i (row)
      # outer_product expects vec1 (col), vec2 (row)
      grad_W_attn_i = outer_product(d_tanh_input, emb) # attn_hidden x embedding_dim
      # Need to add matrices element-wise
      current_grad_W_attn = mat_addition(current_grad_W_attn, grad_W_attn_i)

      # dL/demb_i (indirect path) += dL/dattn_hidden_input_i * d(...)/demb_i = d_tanh_input^T * W_attn
      # Need vec-mat: d_tanh_input (row vec) * W_attn (attn_hidden x embedding_dim)
      d_emb_i_indirect = multiply_vec_mat(d_tanh_input, @W_attn) # size: embedding_dim
      d_embeddings[i] = add_vectors(d_embeddings[i], d_emb_i_indirect)
    end

    {
      grad_W_attn: current_grad_W_attn,
      grad_b_attn: current_grad_b_attn,
      grad_v_attn: current_grad_v_attn,
      d_embeddings: d_embeddings
    }
  end

  # --- Parameter Update ---
  def update_parameters(gradients)
    # Update Embeddings
    gradients[:grad_embeddings].each do |word_ix, grad|
      @embeddings[word_ix] = subtract_vectors(@embeddings[word_ix], scalar_multiply(@learning_rate, grad))
    end

    # Update Attention Layer 
    @W_attn = @W_attn.map.with_index do |row, i|
      subtract_vectors(row, scalar_multiply(@learning_rate, gradients[:grad_W_attn][i]))
    end
    @b_attn = subtract_vectors(@b_attn, scalar_multiply(@learning_rate, gradients[:grad_b_attn]))
    @v_attn = subtract_vectors(@v_attn, scalar_multiply(@learning_rate, gradients[:grad_v_attn]))

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
      W_attn: @W_attn,
      b_attn: @b_attn,
      v_attn: @v_attn
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
      loaded_model.instance_variable_set(:@W_attn, model_data["W_attn"])
      loaded_model.instance_variable_set(:@b_attn, model_data["b_attn"])
      loaded_model.instance_variable_set(:@v_attn, model_data["v_attn"])

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
