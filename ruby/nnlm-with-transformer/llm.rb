#! /usr/bin/env ruby
# frozen_string_literal: true

require "set"
require "cmath" # Using CMath just for tanh convenience, could implement manually
require "msgpack"
require 'forwardable' # For easier access to LinAlg methods

DASHES = [150, 151].map(&:chr) # em & en dash
PARAGRAPH = "[PARAGRAPH]"
MODEL_FILE = "model.msgpack"
# TOKEN_FILE = "tokens.json" # Not used in provided snippets

# Helper functions for basic vector/matrix operations on Ruby Arrays
module BasicLinAlg
  # --- Existing Vector/Matrix Ops ---
  def dot_product(vec1, vec2)
    vec1.zip(vec2).sum { |a, b| a * b }
  end

  # vector * matrix (vec: 1 x n, mat: n x m -> result: 1 x m)
  # NOTE: This implementation assumes mat[i][j] access, which is row-major.
  # The calculation effectively performs vec * mat.
  def multiply_vec_mat(vec, mat)
    raise ArgumentError, "Matrix cannot be empty" if mat.empty?
    n = vec.size
    m = mat[0].size
    raise ArgumentError, "Vector size #{n} != Matrix rows #{mat.size}" if n != mat.size

    result = Array.new(m, 0.0)
    m.times do |j|
      sum = 0.0
      n.times do |i|
        sum += vec[i] * mat[i][j]
      end
      result[j] = sum
    end
    result
  end

  # matrix * vector (mat: m x n, vec: n x 1 -> result: m x 1)
  def multiply_mat_vec(mat, vec)
    raise ArgumentError, "Matrix columns #{mat[0].size} != Vector size #{vec.size}" if mat.empty? || mat[0].size != vec.size
    result = Array.new(mat.size, 0.0)
    mat.size.times do |i|
      result[i] = dot_product(mat[i], vec)
    end
    result
  end

  def add_matrices(mat1, mat2) # Renamed from mat_addition for clarity
    raise ArgumentError, "Matrices must have same dimensions for addition" unless mat1.size == mat2.size && (mat1.empty? || mat1[0].size == mat2[0].size)
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
    raise ArgumentError, "Vectors must be the same size #{vec1.size} != #{vec2.size}" if vec1.size != vec2.size
    vec1.zip(vec2).map { |a, b| a * b }
  end

  def scalar_multiply(scalar, vec)
    vec.map { |x| scalar * x }
  end

  def tanh(vec)
    vec.map { |x| CMath.tanh(x).real }
  end

  def dtanh(tanh_output_vec)
    tanh_output_vec.map { |y| 1.0 - (y**2) }
  end

  def softmax(vec)
    max_val = vec.max || 0.0
    exps = vec.map { |x| Math.exp(x - max_val) }
    sum_exps = exps.sum
    return Array.new(vec.size, 1.0 / vec.size) if sum_exps == 0 || sum_exps.nan? || sum_exps.infinite? # Added stability checks
    exps.map { |e| e / sum_exps }
  end

  # --- Softmax Derivative ---
  def dsoftmax(d_output, output) # Gradient w.r.t SOFTMAX INPUT
    raise ArgumentError, "d_output size #{d_output.size} != output size #{output.size}" if d_output.size != output.size
    n = output.size
    d_input = Array.new(n, 0.0)

    # Less common but potentially more stable calculation:
    # dL/dLogit_i = sum_j[ dL/dProb_j * Prob_j * (delta_ij - Prob_i) ]
    # dL/dLogit_i = sum_j[ dL/dProb_j * Prob_j * delta_ij ] - sum_j[ dL/dProb_j * Prob_j * Prob_i ]
    # dL/dLogit_i = dL/dProb_i * Prob_i - Prob_i * sum_j[ dL/dProb_j * Prob_j ]
    # dL/dLogit_i = Prob_i * (dL/dProb_i - sum_j[ dL/dProb_j * Prob_j ])

    weighted_d_output_sum = dot_product(d_output, output) # sum_j[ dL/dProb_j * Prob_j ]

    n.times do |i|
       d_input[i] = output[i] * (d_output[i] - weighted_d_output_sum)
    end
    d_input
  end

  # --- Weighted Sum (Used in old Attention - keep if needed elsewhere, remove if not) ---
  # def weighted_sum(mat, vec) ...

  # --- Matrix Operations added for Transformer ---

  # Standard Matrix Multiplication: C = A * B
  # A: m x n, B: n x p => C: m x p
  def multiply_mat_mat(mat_a, mat_b)
    return [] if mat_a.empty? || mat_b.empty?
    m = mat_a.size
    n_a = mat_a[0].size
    n_b = mat_b.size
    p = mat_b[0].size
    raise ArgumentError, "Matrix dimensions mismatch: A(cols)=#{n_a} != B(rows)=#{n_b}" if n_a != n_b

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

  def matrix_scalar_multiply(mat, scalar)
    mat.map { |row| scalar_multiply(scalar, row) }
  end

  def add_bias_to_matrix(mat, bias_vec)
    return [] if mat.empty?
    raise ArgumentError, "Matrix cols #{mat[0].size} != bias vector size #{bias_vec.size}" if mat[0].size != bias_vec.size
    mat.map { |row| add_vectors(row, bias_vec) }
  end

  def relu(vec)
    vec.map { |x| [0.0, x].max }
  end

  def matrix_relu(mat)
    mat.map { |row| relu(row) }
  end

  # Derivative of ReLU element-wise for a matrix
  def matrix_drelu(d_output_mat, original_input_mat)
     raise ArgumentError, "Matrices must have same dimensions for dReLU" unless d_output_mat.size == original_input_mat.size && (d_output_mat.empty? || d_output_mat[0].size == original_input_mat[0].size)
     d_output_mat.map.with_index do |d_row, r_idx|
       d_row.map.with_index do |d_val, c_idx|
         original_input_mat[r_idx][c_idx] > 0 ? d_val : 0.0
       end
     end
  end

  def matrix_softmax(mat) # Apply softmax row-wise
    mat.map { |row| softmax(row) }
  end

  # Apply dsoftmax row-wise
  def matrix_dsoftmax(d_output_mat, output_mat)
      raise ArgumentError, "Matrices must have same dimensions for matrix_dsoftmax" unless d_output_mat.size == output_mat.size && (d_output_mat.empty? || d_output_mat[0].size == output_mat[0].size)
      d_output_mat.map.with_index do |d_row, r_idx|
          dsoftmax(d_row, output_mat[r_idx])
      end
  end

  # Layer Normalization Forward
  def layer_norm(mat, gamma, beta, epsilon = 1e-5)
    return [[], [], [], []] if mat.empty? # Return empty intermediates if input is empty
    feature_size = mat[0].size
    raise ArgumentError, "Gamma/Beta size #{gamma.size} must match matrix columns #{feature_size}" if feature_size != gamma.size || feature_size != beta.size

    means = []
    variances = []
    normalized_mat = []
    output_mat = []

    mat.each do |row|
      mean = row.sum / feature_size.to_f
      variance = row.sum { |x| (x - mean)**2 } / feature_size.to_f
      std_dev_inv = 1.0 / Math.sqrt(variance + epsilon)

      normalized_row = row.map { |x| (x - mean) * std_dev_inv }
      output_row = add_vectors(multiply_elementwise(normalized_row, gamma), beta)

      means << mean
      variances << variance
      normalized_mat << normalized_row
      output_mat << output_row
    end

    # Return output and values needed for backward pass
    { output: output_mat, normalized_input: normalized_mat, means: means, variances: variances, gamma: gamma, input_mat: mat, epsilon: epsilon }
  end

  # Layer Normalization Backward (Complex!)
  # Calculates dL/dInput, dL/dGamma, dL/dBeta
  def backward_layer_norm(d_output_mat, ln_intermediates)
    normalized_mat = ln_intermediates[:normalized_input]
    means = ln_intermediates[:means]
    variances = ln_intermediates[:variances]
    gamma = ln_intermediates[:gamma]
    input_mat = ln_intermediates[:input_mat]
    epsilon = ln_intermediates[:epsilon]

    n_rows = input_mat.size
    return [zeros_matrix(n_rows, @embedding_dim), zeros_vector(@embedding_dim), zeros_vector(@embedding_dim)] if n_rows == 0 # Handle empty input case

    feature_size = input_mat[0].size

    # Initialize gradients
    d_input_mat = Array.new(n_rows) { Array.new(feature_size, 0.0) }
    d_gamma = Array.new(feature_size, 0.0)
    d_beta = Array.new(feature_size, 0.0)

    # Calculate dGamma and dBeta by summing over the sequence dimension
    n_rows.times do |i|
        d_beta = add_vectors(d_beta, d_output_mat[i])
        d_gamma_row = multiply_elementwise(d_output_mat[i], normalized_mat[i])
        d_gamma = add_vectors(d_gamma, d_gamma_row)
    end

    # Calculate dL/dInput row by row
    n_rows.times do |i|
        d_normalized_input_row = multiply_elementwise(d_output_mat[i], gamma) # dL/dNormInput = dL/dOutput * gamma
        variance = variances[i]
        mean = means[i]
        input_row = input_mat[i]
        std_dev_inv = 1.0 / Math.sqrt(variance + epsilon)

        # dL/dVariance = sum( dL/dNormInput_j * (x_j - mean) * (-0.5) * (var + eps)^(-1.5) )
        d_variance = dot_product(
            d_normalized_input_row,
            input_row.map { |x| (x - mean) * (-0.5) * (std_dev_inv**3) }
        )

        # dL/dMean = sum( dL/dNormInput_j * (-std_dev_inv) ) + dL/dVar * sum( -2 * (x_j - mean) ) / N
        d_mean_term1 = dot_product(d_normalized_input_row, Array.new(feature_size, -std_dev_inv))
        d_mean_term2 = d_variance * input_row.sum { |x| -2.0 * (x - mean) } / feature_size.to_f
        d_mean = d_mean_term1 + d_mean_term2

        # dL/dInput_j = dL/dNormInput_j * std_dev_inv + dL/dVar * (2 * (x_j - mean) / N) + dL/dMean / N
        feature_size.times do |j|
            term1 = d_normalized_input_row[j] * std_dev_inv
            term2 = d_variance * 2.0 * (input_row[j] - mean) / feature_size.to_f
            term3 = d_mean / feature_size.to_f
            d_input_mat[i][j] = term1 + term2 + term3
        end
    end

    { d_input: d_input_mat, d_gamma: d_gamma, d_beta: d_beta }
  end

  # Mean Pooling Forward (already exists)
  # def mean_pool(mat)...

  # Mean Pooling Backward
  def backward_mean_pool(d_pooled_vec, num_rows)
      return [] if num_rows == 0
      row_grad = scalar_multiply(1.0 / num_rows, d_pooled_vec)
      Array.new(num_rows) { row_grad } # Each original row receives the averaged gradient
  end

  # Helper to sum vectors across rows of a matrix (useful for bias gradients)
  def sum_rows(matrix)
      return [] if matrix.empty?
      num_cols = matrix[0].size
      sum_vec = Array.new(num_cols, 0.0)
      matrix.each { |row| sum_vec = add_vectors(sum_vec, row) }
      sum_vec
  end

  # Helper for zeros initialization (useful for gradients)
   def zeros_vector(size)
      Array.new(size, 0.0)
   end

   def zeros_matrix(rows, cols)
      Array.new(rows) { Array.new(cols, 0.0) }
   end

end


class NNLM
  include BasicLinAlg
  extend Forwardable # Allows calling BasicLinAlg methods without 'self.'

  attr_reader :word_to_ix, :ix_to_word, :vocab_size

  # Forward BasicLinAlg methods to the instance
  def_delegators :self, *BasicLinAlg.instance_methods

  def initialize(embedding_dim:, context_size:, hidden_size:, learning_rate: 0.01, transformer_ff_dim: nil)
    @embedding_dim = embedding_dim
    @context_size = context_size # Max sequence length for transformer input
    @hidden_size = hidden_size
    @transformer_ff_dim = transformer_ff_dim || @embedding_dim * 4 # Common heuristic (use emb dim)

    @learning_rate = learning_rate

    @vocab_size = 0
    @word_to_ix = {}
    @ix_to_word = []

    @embeddings = nil
    # IMPORTANT: W_h input is now aggregated_context (embedding_dim), not concatenated embeddings
    @W_h = nil # Hidden layer weights: embedding_dim x hidden_size
    @b_h = nil # Hidden layer biases: hidden_size
    @W_o = nil # Output layer weights: hidden_size x vocab_size
    @b_o = nil # Output layer biases: vocab_size

    # Transformer Parameters
    @W_Q, @W_K, @W_V = nil, nil, nil
    @ln1_gamma, @ln1_beta = nil, nil
    @W_ff1, @b_ff1 = nil, nil
    @W_ff2, @b_ff2 = nil, nil
    @ln2_gamma, @ln2_beta = nil, nil

    # Remove old Attention parameters if they existed in the original init
    # @W_attn = nil
    # @b_attn = nil
    # @v_attn = nil
    # @attn_hidden_dim = nil # No longer needed
  end

  def build_vocabulary(training_dir)
    tokens = get_input(training_dir).flatten.uniq
    # Add PAD token if not present? Assume tokenizer handles it.
    # Ensure PARAGRAPH token is included?
    tokens.unshift("[PAD]") unless tokens.include?("[PAD]") # Ensure PAD is present and index 0
    tokens.uniq! # Remove duplicates if PAD was added

    puts "Building vocabulary..."
    @ix_to_word = tokens
    @word_to_ix = @ix_to_word.each_with_index.to_h
    @vocab_size = @ix_to_word.size

    puts "Vocabulary size: #{@vocab_size}"
    _initialize_parameters()
  end

  def _initialize_parameters
    puts "Initializing parameters..."

    # Embedding Matrix C (represented as a Hash lookup)
    @embeddings = Hash.new do |h, k|
      # Default init for unknown words? Or raise error? Assuming vocab is fixed.
      # h[k] = Array.new(@embedding_dim) { (rand * 0.1) - 0.05 }
      raise "Accessing embedding for unknown word index: #{k}" unless @word_to_ix.value?(k)
    end
    @vocab_size.times do |i|
      @embeddings[i] = Array.new(@embedding_dim) { (rand * 0.1) - 0.05 }
    end
    pad_index = @word_to_ix["[PAD]"]
    @embeddings[pad_index] = Array.new(@embedding_dim, 0.0) if pad_index

    # Hidden Layer Weights/Biases (Input dimension is now embedding_dim)
    @W_h = Array.new(@embedding_dim) { Array.new(@hidden_size) { (rand * 0.1) - 0.05 } }
    @b_h = Array.new(@hidden_size) { (rand * 0.1) - 0.05 }

    # Output Layer Weights/Biases
    @W_o = Array.new(@hidden_size) { Array.new(@vocab_size) { (rand * 0.1) - 0.05 } }
    @b_o = Array.new(@vocab_size) { (rand * 0.1) - 0.05 }

    # --- Initialize Transformer parameters ---
    puts "Initializing Transformer parameters..."
    init_scale = 0.05 # Keep consistent

    # Self-Attention Weights (emb_dim x emb_dim)
    @W_Q = Array.new(@embedding_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }
    @W_K = Array.new(@embedding_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }
    @W_V = Array.new(@embedding_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }

    # Layer Normalization Parameters (gamma=1, beta=0)
    @ln1_gamma = Array.new(@embedding_dim, 1.0)
    @ln1_beta = Array.new(@embedding_dim, 0.0)
    @ln2_gamma = Array.new(@embedding_dim, 1.0)
    @ln2_beta = Array.new(@embedding_dim, 0.0)

    # Feed-Forward Network Weights/Biases
    @W_ff1 = Array.new(@embedding_dim) { Array.new(@transformer_ff_dim) { (rand * 2 * init_scale) - init_scale } }
    @b_ff1 = Array.new(@transformer_ff_dim, 0.0)
    @W_ff2 = Array.new(@transformer_ff_dim) { Array.new(@embedding_dim) { (rand * 2 * init_scale) - init_scale } }
    @b_ff2 = Array.new(@embedding_dim, 0.0)

    # Remove initialization of old attention parameters
    # @W_attn = ...
    # @b_attn = ...
    # @v_attn = ...

    puts "Parameter initialization complete."
  end


  # --- Forward Pass ---
  def forward(context_indices)
    # 1. Get Embeddings
    context_embeddings = context_embeddings(context_indices) # Shape: [context_size, embedding_dim]

    # 2. Apply Transformer Layer
    # This now returns the final output AND the intermediate values needed for backprop
    transformer_data = transformer_layer(context_embeddings)
    transformer_output = transformer_data[:output] # Shape: [context_size, embedding_dim]

    # 3. Aggregate the sequence output
    aggregated_context = mean_pool(transformer_output) # Shape: [embedding_dim]

    # 4. Hidden Layer
    hidden_input_raw = weigh_and_bias(aggregated_context, @W_h, @b_h)
    hidden_activation = activate(hidden_input_raw) # Tanh activation

    # 5. Output Layer
    output_scores = weigh_and_bias(hidden_activation, @W_o, @b_o)
    probabilities = softmax(output_scores)

    # Return all necessary values for backpropagation
    {
      # Final output
      probabilities: probabilities,
      # Hidden/Output layer intermediates
      hidden_activation: hidden_activation,
      hidden_input_raw: hidden_input_raw, # Needed for dtanh
      aggregated_context: aggregated_context, # Input to hidden layer
      # Transformer intermediates (unpack from transformer_data)
      **transformer_data, # Includes :output, :q, :k, :v, etc.
      # Initial input
      context_embeddings: context_embeddings
    }
  end

  # --- Get Embeddings ---
  def context_embeddings(context_indices)
    context_indices.map { |ix| @embeddings.fetch(ix) { raise "Unknown index: #{ix}"} } # Use fetch for safety
  end

  # --- Transformer Layer Forward ---
  def transformer_layer(input_embeddings)
    seq_len = input_embeddings.size
    emb_dim = @embedding_dim

    # --- 1. Self-Attention ---
    q = multiply_mat_mat(input_embeddings, @W_Q)
    k = multiply_mat_mat(input_embeddings, @W_K)
    v = multiply_mat_mat(input_embeddings, @W_V)
    scores = multiply_mat_mat(q, transpose(k))
    scale_factor = 1.0 / Math.sqrt(emb_dim)
    scaled_scores = matrix_scalar_multiply(scores, scale_factor)
    # TODO: Add masking here if needed (e.g., for padding or causal attention)
    attention_weights = matrix_softmax(scaled_scores)
    attn_output = multiply_mat_mat(attention_weights, v)

    # --- 2. Add & Norm 1 ---
    residual1_input = add_matrices(input_embeddings, attn_output)
    ln1_data = layer_norm(residual1_input, @ln1_gamma, @ln1_beta)
    norm1_output = ln1_data[:output]

    # --- 3. Feed-Forward Network ---
    ff1_output = multiply_mat_mat(norm1_output, @W_ff1)
    ff1_biased = add_bias_to_matrix(ff1_output, @b_ff1)
    ff1_activated = matrix_relu(ff1_biased)
    ff2_output = multiply_mat_mat(ff1_activated, @W_ff2)
    ff_output = add_bias_to_matrix(ff2_output, @b_ff2)

    # --- 4. Add & Norm 2 ---
    residual2_input = add_matrices(norm1_output, ff_output)
    ln2_data = layer_norm(residual2_input, @ln2_gamma, @ln2_beta)
    transformer_block_output = ln2_data[:output]

    # Return final output AND all intermediates needed for backprop
    {
      output: transformer_block_output,
      # Self-Attention intermediates
      q: q, k: k, v: v,
      scaled_scores: scaled_scores,
      attention_weights: attention_weights,
      attn_output: attn_output,
      # Add & Norm 1 intermediates
      input_embeddings: input_embeddings, # Needed for residual and W_Q/K/V grads
      residual1_input: residual1_input,
      ln1_intermediates: ln1_data, # Contains normalized_input, mean, var etc. for LN1 backward
      norm1_output: norm1_output, # Output of LN1, input to FFN and Add&Norm2 residual
      # FFN intermediates
      ff1_biased: ff1_biased, # Input to ReLU
      ff1_activated: ff1_activated, # Output of ReLU, input to second linear
      ff_output: ff_output, # Output of FFN block
      # Add & Norm 2 intermediates
      residual2_input: residual2_input,
      ln2_intermediates: ln2_data # Contains intermediates for LN2 backward
    }
  end

  # --- Basic Affine Transformation ---
  def weigh_and_bias(input_vector, weight_matrix, bias_vector)
    add_vectors(multiply_vec_mat(input_vector, weight_matrix), bias_vector)
  end

  # --- Backprop helper for Affine Transformation (calculates dL/dInput) ---
  def reverse_weigh(d_output_vector, weight_matrix) # dL/dOutput * W^T
    multiply_vec_mat(d_output_vector, transpose(weight_matrix))
  end

  # --- Activation (Tanh) ---
  def activate(raw_input_vec)
    tanh(raw_input_vec)
  end

  # --- Backprop through Activation (Tanh) ---
  # Takes dL/dActivationOutput, returns dL/dActivationInput
  def deactivate(d_activation_output, activation_output)
    # Note: The original deactivate used tanh_output (which is activation_output here)
    # And d_vec (which is d_activation_output). Formula is: dL/dInput = dL/dOutput * dtanh(Output)
    multiply_elementwise(d_activation_output, dtanh(activation_output))
  end

  # --- Backward Pass (Main Orchestrator) ---
  def backward(context_indices, target_index, forward_pass_data)
    # Unpack forward pass data for convenience
    probabilities = forward_pass_data[:probabilities]
    hidden_activation = forward_pass_data[:hidden_activation]
    # hidden_input_raw = forward_pass_data[:hidden_input_raw] # If needed for activation backward
    aggregated_context = forward_pass_data[:aggregated_context]
    context_embeddings = forward_pass_data[:context_embeddings]

    # Initialize gradients for all parameters
    grad_W_Q = zeros_matrix(@embedding_dim, @embedding_dim)
    grad_W_K = zeros_matrix(@embedding_dim, @embedding_dim)
    grad_W_V = zeros_matrix(@embedding_dim, @embedding_dim)
    grad_ln1_gamma = zeros_vector(@embedding_dim)
    grad_ln1_beta = zeros_vector(@embedding_dim)
    grad_W_ff1 = zeros_matrix(@embedding_dim, @transformer_ff_dim)
    grad_b_ff1 = zeros_vector(@transformer_ff_dim)
    grad_W_ff2 = zeros_matrix(@transformer_ff_dim, @embedding_dim)
    grad_b_ff2 = zeros_vector(@embedding_dim)
    grad_ln2_gamma = zeros_vector(@embedding_dim)
    grad_ln2_beta = zeros_vector(@embedding_dim)
    grad_W_h = zeros_matrix(@embedding_dim, @hidden_size)
    grad_b_h = zeros_vector(@hidden_size)
    grad_W_o = zeros_matrix(@hidden_size, @vocab_size)
    grad_b_o = zeros_vector(@vocab_size)
    grad_embeddings = Hash.new { |h, k| h[k] = zeros_vector(@embedding_dim) }

    # 1. Gradient w.r.t. Output Scores (dL/dOutputScores)
    d_output_scores = probabilities.dup
    d_output_scores[target_index] -= 1.0 # Gradient of CrossEntropy + Softmax

    # 2. Gradients for Output Layer (W_o, b_o) and dL/dHiddenActivation
    grad_W_o = outer_product(hidden_activation, d_output_scores) # dL/dW = dL/dOutput * ActivationInput^T
    grad_b_o = d_output_scores # dL/dBias = dL/dOutput
    d_hidden_activation = reverse_weigh(d_output_scores, @W_o) # dL/dActivationInput = dL/dOutput * W^T

    # 3. Gradient through Hidden Layer Activation (Tanh) -> dL/dHiddenInputRaw
    d_hidden_input_raw = deactivate(d_hidden_activation, hidden_activation)

    # 4. Gradients for Hidden Layer (W_h, b_h) and dL/dAggregatedContext
    grad_W_h = outer_product(d_hidden_input_raw, aggregated_context) # dL/dW = dL/dOutput * ActivationInput^T
    grad_b_h = d_hidden_input_raw # dL/dBias = dL/dOutput
    d_aggregated_context = reverse_weigh(d_hidden_input_raw, @W_h) # dL/dInput = dL/dOutput * W^T

    # 5. Gradient through Mean Pooling -> dL/dTransformerOutputMatrix
    d_transformer_output_matrix = backward_mean_pool(d_aggregated_context, @context_size)

    # 6. Backpropagation through the Transformer Layer
    transformer_grads = backward_transformer_layer(d_transformer_output_matrix, forward_pass_data)
    # This returns: { d_input:, grad_W_Q:, grad_W_K:, ... }

    # 7. Accumulate gradients for the original embeddings
    d_input_embeddings_matrix = transformer_grads[:d_input]
    context_indices.each_with_index do |word_ix, i|
      # Important: Add gradients, don't overwrite, if index appears multiple times
      grad_embeddings[word_ix] = add_vectors(grad_embeddings[word_ix], d_input_embeddings_matrix[i])
    end

    # Return all computed gradients
    {
      grad_embeddings: grad_embeddings,
      grad_W_h: grad_W_h, grad_b_h: grad_b_h,
      grad_W_o: grad_W_o, grad_b_o: grad_b_o,
      # Merge transformer gradients (excluding the input gradient :d_input)
      **transformer_grads.reject { |k, _v| k == :d_input }
    }
  end

  # --- Backpropagation through Transformer Layer ---
  def backward_transformer_layer(d_transformer_block_output, forward_data)
     # Unpack intermediate values needed from forward pass
     ln2_intermediates = forward_data[:ln2_intermediates]
     residual2_input = forward_data[:residual2_input] # = norm1_output + ff_output
     norm1_output = forward_data[:norm1_output]
     ff_output = forward_data[:ff_output]
     ff1_activated = forward_data[:ff1_activated]
     ff1_biased = forward_data[:ff1_biased]
     ln1_intermediates = forward_data[:ln1_intermediates]
     residual1_input = forward_data[:residual1_input] # = input_embeddings + attn_output
     attn_output = forward_data[:attn_output]
     attention_weights = forward_data[:attention_weights]
     scaled_scores = forward_data[:scaled_scores]
     q, k, v = forward_data[:q], forward_data[:k], forward_data[:v]
     input_embeddings = forward_data[:input_embeddings] # Original input to the layer
     emb_dim = @embedding_dim

     # --- 1. Backprop through Add & Norm 2 ---
     ln2_grads = backward_layer_norm(d_transformer_block_output, ln2_intermediates)
     d_residual2_input = ln2_grads[:d_input]
     grad_ln2_gamma = ln2_grads[:d_gamma]
     grad_ln2_beta = ln2_grads[:d_beta]
     # Gradient flows back to both inputs of the addition
     d_norm1_output_from_res2 = d_residual2_input
     d_ff_output = d_residual2_input

     # --- 2. Backprop through Feed-Forward Network ---
     # Backprop through second linear layer (FFN output -> FF1 activated)
     # dL/dB2 = sum(dL/dOutput) over seq len
     grad_b_ff2 = sum_rows(d_ff_output)
     # dL/dW2 = Input^T * dL/dOutput
     grad_W_ff2 = multiply_mat_mat(transpose(ff1_activated), d_ff_output)
     # dL/dInput = dL/dOutput * W2^T
     d_ff1_activated = multiply_mat_mat(d_ff_output, transpose(@W_ff2))

     # Backprop through ReLU
     d_ff1_biased = matrix_drelu(d_ff1_activated, ff1_biased)

     # Backprop through first linear layer (FF1 biased -> Norm1 output)
     # dL/dB1 = sum(dL/dOutput) over seq len
     grad_b_ff1 = sum_rows(d_ff1_biased)
     # dL/dW1 = Input^T * dL/dOutput
     grad_W_ff1 = multiply_mat_mat(transpose(norm1_output), d_ff1_biased)
     # dL/dInput = dL/dOutput * W1^T
     d_norm1_output_from_ffn = multiply_mat_mat(d_ff1_biased, transpose(@W_ff1))

     # --- 3. Backprop through Add & Norm 1 ---
     # Combine gradients flowing back to norm1_output
     d_norm1_output = add_matrices(d_norm1_output_from_res2, d_norm1_output_from_ffn)

     ln1_grads = backward_layer_norm(d_norm1_output, ln1_intermediates)
     d_residual1_input = ln1_grads[:d_input]
     grad_ln1_gamma = ln1_grads[:d_gamma]
     grad_ln1_beta = ln1_grads[:d_beta]
     # Gradient flows back to both inputs of the addition
     d_input_embeddings_from_res1 = d_residual1_input
     d_attn_output = d_residual1_input

     # --- 4. Backprop through Self-Attention ---
     # Backprop through AttnWeights * V -> dL/dAttnWeights, dL/dV
     # dL/dV = AttnWeights^T * dL/dAttnOutput
     d_v = multiply_mat_mat(transpose(attention_weights), d_attn_output)
     # dL/dAttnWeights = dL/dAttnOutput * V^T
     d_attention_weights = multiply_mat_mat(d_attn_output, transpose(v))

     # Backprop through Softmax -> dL/dScaledScores
     # IMPORTANT: Use matrix_dsoftmax which applies dsoftmax row-wise
     d_scaled_scores = matrix_dsoftmax(d_attention_weights, attention_weights)

     # Backprop through Scaling -> dL/dScores
     scale_factor = 1.0 / Math.sqrt(emb_dim)
     d_scores = matrix_scalar_multiply(d_scaled_scores, scale_factor)

     # Backprop through Q * K^T -> dL/dQ, dL/dK
     # dL/dK = Q^T * dL/dScores^T = Q^T * dScores (dL/dScores is symmetric if calculated right?) <= Check this assumption
     # Let's recalculate using standard formulas:
     # dL/dK = dL/dScores^T * Q = transpose(d_scores) * Q <= Incorrect dim
     # dL/dK = Q^T * d_scores <= Incorrect dim
     # Correct: dL/dK = dL/dScores^T * Q . Need dL/dScores^T which is transpose(d_scores)
     d_k = multiply_mat_mat(transpose(q), transpose(d_scores)) # (emb x seq) * (seq x seq) -> (emb x seq). Needs transpose.
     # Let's rethink: dL/dK_kj = sum_i (dL/dScore_ij * Q_ik) -> dK = Q^T * dScores ?
     d_k = multiply_mat_mat(transpose(q), d_scores) # (emb x seq) * (seq x seq) -> (emb x seq) => This seems right for dK^T. So transpose it.
     d_k = transpose(d_k) # Shape: (seq x emb)

     # dL/dQ = dL/dScores * K
     d_q = multiply_mat_mat(d_scores, k) # (seq x seq) * (seq x emb) -> (seq x emb)

     # Backprop through Q, K, V projections to get dL/dW_Q, dL/dW_K, dL/dW_V and dL/dInputEmbeddings
     # dL/dW = Input^T * dL/dOutputLayer
     grad_W_Q = multiply_mat_mat(transpose(input_embeddings), d_q)
     grad_W_K = multiply_mat_mat(transpose(input_embeddings), d_k)
     grad_W_V = multiply_mat_mat(transpose(input_embeddings), d_v)

     # dL/dInput = dL/dOutputLayer * W^T
     d_input_embeddings_from_q = multiply_mat_mat(d_q, transpose(@W_Q))
     d_input_embeddings_from_k = multiply_mat_mat(d_k, transpose(@W_K))
     d_input_embeddings_from_v = multiply_mat_mat(d_v, transpose(@W_V))

     # Combine gradients flowing back to the input embeddings
     d_input_embeddings = add_matrices(
         d_input_embeddings_from_res1,
         add_matrices(
             d_input_embeddings_from_q,
             add_matrices(d_input_embeddings_from_k, d_input_embeddings_from_v)
         )
     )

     # --- 5. Return all gradients ---
     {
         d_input: d_input_embeddings,
         grad_W_Q: grad_W_Q, grad_W_K: grad_W_K, grad_W_V: grad_W_V,
         grad_ln1_gamma: grad_ln1_gamma, grad_ln1_beta: grad_ln1_beta,
         grad_W_ff1: grad_W_ff1, grad_b_ff1: grad_b_ff1,
         grad_W_ff2: grad_W_ff2, grad_b_ff2: grad_b_ff2,
         grad_ln2_gamma: grad_ln2_gamma, grad_ln2_beta: grad_ln2_beta
     }
  end


  # --- Parameter Update ---
  def update_parameters(gradients)
    lr = @learning_rate

    # Update Embeddings
    gradients.fetch(:grad_embeddings, {}).each do |word_ix, grad|
       @embeddings[word_ix] = subtract_vectors(@embeddings.fetch(word_ix), scalar_multiply(lr, grad))
    end

    # Update Transformer Parameters (Check if gradients exist)
    if gradients[:grad_W_Q]
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
    end

    # Update Hidden Layer
    if gradients[:grad_W_h]
      @W_h = @W_h.map.with_index do |row, i|
        subtract_vectors(row, scalar_multiply(lr, gradients[:grad_W_h][i]))
      end
      @b_h = subtract_vectors(@b_h, scalar_multiply(lr, gradients[:grad_b_h]))
    end

    # Update Output Layer
    if gradients[:grad_W_o]
      @W_o = @W_o.map.with_index do |row, i|
        subtract_vectors(row, scalar_multiply(lr, gradients[:grad_W_o][i]))
      end
      @b_o = subtract_vectors(@b_o, scalar_multiply(lr, gradients[:grad_b_o]))
    end

    # Remove updates for old attention parameters if they were here
  end

  # --- Training Loop ---
  def process_context(input, i)
    # Ensure context indices are valid before proceeding
    context_indices = input[i...(i + @context_size)].map do |idx|
        raise "Invalid index #{idx} in context window" unless @ix_to_word[idx]
        idx
    end
    target_index = input[i + @context_size]
    raise "Invalid target index #{target_index}" unless @ix_to_word[target_index]


    # Forward pass
    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    # Calculate Loss
    prob_target = probabilities[target_index]
    loss = -Math.log([prob_target, 1e-9].max) # Avoid log(0)

    # Backward pass only if loss is finite
    if loss.finite?
        gradients = backward(context_indices, target_index, forward_data)
        # Update parameters
        update_parameters(gradients)
    else
        puts "Warning: Infinite or NaN loss encountered for context #{context_indices.inspect} -> #{target_index}. Skipping update."
        # Optionally return a large loss value or handle differently
        loss = 100.0 # Assign a large finite loss for reporting
    end

    loss
  end

  def train(training_dir, epochs: 10)
    raise "Vocabulary not built!" unless @vocab_size > 0

    padding_ix = @word_to_ix["[PAD]"]
    sentences = get_input(training_dir) # Assumes this returns arrays of tokens

    puts "\nStarting training..."
    epochs.times do |epoch|
      total_loss = 0.0
      example_count = 0
      processed_count = 0

      sentences.each_with_index do |sentence_tokens, s_idx|
        print "\rEpoch #{epoch + 1}/#{epochs}, Sentence #{s_idx + 1}/#{sentences.size}..." # Progress indicator

        encoded_sentence = encode(sentence_tokens) # Convert tokens to indices
        padded_sentence = Array.new(@context_size, padding_ix) + encoded_sentence

        (padded_sentence.size - @context_size).times do |i|
          # Basic check to ensure target is not PAD, unless context is also all PAD
          context_indices = padded_sentence[i...(i + @context_size)]
          target_index = padded_sentence[i + @context_size]
          next if target_index == padding_ix && !context_indices.all? { |ci| ci == padding_ix } # Skip predicting PAD unless context is just PADs

          begin
            loss = process_context(padded_sentence, i)
            total_loss += loss if loss.finite? # Only add finite loss
            example_count += 1 if loss.finite?
          rescue => e
             puts "\nError during processing context #{context_indices} -> #{target_index}: #{e.message}"
             puts e.backtrace.take(5).join("\n")
             # Decide whether to continue or stop training
          end
          processed_count += 1

        end
      end
      avg_loss = example_count > 0 ? total_loss / example_count : Float::INFINITY
      perplexity = example_count > 0 && avg_loss.finite? ? (Math::E**avg_loss) : Float::INFINITY
      print "\r" + " " * 80 + "\r" # Clear line
      puts "Epoch #{epoch + 1}/#{epochs}, Average Loss: #{avg_loss.round(4)}, Perplexity: #{perplexity.round(4)} (#{example_count}/#{processed_count} examples used)"

    end
    puts "Training finished."
  end

  # --- Prediction ---
  def predict_next_word(prompt_tokens) # Takes already tokenized prompt
    raise "Vocabulary not built!" unless @vocab_size > 0
    raise ArgumentError, "Prompt size must equal context size (#{@context_size})" unless prompt_tokens.size == @context_size

    # Encode prompt tokens to indices, handle unknown tokens if necessary
    context_indices = prompt_tokens.map do |token|
        @word_to_ix.fetch(token) { raise "Token '#{token}' not in vocabulary during prediction." }
    end
    # puts "CONTEXT INDICES: #{context_indices}"

    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    predicted_index = probabilities.each_with_index.max_by { |prob, _ix| prob }[1]

    @ix_to_word[predicted_index]
  end


  # --- Save/Load Model ---
  def save_model(filepath)
    puts "Saving model to #{filepath}..."
    model_data = {
      # Hyperparameters
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      transformer_ff_dim: @transformer_ff_dim, # New
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
      # Transformer Parameters
      W_Q: @W_Q, W_K: @W_K, W_V: @W_V,
      ln1_gamma: @ln1_gamma, ln1_beta: @ln1_beta,
      W_ff1: @W_ff1, b_ff1: @b_ff1,
      W_ff2: @W_ff2, b_ff2: @b_ff2,
      ln2_gamma: @ln2_gamma, ln2_beta: @ln2_beta
    }
    # Remove old attention params if they were here
    # model_data.delete(:attn_hidden_dim) # If it was saved before

    begin
      File.open(filepath, "wb") do |file|
        MessagePack.pack(model_data, file)
      end
      puts "Model saved successfully."
    rescue => e
      puts "Error saving model: #{e.message}"
      puts e.backtrace.take(5).join("\n")
    end
  end

  def self.load_model(filepath)
    puts "Loading model from #{filepath}..."
    begin
      packed_data = File.binread(filepath)
      # Ensure MessagePack uses symbol keys if they were used for saving
      # By default, it uses string keys when unpacking. We'll access via strings.
      model_data = MessagePack.unpack(packed_data)

      # Create instance with saved hyperparameters
      loaded_model = NNLM.new(
        embedding_dim: model_data.fetch("embedding_dim"),
        context_size: model_data.fetch("context_size"),
        hidden_size: model_data.fetch("hidden_size"),
        transformer_ff_dim: model_data.fetch("transformer_ff_dim") # New
        # attn_hidden_dim removed
      )

      # Load state
      loaded_model.instance_variable_set(:@vocab_size, model_data.fetch("vocab_size"))
      loaded_model.instance_variable_set(:@word_to_ix, model_data.fetch("word_to_ix"))
      loaded_model.instance_variable_set(:@ix_to_word, model_data.fetch("ix_to_word"))
      loaded_model.instance_variable_set(:@embeddings, model_data.fetch("embeddings"))
      loaded_model.instance_variable_set(:@W_h, model_data.fetch("W_h"))
      loaded_model.instance_variable_set(:@b_h, model_data.fetch("b_h"))
      loaded_model.instance_variable_set(:@W_o, model_data.fetch("W_o"))
      loaded_model.instance_variable_set(:@b_o, model_data.fetch("b_o"))
      # Load Transformer parameters
      loaded_model.instance_variable_set(:@W_Q, model_data.fetch("W_Q"))
      loaded_model.instance_variable_set(:@W_K, model_data.fetch("W_K"))
      loaded_model.instance_variable_set(:@W_V, model_data.fetch("W_V"))
      loaded_model.instance_variable_set(:@ln1_gamma, model_data.fetch("ln1_gamma"))
      loaded_model.instance_variable_set(:@ln1_beta, model_data.fetch("ln1_beta"))
      loaded_model.instance_variable_set(:@W_ff1, model_data.fetch("W_ff1"))
      loaded_model.instance_variable_set(:@b_ff1, model_data.fetch("b_ff1"))
      loaded_model.instance_variable_set(:@W_ff2, model_data.fetch("W_ff2"))
      loaded_model.instance_variable_set(:@b_ff2, model_data.fetch("b_ff2"))
      loaded_model.instance_variable_set(:@ln2_gamma, model_data.fetch("ln2_gamma"))
      loaded_model.instance_variable_set(:@ln2_beta, model_data.fetch("ln2_beta"))

      # Remove loading of old attention params if they were here

      puts "Model loaded successfully."
      loaded_model
    rescue KeyError => e
       puts "Error loading model: Missing key #{e.key}"
       nil
    rescue => e
      puts "Error loading model: #{e.message}"
      puts e.backtrace.take(5).join("\n")
      nil
    end
  end

  # --- Input Processing / Tokenization ---
  def get_files(training_dir)
    Dir.glob(File.join(training_dir, "*")).reject { |p| File.directory?(p) || File.basename(p).start_with?(".") }
  end

  def get_input(training_dir)
    files = get_files(training_dir)
    puts "TRAINING ON THESE FILES: #{files}"
    all_tokens = []
    files.each do |f|
        # Returns array of sentences, where each sentence is array of tokens
        sentences_from_file = tokenize_into_sentences(File.read(f))
        all_tokens.concat(sentences_from_file)
    end
    all_tokens # Should be an array of arrays of tokens
  end

  # Updated Tokenizer to split into sentences based on PARAGRAPH marker
  # and handle punctuation slightly better. Returns array of sentences (arrays of tokens).
  def tokenize_into_sentences(str)
      processed_str = str
         .downcase
         .gsub(/[“”]/, '"')
         .gsub(/[‘’]/, "'")
         .gsub(/[#{DASHES.join}]|—/, "-") # Consolidate dashes
         .gsub(/[{[]/, "(")
         .gsub(/[}\]]/, ")")
         .gsub(/€/, "$")
         .gsub(/([a-z])'([a-z])/, '\1 \' \2') # Split contractions like "don't" -> "don ' t"
         .gsub(/([^\w\s.'-])/, ' \1 ') # Add space around most punctuation
         .gsub(/(\d)\s*\.\s*(\d)/, '\1.\2') # Re-join numbers like 3 . 14
         .gsub(/\s*\n+\s*/, " #{PARAGRAPH} ") # Replace newlines/paragraphs
         .strip

      # Split into sentences/segments based on PARAGRAPH marker
      segments = processed_str.split(PARAGRAPH).map(&:strip).reject(&:empty?)

      # Tokenize each segment
      sentences_as_tokens = segments.map do |segment|
          segment.split(/\s+/) # Split by whitespace
                  .reject(&:empty?)
                  # .take(500) # Limit sentence length if needed, maybe apply globally later
      end
      sentences_as_tokens
  end

  def encode(tokens)
    # Map tokens to indices, handling potential unknowns if vocabulary isn't exhaustive
    tokens.map { |t| @word_to_ix.fetch(t) { @word_to_ix['[UNK]'] } } # Assuming UNK token exists if needed
  end
end

