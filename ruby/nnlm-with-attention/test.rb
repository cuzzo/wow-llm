#! /usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require "minitest/rg"
require "cmath"
require "set"

require_relative "llm"

# Define a helper for comparing floating point arrays/matrices
module Minitest::Assertions
  # Asserts that two arrays (vectors) of floats are element-wise equal within a delta.
  def assert_vector_in_delta(expected_vec, actual_vec, delta = 1e-6, msg = nil)
    msg ||= "Expected vectors to be element-wise equal within delta #{delta}"
    assert_equal(expected_vec.size, actual_vec.size, "#{msg} (different sizes)")
    expected_vec.zip(actual_vec).each_with_index do |(exp, act), i|
      assert_in_delta(exp, act, delta, "#{msg} (difference at index #{i})")
    end
  end

  # Asserts that two nested arrays (matrices) of floats are element-wise equal within a delta.
  def assert_matrix_in_delta(expected_mat, actual_mat, delta = 1e-6, msg = nil)
    msg ||= "Expected matrices to be element-wise equal within delta #{delta}"
    assert_equal(expected_mat.size, actual_mat.size, "#{msg} (different number of rows)")
    expected_mat.each_with_index do |expected_row, i|
      assert_vector_in_delta(expected_row, actual_mat[i], delta, "#{msg} (difference in row #{i})")
    end
  end

  # Asserts that two hashes (like embeddings) have the same keys and corresponding
  # vector values are element-wise equal within a delta.
  def assert_embedding_hash_in_delta(expected_hash, actual_hash, delta = 1e-6, msg = nil)
    msg ||= "Expected embedding hashes to be equal within delta #{delta}"
    assert_equal(expected_hash.keys.sort, actual_hash.keys.sort, "#{msg} (keys differ)")
    expected_hash.each do |key, expected_vec|
      assert(actual_hash.key?(key), "#{msg} (actual hash missing key #{key})")
      assert_vector_in_delta(expected_vec, actual_hash[key], delta, "#{msg} (difference for key #{key})")
    end
  end
end

module BasicLinAlg
  # Creates a vector (Array) filled with zeros.
  #
  # @param size [Integer] The desired length of the vector.
  # @return [Array<Integer>] An array of the specified size filled with 0.
  def zeros_vector(size)
    # Input validation (optional but recommended)
    raise ArgumentError, "Size must be a non-negative integer" unless size.is_a?(Integer) && size >= 0
    
    Array.new(size, 0)
  end
  
  # Creates a vector (Array) filled with ones.
  #
  # @param size [Integer] The desired length of the vector.
  # @return [Array<Integer>] An array of the specified size filled with 1.
  def ones_vector(size)
    # Input validation (optional but recommended)
    raise ArgumentError, "Size must be a non-negative integer" unless size.is_a?(Integer) && size >= 0
    
    Array.new(size, 1)
  end
  
  # Creates a matrix (Array of Arrays) filled with zeros.
  #
  # @param rows [Integer] The desired number of rows.
  # @param cols [Integer] The desired number of columns.
  # @return [Array<Array<Integer>>] A matrix (rows x cols) filled with 0.
  def zeros_matrix(rows, cols)
    # Input validation (optional but recommended)
    raise ArgumentError, "Rows must be a non-negative integer" unless rows.is_a?(Integer) && rows >= 0
    raise ArgumentError, "Cols must be a non-negative integer" unless cols.is_a?(Integer) && cols >= 0
    
    # Use the block form of Array.new to ensure each row is a *new* array instance
    Array.new(rows) { Array.new(cols, 0) }
  end
  
  # Creates a matrix (Array of Arrays) filled with ones.
  #
  # @param rows [Integer] The desired number of rows.
  # @param cols [Integer] The desired number of columns.
  # @return [Array<Array<Integer>>] A matrix (rows x cols) filled with 1.
  def ones_matrix(rows, cols)
    # Input validation (optional but recommended)
    raise ArgumentError, "Rows must be a non-negative integer" unless rows.is_a?(Integer) && rows >= 0
    raise ArgumentError, "Cols must be a non-negative integer" unless cols.is_a?(Integer) && cols >= 0
    
    # Use the block form of Array.new to ensure each row is a *new* array instance
    Array.new(rows) { Array.new(cols, 1) }
  end

  # Creates a one-hot matrix from a list of indices.
  # Each row in the output matrix corresponds to an index from the input list,
  # represented as a one-hot vector.
  #
  # Example:
  #   indices = [0, 2, 1]
  #   num_classes = 3
  #   Result:
  #   [
  #     [1, 0, 0],  # one-hot encoding for index 0
  #     [0, 0, 1],  # one-hot encoding for index 2
  #     [0, 1, 0]   # one-hot encoding for index 1
  #   ]
  #
  # @param indices [Array<Integer>] An array of non-negative integer indices.
  #   Each index specifies the position of the '1' in its corresponding row.
  # @param num_classes [Integer] The total number of classes. This determines the
  #   length (number of columns) of each one-hot vector/row. Must be positive.
  # @return [Array<Array<Integer>>] A matrix (Array of Arrays) where each row
  #   is a one-hot encoding of the corresponding input index.
  # @raise [ArgumentError] if inputs are invalid (e.g., num_classes <= 0,
  #   indices is not an array, or an invalid index value is found).
  def one_hot_matrix(indices, num_classes)
    one_hot_matrix = []
  
    indices.each_with_index do |index, list_pos|
      # Validate the value of each index from the input list
      unless index.is_a?(Integer) && index >= 0 && index < num_classes
        raise ArgumentError, "Invalid index value '#{index}' found at position #{list_pos} in the indices array. " \
                             "All indices must be integers between 0 and #{num_classes - 1} (inclusive)."
      end
  
      # Create a base row vector filled with zeros
      row = Array.new(num_classes, 0)
  
      # Set the 'hot' element (the '1') at the specified index position
      row[index] = 1
  
      # Add the completed one-hot row to the matrix
      one_hot_matrix << row
    end
  
    one_hot_matrix
  end

  # returns a vector where each element is 1 divided by it's 1-based index * PI
  def pseudo_random_vector(size)
    (1..size).map { |i| 1 / i*Math::PI }
  end
end

class TestNNLM < Minitest::Test
  DELTA = 1e-6 # Tolerance for float comparisons

  def setup
    # --- Fixed Hyperparameters for Predictable Tests ---
    @embedding_dim = 2
    @context_size = 2
    @hidden_size = 3
    @attn_hidden_dim = 3
    @vocab_size = 5 # Includes [PAD], hello, world, foo, bar

    @learning_rate = 0.1 # Not used directly in forward/backward, but part of NNLM

    @nnlm = NNLM.new(
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      learning_rate: @learning_rate,
      attn_hidden_dim: @attn_hidden_dim
    )

    # --- Manually Set Vocabulary (Bypass build_vocabulary for test isolation) ---
    # Usually tokenizer sets this, but we override for deterministic tests
    vocab = { "[PAD]" => 0, "hello" => 1, "world" => 2, "foo" => 3, "bar" => 4 }
    ix_to_word = vocab.keys
    word_to_ix = ix_to_word.each_with_index.to_h

    @nnlm.instance_variable_set(:@vocab_size, @vocab_size)
    @nnlm.instance_variable_set(:@word_to_ix, word_to_ix)
    @nnlm.instance_variable_set(:@ix_to_word, ix_to_word)

    # --- Manually Set Fixed Parameters (Crucial for Predictability) ---
    # Use simple values for easy manual calculation/verification.
    # Note: Hashes need default procs if they might be accessed with unknown keys during tests,
    # but here we ensure accesses are within the initialized keys.
    @embeddings = {
      0 => [0.0, 0.0],          # [PAD] embedding (often zeros)
      1 => [0.1, 0.2],          # hello
      2 => [0.3, -0.1],         # world
      3 => [0.2, 0.4],          # foo
      4 => [-0.2, 0.1]          # bar
    }

    # Dimensions: embedding_dim x hidden_size (2 x 3)
    @W_h = [
      [0.1, 0.2, 0.3],
      [-0.1, 0.3, 0.1],
    ]
    # Dimensions: hidden_size (3)
    @b_h = [0.05, -0.05, 0.1]

    # Dimensions: hidden_size x vocab_size (3 x 5)
    @W_o = [
      [0.2, 0.1, -0.1, 0.3, 0.4],
      [-0.2, 0.4, 0.2, -0.1, 0.1],
      [0.3, -0.3, 0.1, 0.2, -0.2]
    ]
    # Dimensions: vocab_size (5)
    @b_o = [0.1, 0.0, -0.1, 0.2, 0.05]

    # attn_hidden_dim x embedding_dim
    @W_attn = [
      [0.1, 0.2],    # Row 1
      [-0.1, 0.1],   # Row 2
      [0.3, 0.1],    # Row 3
    ]
    @b_attn = [0.05, -0.05, 0.2] # size: attn_hidden_size
    @v_attn = [0.3, -0.2, 0.1] # size: attn_hidden_size

    @nnlm.instance_variable_set(:@embeddings, @embeddings)
    @nnlm.instance_variable_set(:@W_h, @W_h)
    @nnlm.instance_variable_set(:@b_h, @b_h)
    @nnlm.instance_variable_set(:@W_o, @W_o)
    @nnlm.instance_variable_set(:@b_o, @b_o)
    @nnlm.instance_variable_set(:@W_attn, @W_attn)
    @nnlm.instance_variable_set(:@b_attn, @b_attn)
    @nnlm.instance_variable_set(:@v_attn, @v_attn)

    # --- Define Fixed Input for Tests ---
    # Context: "hello world" -> indices [1, 2]
    @test_context_indices = [1, 2]
    # Target: "foo" -> index 3
    @test_target_index = 3
    @test_padded_sentence = [0, 0, 1, 2, 3, 4]
    @test_processing_index = 2
  end

  def test_forward_input_layer_concatenates_embeddings
    result = @nnlm.forward([1, 2])  # hello, world

    # The input layer should be the concatenation of the embeddings for "hello" and "world"
    # [0.1, 0.2] = features / dims of "hello"
    # [0.3, -0.1] = features / dims of "world"
    expected_context_embedding = [[0.1, 0.2], [0.3, -0.1]]
    assert_equal expected_context_embedding, result[:context_embeddings]
  end

  def test_forward_probabilities_sum_to_one
    result = @nnlm.forward([1, 2])  # hello, world

    # Probabilities should sum to approximately 1.0
    assert_in_delta 1.0, result[:probabilities].sum, 0.0001

    # All probabilities should be between 0 and 1
    result[:probabilities].each do |prob|
      assert prob.between?(0.0, 1.0), "Probability should be between 0 and 1"
    end
  end

  def test_forward_hidden_activation_values_bounded
    result = @nnlm.forward([1, 2])  # hello, world

    # tanh values should be between -1 and 1
    result[:hidden_activation].each do |val|
      assert val.between?(-1.0, 1.0), "Hidden activation should be between -1 and 1"
    end
  end

  def test_forward_different_inputs_produce_different_outputs
    result1 = @nnlm.forward([1, 2])  # hello, world
    result2 = @nnlm.forward([3, 4])  # foo, bar

    # Different inputs should produce different probabilities
    refute_equal result1[:probabilities], result2[:probabilities]

    # Different inputs should produce different hidden activations
    refute_equal result1[:hidden_activation], result2[:hidden_activation]
  end

  def test_forward_pad_tokens_produce_muted_output
    # When using all padding tokens, the input layer should be all zeros
    result_with_pads = @nnlm.forward([0, 0])  # [PAD], [PAD]

    # Input layer should be all zeros
    assert_equal [ [0, 0], [0, 0] ], result_with_pads[:context_embeddings]

    # Compare with non-pad result
    result_with_words = @nnlm.forward([1, 2])  # hello, world

    # The outputs should be different
    refute_equal result_with_pads[:probabilities], result_with_words[:probabilities]
  end

  def test_forward_consistent_outputs_for_same_inputs
    result1 = @nnlm.forward([1, 2])  # hello, world
    result2 = @nnlm.forward([1, 2])  # hello, world again

    # Same inputs should produce identical outputs
    assert_equal result1, result2
  end
  
  # Goal: Without re-writing the entire function in the test
  # And checking that each operation is called in the correct
  # Order -- instead use one-hot values to hack each step
  # So that the output is simple IFF the operations are in
  # the correct order.
  def test_attention_layer_weights
    # One-hot matrix, keep only first item from embeddings
    w_attn = [
      [1, 0],    # Row 1
      [1, 0],    # Row 2
      [1, 0],    # Row 3
    ]
    # Bias - simple - do nothing
    b_attn = [0.0, 0.0, 0.0] # size: attn_hidden_size
    # One-hot, simply take 
    v_attn = [1.0, 0.0, 0.0] # size: attn_hidden_size
    
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)

    emb = [[1.0, 0.2], [1.0, 0.3]]

    attn_scores = [1.0, 1.0]
    attn_weights = @nnlm.softmax(attn_scores)  # [0.5, 0.5]

    actual_attn_layer = @nnlm.attention_layer(emb)

    assert_equal actual_attn_layer[:attention_weights], attn_weights

    # Context vector = [v1, v2]
    #
    # v1 =
    # (emb[0][0] + emb[1][0]) * attn_weights[0] =
    # (1 + 1) * 0.5
    #
    # v2 =
    # (emb[0][1] + emb[1][1]) * attn_weights[1] =
    # (0.2 + 0.3) * 0.5
    assert_equal actual_attn_layer[:context_vector], [1.0, 0.5 * 0.5]
  end

  # To keep the previous function simple, the biases were zeroed out.
  # Here, we will zero out the attention weights, and have the biases
  # Add in values good for one-hot values later, to get similar results.
  def test_attention_layer_biases
    # Zeros matrix, strip everything from embeddings
    w_attn = [
      [0, 0],    # Row 1
      [0, 0],    # Row 2
      [0, 0],    # Row 3
    ]
    # Bias - simple, set to one-hot vector
    b_attn = [1.0, 0.0, 0.0] # size: attn_hidden_size
    # One-hot, simply take 
    v_attn = [1.0, 0.0, 0.0] # size: attn_hidden_size
    
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)

    emb = [[1.0, 0.5], [1.0, 0.5]]

    attn_scores = [1.0, 1.0]
    attn_weights = @nnlm.softmax(attn_scores)  # [0.5, 0.5]

    actual_attn_layer = @nnlm.attention_layer(emb)

    assert_equal actual_attn_layer[:attention_weights], attn_weights

    # Context vector = [v1, v2]
    #
    # v1 =
    # (emb[0][0] + emb[1][0]) * attn_weights[0] =
    # (1 + 1) * 0.5
    #
    # v2 =
    # (emb[0][1] + emb[1][1]) * attn_weights[1] =
    # (0.2 + 0.3) * 0.5
    assert_equal actual_attn_layer[:context_vector], [1.0, 0.5]
  end

  # Now test that weights and biases work together
  def test_attention_layer_weights_and_biases
    # One-hot matrix, keep only first item from embeddings
    w_attn = [
      [0.9, 0],    # Row 1
      [0.9, 0],    # Row 2
      [0.9, 0],    # Row 3
    ]
    # Bias - create one-hot vectors
    b_attn = [0.1, -0.9, -0.9] # size: attn_hidden_size
    # One-hot, simply take 
    v_attn = [1.0, 0.0, 0.0] # size: attn_hidden_size
    
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)

    emb = [[1.0, 0.5], [1.0, 0.5]]

    attn_scores = [1.0, 1.0]
    attn_weights = @nnlm.softmax(attn_scores)  # [0.5, 0.5]

    actual_attn_layer = @nnlm.attention_layer(emb)

    assert_equal actual_attn_layer[:attention_weights], attn_weights

    # Context vector = [v1, v2]
    #
    # v1 =
    # (emb[0][0] + emb[1][0]) * attn_weights[0] =
    # (1 + 1) * 0.5
    #
    # v2 =
    # (emb[0][1] + emb[1][1]) * attn_weights[1] =
    # (0.2 + 0.3) * 0.5
    assert_equal actual_attn_layer[:context_vector], [1.0, 0.5]
  end

  # Now test that weights and biases work together
  # Are fed to dot-product correctly.
  def test_attention_layer_dot_product
    # Keep only first item from embeddings
    w_attn = [
      [0.9, 0],    # Row 1
      [0.9, 0],    # Row 2
      [0.9, 0],    # Row 3
    ]
    # Bias - simple - do nothing
    b_attn = [-0.4, -0.4, -0.9] # size: attn_hidden_size
    # One-hot, simply take 
    v_attn = [1.0, 1.0, 0.0] # size: attn_hidden_size
   
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)

    emb = [[1.0, 0.5], [1.0, 0.5]]

    attn_scores = [1.0, 1.0]
    attn_weights = @nnlm.softmax(attn_scores)  # [0.5, 0.5]

    actual_attn_layer = @nnlm.attention_layer(emb)

    assert_equal actual_attn_layer[:attention_weights], attn_weights

    # Context vector = [v1, v2]
    #
    # v1 =
    # (emb[0][0] + emb[1][0]) * attn_weights[0] =
    # (1 + 1) * 0.5
    #
    # v2 =
    # (emb[0][1] + emb[1][1]) * attn_weights[1] =
    # (0.2 + 0.3) * 0.5
    assert_equal actual_attn_layer[:context_vector], [1.0, 0.5]
  end

  def test_weigh_and_bias_order_simple_one_by_two
    resp = @nnlm.weigh_and_bias([1.0], [[1.0, 0.5]], [-0.1, 0.1])
                       
    assert_equal resp, [
       0.9,  # 1.0 * 1.0 - 0.1
       0.6   # 1.0 * 0.5 + 0.1
    ] 
  end

  def test_weigh_and_bias_order_simple_two_by_three_zero_weights
    weights = @nnlm.zeros_matrix(2, 3)
    resp = @nnlm.weigh_and_bias([1.0, 1.0], weights, [-0.1, 0.1, 0.6])
                       
    assert_equal resp, [
       -0.1,  # (1.0 * 0.0 + 1.0 * 0.0) - 0.1
       0.1,   # (1.0 * 0.0 + 1.0 * 0.0) + 0.1
       0.6    # (1.0 * 0.0 + 1.0 * 0.0) + 0.6
    ] 
  end

  def test_weigh_and_bias_order_simple_two_by_three_one_hot_weights
    weights = @nnlm.one_hot_matrix([0, 0], 3)
    resp = @nnlm.weigh_and_bias([1.0, 1.0], weights, [-0.1, 0.1, 0.6])
                    
    # [1.9, 0.1, 0.6]   
    assert_equal resp, [
       1.9,  # (1.0 * 1.0 + 1.0 * 1.0) - 0.1
       0.1,   # (1.0 * 0.0 + 1.0 * 0.0) + 0.1
       0.6    # (1.0 * 0.0 + 1.0 * 0.0) + 0.6
    ] 
  end

  # The hidden signal simply `activates` the weighted and biased raw input.
  #
  # Our activation function is tanh.
  #
  # Since we have sufficiently tested the weigh and bias function, we will simply
  # check that tanh is being called.
  #
  # tanh(1) is a number unlikely to be returned randomly via any other function.
  def test_forward_hidden_signals
    fake_hidden_input = @nnlm.ones_vector(@hidden_size) 
    @nnlm.stub(:weigh_and_bias, ->(input_vector, weight_matrix, bias_vector) {
      @nnlm.ones_vector(bias_vector.size)
    }) do
      forward_data = @nnlm.forward(@test_context_indices)
      assert forward_data[:hidden_activation], @nnlm.tanh(fake_hidden_input)
    end
  end

  # The output simply `softmaxes` the weighted and biased raw output.
  #
  # Since we have sufficiently tested the weigh and bias function, we will simply
  # check that softmax is being called.
  #
  # softmax(1) for a 5 dimensional vector return 1/5 - which is not sufficently random
  # Instead, we will use `pseudo_random_vector`.
  def test_forward_output_probabilities
    fake_hidden_output = @nnlm.pseudo_random_vector(@b_o.size) 
    @nnlm.stub(:weigh_and_bias, ->(input_vector, weight_matrix, bias_vector) {
      @nnlm.pseudo_random_vector(bias_vector.size)
    }) do
      forward_data = @nnlm.forward(@test_context_indices)
      assert forward_data[:probabilities], @nnlm.softmax(fake_hidden_output)
    end
  end

  # ============================================================================
  # Test Forward Pass
  # ============================================================================
  # 
  # Since we are sufficiently satisifed with attention_layer tests,
  # Re-use the existing test, but stubbing attention_layer to return the expected
  # previous values
  def test_forward_pass_calculations
    # --- 1. Expected Projection/Concatenation ---
    # Embeddings for indices 1 and 2 are [0.1, 0.2] and [0.3, -0.1]
    expected_context_vector = [0.1, 0.2] # 0.3, -0.1]

    # --- 2. Expected Hidden Layer Input ---
    expected_hidden_input = @nnlm.weigh_and_bias(
      expected_context_vector,
      @nnlm.instance_variable_get(:@W_h),
      @nnlm.instance_variable_get(:@b_h)
    )

    # --- 3. Expected Hidden Layer Activation (Tanh) ---
    expected_hidden_activation = expected_hidden_input.map { |x| CMath.tanh(x).real }
    # expected_hidden_activation approx = [0.139, -0.040, 0.235] (using more precision below)

    # --- 4. Expected Output Layer Scores ---
    # output_scores = hidden_activation * W_o + b_o
    # hidden_activation (1x3) * W_o (3x5) -> (1x5)
    # W_o = [[0.2, 0.1, -0.1, 0.3, 0.4], [-0.2, 0.4, 0.2, -0.1, 0.1], [0.3, -0.3, 0.1, 0.2, -0.2]]
    # b_o = [0.1, 0.0, -0.1, 0.2, 0.05]
    # Manually calculate hidden_activation * W_o + b_o using expected_hidden_activation
    manual_output_scores = @nnlm.weigh_and_bias(
      expected_hidden_activation, 
      @nnlm.instance_variable_get(:@W_o),
      @nnlm.instance_variable_get(:@b_o)
    )
    expected_output_scores = manual_output_scores # Use calculated value for precision

    ## --- 5. Expected Probabilities (Softmax) ---
    expected_probabilities = @nnlm.softmax(expected_output_scores)

    ## --- Action: Call the actual forward method ---
    @nnlm.stub(:attention_layer, ->(context_embeddings) {
      {
        context_vector: expected_context_vector,
        context_weights: [],
        projected_embeddings: []
      }
    }) do
      forward_data = @nnlm.forward(@test_context_indices)

      ## --- Assertions ---
      assert_instance_of Hash, forward_data, "Forward pass should return a Hash"

      resp_keys = [:probabilities, :hidden_activation, :context_vector, :attention_weights, :projected_embeddings, :context_embeddings]
      assert_equal resp_keys.to_set, forward_data.keys.to_set, "Forward data keys mismatch"

      assert_vector_in_delta expected_context_vector, forward_data[:context_vector], DELTA, "Input layer (concatenated embeddings) mismatch"

      assert_vector_in_delta expected_hidden_activation, forward_data[:hidden_activation], DELTA, "Hidden activation (tanh) mismatch"

      assert_vector_in_delta expected_probabilities, forward_data[:probabilities], DELTA, "Probabilities (softmax) mismatch"
      assert_in_delta 1.0, forward_data[:probabilities].sum, DELTA * 10, "Probabilities should sum to 1.0" # Allow slightly larger delta for sum
    end
  end

  # Backwards supplies gradients (errors) for hidden wieghts & biases, output weights and biases, and context indices
  def test_backward_only_updates_used_embeddings
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: [0.2, 0.1, 0.1, 0.2, 0.3],
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # Only embeddings for words in the context should have non-zero gradients
    context_indices.each do |idx|
      assert gradients[:grad_embeddings].key?(idx), "Should have gradient for word #{idx}"
      refute_equal Array.new(@embedding_dim, 0.0), 
gradients[:grad_embeddings][idx],
                   "Gradient for used word #{idx} should not be all zeros"
    end

    # Words not in context should not have gradients
    all_word_idxs = @nnlm.instance_variable_get(:@embeddings).keys
    unused_idxs = (all_word_idxs - context_indices)
    unused_idxs.each do |idx|
      assert_equal gradients[:grad_embeddings][idx], [0.0, 0.0], "Unused word #{idx} should not have gradient"
    end
  end

  # Backward sets the error for the Output bias (grad_b_o)
  # These are the biases for the outputs (each word)
  # The `probabilities` in forward_data are the probabilities (each word)
  #
  # In backward, we are passing in the observed word ("foo")
  # Since this is an observation, it is 100% probability for this occurence
  # backward subtracts 1 from the probability to get the error for the output
  # ONLY for the target word
  #
  # Here we ensure that only the target word's output biases will be modified
  def test_backward_error_signal_direction_b_o
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: [0.2, 0.1, 0.1, 0.2, 0.3],
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    # When target is "foo" (index 3), the error for "foo" should be negative
    # and error for other words should be positive
    gradients = @nnlm.backward(context_indices, target_index, forward_data)


    # Check output layer error signal (d_output_scores)
    # This is stored directly in grad_b_o
    output_errors = gradients[:grad_b_o]

    # Error for target word should be negative (probabilities[target] - 1.0)
    assert output_errors[target_index] < 0,
           "Error for target word should be negative (want higher probability)"

    # Error for at least one non-target word should be positive (want lower probability)
    non_target_errors = output_errors.each_with_index.reject { |_, i| i == target_index }.map(&:first)
    assert non_target_errors.any? { |e| e > 0 },
           "Error for at least one non-target word should be positive"
  end

  # grad_W_o tells us how to adjust the connections from the hidden layer to the output layer.
  # grad_W_o moves in the *same* direction as activation for non-target words.
  # grad_W_o moves in the *opposite* direction as activation for target words.
  def test_backward_error_signal_direction_W_o
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: [0.2, 0.1, 0.1, 0.2, 0.3],
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    # When target is "foo" (index 3), the error for "foo" should be negative
    # and error for other words should be positive
    gradients = @nnlm.backward(context_indices, target_index, forward_data)


    # Check output weights error signal stored directly in grad_W_o
    output_errors = gradients[:grad_W_o]

    # Iterate through each hidden neuron
    @hidden_size.times do |h_idx|
      activation = forward_data[:hidden_activation][h_idx]

      # Skip check if activation is zero (gradient contribution is zero)
      next if activation.abs < DELTA

      # --- Check column for the TARGET word (index 3) ---
      grad_target = output_errors[h_idx][target_index]

      # If activation is positive, gradient should be negative (to increase W_o)
      # If activation is negative, gradient should be positive (to decrease W_o, making it less negative)
      # They should have opposite signs
      assert activation * grad_target <= 0,
             "grad_W_o[#{h_idx}][#{target_index}] sign should oppose hidden_activation[#{h_idx}] sign"

      non_target_idxs = (@nnlm.instance_variable_get(:@embeddings).keys - [target_index])
      non_target_idxs.each do |non_target_idx|
        grad_non_target = output_errors[h_idx][non_target_idx]

        # If activation is positive, gradient should be positive (to decrease W_o)
        # If activation is negative, gradient should be negative (to increase W_o, making it less negative)
        # They should have the same sign
        assert activation * grad_non_target >= 0,
               "grad_W_o[#{h_idx}][#{non_target_idx}] sign should match hidden_activation[#{h_idx}] sign"
      end
    end
  end

  # Unlikely event, but test that when there is a 100% prediction for a word
  # There is no error correction.
  def test_backward_with_perfect_prediction_produces_zero_gradients
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: [0.0, 0.0, 0.0, 1.0, 0.0],
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # Output bias gradients should be zeros (probabilities - target_one_hot = 0)
    assert_equal Array.new(@nnlm.instance_variable_get(:@vocab_size), 0.0), 
gradients[:grad_b_o],
                 "Output bias gradients should be zero when prediction is perfect"
  end

  def test_backward_produces_non_zero_gradients
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: [0.2, 0.1, 0.1, 0.2, 0.3], 
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # Helper to check if a matrix has any non-zero elements
    def non_zero?(matrix)
      matrix.flatten.any? { |x| x != 0 }
    end

    # All gradient matrices should have at least some non-zero values
    assert non_zero?(gradients[:grad_W_h]), "Hidden weights gradient should not be all zeros"
    assert non_zero?(gradients[:grad_b_h]), "Hidden bias gradient should not be all zeros"
    assert non_zero?(gradients[:grad_W_o]), "Output weights gradient should not be all zeros"
    assert non_zero?(gradients[:grad_b_o]), "Output bias gradient should not be all zeros"

    # At least some embedding gradients should be non-zero
    used_embedding_grads = context_indices.map { |idx| gradients[:grad_embeddings][idx] }.flatten
    assert used_embedding_grads.any? { |x| x != 0 }, "Embedding gradients should not be all zeros"
  end

  def test_backward_with_different_targets_produces_different_gradients
    context_indices = [1, 2]  # "hello", "world"
    forward_data = {
      probabilities: [0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    # Compare gradients when target is "foo" vs "bar"
    gradients_foo = @nnlm.backward(context_indices, 3, forward_data)  # target: foo
    gradients_bar = @nnlm.backward(context_indices, 4, forward_data)  # target: bar

    # Gradients should be different
    assert_equal false, 
gradients_foo[:grad_b_o] == gradients_bar[:grad_b_o],
                 "Different targets should produce different output gradients"
    assert_equal false, 
gradients_foo[:grad_b_h] == gradients_bar[:grad_b_h],
                 "Different targets should produce different hidden gradients"
  end

  def test_backward_generates_same_gradients_for_same_inputs
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3
    forward_data = {
      probabilities: [0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: [0.1, -0.1, 0.2],
      context_vector: @nnlm.ones_vector(@hidden_size),
      attention_weights: @nnlm.ones_vector(@context_size),
      projected_embeddings: @nnlm.ones_matrix(@context_size, @attn_hidden_dim),
      context_embeddings: [[0.1, 0.2], [0.3, -0.1]]
    }

    # Run backward twice with same inputs
    gradients1 = @nnlm.backward(context_indices, target_index, forward_data)
    gradients2 = @nnlm.backward(context_indices, target_index, forward_data)

    # Gradients should be identical
    assert_equal gradients1, gradients2, "Same inputs should produce identical gradients"
  end

  # TODO:
  #  1) Add tests for reverse weigh
  #  2) Add tests for gradient calculation for attn layer

  # ============================================================================
  # Test Backward Pass
  # ============================================================================
  #
  # Since backward works in reverse, we can re-use the existing test exactly as is
  # until the final stage calculates the attn_layer.
  #
  # We need to add another layer of tests to ensure that gradients update the
  # attention gradients are calculated as expected.
  def test_backward_pass_gradients
    # --- Prerequisite: Run forward pass to get intermediate values ---
    # Use the same fixed inputs and parameters from setup
    forward_data = @nnlm.forward(@test_context_indices)
    probabilities = forward_data[:probabilities]
    hidden_activation = forward_data[:hidden_activation]
    context_embeddings = forward_data[:context_embeddings] #  [[0.1, 0.2], [0.3, -0.1]]
    context_vector = forward_data[:context_vector]

    target_index = @test_target_index # Target index is 3 ("foo")

    # --- 1. Expected Gradient of Loss w.r.t Output Scores (d_output_scores) ---
    # dL/dOutput_Scores = probabilities - target_one_hot
    # target_one_hot = [0, 0, 0, 1, 0] for target_index 3
    expected_d_output_scores = probabilities.dup
    expected_d_output_scores[target_index] -= 1.0

    # --- 2. Expected Gradients for Output Layer (W_o, b_o) ---
    # grad_b_o = d_output_scores
    expected_grad_b_o = expected_d_output_scores

    # grad_W_o = outer_product(hidden_activation, d_output_scores) (hidden_size x vocab_size) -> (3 x 5)
    expected_grad_w_o = @nnlm.outer_product(hidden_activation, expected_d_output_scores)

    # --- 3. Expected Gradient w.r.t Hidden Activation Input Signal ---
    # d_hidden_input_signal = d_output_scores * W_o^T
    # d_output_scores (1x5) * W_o^T (5x3) -> (1x3)
    expected_d_hidden_input_signal = @nnlm.reverse_weigh(expected_d_output_scores, @W_o)

    # --- 4. Expected Gradient w.r.t Hidden Input (d_hidden_input) ---
    # d_hidden_input = d_hidden_input_signal * dtanh(hidden_activation)
    # dtanh(y) = 1 - y^2
    d_tanh = @nnlm.dtanh(hidden_activation)
    expected_d_hidden_input = @nnlm.multiply_elementwise(expected_d_hidden_input_signal, d_tanh)

    # --- 5. Expected Gradients for Hidden Layer (W_h, b_h) ---
    # grad_b_h = d_hidden_input
    expected_grad_b_h = expected_d_hidden_input

    # grad_W_h = outer_product(input_layer, d_hidden_input) (input_concat_size x hidden_size) -> (4 x 3)
    expected_grad_w_h = @nnlm.outer_product(context_vector, expected_d_hidden_input)

    ## --- 6. Expected Gradient w.r.t Input Layer (d_input_layer) ---
    ## d_input_layer = d_hidden_input * W_h^T
    ## d_hidden_input (1x3) * W_h^T (3x4) -> (1x4)
    # TODO: Not currently used.
    expected_d_context_vector = @nnlm.reverse_weigh(expected_d_hidden_input, @W_h) # Result size: embedding_dim

    # --- 7. Expected Gradients for Embeddings ---
    # Distribute d_input_layer back to the embeddings used in the context
    # Context indices were [1, 2]
    #expected_grad_embeddings = Hash.new { |h, k| h[k] = Array.new(@embedding_dim, 0.0) }
    #context_indices = @test_context_indices
    #context_indices.each_with_index do |word_ix, i|
    #  start_idx = i * @embedding_dim
    #  end_idx = start_idx + @embedding_dim - 1
    #  embedding_grad_slice = expected_d_input_layer[start_idx..end_idx]
    #   # Important: Use add_vectors for accumulation if the same index appeared multiple times
    #  expected_grad_embeddings[word_ix] = @nnlm.add_vectors(expected_grad_embeddings[word_ix], embedding_grad_slice)
    #end

    # --- Action: Call the actual backward method ---
    gradients = @nnlm.backward(@test_context_indices, target_index, forward_data)

    # --- Assertions ---
    assert_instance_of Hash, gradients, "Backward pass should return a Hash"
    expected_keys = [:grad_embeddings, :grad_W_h, :grad_b_h, :grad_W_o, :grad_b_o, :grad_W_attn, :grad_b_attn, :grad_v_attn].to_set
    assert_equal expected_keys, gradients.keys.to_set, "Backward gradients keys mismatch"

    # Output Layer Gradients
    assert_vector_in_delta expected_grad_b_o, gradients[:grad_b_o], DELTA, "Gradient b_o mismatch"
    assert_matrix_in_delta expected_grad_w_o, gradients[:grad_W_o], DELTA, "Gradient W_o mismatch"

    # Hidden Layer Gradients
    assert_vector_in_delta expected_grad_b_h, gradients[:grad_b_h], DELTA, "Gradient b_h mismatch"
    assert_matrix_in_delta expected_grad_w_h, gradients[:grad_W_h], DELTA, "Gradient W_h mismatch"
  end

  # TEST BACK-PROPAGATION THROUGH THE ATTENTION LAYER!
  #
  def test_backward_indirect_path_only

    # Use non-trivial weights for W
    w_attn = [[1.0, 2.0]] # 1x2
    b_attn = [0.0]       # size 1
    v_attn = [1.0]       # size 1

    # --- Setup instance variables for this test ---
    @nnlm.instance_variable_set(:@attn_hidden_dim, 1) # Ensure correct dim
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)
    # Ensure embedding_dim is also set if needed by backward_attention init
    @nnlm.instance_variable_set(:@embedding_dim, 2)

    # --- Inputs to backward_attention ---
    context_embeddings = [[1.0, 0.1], [2.0, 0.2]] # Shape: (2, 2)
    # Projected embeddings (Values only needed for grad_v calc and deactivate stub input)
    projected_embeddings = [[0.5], [0.6]]         # Shape: (2, 1), simple values

    attention_weights = [0.5, 0.5] # Assumed weights
    d_context_vector = [0.0, 0.0]  # ZERO incoming gradient - isolates indirect path

    # --- Define expected values (based on Scenario 3 logic) ---
    # dC=0 -> dAlpha=0.
    # Stub dsoftmax returns [1.0, 0.0] (forcing score 0 gradient).
    # Stub deactivate returns [1.0] if input grad > 0, else [0.0].
    # Loop i=0: dScore=1.0 -> dProjEmb=[1.0] -> dTanhInput=[1.0] -> contributes to grads
    # Loop i=1: dScore=0.0 -> dProjEmb=[0.0] -> dTanhInput=[0.0] -> zero contribution
    expected_grad_W = [[1.0, 0.1]] # outer([1.0], emb0) + outer([0.0], emb1)
    expected_grad_b = [1.0]       # [1.0] + [0.0]
    expected_grad_v = [0.5]       # 1.0 * proj_emb0[0] + 0.0 * proj_emb1[0] = 1.0 * 0.5
    # dEmb0 = direct + indirect = [0,0] + vec_mat([1.0], W_attn) = [1.0, 2.0]
    # dEmb1 = direct + indirect = [0,0] + vec_mat([0.0], W_attn) = [0.0, 0.0]
    expected_d_emb = [[1.0, 2.0], [0.0, 0.0]]

    # --- Nest the stubs and run calculation ---
    @nnlm.stub(:deactivate, ->(d_vec, o_vec) { d_vec.sum.abs > 1e-9 ? [1.0] : [0.0] }) do
      @nnlm.stub(:dsoftmax, ->(d_output, output) { [1.0, 0.0] }) do
        # --- Actual Calculation ---
        actual_grads = @nnlm.backward_attention(
          context_embeddings,
          projected_embeddings,
          d_context_vector,
          attention_weights
        )

        # --- Assertions ---
        assert_matrix_in_delta expected_grad_W, actual_grads[:grad_W_attn], DELTA
        assert_vector_in_delta expected_grad_b, actual_grads[:grad_b_attn], DELTA
        assert_vector_in_delta expected_grad_v, actual_grads[:grad_v_attn], DELTA
        assert_matrix_in_delta expected_d_emb, actual_grads[:d_embeddings], DELTA
      end
    end
  end

  # Two embeddings, simple dC, stubbed derivatives
  def test_backward_two_embeddings_simple_gradient
    w_attn = [[1.0, 0.0]] # Selects first element
    b_attn = [0.0]
    v_attn = [1.0]

    # --- Setup ---
    @nnlm.instance_variable_set(:@attn_hidden_dim, 1)  
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)

    # Inputs to backward_attention
    context_embeddings = [[1.0, 0.1], [2.0, 0.2]] # Shape: (2, 2)
    # Projected embeddings (consistent with W,b above, needed for deactivate stub + grad_v_attn)
    proj_emb_0 = @nnlm.tanh(@nnlm.multiply_mat_vec(w_attn, context_embeddings[0])) # tanh(1.0)
    proj_emb_1 = @nnlm.tanh(@nnlm.multiply_mat_vec(w_attn, context_embeddings[1])) # tanh(2.0)
    projected_embeddings = [proj_emb_0, proj_emb_1] # Shape: (2, 1)

    attention_weights = [0.5, 0.5] # Assumed weights
    d_context_vector = [1.0, 0.0]  # Incoming gradient

    # --- Expected Intermediate Values (based on planning doc & stubs) ---
    # dL/dAlpha = [dot(dC, emb0), dot(dC, emb1)] = [1.0, 2.0]
    # dScores = dsoftmax([1.0, 2.0], [0.5, 0.5]) = [0.2, -0.2] (from stub)
    # --- Loop i = 0 ---
    # dScore_0 = 0.2
    # grad_v_attn += 0.2 * proj_emb_0 = [0.2 * tanh(1.0)]
    # dProjEmb = 0.2 * v_attn = [0.2]
    # dTanhInput = deactivate([0.2], proj_emb_0) = [0.2 * 0.5] = [0.1] (stub)
    # grad_b_attn += [0.1]
    # grad_W_attn += outer([0.1], emb0) = [[0.1, 0.01]]
    # dEmb0_indirect = multiply_vec_mat([0.1], W_attn) = [0.1, 0.0]
    # dEmb0_direct = alpha0 * dC = 0.5 * [1.0, 0.0] = [0.5, 0.0]
    # Total dEmb0 = [0.5, 0.0] + [0.1, 0.0] = [0.6, 0.0]
    # --- Loop i = 1 ---
    # dScore_1 = -0.2
    # grad_v_attn += -0.2 * proj_emb_1 = [0.2*tanh(1.0) - 0.2*tanh(2.0)]
    # dProjEmb = -0.2 * v_attn = [-0.2]
    # dTanhInput = deactivate([-0.2], proj_emb_1) = [-0.2 * 0.5] = [-0.1] (stub)
    # grad_b_attn += [-0.1] = [0.1 - 0.1] = [0.0]
    # grad_W_attn += outer([-0.1], emb1) = [[0.1, 0.01]] + [[-0.2, -0.02]] = [[-0.1, -0.01]]
    # dEmb1_indirect = multiply_vec_mat([-0.1], W_attn) = [-0.1, 0.0]
    # dEmb1_direct = alpha1 * dC = 0.5 * [1.0, 0.0] = [0.5, 0.0]
    # Total dEmb1 = [0.5, 0.0] + [-0.1, 0.0] = [0.4, 0.0]

    expected_grad_W = [[-0.1, -0.01]]
    expected_grad_b = [0.0]
    expected_grad_v = @nnlm.add_vectors(
                         @nnlm.scalar_multiply(0.2, proj_emb_0),
                         @nnlm.scalar_multiply(-0.2, proj_emb_1)
                       ) # [0.2*tanh(1.0) - 0.2*tanh(2.0)] => This is stubbed below
    expected_d_emb = [[0.6, 0.0], [0.4, 0.0]]

    @nnlm.stub(:deactivate, ->(d_vec, o_vec) { @nnlm.scalar_multiply(0.5, d_vec) }) do
      @nnlm.stub(:dsoftmax, ->(d_output, output) { [0.2, -0.2] }) do
        # --- Actual Calculation ---
        actual_grads = @nnlm.backward_attention(
          context_embeddings,
          projected_embeddings,
          d_context_vector,
          attention_weights
        )

        # --- Assertions ---
        assert_matrix_in_delta expected_grad_W, actual_grads[:grad_W_attn], DELTA
        assert_vector_in_delta expected_grad_b, actual_grads[:grad_b_attn], DELTA
        assert_vector_in_delta expected_grad_v, actual_grads[:grad_v_attn], DELTA
        assert_matrix_in_delta expected_d_emb, actual_grads[:d_embeddings], DELTA
      end
    end
  end

  # Test case where dC is zero AND dScores are zero (all gradients should be zero)
  def test_backward_zero_input_gradient

    # Define parameters (values don't matter much as gradients should be zero)
    w_attn = [[1.0, 0.0]]
    b_attn = [0.0]
    v_attn = [1.0]

    # --- Setup instance variables ---
    @nnlm.instance_variable_set(:@attn_hidden_dim, 1)
    @nnlm.instance_variable_set(:@W_attn, w_attn)
    @nnlm.instance_variable_set(:@b_attn, b_attn)
    @nnlm.instance_variable_set(:@v_attn, v_attn)
    @nnlm.instance_variable_set(:@embedding_dim, 2)

    # --- Inputs ---
    context_embeddings = [[1.0, 0.1], [2.0, 0.2]]
    projected_embeddings = [[0.5], [0.6]] # Dummy values
    attention_weights = [0.5, 0.5]
    d_context_vector = [0.0, 0.0] # ZERO incoming gradient

    # --- Expected Output: All gradients should be zero ---
    expected_grad_W = [[0.0, 0.0]]
    expected_grad_b = [0.0]
    expected_grad_v = [0.0]
    expected_d_emb = [[0.0, 0.0], [0.0, 0.0]]

    # --- Nest stubs and run calculation ---
    @nnlm.stub(:deactivate, ->(d_vec, o_vec) { @nnlm.scalar_multiply(0.5, d_vec) }) do
      @nnlm.stub(:dsoftmax, ->(d_output, output) { @nnlm.zeros_vector(2) }) do
        # --- Actual Calculation ---
        actual_grads = @nnlm.backward_attention(
          context_embeddings,
          projected_embeddings,
          d_context_vector,
          attention_weights
        )

        # --- Assertions ---
        assert_matrix_in_delta expected_grad_W, actual_grads[:grad_W_attn], DELTA
        assert_vector_in_delta expected_grad_b, actual_grads[:grad_b_attn], DELTA
        assert_vector_in_delta expected_grad_v, actual_grads[:grad_v_attn], DELTA
        assert_matrix_in_delta expected_d_emb, actual_grads[:d_embeddings], DELTA
      end
    end
  end

  # process idx=2 (hello), should update all unique words in the context_size (2) = (hello, world)
  def test_process_context_hello
    padded_sentence = [0, 0, 1, 2, 3, 4] # pad, pad, hello, world, foo, bar
    idx = 2 # => hello

    t_pad_idx = 0 # pad
    t_hello_idx = 1 # hello
    t_world_idx = 2 # world
    t_foo_idx = 3 # foo
    t_bar_idx = 4 # bar

    predict_sentence = [t_hello_idx, t_world_idx]

    initial_pred = @nnlm.forward(predict_sentence)[:probabilities]
    initial_embeddings = @nnlm.instance_variable_get(:@embeddings).dup

    @nnlm.process_context(padded_sentence, idx)

    post_embeddings = @nnlm.instance_variable_get(:@embeddings).dup
    post_pred = @nnlm.forward(predict_sentence)[:probabilities]

    assert_equal initial_embeddings[t_pad_idx], post_embeddings[t_pad_idx] # pad should be unchanged
    assert_equal initial_embeddings[t_foo_idx], post_embeddings[t_foo_idx] # foo should be unchanged
    assert_equal initial_embeddings[t_bar_idx], post_embeddings[t_bar_idx] # bar should be unchanged

    refute_equal initial_embeddings[t_hello_idx], post_embeddings[t_hello_idx] # hello should be unchanged
    refute_equal initial_embeddings[t_world_idx], post_embeddings[t_world_idx] # hello should be unchanged

    assert post_pred[t_pad_idx] <= initial_pred[t_pad_idx] # probability of picking pad decreased
    assert post_pred[t_hello_idx] <= initial_pred[t_hello_idx] # probability of picking hello decreased
    assert post_pred[t_world_idx] <= initial_pred[t_world_idx] # probability of picking world decreased
    assert post_pred[t_bar_idx] <= initial_pred[t_bar_idx] # probability of picking bar decreased

    assert post_pred[t_foo_idx] >= initial_pred[t_foo_idx] # probability of picking foo increased
  end

  # Ensure that at each iteration, predictions continue to improve
  def test_process_context_hello_iterative
    padded_sentence = [0, 0, 1, 2, 3, 4] # pad, pad, hello, world, foo, bar
    idx = 2 # => hello

    t_pad_idx = 0 # pad
    t_hello_idx = 1 # hello
    t_world_idx = 2 # world
    t_foo_idx = 3 # foo
    t_bar_idx = 4 # bar

    predict_sentence = [t_hello_idx, t_world_idx]

    10.times do |_i|
      @nnlm.process_context(padded_sentence, idx)

      initial_pred = @nnlm.forward(predict_sentence)[:probabilities]
      initial_embeddings = @nnlm.instance_variable_get(:@embeddings).dup

      @nnlm.process_context(padded_sentence, idx)

      post_embeddings = @nnlm.instance_variable_get(:@embeddings).dup
      post_pred = @nnlm.forward(predict_sentence)[:probabilities]

      assert_equal initial_embeddings[t_pad_idx], post_embeddings[t_pad_idx] # pad should be unchanged
      assert_equal initial_embeddings[t_foo_idx], post_embeddings[t_foo_idx] # foo should be unchanged
      assert_equal initial_embeddings[t_bar_idx], post_embeddings[t_bar_idx] # bar should be unchanged

      refute_equal initial_embeddings[t_hello_idx], post_embeddings[t_hello_idx] # hello should be unchanged
      refute_equal initial_embeddings[t_world_idx], post_embeddings[t_world_idx] # hello should be unchanged

      assert post_pred[t_pad_idx] <= initial_pred[t_pad_idx] # probability of picking pad decreased
      assert post_pred[t_hello_idx] <= initial_pred[t_hello_idx] # probability of picking hello decreased
      assert post_pred[t_world_idx] <= initial_pred[t_world_idx] # probability of picking world decreased
      assert post_pred[t_bar_idx] <= initial_pred[t_bar_idx] # probability of picking bar decreased

      assert post_pred[t_foo_idx] >= initial_pred[t_foo_idx] # probability of picking foo increased
    end
  end

  # process idx=0 (pad), should update all unique words in the context_size (2) = (pad)
  def test_process_context_pad
    padded_sentence = [0, 0, 1, 2, 3, 4] # pad, pad, hello, world, foo, bar
    idx = 0 # => pad

    t_pad_idx = 0 # pad
    t_hello_idx = 1 # hello
    t_world_idx = 2 # world
    t_foo_idx = 3 # foo
    t_bar_idx = 4 # bar

    predict_sentence = [t_pad_idx, t_pad_idx]

    initial_pred = @nnlm.forward(predict_sentence)[:probabilities]
    initial_embeddings = @nnlm.instance_variable_get(:@embeddings).dup

    @nnlm.process_context(padded_sentence, idx)

    post_embeddings = @nnlm.instance_variable_get(:@embeddings).dup
    post_pred = @nnlm.forward(predict_sentence)[:probabilities]

    assert_equal initial_embeddings[t_foo_idx], post_embeddings[t_foo_idx] # foo should be ununchanged
    assert_equal initial_embeddings[t_bar_idx], post_embeddings[t_bar_idx] # bar should be unchanged
    assert_equal initial_embeddings[t_hello_idx], post_embeddings[t_hello_idx] # hello should be unchanged
    assert_equal initial_embeddings[t_world_idx], post_embeddings[t_world_idx] # hello should be unchanged

    refute_equal initial_embeddings[t_pad_idx], post_embeddings[t_pad_idx] # pad should be changed

    assert post_pred[t_pad_idx] <= initial_pred[t_pad_idx] # probability of picking pad decreased
    assert post_pred[t_world_idx] <= initial_pred[t_world_idx] # probability of picking world decreased
    assert post_pred[t_foo_idx] <= initial_pred[t_foo_idx] # probability of picking foo decreased
    assert post_pred[t_bar_idx] <= initial_pred[t_bar_idx] # probability of picking bar decreased

    assert post_pred[t_hello_idx] >= initial_pred[t_hello_idx] # probability of picking hello increased
  end

  # ============================================================================
  # Test process_context Return Value (Loss) and Side Effects (Parameter Updates)
  # ============================================================================
  #
  # Ensure that the result of forward is actually applied.
  def test_process_context_returns_loss_and_updates_parameters
    # --- Arrange ---
    # 1. Get initial parameter state (done in setup, stored in @*)

    # 2. Determine expected loss
    #    Requires running forward pass with initial parameters
    forward_data_for_loss = @nnlm.forward(@test_context_indices)
    probabilities_for_loss = forward_data_for_loss[:probabilities]
    # Loss = -log(probability of target_index)
    expected_loss = -Math.log(probabilities_for_loss[@test_target_index] + 1e-9) # Use target_index=3

    # 3. Determine expected gradients (run backward pass conceptually or actually)
    #    Use the same forward data as calculated above for consistency
    expected_gradients = @nnlm.backward(@test_context_indices, @test_target_index, forward_data_for_loss)

    # 4. Calculate expected *updated* parameters based on initial state, gradients, and LR
    lr = @nnlm.instance_variable_get(:@learning_rate)

    expected_embeddings = Marshal.load(Marshal.dump(@embeddings))
    expected_gradients[:grad_embeddings].each do |word_ix, grad|
      expected_embeddings[word_ix] = @nnlm.subtract_vectors(expected_embeddings[word_ix], @nnlm.scalar_multiply(lr, grad))
    end

    expected_W_attn = @W_attn.map.with_index do |row, i|
      @nnlm.subtract_vectors(row, @nnlm.scalar_multiply(@learning_rate, expected_gradients[:grad_W_attn][i]))
    end
    expected_b_attn = @nnlm.subtract_vectors(@b_attn, @nnlm.scalar_multiply(@learning_rate, expected_gradients[:grad_b_attn]))
    expected_v_attn = @nnlm.subtract_vectors(@v_attn, @nnlm.scalar_multiply(@learning_rate, expected_gradients[:grad_v_attn]))

    expected_W_h = @W_h.map.with_index do |row, i|
      @nnlm.subtract_vectors(row, @nnlm.scalar_multiply(lr, expected_gradients[:grad_W_h][i]))
    end
    expected_b_h = @nnlm.subtract_vectors(@b_h, @nnlm.scalar_multiply(lr, expected_gradients[:grad_b_h]))

    expected_W_o = @W_o.map.with_index do |row, i|
      @nnlm.subtract_vectors(row, @nnlm.scalar_multiply(lr, expected_gradients[:grad_W_o][i]))
    end
    expected_b_o = @nnlm.subtract_vectors(@b_o, @nnlm.scalar_multiply(lr, expected_gradients[:grad_b_o]))

    # --- Act ---
    # Call process_context with the test sentence and index
    # Store the returned loss value
    actual_loss = @nnlm.process_context(@test_padded_sentence, @test_processing_index)

    # 1. Assert the returned loss value
    assert_in_delta expected_loss, actual_loss, DELTA, "Returned loss value mismatch"

    # 2. Get the actual parameters *after* process_context has run
    actual_embeddings = @nnlm.instance_variable_get(:@embeddings)
    actual_W_attn = @nnlm.instance_variable_get(:@W_attn)
    actual_b_attn = @nnlm.instance_variable_get(:@b_attn)
    actual_v_attn = @nnlm.instance_variable_get(:@v_attn)

    actual_W_h = @nnlm.instance_variable_get(:@W_h)
    actual_b_h = @nnlm.instance_variable_get(:@b_h)
    actual_W_o = @nnlm.instance_variable_get(:@W_o)
    actual_b_o = @nnlm.instance_variable_get(:@b_o)

    # Compare actual parameters with the expected final parameters
    assert_embedding_hash_in_delta expected_embeddings, actual_embeddings, DELTA, "Embeddings mismatch after update"
    assert_matrix_in_delta expected_W_attn, actual_W_attn, DELTA, "W_attn mismatch after update"
    assert_vector_in_delta expected_b_attn, actual_b_attn, DELTA, "b_attn mismatch after update"
    assert_vector_in_delta expected_v_attn, actual_v_attn, DELTA, "v_attn mismatch after update"

    assert_matrix_in_delta expected_W_h, actual_W_h, DELTA, "W_h mismatch after update"
    assert_vector_in_delta expected_b_h, actual_b_h, DELTA, "b_h mismatch after update"

    assert_matrix_in_delta expected_W_o, actual_W_o, DELTA, "W_o mismatch after update"
    assert_vector_in_delta expected_b_o, actual_b_o, DELTA, "b_o mismatch after update"

    # Sanity check: ensure embeddings not part of the context gradient didn't change
    assert_vector_in_delta @embeddings[0], actual_embeddings[0], DELTA, "Embedding for index 0 (PAD) should not change"
    assert_vector_in_delta @embeddings[4], actual_embeddings[4], DELTA, "Embedding for index 4 (bar) should not change"
  end

  # Test that update works as expected
  # Zeros do nothing
  def test_update_parameters_zeros_unchanged
    @nnlm.update_parameters({
      grad_embeddings: {0 => [0.0, 0.0] },
      grad_W_attn: @nnlm.zeros_matrix(@attn_hidden_dim, @embedding_dim),
      grad_b_attn: @nnlm.zeros_vector(@attn_hidden_dim),
      grad_v_attn: @nnlm.zeros_vector(@attn_hidden_dim),
      grad_W_h: @nnlm.zeros_matrix(@attn_hidden_dim, @hidden_size),
      grad_b_h: @nnlm.zeros_vector(@hidden_size),
      grad_W_o: @nnlm.zeros_matrix(@hidden_size, @vocab_size),
      grad_b_o: @nnlm.zeros_vector(@vocab_size)
    })

    # Get the actual parameters *after* process_context has run
    actual_embeddings = @nnlm.instance_variable_get(:@embeddings)
    actual_W_attn = @nnlm.instance_variable_get(:@W_attn)
    actual_b_attn = @nnlm.instance_variable_get(:@b_attn)
    actual_v_attn = @nnlm.instance_variable_get(:@v_attn)

    actual_W_h = @nnlm.instance_variable_get(:@W_h)
    actual_b_h = @nnlm.instance_variable_get(:@b_h)
    actual_W_o = @nnlm.instance_variable_get(:@W_o)
    actual_b_o = @nnlm.instance_variable_get(:@b_o)

    # Compare actual parameters with the expected final unchanged parameters
    assert_embedding_hash_in_delta @embeddings, actual_embeddings, DELTA, "Embeddings mismatch after update"
    assert_matrix_in_delta @W_attn, actual_W_attn, DELTA, "W_attn mismatch after update"
    assert_vector_in_delta @b_attn, actual_b_attn, DELTA, "b_attn mismatch after update"
    assert_vector_in_delta @v_attn, actual_v_attn, DELTA, "v_attn mismatch after update"

    assert_matrix_in_delta @W_h, actual_W_h, DELTA, "W_h mismatch after update"
    assert_vector_in_delta @b_h, actual_b_h, DELTA, "b_h mismatch after update"

    assert_matrix_in_delta @W_o, actual_W_o, DELTA, "W_o mismatch after update"
    assert_vector_in_delta @b_o, actual_b_o, DELTA, "b_o mismatch after update"
  end

  def test_update_parameters
    embedding_grad = [0.1, -0.33]

    b_attn_grad = [ 0.2, -0.2, 0.0 ]
    v_attn_grad = [ -0.2, 0.5, -0.05 ]

    b_h_grad = [ 0.6, 0.3, 0.1 ]
    b_o_grad = [ -0.3, 0.1, 0.2, 0.1, 0.5 ]

    @nnlm.update_parameters({
      grad_embeddings: {0 => embedding_grad },
      grad_W_attn: @nnlm.zeros_matrix(@attn_hidden_dim, @embedding_dim),
      grad_b_attn: b_attn_grad,
      grad_v_attn: v_attn_grad,
      grad_W_h: @nnlm.zeros_matrix(@attn_hidden_dim, @hidden_size),
      grad_b_h: b_h_grad,
      grad_W_o: @nnlm.zeros_matrix(@hidden_size, @vocab_size),
      grad_b_o: b_o_grad 
    })

    # Get the actual parameters *after* process_context has run
    actual_embeddings = @nnlm.instance_variable_get(:@embeddings)

    @embeddings[0] = @nnlm.subtract_vectors(@embeddings[0], @nnlm.scalar_multiply(@learning_rate, embedding_grad))

    actual_W_attn = @nnlm.instance_variable_get(:@W_attn)
    actual_b_attn = @nnlm.instance_variable_get(:@b_attn)
    actual_v_attn = @nnlm.instance_variable_get(:@v_attn)

    expected_b_attn = @nnlm.subtract_vectors(@b_attn, @nnlm.scalar_multiply(@learning_rate, b_attn_grad))
    expected_v_attn = @nnlm.subtract_vectors(@v_attn, @nnlm.scalar_multiply(@learning_rate, v_attn_grad))

    actual_W_h = @nnlm.instance_variable_get(:@W_h)
    actual_b_h = @nnlm.instance_variable_get(:@b_h)
    actual_W_o = @nnlm.instance_variable_get(:@W_o)
    actual_b_o = @nnlm.instance_variable_get(:@b_o)

    expected_b_h = @nnlm.subtract_vectors(@b_h, @nnlm.scalar_multiply(@learning_rate, b_h_grad))
    expected_b_o = @nnlm.subtract_vectors(@b_o, @nnlm.scalar_multiply(@learning_rate, b_o_grad))

    # Compare actual parameters with the expected final unchanged parameters
    assert_embedding_hash_in_delta @embeddings, actual_embeddings, DELTA, "Embeddings mismatch after update"
    assert_matrix_in_delta @W_attn, actual_W_attn, DELTA, "W_attn mismatch after update"
    assert_vector_in_delta expected_b_attn, actual_b_attn, DELTA, "b_attn mismatch after update"
    assert_vector_in_delta expected_v_attn, actual_v_attn, DELTA, "v_attn mismatch after update"

    assert_matrix_in_delta @W_h, actual_W_h, DELTA, "W_h mismatch after update"
    assert_vector_in_delta expected_b_h, actual_b_h, DELTA, "b_h mismatch after update"

    assert_matrix_in_delta @W_o, actual_W_o, DELTA, "W_o mismatch after update"
    assert_vector_in_delta expected_b_o, actual_b_o, DELTA, "b_o mismatch after update"

    assert_embedding_hash_in_delta @embeddings, actual_embeddings, DELTA, "Embeddings mismatch after update"
  end
end
