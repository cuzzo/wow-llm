#! /usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require "minitest/rg"
require "numo/narray"
require "cmath"
require "set"

require_relative "tokenizer"
require_relative "llm"

# Define a helper for comparing floating point arrays/matrices
module Minitest::Assertions
  # Asserts that two arrays (vectors) of floats are element-wise equal within a delta.
  def assert_vector_in_delta(expected_vec, actual_vec, delta = 1e-6, msg = nil)
    msg ||= "Expected vectors to be element-wise equal within delta #{delta}"
    assert_equal(expected_vec.shape, actual_vec.shape, "#{msg} (different sizes)")
    expected_vec.to_a.zip(actual_vec.to_a).each_with_index do |(exp, act), i|
      assert_in_delta(exp, act, delta, "#{msg} (difference at index #{i})")
    end
  end

  # Asserts that two nested arrays (matrices) of floats are element-wise equal within a delta.
  def assert_matrix_in_delta(expected_mat, actual_mat, delta = 1e-6, msg = nil)
    msg ||= "Expected matrices to be element-wise equal within delta #{delta}"
    assert_equal(expected_mat.shape, actual_mat.shape, "#{msg} (different number of rows)")
    expected_mat.shape.first.times do |i|
      assert_vector_in_delta(expected_mat[i, true], actual_mat[i, true], delta, "#{msg} (difference in row #{i})")
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


class TestNNLM < Minitest::Test
  DELTA = 1e-6 # Tolerance for float comparisons

  def setup
    # --- Fixed Hyperparameters for Predictable Tests ---
    @embedding_dim = 2
    @context_size = 2
    @hidden_size = 3
    @vocab_size = 5 # Includes [PAD], hello, world, foo, bar
    @learning_rate = 0.1 # Not used directly in forward/backward, but part of NNLM

    @nnlm = NNLM.new(
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      learning_rate: @learning_rate
      # tokenizer will be the dummy one
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
    embeddings = Numo::DFloat[
      [0.0, 0.0],          # [PAD] embedding (often zeros)
      [0.1, 0.2],          # hello
      [0.3, -0.1],         # world
      [0.2, 0.4],          # foo
      [-0.2, 0.1]          # bar
    ]
    input_concat_size = @context_size * @embedding_dim # 2 * 2 = 4

    # Dimensions: input_concat_size x hidden_size (4 x 3)
    w_h = Numo::DFloat[
      [0.1, 0.2, 0.3],
      [-0.1, 0.3, 0.1],
      [0.4, -0.2, 0.2],
      [0.2, 0.1, -0.3]
    ]
    # Dimensions: hidden_size (3)
    b_h = Numo::DFloat[0.05, -0.05, 0.1]

    # Dimensions: hidden_size x vocab_size (3 x 5)
    w_o = Numo::DFloat[
      [0.2, 0.1, -0.1, 0.3, 0.4],
      [-0.2, 0.4, 0.2, -0.1, 0.1],
      [0.3, -0.3, 0.1, 0.2, -0.2]
    ]
    # Dimensions: vocab_size (5)
    b_o = Numo::DFloat[0.1, 0.0, -0.1, 0.2, 0.05]

    @nnlm.instance_variable_set(:@embeddings, embeddings)
    @nnlm.instance_variable_set(:@W_h, w_h)
    @nnlm.instance_variable_set(:@b_h, b_h)
    @nnlm.instance_variable_set(:@W_o, w_o)
    @nnlm.instance_variable_set(:@b_o, b_o)

    # --- Define and Store Initial Fixed Parameters as INSTANCE Variables ---
    @initial_embeddings = Marshal.load(Marshal.dump(embeddings))
    @initial_W_h = w_h.copy
    @initial_b_h = b_h.copy
    @initial_W_o = w_o.copy
    @initial_b_o = b_o.copy

    # --- Set NNLM's state using DEEP COPIES of the initial parameters ---
    # This ensures the @nnlm instance can be modified without affecting @initial_* vars
    @nnlm.instance_variable_set(:@embeddings, Marshal.load(Marshal.dump(@initial_embeddings)))
    @nnlm.instance_variable_set(:@W_h, @initial_W_h.copy)
    @nnlm.instance_variable_set(:@b_h, @initial_b_h.copy)
    @nnlm.instance_variable_set(:@W_o, @initial_W_o.copy)
    @nnlm.instance_variable_set(:@b_o, @initial_b_o.copy)

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
    expected_input = Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    assert_equal expected_input, result[:input_layer]
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
    assert_equal Numo::DFloat.cast([0] * (@context_size * @embedding_dim)), result_with_pads[:input_layer]

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

  # ============================================================================
  # Test Forward Pass
  # ============================================================================
  def test_forward_pass_calculations
    puts "\n--- Testing Forward Pass ---"

    # --- 1. Expected Projection/Concatenation ---
    # Embeddings for indices 1 and 2 are [0.1, 0.2] and [0.3, -0.1]
    expected_input_layer = Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    puts "Expected Input Layer (Concatenated Embeddings): #{expected_input_layer.inspect}"

    # --- 2. Expected Hidden Layer Input ---
    # hidden_input = input_layer * W_h + b_h
    # input_layer (1x4) * W_h (4x3) -> (1x3)
    # [0.1, 0.2, 0.3, -0.1] * [[0.1, 0.2, 0.3], [-0.1, 0.3, 0.1], [0.4, -0.2, 0.2], [0.2, 0.1, -0.3]]
    # = [ (0.1*0.1 + 0.2*-0.1 + 0.3*0.4 + -0.1*0.2),  -> 0.01 - 0.02 + 0.12 - 0.02 = 0.09
    #     (0.1*0.2 + 0.2*0.3 + 0.3*-0.2 + -0.1*0.1),  -> 0.02 + 0.06 - 0.06 - 0.01 = 0.01
    #     (0.1*0.3 + 0.2*0.1 + 0.3*0.2 + -0.1*-0.3) ] -> 0.03 + 0.02 + 0.06 + 0.03 = 0.14
    # = [0.09, 0.01, 0.14]
    # Add bias b_h = [0.05, -0.05, 0.1]
    # hidden_input = [0.09+0.05, 0.01-0.05, 0.14+0.1] = [0.14, -0.04, 0.24]
    expected_hidden_input = Numo::DFloat[0.14, -0.04, 0.24]
    puts "Expected Hidden Input (Before Tanh): #{expected_hidden_input.inspect}"

    # --- 3. Expected Hidden Layer Activation (Tanh) ---
    expected_hidden_activation = expected_hidden_input.map { |x| CMath.tanh(x).real }
    # expected_hidden_activation approx = [0.139, -0.040, 0.235] (using more precision below)
    puts "Expected Hidden Activation (Tanh Output): #{expected_hidden_activation.inspect}"

    # --- 4. Expected Output Layer Scores ---
    # output_scores = hidden_activation * W_o + b_o
    # hidden_activation (1x3) * W_o (3x5) -> (1x5)
    # W_o = [[0.2, 0.1, -0.1, 0.3, 0.4], [-0.2, 0.4, 0.2, -0.1, 0.1], [0.3, -0.3, 0.1, 0.2, -0.2]]
    # b_o = [0.1, 0.0, -0.1, 0.2, 0.05]
    # Manually calculate hidden_activation * W_o + b_o using expected_hidden_activation
    manual_output_scores = @nnlm.add_vectors(
      @nnlm.multiply_vec_mat(expected_hidden_activation, @nnlm.instance_variable_get(:@W_o)),
      @nnlm.instance_variable_get(:@b_o)
    )
    expected_output_scores = manual_output_scores # Use calculated value for precision
    puts "Expected Output Scores (Before Softmax): #{expected_output_scores.inspect}"

    # --- 5. Expected Probabilities (Softmax) ---
    expected_probabilities = @nnlm.softmax(expected_output_scores)
    puts "Expected Probabilities (Softmax Output): #{expected_probabilities.inspect}"
    puts "Sum of Probabilities: #{expected_probabilities.sum}" # Should be close to 1.0

    # --- Action: Call the actual forward method ---
    forward_data = @nnlm.forward(@test_context_indices)

    # --- Assertions ---
    assert_instance_of Hash, forward_data, "Forward pass should return a Hash"
    assert_equal [:probabilities, :hidden_activation, :input_layer].to_set, forward_data.keys.to_set, "Forward data keys mismatch"

    puts "\nVerifying Forward Pass Results..."
    assert_vector_in_delta expected_input_layer, forward_data[:input_layer], DELTA, "Input layer (concatenated embeddings) mismatch"
    puts "Input Layer OK"

    assert_vector_in_delta expected_hidden_activation, forward_data[:hidden_activation], DELTA, "Hidden activation (tanh) mismatch"
    puts "Hidden Activation OK"

    assert_vector_in_delta expected_probabilities, forward_data[:probabilities], DELTA, "Probabilities (softmax) mismatch"
    assert_in_delta 1.0, forward_data[:probabilities].sum, DELTA * 10, "Probabilities should sum to 1.0" # Allow slightly larger delta for sum
    puts "Probabilities OK"
    puts "--- Forward Pass Test Complete ---"
  end

  # Backwards supplies gradients (errors) for hidden wieghts & biases, output weights and biases, and context indices
  def test_backward_only_updates_used_embeddings
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: Numo::DFloat[0.2, 0.1, 0.1, 0.2, 0.3],
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    }

    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # Only embeddings for words in the context should have non-zero gradients
    context_indices.each do |idx|
      assert gradients[:grad_embeddings].key?(idx), "Should have gradient for word #{idx}"
      refute_equal Array.new(@embedding_dim, 0.0), gradients[:grad_embeddings][idx],
                  "Gradient for used word #{idx} should not be all zeros"
    end

    # Words not in context should not have gradients
    all_word_idxs = @nnlm.instance_variable_get(:@embeddings).shape.first.times.to_a
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
      probabilities: Numo::DFloat[0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
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
    non_target_errors = output_errors.to_a.each_with_index.reject { |_, i| i == target_index }.map(&:first)
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
      probabilities: Numo::DFloat[0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
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
      grad_target = output_errors[h_idx, target_index]

      # If activation is positive, gradient should be negative (to increase W_o)
      # If activation is negative, gradient should be positive (to decrease W_o, making it less negative)
      # They should have opposite signs
      assert activation * grad_target <= 0,
             "grad_W_o[#{h_idx}][#{target_index}] sign should oppose hidden_activation[#{h_idx}] sign"

      non_target_idxs = (@nnlm.instance_variable_get(:@embeddings).shape.first.times.to_a - [target_index])
      non_target_idxs.each do |non_target_idx|
        grad_non_target = output_errors[h_idx, non_target_idx]

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
      probabilities: Numo::DFloat[0.0, 0.0, 0.0, 1.0, 0.0], # softmax guarantees probabilities are positive
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    }

    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # Output bias gradients should be zeros (probabilities - target_one_hot = 0)
    assert_equal Numo::DFloat.cast([0.0] * @nnlm.instance_variable_get(:@vocab_size)), gradients[:grad_b_o],
                "Output bias gradients should be zero when prediction is perfect"
  end

  def test_backward_produces_non_zero_gradients
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3  # "foo"
    forward_data = {
      probabilities: Numo::DFloat[0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    }

    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # Helper to check if a matrix has any non-zero elements
    def non_zero?(matrix)
      matrix.flatten.to_a.any? { |x| x != 0 }
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
      probabilities: Numo::DFloat[0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    }

    # Compare gradients when target is "foo" vs "bar"
    gradients_foo = @nnlm.backward(context_indices, 3, forward_data)  # target: foo
    gradients_bar = @nnlm.backward(context_indices, 4, forward_data)  # target: bar

    # Gradients should be different
    assert_equal false, gradients_foo[:grad_b_o] == gradients_bar[:grad_b_o],
                 "Different targets should produce different output gradients"
    assert_equal false, gradients_foo[:grad_b_h] == gradients_bar[:grad_b_h],
                 "Different targets should produce different hidden gradients"
  end

  def test_backward_generates_same_gradients_for_same_inputs
    context_indices = [1, 2]  # "hello", "world"
    target_index = 3
    forward_data = {
      probabilities: Numo::DFloat[0.2, 0.1, 0.1, 0.2, 0.3], # softmax guarantees probabilities are positive
      hidden_activation: Numo::DFloat[0.1, -0.1, 0.2],
      input_layer: Numo::DFloat[0.1, 0.2, 0.3, -0.1]
    }

    # Run backward twice with same inputs
    gradients1 = @nnlm.backward(context_indices, target_index, forward_data)
    gradients2 = @nnlm.backward(context_indices, target_index, forward_data)

    # Gradients should be identical
    assert_equal gradients1, gradients2, "Same inputs should produce identical gradients"
  end

  # ============================================================================
  # Test Backward Pass
  # ============================================================================
  def test_backward_pass_gradients
    puts "\n--- Testing Backward Pass ---"

    # --- Prerequisite: Run forward pass to get intermediate values ---
    # Use the same fixed inputs and parameters from setup
    forward_data = @nnlm.forward(@test_context_indices)
    probabilities = forward_data[:probabilities]
    hidden_activation = forward_data[:hidden_activation]
    input_layer = forward_data[:input_layer] # Concatenated [0.1, 0.2, 0.3, -0.1]

    target_index = @test_target_index # Target index is 3 ("foo")

    # --- 1. Expected Gradient of Loss w.r.t Output Scores (d_output_scores) ---
    # dL/dOutput_Scores = probabilities - target_one_hot
    # target_one_hot = [0, 0, 0, 1, 0] for target_index 3
    expected_d_output_scores = probabilities.dup
    expected_d_output_scores[target_index] -= 1.0
    puts "Expected dL/dOutput_Scores: #{expected_d_output_scores.inspect}"

    # --- 2. Expected Gradients for Output Layer (W_o, b_o) ---
    # grad_b_o = d_output_scores
    expected_grad_b_o = expected_d_output_scores
    puts "Expected grad_b_o: #{expected_grad_b_o.inspect}"

    # grad_W_o = outer_product(hidden_activation, d_output_scores) (hidden_size x vocab_size) -> (3 x 5)
    expected_grad_w_o = @nnlm.outer_product(hidden_activation, expected_d_output_scores)
    puts "Expected grad_W_o (shape #{expected_grad_w_o.shape}x#{expected_grad_w_o.shape}): #{expected_grad_w_o.inspect}"

    # --- 3. Expected Gradient w.r.t Hidden Activation Input Signal ---
    # d_hidden_input_signal = d_output_scores * W_o^T
    # d_output_scores (1x5) * W_o^T (5x3) -> (1x3)
    w_o_transpose = @nnlm.transpose(@nnlm.instance_variable_get(:@W_o))
    expected_d_hidden_input_signal = @nnlm.multiply_vec_mat(expected_d_output_scores, w_o_transpose)
    puts "Expected dL/dHiddenActivation (Signal before dtanh): #{expected_d_hidden_input_signal.inspect}"

    # --- 4. Expected Gradient w.r.t Hidden Input (d_hidden_input) ---
    # d_hidden_input = d_hidden_input_signal * dtanh(hidden_activation)
    # dtanh(y) = 1 - y^2
    d_tanh = @nnlm.dtanh(hidden_activation)
    expected_d_hidden_input = @nnlm.multiply_elementwise(expected_d_hidden_input_signal, d_tanh)
    puts "Expected dL/dHiddenInput (After dtanh): #{expected_d_hidden_input.inspect}"

    # --- 5. Expected Gradients for Hidden Layer (W_h, b_h) ---
    # grad_b_h = d_hidden_input
    expected_grad_b_h = expected_d_hidden_input
    puts "Expected grad_b_h: #{expected_grad_b_h.inspect}"

    # grad_W_h = outer_product(input_layer, d_hidden_input) (input_concat_size x hidden_size) -> (4 x 3)
    expected_grad_w_h = @nnlm.outer_product(input_layer, expected_d_hidden_input)
    puts "Expected grad_W_h (shape #{expected_grad_w_h.shape}x#{expected_grad_w_h.shape}): #{expected_grad_w_h.inspect}"

    # --- 6. Expected Gradient w.r.t Input Layer (d_input_layer) ---
    # d_input_layer = d_hidden_input * W_h^T
    # d_hidden_input (1x3) * W_h^T (3x4) -> (1x4)
    w_h_transpose = @nnlm.transpose(@nnlm.instance_variable_get(:@W_h))
    expected_d_input_layer = @nnlm.multiply_vec_mat(expected_d_hidden_input, w_h_transpose)
    puts "Expected dL/dInputLayer (Gradient for concatenated embeddings): #{expected_d_input_layer.inspect}"

    # --- 7. Expected Gradients for Embeddings ---
    # Distribute d_input_layer back to the embeddings used in the context
    # Context indices were [1, 2]
    expected_grad_embeddings = Hash.new { |h, k| h[k] = Numo::DFloat.cast([0.0] * @embedding_dim) }
    context_indices = @test_context_indices
    context_indices.each_with_index do |word_ix, i|
      start_idx = i * @embedding_dim
      end_idx = start_idx + @embedding_dim - 1
      embedding_grad_slice = expected_d_input_layer[start_idx..end_idx]
      # Important: Use add_vectors for accumulation if the same index appeared multiple times
      expected_grad_embeddings[word_ix] = @nnlm.add_vectors(expected_grad_embeddings[word_ix], embedding_grad_slice)
    end
    puts "Expected grad_embeddings: #{expected_grad_embeddings.inspect}"


    # --- Action: Call the actual backward method ---
    gradients = @nnlm.backward(context_indices, target_index, forward_data)

    # --- Assertions ---
    assert_instance_of Hash, gradients, "Backward pass should return a Hash"
    expected_keys = [:grad_embeddings, :grad_W_h, :grad_b_h, :grad_W_o, :grad_b_o].to_set
    assert_equal expected_keys, gradients.keys.to_set, "Backward gradients keys mismatch"

    puts "\nVerifying Backward Pass Results..."
    # Output Layer Gradients
    assert_vector_in_delta expected_grad_b_o, gradients[:grad_b_o], DELTA, "Gradient b_o mismatch"
    puts "grad_b_o OK"
    assert_matrix_in_delta expected_grad_w_o, gradients[:grad_W_o], DELTA, "Gradient W_o mismatch"
    puts "grad_W_o OK"

    # Hidden Layer Gradients
    assert_vector_in_delta expected_grad_b_h, gradients[:grad_b_h], DELTA, "Gradient b_h mismatch"
    puts "grad_b_h OK"
    assert_matrix_in_delta expected_grad_w_h, gradients[:grad_W_h], DELTA, "Gradient W_h mismatch"
    puts "grad_W_h OK"

    # Embedding Gradients
    # Check that only expected keys have non-zero gradients if applicable
    assert_equal expected_grad_embeddings.keys.sort, gradients[:grad_embeddings].keys.sort, "Gradient embeddings keys mismatch"
    assert_embedding_hash_in_delta expected_grad_embeddings, gradients[:grad_embeddings], DELTA, "Gradient embeddings values mismatch"
    puts "grad_embeddings OK"

    puts "--- Backward Pass Test Complete ---"
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

    actual_loss = @nnlm.process_context(padded_sentence, idx)

    post_embeddings = @nnlm.instance_variable_get(:@embeddings).dup
    post_pred = @nnlm.forward(predict_sentence)[:probabilities]

    assert_equal initial_embeddings[t_pad_idx, true], post_embeddings[t_pad_idx, true] # pad should be unchanged
    assert_equal initial_embeddings[t_foo_idx, true], post_embeddings[t_foo_idx, true] # foo should be unchanged
    assert_equal initial_embeddings[t_bar_idx, true], post_embeddings[t_bar_idx, true] # bar should be unchanged

    refute_equal initial_embeddings[t_hello_idx, true], post_embeddings[t_hello_idx, true] # hello should be unchanged
    refute_equal initial_embeddings[t_world_idx, true], post_embeddings[t_world_idx, true] # hello should be unchanged

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

    10.times do |i|
      actual_loss = @nnlm.process_context(padded_sentence, idx)

      initial_pred = @nnlm.forward(predict_sentence)[:probabilities]
      initial_embeddings = @nnlm.instance_variable_get(:@embeddings).dup

      actual_loss = @nnlm.process_context(padded_sentence, idx)

      post_embeddings = @nnlm.instance_variable_get(:@embeddings).dup
      post_pred = @nnlm.forward(predict_sentence)[:probabilities]

      assert_equal initial_embeddings[t_pad_idx, true], post_embeddings[t_pad_idx, true] # pad should be unchanged
      assert_equal initial_embeddings[t_foo_idx, true], post_embeddings[t_foo_idx, true] # foo should be unchanged
      assert_equal initial_embeddings[t_bar_idx, true], post_embeddings[t_bar_idx, true] # bar should be unchanged

      refute_equal initial_embeddings[t_hello_idx, true], post_embeddings[t_hello_idx, true] # hello should be unchanged
      refute_equal initial_embeddings[t_world_idx, true], post_embeddings[t_world_idx, true] # hello should be unchanged

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

    actual_loss = @nnlm.process_context(padded_sentence, idx)

    post_embeddings = @nnlm.instance_variable_get(:@embeddings).dup
    post_pred = @nnlm.forward(predict_sentence)[:probabilities]

    assert_equal initial_embeddings[t_foo_idx, true], post_embeddings[t_foo_idx, true] # foo should be ununchanged
    assert_equal initial_embeddings[t_bar_idx, true], post_embeddings[t_bar_idx, true] # bar should be unchanged
    assert_equal initial_embeddings[t_hello_idx, true], post_embeddings[t_hello_idx, true] # hello should be unchanged
    assert_equal initial_embeddings[t_world_idx, true], post_embeddings[t_world_idx, true] # hello should be unchanged

    refute_equal initial_embeddings[t_pad_idx, true], post_embeddings[t_pad_idx, true] # pad should be changed

    assert post_pred[t_pad_idx] <= initial_pred[t_pad_idx] # probability of picking pad decreased
    assert post_pred[t_world_idx] <= initial_pred[t_world_idx] # probability of picking world decreased
    assert post_pred[t_foo_idx] <= initial_pred[t_foo_idx] # probability of picking foo decreased
    assert post_pred[t_bar_idx] <= initial_pred[t_bar_idx] # probability of picking bar decreased

    assert post_pred[t_hello_idx] >= initial_pred[t_hello_idx] # probability of picking hello increased
  end

  # ============================================================================
  # Test process_context Return Value (Loss) and Side Effects (Parameter Updates)
  # ============================================================================
  def test_process_context_returns_loss_and_updates_parameters
    puts "\n--- Testing process_context (Return Value and Side Effect) ---"

    # --- Arrange ---
    # 1. Get initial parameter state (done in setup, stored in @initial_*)

    # 2. Determine expected loss
    #    Requires running forward pass with initial parameters
    forward_data_for_loss = @nnlm.forward(@test_context_indices)
    probabilities_for_loss = forward_data_for_loss[:probabilities]
    # Loss = -log(probability of target_index)
    expected_loss = -Math.log(probabilities_for_loss[@test_target_index] + 1e-9) # Use target_index=3
    puts "Expected Loss: #{expected_loss}"

    # 3. Determine expected gradients (run backward pass conceptually or actually)
    #    Use the same forward data as calculated above for consistency
    expected_gradients = @nnlm.backward(@test_context_indices, @test_target_index, forward_data_for_loss)

    # 4. Calculate expected *updated* parameters based on initial state, gradients, and LR
    lr = @nnlm.instance_variable_get(:@learning_rate)

    expected_embeddings = Marshal.load(Marshal.dump(@initial_embeddings))
    expected_gradients[:grad_embeddings].each do |word_ix, grad|
      expected_embeddings[word_ix, true] -= lr * grad
    end

    expected_W_h = @initial_W_h - (lr * expected_gradients[:grad_W_h])
    expected_b_h = @initial_b_h - (lr * expected_gradients[:grad_b_h])
    expected_W_o = @initial_W_o - (lr * expected_gradients[:grad_W_o])
    expected_b_o = @initial_b_o - (lr * expected_gradients[:grad_b_o])

    puts "Initial b_h: #{@initial_b_h.inspect}"
    puts "Expected grad_b_h: #{expected_gradients[:grad_b_h].inspect}"
    puts "Expected final b_h: #{expected_b_h.inspect}"

    # --- Act ---
    # Call process_context with the test sentence and index
    # Store the returned loss value
    actual_loss = @nnlm.process_context(@test_padded_sentence, @test_processing_index)

    # --- Assert ---
    puts "\nVerifying process_context Results..."

    # 1. Assert the returned loss value
    assert_in_delta expected_loss, actual_loss, DELTA, "Returned loss value mismatch"
    puts "Returned Loss OK"

    # 2. Assert the parameter update side effect
    # Get the actual parameters *after* process_context has run
    actual_embeddings = @nnlm.instance_variable_get(:@embeddings)
    actual_W_h = @nnlm.instance_variable_get(:@W_h)
    actual_b_h = @nnlm.instance_variable_get(:@b_h)
    actual_W_o = @nnlm.instance_variable_get(:@W_o)
    actual_b_o = @nnlm.instance_variable_get(:@b_o)

    # Compare actual parameters with the expected final parameters
    assert_matrix_in_delta expected_embeddings, actual_embeddings, DELTA, "Embeddings mismatch after update"
    puts "Embeddings Update OK"

    assert_matrix_in_delta expected_W_h, actual_W_h, DELTA, "W_h mismatch after update"
    puts "W_h Update OK"
    assert_vector_in_delta expected_b_h, actual_b_h, DELTA, "b_h mismatch after update"
    puts "b_h Update OK"

    assert_matrix_in_delta expected_W_o, actual_W_o, DELTA, "W_o mismatch after update"
    puts "W_o Update OK"
    assert_vector_in_delta expected_b_o, actual_b_o, DELTA, "b_o mismatch after update"
    puts "b_o Update OK"

    # Sanity check: ensure embeddings not part of the context gradient didn't change
    assert_vector_in_delta @initial_embeddings[0, true], actual_embeddings[0, true], DELTA, "Embedding for index 0 (PAD) should not change"
    assert_vector_in_delta @initial_embeddings[4, true], actual_embeddings[4, true], DELTA, "Embedding for index 4 (bar) should not change"
    puts "Unrelated Embeddings Unchanged OK"

    puts "--- process_context Test Complete ---"
  end
end
