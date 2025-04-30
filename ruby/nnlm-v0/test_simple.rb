#! /usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require "minitest/rg"
require "cmath"
require "set"
require "byebug"

require_relative "present"

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


# Deep copy helper using JSON serialization (more robust than Marshal for simple structures)
def deep_copy(obj)
  # Ensure floating point precision is reasonably maintained during JSON conversion
  Marshal.load(Marshal.dump(obj))
end

# - If exp_input > 0, tanh(exp_input) > 0.
# - If exp_input < 0, tanh(exp_input) < 0.
# - If exp_input == 0, tanh(exp_input) == 0.
# Multiplying them is a concise way to check this:
# The product is non-negative (>= 0) if signs match or one/both are zero.
# The product is negative (< 0) only if the signs differ.
def signs_match?(a, b)
  (a * b) >= 0
end

class TestSimpleNNLM < Minitest::Test
  include BasicLinAlg # Make helper functions available if needed directly
  DELTA = 1e-6 # Tolerance for float comparisons

  def setup
    # --- Simplified Hyperparameters ---
    @embedding_dim = 2
    @context_size = 1
    @hidden_size = 2 # Number of neurons
    @vocab_size = 2
    @learning_rate = 0.1

    @nnlm = NNLMPresenter.new(
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      learning_rate: @learning_rate
    )

    # --- Manually Set Vocabulary ---
    vocab = { "false" => 0, "true" => 1 }
    ix_to_word = vocab.keys.sort_by { |k| vocab[k] } # Ensure order [false, true]
    word_to_ix = ix_to_word.each_with_index.to_h

    @nnlm.instance_variable_set(:@vocab_size, @vocab_size)
    @nnlm.instance_variable_set(:@word_to_ix, word_to_ix)
    @nnlm.instance_variable_set(:@ix_to_word, ix_to_word)

    # --- Manually Set Fixed Parameters for Predictability ---

    embeddings = {
      0 => [0.0, 0.0], # "cat"
      1 => [0.0, 1.0]  # "dog"
    }

    # z0 = w10 + b0 => 0.5 + (-0.5) = 0
    # z1 = w11 + b1 => 1.5 + 0.5 = 2
  
    # COLUMN FOR EACH INPUT IN INPUT LAYER
    # ROW FOR EACH NUERON
    w_h = [
      [0.1, 0.2],  # w00, w01 (less critical for context [0] calculation)
      [0.5, 1.5]   # w10, w11 (Chosen for target z0, z1)
    ]
    b_h = [-0.5, 0.5] # b0, b1 (Chosen for target z0, z1)

    # Output Weights W_o: hidden_size x vocab_size (2 x 2) - Keep as before
    w_o = [
      [ 0.6, -0.4],
      [-0.1,  0.7]
    ]
    # Output Biases b_o: vocab_size (2) - Keep as before
    b_o = [0.1, -0.1]

    # --- Store Initial Fixed Parameters (Using DEEP COPIES) ---
    @embeddings = deep_copy(embeddings)
    @W_h = deep_copy(w_h)
    @b_h = deep_copy(b_h)
    @W_o = deep_copy(w_o)
    @b_o = deep_copy(b_o)

    # --- Set NNLM's state using DEEP COPIES ---
    # Ensure the instance variables get the *new* embeddings
    @nnlm.instance_variable_set(:@embeddings, deep_copy(embeddings))
    @nnlm.instance_variable_set(:@W_h, deep_copy(w_h))
    @nnlm.instance_variable_set(:@b_h, deep_copy(b_h))
    @nnlm.instance_variable_set(:@W_o, deep_copy(w_o))
    @nnlm.instance_variable_set(:@b_o, deep_copy(b_o))

    # --- Define Fixed Input for Tests ---
    # Context: "dog" -> index [1]
    @test_target_index = 1
    @test_context_indices = [@test_target_index]
    # Target: "false" -> index 1
    # A simple sentence: "cat dog" -> [0, 1]
    @test_sentence_indices = [0, 1]
    @test_processing_index = 0
  end
#
#  # ============================================================================
#  # Test Forward Pass
#  # ============================================================================
#  def test_forward_input_layer_is_correct_embedding
#    result = @nnlm.forward([0]) # Context: "cat"
#    expected_input = [0.0, 0.0]
#    assert_vector_in_delta expected_input, result[:input_layer], DELTA
#  end
#
#  def test_forward_probabilities_sum_to_one
#    result = @nnlm.forward([0]) # Context: "cat"
#    assert_in_delta 1.0, result[:probabilities].sum, DELTA * 10 # Allow slightly larger delta for sum
#    assert_equal @vocab_size, result[:probabilities].size
#    result[:probabilities].each do |prob|
#      assert prob.between?(0.0, 1.0), "Probability #{prob} should be between 0 and 1"
#    end
#
#    result = @nnlm.forward([1]) # Context: "dog"
#    assert_in_delta 1.0, result[:probabilities].sum, DELTA * 10
#    assert_equal @vocab_size, result[:probabilities].size
#    result[:probabilities].each do |prob|
#      assert prob.between?(0.0, 1.0), "Probability #{prob} should be between 0 and 1"
#    end
#  end
#
#  def test_forward_hidden_activation_values_bounded
#    result = @nnlm.forward([0]) # Context: "cat"
#    assert_equal @hidden_size, result[:hidden_activation].size
#    result[:hidden_activation].each do |val|
#      assert val.between?(-1.0, 1.0), "Hidden activation #{val} should be between -1 and 1"
#    end
#  end
#
#  def test_forward_different_inputs_produce_different_outputs
#    result_true = @nnlm.forward([0]) # Context: "cat"
#    result_false = @nnlm.forward([1]) # Context: "dog"
#
#    refute_equal result_true[:probabilities], result_false[:probabilities]
#    refute_equal result_true[:hidden_activation], result_false[:hidden_activation]
#    refute_equal result_true[:input_layer], result_false[:input_layer]
#  end
#
#  def test_forward_consistent_outputs_for_same_inputs
#    result1 = @nnlm.forward([0]) # Context: "cat"
#    result2 = @nnlm.forward([0]) # Context: "cat" again
#    assert_equal result1, result2
#  end
#
#  # It is crucial that hidden_activation matches 
#  # While `tanh` changes the size (magnitude) of the number, it's crucial that it doesn't flip 
#  # The basic direction (the sign).
#  #
#  # If the "raw signal" calculated in step 1 is positive, 
#  #   It means the combined inputs were leaning in a "positive" direction for that hidden neuron. 
#  #   The final output after `tanh` should still be positive (though maybe smaller, squashed closer towards +1).
#  # If the "raw signal" was negative:
#  #   The inputs were leaning "negative". 
#  #   The final output after `tanh` should still be negative (squashed closer towards -1).
#  #
#  # After `tanh` we get how steep the curve is at a particular point (nueron).
#  # This steepness tells us how much the output (activation) changes if we make a tiny change
#  # to the input / raw signal.
#  #
#  # The cool thing about `tanh` is that you can figure out it's slope at any given point by
#  # knowing only the activation (which is all we know)
#  # 
#  # The formula is `slope = 1 - activation²`
#  #
#  # What does this mean? If the activation is near 0, it means that number squared is still tiny.
#  # The slope is 1 - (tiny number), which is close to 1. 
#  # This is the steepest part of the tanh curve (remember it squishes between -1 and 1).
#  #
#  # A small change in the input causes a relatively large change in the output.
#  #
#  # If the activation value is close to +1 or -1 (meaning the "raw signal" input was large positive or large negative), 
#  # then activation² is close to 1. 
#  # The slope is 1 - (number close to 1), which is close to 0. 
#  # This is where the tanh curve flattens out. 
#  # Even a significant change in the input causes only a very tiny change in the output 
#  # because it's already near its max/min.
#  def test_forward_hidden_activation_sign_matches_input_sign
#    # Test both possible contexts in this simple setup
#    contexts_to_test = [[0], [1]] # Contexts: ["cat"], ["dog"]
#
#    contexts_to_test.each do |context_indices|
#      context_idx = context_indices[0]
#
#      # --- Arrange ---
#      # 1. Get the input layer for this context using the initial parameters
#      #    (Using @initial_embeddings ensures we test against the known start state)
#      input_layer = @initial_embeddings[context_idx]
#      unless input_layer
#        raise "Test setup error: Could not find initial embedding for index #{context_idx}"
#      end
#
#      # 2. Calculate the expected input *to* the tanh function manually
#      #    hidden_input = input_layer * W_h + b_h
#      expected_hidden_input_pre_tanh = add_vectors(
#        multiply_vec_mat(input_layer, @initial_W_h), # Use initial weights
#        @initial_b_h                                 # Use initial bias
#      )
#
#      # --- Act ---
#      # 3. Run the actual forward pass using the @nnlm instance
#      #    (which should also be initialized with the same initial parameters)
#      forward_data = @nnlm.forward(context_indices)
#      actual_hidden_activation_post_tanh = forward_data[:hidden_activation]
#
#      # --- Assert ---
#      # 4. Verify that the sign of each activation matches the sign of its input
#      assert_equal expected_hidden_input_pre_tanh.size, actual_hidden_activation_post_tanh.size,
#                     "Hidden vector sizes should match for context #{context_indices}"
#
#      expected_hidden_input_pre_tanh.zip(actual_hidden_activation_post_tanh).each_with_index do |(exp_input, act_activation), i|
#        # Check if the signs match.
#        assert signs_match?(exp_input, act_activation),
#                 "Sign mismatch at hidden index #{i} for context #{context_indices}. Input to tanh was #{exp_input.round(6)}, resulting activation was #{act_activation.round(6)}. Signs should match."
#
#        # Additionally, check the near-zero case more explicitly for robustness
#        if exp_input.abs < DELTA
#          assert_in_delta 0.0, act_activation, DELTA,
#                            "Activation at hidden index #{i} for context #{context_indices} should be near zero when input is near zero (#{exp_input}), but got #{act_activation}"
#
#        # Optional: Check non-zero cases explicitly if needed
#        # elsif exp_input > 0
#        #     assert act_activation > 0.0, "Activation should be > 0 for positive input"
#        # else # exp_input < 0
#        #     assert act_activation < 0.0, "Activation should be < 0 for negative input"
#
#        end
#      end
#    end
#  end
#
#  def test_forward_hidden_slope_is_steepest_near_zero_input
#    # This test relies on the specific W_h and b_h set in setup,
#    # which ensure that for input context [0] ("true" -> embedding [0.0, 1.0]),
#    # hidden neuron 0 receives input z=0 (pre-tanh), resulting in slope=1 (steepest),
#    # while hidden neuron 1 receives input z=2 (pre-tanh), resulting in slope ~= 0.07 (flat).
#
#    context_idx = 1 # Input "cat" -> embedding [0.0, 1.0]
#    context_indices = [context_idx] 
#
#    # --- Verify Pre-Calculation (using initial parameters) ---
#    # This step recalculates z to ensure our setup modification worked as intended.
#    input_layer = @initial_embeddings[context_idx]
#    expected_hidden_input_pre_tanh = add_vectors(
#        multiply_vec_mat(input_layer, @initial_W_h),
#        @initial_b_h
#    )
#
#    # Check if our setup created the desired pre-tanh inputs (z0=0, z1=2)
#    assert_in_delta(0.0, expected_hidden_input_pre_tanh[0], DELTA,
#                      "Test setup check failed: Input to Neuron 0 (z0) should be near 0.0")
#    assert_in_delta(2.0, expected_hidden_input_pre_tanh[1], DELTA,
#                      "Test setup check failed: Input to Neuron 1 (z1) should be near 2.0")
#
#    # --- Act: Get actual activations from the forward pass ---
#    forward_data = @nnlm.forward(context_indices)
#    actual_hidden_activation_post_tanh = forward_data[:hidden_activation]
#
#    # --- Calculate Slopes from Actual Activations ---
#    # Recall: Slope = 1 - activation^2
#    actual_slopes = actual_hidden_activation_post_tanh.map { |a| 1.0 - a**2 }
#
#    # --- Assert ---
#    # Neuron 0: Had input z=0. Expect activation a=0. Expect slope s=1 - 0^2 = 1.
#    assert_in_delta(0.0, actual_hidden_activation_post_tanh[0], DELTA,
#                      "Activation for Neuron 0 (input z=0) should be near 0")
#    assert_in_delta(1.0, actual_slopes[0], DELTA,
#                      "Slope for Neuron 0 (input z=0) should be near 1.0 (steepest), but got #{actual_slopes[0].round(6)}")
#
#    # Neuron 1: Had input z=2. Expect activation a=tanh(2) ~= 0.964. Expect slope s=1 - tanh(2)^2 ~= 0.0707.
#    expected_activation_1 = CMath.tanh(expected_hidden_input_pre_tanh[1]).real
#    expected_slope_1 = 1.0 - expected_activation_1**2
#    assert_in_delta(expected_activation_1, actual_hidden_activation_post_tanh[1], DELTA,
#                      "Activation for Neuron 1 (input z=2) should be near tanh(2) ~= #{expected_activation_1.round(6)}")
#    assert_in_delta(expected_slope_1, actual_slopes[1], DELTA,
#                      "Slope for Neuron 1 (input z=2) should be near #{expected_slope_1.round(6)} (flat), but got #{actual_slopes[1].round(6)}")
#    # Also check that the slope for Neuron 1 is indeed small
#    assert actual_slopes[1] < 0.1,
#           "Slope for Neuron 1 (#{actual_slopes[1].round(6)}) should be significantly less than 1 (flat)."
#
#    # The core check: Neuron 0's slope should be greater than Neuron 1's slope
#    assert actual_slopes[0] > actual_slopes[1],
#           "Slope for Neuron 0 (#{actual_slopes[0].round(6)}) should be steeper than Neuron 1 (#{actual_slopes[1].round(6)}) " +
#           "because Neuron 0's input (z=#{expected_hidden_input_pre_tanh[0].round(1)}) is closer to 0 " +
#           "than Neuron 1's input (z=#{expected_hidden_input_pre_tanh[1].round(1)})."
#  end
#

  # Goal: At every stage, we want to amplify the error
  def test_binary_nn
    # --- Simplified Hyperparameters ---
    @embedding_dim = 1
    @context_size = 2
    @hidden_size = 2 # Number of neurons
    @vocab_size = 2
    @learning_rate = 1.0

    @nnlm = NNLMPresenter.new(
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      learning_rate: @learning_rate
    )

    # --- Manually Set Vocabulary ---
    vocab = { "false" => 0, "true" => 1 }
    ix_to_word = vocab.keys.sort_by { |k| vocab[k] } # Ensure order [false, true]
    word_to_ix = ix_to_word.each_with_index.to_h

    @nnlm.instance_variable_set(:@vocab_size, @vocab_size)
    @nnlm.instance_variable_set(:@word_to_ix, word_to_ix)
    @nnlm.instance_variable_set(:@ix_to_word, ix_to_word)

    # --- Manually Set Fixed Parameters for Predictability ---

    embeddings = {
      0 => [0.5], # "false"
      1 => [1.0]  # "true"
    }

    # z0 = w10 + b0 => 0.5 + (-0.5) = 0
    # z1 = w11 + b1 => 1.5 + 0.5 = 2
    
    w_h = [ [0.01, -0.5], [ 0.1, 0.1] ] # Moves column 0 in wrong direction (still positive), moves column 1 in wrong direction (negative)
    b_h = [0.5, -0.1] # Moves column 0 in wrong direction (more positive), moveces column 1 in wrong direction (more negative)

    w_o = [ [1.5, 1.25], [2.0, 3.0] ]  # Amplifies the positive even more for 0, Amplifes the negative to be worse for 1
    # Output Biases b_o: vocab_size (2) - Keep as before
    b_o = [1.0, -2.0] # Amplifies positives for 0, Amplifies negative to be worse for 1

    @nnlm.instance_variable_set(:@embeddings, deep_copy(embeddings))
    @nnlm.instance_variable_set(:@W_h, deep_copy(w_h))
    @nnlm.instance_variable_set(:@b_h, deep_copy(b_h))
    @nnlm.instance_variable_set(:@W_o, deep_copy(w_o))
    @nnlm.instance_variable_set(:@b_o, deep_copy(b_o))

    # hi = [0.25, 1.0] [ 0.5 * 1.0 + 0.5 * -0.5, 1.0 ]
    # ha = tanh(hi) ~~ [ .25, .75 ]
    # ho = [1.625, -2 ]  [ .25 * 1.0 + .25 * 0.5 + 1, - 2.0 ] 
    # p = softmax(ho)  ~~ [ .975, .025 ]
    fd = @nnlm.forward([0, 0])
    errs = @nnlm.backward([0], 1, fd)

    puts "BEFORE:"
    puts @nnlm.to_a

    # each cell in vector sum product of colum in matrix

    # W_h column 0 unchanged, why?
    # b_h item 0 unchanged, why?
    @nnlm.update_parameters(errs) 

    puts "AFTER:"
    puts @nnlm.to_a
    byebug


    assert_equal true, false
    1
  end
end
