#! /usr/bin/env ruby

require 'minitest/autorun'
require 'minitest/pride'
require_relative 'llm'

class NGramLLMTest < Minitest::Test
  TOKEN_A = 1
  TOKEN_B = 2
  TOKEN_C = 3

  def setup
    @llm = NGramLLM.new(3)  # Default trigram model
  end

  def test_initialization
    assert_equal 3, @llm.n
    assert_empty @llm.model
    assert_empty @llm.vocab
  end

  def test_initialization_with_invalid_n
    assert_raises(ArgumentError) do
      NGramLLM.new(1)
    end
  end

  def test_training
    def context_key(id)
      @llm.send(:context_id, @llm.send(:tokenize, id))
    end

    def char_key(id)
      @llm.send(:tokenize, id).first
    end

    expected_model = {
      # Bi-grams
      context_key("h") => { char_key("e") => 2},
      context_key("e") => { char_key("y") => 1, char_key("l") => 1},
      context_key("l") => { char_key("l") => 1, char_key("o") => 1, char_key("d") => 1},
      context_key("o") => { char_key(" ") => 1, char_key("r") => 1},
      context_key(" ") => { char_key("w") => 1, char_key("h") => 1},
      context_key("w") => { char_key("o") => 1},
      context_key("r") => { char_key("l") => 1},
      context_key("d") => { char_key(" ") => 1},

      # Tri-grams
      context_key("he") => { char_key("l") => 1, char_key("y") => 1 },
      context_key("el") => { char_key("l") => 1 },
      context_key("ll") => { char_key("o") => 1 },
      context_key("lo") => { char_key(" ") => 1 },
      context_key("o ") => { char_key("w") => 1 },
      context_key(" w") => { char_key("o") => 1 },
      context_key("wo") => { char_key("r") => 1 },
      context_key("or") => { char_key("l") => 1 },
      context_key("rl") => { char_key("d") => 1 },
      context_key("ld") => { char_key(" ") => 1 },
      context_key("d ") => { char_key("h") => 1 },
      context_key(" h") => { char_key("e") => 1 },

      # Quad-grams
      context_key("hel") => { char_key("l") => 1 },
      context_key("ell") => { char_key("o") => 1 },
      context_key("llo") => { char_key(" ") => 1 },
      context_key("lo ") => { char_key("w") => 1 },
      context_key("o w") => { char_key("o") => 1 },
      context_key(" wo") => { char_key("r") => 1 },
      context_key("wor") => { char_key("l") => 1 },
      context_key("orl") => { char_key("d") => 1 },
      context_key("rld") => { char_key(" ") => 1 },
      context_key("ld ") => { char_key("h") => 1 },
      context_key("d h") => { char_key("e") => 1 },
      context_key(" he") => { char_key("y") => 1 },
    }
    llm4 = NGramLLM.new(4)
    llm4.train("hello world hey")

    assert_equal expected_model, llm4.model
    assert_equal 9, llm4.vocab.size  # 10 unique bytes in "hello world hey"

    @llm.train("abcde fghij")
    assert_equal 11, @llm.vocab.size # 11 unique bytes in "abcde fghij"
    
    # Verify model structure
    assert @llm.model.keys.size > 0
    assert @llm.model.values.all? { |dict| dict.is_a?(Hash) }
  end

  def test_context_id_consistent
    # Test that the same context always produces the same ID
    context = "abc".bytes
    id1 = @llm.send(:context_id, context)
    id2 = @llm.send(:context_id, context)
    assert_equal id1, id2
  end

  def test_context_id_different
    # Test that different contexts produce different IDs
    context1 = "abc".bytes
    context2 = "xyz".bytes
    id1 = @llm.send(:context_id, context1)
    id2 = @llm.send(:context_id, context2)
    refute_equal id1, id2
  end

  def test_weighted_choice
    # Create a deterministic distribution for testing
    options = { 1 => 10, 2 => 0, 3 => 0 }
    # If weight of option 1 is 10 and others are 0, it should always choose 1
    10.times do
      assert_equal 1, @llm.send(:weighted_choice, options)
    end
  end

  def test_weighted_choice_distribution
    # Test that weighted_choice respects probability distribution
    options = { 1 => 100, 2 => 0 }
    100.times do
      assert_equal 1, @llm.send(:weighted_choice, options)
    end

    # For a more balanced distribution, we would need to mock rand
    # or use statistical tests, but that's beyond the scope of this test
  end

  def test_generate_with_short_prompt
    @llm.train("hello world")
    
    assert_raises(RuntimeError) do
      @llm.generate("h", 10)  # Prompt shorter than context_size
    end
  end

  def test_generate_without_training
    assert_raises(RuntimeError) do
      @llm.generate("hello", 10)
    end
  end

  def test_load_model
    # Create and train a model
    original_llm = NGramLLM.new(3)
    original_llm.train("llhello world")
    
    # Create a new model and load the trained model's data
    new_llm = NGramLLM.new(3)
    
    new_llm.load(original_llm.model)

    assert_equal original_llm.model.keys.size, new_llm.model.keys.size

    # TODO: vocab does not load perfectly, as the first n tokens in the string may not be included.
    # If the test string is change from "llhello world" to "hello world",
    # The loaded model will not include "he" in the vocab.
    #
    # Currently, vocab is only used in sampling in our fallback.
    # When we introduce backoff and other techniques, this won't be used.
    assert_equal original_llm.vocab.size, new_llm.vocab.size
  end

  def test_model_normalization
    # Test that the model normalizes to lowercase
    uppercase_llm = NGramLLM.new(3)
    lowercase_llm = NGramLLM.new(3)
    
    uppercase_llm.train("HELLO")
    lowercase_llm.train("hello")
    
    # Both models should have the same vocabulary and transitions
    assert_equal uppercase_llm.vocab, lowercase_llm.vocab
    
    # Check if the context IDs are the same
    uppercase_context = "HE".downcase.bytes
    lowercase_context = "he".bytes
    
    uppercase_id = uppercase_llm.send(:context_id, uppercase_context)
    lowercase_id = lowercase_llm.send(:context_id, lowercase_context)
    
    assert_equal uppercase_id, lowercase_id
  end

  def test_different_n_values
    # Test with different n-gram lengths
    bigram = NGramLLM.new(2)
    trigram = NGramLLM.new(3)
    fourgram = NGramLLM.new(4)
    
    sample_text = "hello world"
    
    bigram.train(sample_text)
    trigram.train(sample_text)
    fourgram.train(sample_text)
    
    # n-1 should be the context size
    assert_equal 1, bigram.instance_variable_get(:@context_size)
    assert_equal 2, trigram.instance_variable_get(:@context_size)
    assert_equal 3, fourgram.instance_variable_get(:@context_size)
    
    # Each model should have different context patterns
    assert_operator bigram.model.keys.size, :>, 0
    assert_operator trigram.model.keys.size, :>, 0
    assert_operator fourgram.model.keys.size, :>, 0
    
    # Different n values should lead to different model structures
    refute_equal bigram.model.keys.size, trigram.model.keys.size
  end

  ## Test weights function

  def test_weights_returns_correct_number_of_elements
    # Test with n=3 (from setup)
    weights_n3 = @llm.send(:weights)
    assert_equal 3, weights_n3.size, "Should return n=3 weights for n=3 model"
    assert_equal @llm.n, weights_n3.size

    # Test explicitly with n=5
    llm_n5 = NGramLLM.new(5)
    weights_n5 = llm_n5.send(:weights)
    assert_equal 5, weights_n5.size, "Should return n=5 weights for n=5 model"
    assert_equal llm_n5.n, weights_n5.size
  end

  # Test that the calculated weights sum to approximately 1.0
  def test_weights_sum_to_one
    weights_array = @llm.send(:weights)
    # Use assert_in_delta for floating-point comparisons
    assert_in_delta 1.0, weights_array.sum, 1e-9, "Weights should sum to 1.0"
  end

  # Test that weights are ordered correctly (descending) for the default bias
  def test_weights_are_ordered_descending_by_default
    # Weights should be highest for highest order n-gram (index 0)
    # and decrease for lower orders when bias_factor > 0. Default is 2.5.
    weights_array = @llm.send(:weights)

    assert_equal @llm.n, weights_array.size # Pre-condition check

    # Check adjacent elements
    (weights_array.size - 1).times do |i|
      # With default bias > 0, weights should be strictly decreasing
      assert_operator weights_array[i], :>, weights_array[i+1],
                      "Weight at index #{i} (#{weights_array[i]}) should be > weight at index #{i+1} (#{weights_array[i+1]})"
    end
  end

  # Test how changing the bias factor affects the weight distribution
  def test_weights_bias_factor_effect
    # Need separate instances because weights are cached
    llm_low_bias = NGramLLM.new(3)
    # Pass bias_factor argument directly to the method using send
    weights_low = llm_low_bias.send(:weights, 1.0) # Lower bias

    llm_high_bias = NGramLLM.new(3)
    weights_high = llm_high_bias.send(:weights, 5.0) # Higher bias

    # Higher bias should give MORE weight to the highest order n-gram (index 0)
    assert_operator weights_high[0], :>, weights_low[0],
                    "Higher bias should increase weight of highest order n-gram"

    # Higher bias should give LESS weight to the lowest order n-gram (last index)
    assert_operator weights_high.last, :<, weights_low.last,
                    "Higher bias should decrease weight of lowest order n-gram"

    # Both sets should still sum to 1
    assert_in_delta 1.0, weights_low.sum, 1e-9
    assert_in_delta 1.0, weights_high.sum, 1e-9
  end

  ## Interpolation tests

  def test_interpolate_full_match
    # unigram(a->b)
    # bigram(ab->c)
    # unigram(b->c)
    # trigram(abc->d)
    # bigram(bc->d)
    # unigram(c->d)
    # trigram(bcd-> )
    # bigram(cd-> )
    # unigram(d-> )
    llm4 = NGramLLM.new(4)
    llm4.train("abcd bcd2 bcd2 dx")

    current_context = @llm.send(:tokenize, "bcd")
    # @model["bcd"] => {" " => 1, "2" => 2}
    # @model["cd"] => {" " => 1, "2" => 2}
    # @model["d"] => {" " => 1, "2" => 2, "x" => 1}

    weights = llm4.send(:weights)
    k_sp = @llm.send(:tokenize, " ").first
    k_2 = @llm.send(:tokenize, "2").first
    k_x = @llm.send(:tokenize, "x").first

    expected_options = {
      k_sp => 1 * weights[0] + 1 * weights[1] + 1 * weights[2], 
      k_2 => 2 * weights[0] + 2 * weights[1] + 2 * weights[2],
      k_x => 1 * weights[2],
    }

    actual_options = llm4.send(:interpolate_options, current_context)
    assert_equal actual_options.keys, expected_options.keys
    assert_in_delta actual_options[k_sp], expected_options[k_sp], 1e-9
    assert_in_delta actual_options[k_2], expected_options[k_2], 1e-9
    assert_in_delta actual_options[k_x], expected_options[k_x], 1e-9
  end

  def test_interpolate_partial_match
    # unigram(a->b)
    # bigram(ab->c)
    # unigram(b->c)
    # trigram(abc->d)
    # bigram(bc->d)
    # unigram(c->d)
    # trigram(bcd-> )
    # bigram(cd-> )
    # unigram(d-> )
    llm4 = NGramLLM.new(4)
    llm4.train("abcd bcd2 bcd2 dx")

    current_context = @llm.send(:tokenize, "acd")
    # @model["acd"] => {}
    # @model["cd"] => {" " => 1, "2" => 2}
    # @model["d"] => {" " => 1, "2" => 2, "x" => 1}

    weights = llm4.send(:weights)
    k_sp = @llm.send(:tokenize, " ").first
    k_2 = @llm.send(:tokenize, "2").first
    k_x = @llm.send(:tokenize, "x").first

    expected_options = {
      k_sp => 1 * weights[1] + 1 * weights[2], 
      k_2 => 2 * weights[1] + 2 * weights[2],
      k_x => 1 * weights[2],
    }

    actual_options = llm4.send(:interpolate_options, current_context)
    assert_equal actual_options.keys, expected_options.keys
    assert_in_delta actual_options[k_sp], expected_options[k_sp], 0.1
    assert_in_delta actual_options[k_2], expected_options[k_2], 0.1
    assert_in_delta actual_options[k_x], expected_options[k_x], 0.1
  end

  def test_interpolate_no_match
    # unigram(a->b)
    # bigram(ab->c)
    # unigram(b->c)
    # trigram(abc->d)
    # bigram(bc->d)
    # unigram(c->d)
    # trigram(bcd-> )
    # bigram(cd-> )
    # unigram(d-> )
    llm4 = NGramLLM.new(4)
    llm4.train("abcd bcd2 bcd2 dx")

    current_context = @llm.send(:tokenize, "acy")
    # @model["acy"] => {}
    # @model["cy"] => {}
    # @model["y"] => {}

    actual_options = llm4.send(:interpolate_options, current_context)
    assert_empty actual_options
  end

  ## Test Temperature

  def test_temper_baseline
    baseline_temp = 1.0
    options = { TOKEN_A => 60, TOKEN_B => 30, TOKEN_C => 10 }
    total_count = options.values.sum.to_f
    
    expected_probs = {
      TOKEN_A => 60 / total_count,
      TOKEN_B => 30 / total_count,
      TOKEN_C => 10 / total_count
    }

    actual_probs = @llm.send(:temper_options, options, baseline_temp)

    assert_equal expected_probs.keys.sort, actual_probs.keys.sort, "Should have the same tokens"
    expected_probs.each do |token, expected_prob|
      assert_in_delta expected_prob, actual_probs[token], 1e-9,
        "Probability for token #{token} should match original distribution for T=1.0"
    end
    assert_in_delta 1.0, actual_probs.values.sum, 1e-9, "Probabilities must sum to 1.0"
  end

  def test_temper_with_low_temp
    low_temp = 0.5
    options = { TOKEN_A => 60, TOKEN_B => 30, TOKEN_C => 10 }

    adjust = lambda { |prob| prob**(1.0 / low_temp) }

    total_count = options.values.map { |v| adjust.call(v) }.sum.to_f

    expected_probs = {
      TOKEN_A => adjust.call(60) / total_count,
      TOKEN_B => adjust.call(30) / total_count,
      TOKEN_C => adjust.call(10) / total_count
    }

    actual_probs = @llm.send(:temper_options, options, low_temp)

    assert_equal expected_probs.keys.sort, actual_probs.keys.sort, "Should have the same tokens"
    expected_probs.each do |token, expected_prob|
      assert_in_delta expected_prob, actual_probs[token], 1e-9,
        "Probability for token #{token} should match original distribution for T=1.0"
    end
    assert_in_delta 1.0, actual_probs.values.sum, 1e-9, "Probabilities must sum to 1.0"

    # Check that the distribution became sharper compared to T=1 (0.6, 0.3, 0.1)
    assert actual_probs[TOKEN_A] > 0.6, "Highest probability should increase for T < 1"
    assert actual_probs[TOKEN_B] < 0.3, "Lower probabilities should decrease for T < 1"
    assert actual_probs[TOKEN_C] < 0.1, "Lowest probability should decrease significantly for T < 1"
  end

  def test_temper_with_high_temp
    high_temp = 2.0
    options = { TOKEN_A => 60, TOKEN_B => 30, TOKEN_C => 10 }

    adjust = lambda { |prob| prob**(1.0 / high_temp) }

    total_count = options.values.map { |v| adjust.call(v) }.sum.to_f

    expected_probs = {
      TOKEN_A => adjust.call(60) / total_count,
      TOKEN_B => adjust.call(30) / total_count,
      TOKEN_C => adjust.call(10) / total_count
    }

    actual_probs = @llm.send(:temper_options, options, high_temp)

    assert_equal expected_probs.keys.sort, actual_probs.keys.sort, "Should have the same tokens"
    expected_probs.each do |token, expected_prob|
      assert_in_delta expected_prob, actual_probs[token], 1e-9,
        "Probability for token #{token} should match original distribution for T=1.0"
    end
    assert_in_delta 1.0, actual_probs.values.sum, 1e-9, "Probabilities must sum to 1.0"

    # Check that the distribution became flatter compared to T=1 (0.6, 0.3, 0.1)
    assert actual_probs[TOKEN_A] < 0.6, "Highest probability should decrease for T > 1"
    assert actual_probs[TOKEN_B] > 0.3, "Middle probability should increase towards uniform for T > 1"
    assert actual_probs[TOKEN_C] > 0.1, "Lowest probability should increase towards uniform for T > 1"
  end

  def test_temper_with_single_option
    # With only one option, the probability must be 1.0 regardless of temperature
    temp = 0.5 # Use a non-1.0 temperature
    options = { TOKEN_A => 50 }

    expected_probs = { TOKEN_A => 1.0 }

    actual_probs = @llm.send(:temper_options, options, temp)

    assert_equal expected_probs.keys.sort, actual_probs.keys.sort, "Should have the single token"
    expected_probs.each do |token, expected_prob|
      assert_in_delta expected_prob, actual_probs[token], 1e-9,
        "Probability for single token #{token} should be 1.0"
    end
    assert_in_delta 1.0, actual_probs.values.sum, 1e-9, "Probability must sum to 1.0"
  end
end
