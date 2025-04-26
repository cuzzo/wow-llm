#! /usr/bin/env ruby

require 'minitest/autorun'
require 'minitest/rg'
require_relative 'llm'

class NGramLLMTest < Minitest::Test
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

  def test_train_and_generate
    # Simple training text
    @llm.train("hello hello hello world")
    
    # Generate text with a known prompt
    generated = @llm.generate("he", 10)
    
    # The model should generate text that follows the patterns in the training data
    assert_kind_of String, generated
    assert_equal 12, generated.length  # "he" + 10 more characters
    
    # Since "he" is always followed by "llo" in the training data, the next characters
    # should start with "llo"
    assert_match(/^hello/, generated)
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

  ## Backoff Testing

  def test_options_for_context_exact_match
    # Test when the exact context exists in the model
    context = "he".bytes
    context_id = @llm.send(:context_id, context)
    expected_options = { "l".bytes.first => 5 }

    # Manually populate the model for this test case
    @llm.model[context_id] = expected_options

    result = @llm.send(:options_for_current_context, context)
    assert_equal expected_options, result
  end

  def test_options_for_context_backoff_single_step
    # Test when the exact context (len 2) doesn't exist, but the backed-off context (len 1) does
    full_context = "he".bytes # context_size = 2 for n=3
    backed_off_context = "e".bytes # After removing 'h'
    backed_off_context_id = @llm.send(:context_id, backed_off_context)
    expected_options = { "y".bytes.first => 3 }

    # Ensure the full context ID isn't present
    refute @llm.model.key?(@llm.send(:context_id, full_context))

    # Manually populate the model with the backed-off context
    @llm.model[backed_off_context_id] = expected_options

    result = @llm.send(:options_for_current_context, full_context)
    assert_equal expected_options, result, "Should have returned options for backed-off context 'e'"
  end

  def test_options_for_context_backoff_multiple_steps
    # Test backoff over more than one step (requires n > 3)
    llm4 = NGramLLM.new(4) # n=4, context_size=3
    full_context = "abc".bytes
    context_step1 = "bc".bytes # Backoff 1
    context_step2 = "c".bytes  # Backoff 2
    context_step2_id = llm4.send(:context_id, context_step2)
    expected_options = { "d".bytes.first => 10 }

    # Ensure full and intermediate contexts are not present
    refute llm4.model.key?(llm4.send(:context_id, full_context))
    refute llm4.model.key?(llm4.send(:context_id, context_step1))

    # Manually populate the model with the final backed-off context
    llm4.model[context_step2_id] = expected_options

    result = llm4.send(:options_for_current_context, full_context)
    assert_equal expected_options, result, "Should have backed off twice to find context 'c'"
  end

  def test_options_for_context_not_found
    # Test when neither the context nor any backed-off versions exist
    context = "xy".bytes
    backed_off_context = "y".bytes

    # Add some unrelated data to ensure the model isn't empty
    @llm.model[@llm.send(:context_id, "ab".bytes)] = { "c".bytes.first => 1 }

    # Ensure the target contexts aren't present
    refute @llm.model.key?(@llm.send(:context_id, context))
    refute @llm.model.key?(@llm.send(:context_id, backed_off_context))

    result = @llm.send(:options_for_current_context, context)
    assert_empty result, "Should return nil when no context is found after backoff"
  end

  def test_options_for_context_n_equals_2
     # Test behavior with a bigram model (n=2, context_size=1) where backoff is simpler
    llm2 = NGramLLM.new(2)
    context_a = "a".bytes
    context_x = "x".bytes
    context_a_id = llm2.send(:context_id, context_a)
    expected_options = { "b".bytes.first => 5 }

    # Manually populate the model
    llm2.model[context_a_id] = expected_options

    # Case 1: Exact match found
    result_match = llm2.send(:options_for_current_context, context_a)
    assert_equal expected_options, result_match, "Should find exact match for n=2"

    # Case 2: No match found (backoff loop condition c2.length > 0 prevents checking empty context)
    result_no_match = llm2.send(:options_for_current_context, context_x)
    assert_empty result_no_match, "Should return empty hash when context not found."
  end
end
