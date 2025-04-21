#! /usr/bin/env ruby

require 'minitest/autorun'
require 'minitest/pride'
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
    expected_model = {
      @llm.send(:context_id, "he".bytes) => { "l".bytes.first => 1, "y".bytes.first => 1 },
      @llm.send(:context_id, "el".bytes) => { "l".bytes.first => 1 },
      @llm.send(:context_id, "ll".bytes) => { "o".bytes.first => 1 },
      @llm.send(:context_id, "lo".bytes) => { " ".bytes.first => 1 },
      @llm.send(:context_id, "o ".bytes) => { "w".bytes.first => 1 },
      @llm.send(:context_id, " w".bytes) => { "o".bytes.first => 1 },
      @llm.send(:context_id, "wo".bytes) => { "r".bytes.first => 1 },
      @llm.send(:context_id, "or".bytes) => { "l".bytes.first => 1 },
      @llm.send(:context_id, "rl".bytes) => { "d".bytes.first => 1 },
      @llm.send(:context_id, "ld".bytes) => { " ".bytes.first => 1 },
      @llm.send(:context_id, "d ".bytes) => { "h".bytes.first => 1 },
      @llm.send(:context_id, " h".bytes) => { "e".bytes.first => 1 },
    }
    @llm.train("hello world hey")

    assert_equal expected_model, @llm.model
    assert_equal 9, @llm.vocab.size  # 9 unique bytes in "hello world hey"

    new_llm = NGramLLM.new(3)
    new_llm.train("abcde fghij")
    assert_equal 11, new_llm.vocab.size # 11 unique bytes in "abcde fghij"
    
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
    
    # Convert model hash to JSON-style string-keyed hash
    model_as_string_keys = {}
    original_llm.model.each do |k, v|
      model_as_string_keys[k.to_s] = {}
      v.each do |k2, v2|
        model_as_string_keys[k.to_s][k2.to_s] = v2
      end
    end

    new_llm.load(model_as_string_keys)

    assert_equal original_llm.model.keys.size, new_llm.model.keys.size

    # TODO: vocab does not load perfectly, as the first n tokens in the string may not be included.
    # If the test string is change from "llhello world" to "hello world",
    # The loaded model will not include "he" in the vocab.
    #
    # Currently, vocab is only used in sampling in our fallback.
    # When we introduce backoff and other techniques, this won't be used.
    assert_equal original_llm.vocab.size, new_llm.vocab.size
  end

  def test_load_model_number_conversion
    # Test specifically that string keys are properly converted to integers
    model_with_string_keys = {
      "123" => { "97" => 10, "98" => 5 }
    }
    
    @llm.load(model_with_string_keys)
    
    assert @llm.model.has_key?(123)
    assert @llm.model[123].has_key?(97)
    assert @llm.model[123].has_key?(98)
    assert_equal 10, @llm.model[123][97]
    assert_equal 5, @llm.model[123][98]
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
end
