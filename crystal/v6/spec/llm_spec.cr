require "spec"
require "../src/llm" 

describe NGramLLM do
  # --- Test Data ---
  options1 = {1_u8 => 100_i64, 2_u8 => 101_i64} 
  options2 = {3_u8 => 200_i64} 
  options3 = {4_u8 => 300_i64} 

  context "context_id" do
    n = 3
    it "returns for single byte" do
      input = [1_u32]
      llm = NGramLLM.new(n)
      llm.context_id(input).should eq(1)
    end

    it "returns for single byte" do
      input = [1_u32, 2_u32]
      llm = NGramLLM.new(n)
      # most significant byte = 50 "2", then 1 "49"
      result = (1_u128 << NGramLLM::TOKEN_BITS) | 2_u128
      llm.context_id(input).should eq(result)
    end
  end

  context "#interpolate_options" do
    n = 4
    training_data = "a b c d * b c d 2 * b c d 2 * d x" 

    it "calculates interpolated scores with full context match" do
      llm = NGramLLM.new(n)
      llm.train(training_data)

      context_id = [
        llm.@token_to_id["b"],
        llm.@token_to_id["c"],
        llm.@token_to_id["d"]
      ]

      # Context: "bcd" (n-1 = 3 bytes)
      current_context = llm.context_id(context_id)

      # Assuming weights are [trigram_weight, bigram_weight, unigram_weight]
      weights = llm.@weights
      weights.size.should eq(n - 1) # Expect weights for 3 levels (3, 2, 1 grams)

      # Expected predictions based on Ruby comments:
      # Trigram ("bcd"): {" " => 1, "2" => 2}
      # Bigram  ("cd"):  {" " => 1, "2" => 2}
      # Unigram ("d"):   {" " => 1, "2" => 2, "x" => 1}

      k_sp = llm.@token_to_id["*"]
      k_2 = llm.@token_to_id["2"]
      k_x = llm.@token_to_id["x"]

      # Calculate expected scores based on interpolation formula
      expected_options = WeightedOptions.new()
      expected_options[k_sp] = 1 * weights[0] + 1 * weights[1] + 1 * weights[2]
      expected_options[k_2] = 2 * weights[0] + 2 * weights[1] + 2 * weights[2]
      expected_options[k_x] = 0 * weights[0] + 0 * weights[1] + 1 * weights[2] # Only unigram predicts 'x'

      actual_options = llm.interpolate_options(current_context)

      # Sort keys for reliable comparison
      actual_options.keys.sort.should eq(expected_options.keys.sort)

      # Compare scores with tolerance
      actual_options[k_sp].should be_close(expected_options[k_sp], 1e-9)
      actual_options[k_2].should be_close(expected_options[k_2], 1e-9)
      actual_options[k_x].should be_close(expected_options[k_x], 1e-9)
    end

    it "calculates interpolated scores with partial context match" do
      llm = NGramLLM.new(n)
      llm.train(training_data)

      context_id = [
        llm.@token_to_id["a"],
        llm.@token_to_id["c"],
        llm.@token_to_id["d"]
      ]

      # Context: "acd" (n-1 = 3 bytes)
      current_context = llm.context_id(context_id)

      # Assuming weights are [trigram_weight, bigram_weight, unigram_weight]
      weights = llm.@weights
      weights.size.should eq(n - 1)

      # Expected predictions based on Ruby comments:
      # Trigram ("acd"): {} (No match)
      # Bigram  ("cd"):  {" " => 1, "2" => 2}
      # Unigram ("d"):   {" " => 1, "2" => 2, "x" => 1}

      k_sp = llm.@token_to_id["*"]
      k_2 = llm.@token_to_id["2"]
      k_x = llm.@token_to_id["x"]

      # Calculate expected scores based on interpolation formula
      expected_options = WeightedOptions.new
      expected_options[k_sp] = 0 * weights[0] + 1 * weights[1] + 1 * weights[2] # No trigram contribution
      expected_options[k_2] = 0 * weights[0] + 2 * weights[1] + 2 * weights[2] # No trigram contribution
      expected_options[k_x] = 0 * weights[0] + 0 * weights[1] + 1 * weights[2] # Only unigram predicts 'x'

      actual_options = llm.interpolate_options(current_context)

      # Sort keys for reliable comparison
      actual_options.keys.sort.should eq(expected_options.keys.sort)

      # Compare scores with tolerance (using 0.1 as in the Ruby test)
      # Note: 0.1 is a large tolerance, might indicate probabilities or lenient testing.
      actual_options[k_sp].should be_close(expected_options[k_sp], 0.1)
      actual_options[k_2].should be_close(expected_options[k_2], 0.1)
      actual_options[k_x].should be_close(expected_options[k_x], 0.1)
    end

    it "returns empty options when no context levels match" do
      llm = NGramLLM.new(n)
      llm.train(training_data)

      context_id = [
        llm.@token_to_id["a"],
        llm.@token_to_id["c"],
        llm.@token_to_id["y"]
      ]

      # Context: "acy" (n-1 = 3 bytes)
      current_context = llm.context_id(context_id)

      # Expected predictions based on Ruby comments:
      # Trigram ("acy"): {} (No match)
      # Bigram  ("cy"):  {} (No match)
      # Unigram ("y"):   {} (No match)

      actual_options = llm.interpolate_options(current_context)

      actual_options.should be_empty
    end
  end

  context "context_shift" do
    n = 3
    it "shifts and adds" do
      input = [1_u32, 2_u32, 3_u32] 
      llm = NGramLLM.new(n)
      cc_id = llm.context_id(input[0..0])
      shifted_cc_id = llm.context_shift(cc_id, input[1])
      expected_cc_id = llm.context_id(input[0..1])
      shifted_cc_id.should eq(expected_cc_id)
    end

    it "shifts out of bounds and adds" do
      input = [1_u32, 2_u32]
      llm = NGramLLM.new(n)
      cc_id = llm.context_id(input)
      shifted_cc_id = llm.context_shift(cc_id, 3_u32)
      expected_cc_id = llm.context_id([2_u32, 3_u32])
      shifted_cc_id.should eq(expected_cc_id)
    end
  end

  context "train" do
    n = 3
    it "backoffs properly" do
      input = "1 2 3 4"

      llm = NGramLLM.new(n)
      llm.train(input) 

      k_1 = llm.@token_to_id["1"]
      k_2 = llm.@token_to_id["2"]
      k_3 = llm.@token_to_id["3"]
      k_4 = llm.@token_to_id["4"]

      model = llm.@model
      model[llm.context_id([k_1])].should eq({
        k_2 => 1
      })
      model[llm.context_id([k_1, k_2])].should eq({
        k_3 => 1
      })
      model[llm.context_id([k_2])].should eq({
        k_3 => 1
      })
      model[llm.context_id([k_2, k_3])].should eq({
        k_4 => 1
      })
      model[llm.context_id([k_3])].should eq({
        k_4 => 1
      })
    end
  end

  context "generate" do
    n = 3
    it "generates with direct hit" do
      input = "1 2 3 4"
      llm = NGramLLM.new(n)
      llm.train(input)
      output = llm.generate("x 3", 1)
      output.should eq("x 3 4")
    end
  end
end

