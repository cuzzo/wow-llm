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
      input = "1".to_slice
      llm = NGramLLM.new(n)
      llm.context_id(input).should eq(49) # 49 = "1"
    end

    it "returns for single byte" do
      input = "12".to_slice
      llm = NGramLLM.new(n)
      # most significant byte = 50 "2", then 1 "49"
      result = 50 | (49 << 8)
      llm.context_id(input).should eq(result)
    end
  end

  context "next_options" do
    context "when n = 5" do
      n = 5

      it "returns options on direct hit" do
        input = "1234".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        model = {ctx_full => options1}
        llm.load(model)
        llm.next_options(ctx_full).should eq(options1)
      end

      it "returns options on first backoff hit" do
        input = "1234".to_slice
        input1 = "234".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        ctx_back1 = llm.context_id(input1)
        model = {ctx_back1 => options2}
        llm.load(model)
        llm.next_options(ctx_full).should eq(options2)
      end

      it "returns options on second backoff hit" do
        input = "1234".to_slice
        input2 = "34".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        ctx_back2 = llm.context_id(input2)
        model = {ctx_back2 => options3}
        llm.load(model)
        llm.next_options(ctx_full).should eq(options3)
      end

      it "returns options on last backoff hit (single byte context)" do
        input = "1234".to_slice
        input3 = "4".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        ctx_back3 = llm.context_id(input3)
        model = {ctx_back3 => options3}
        llm.load(model)
        llm.next_options(ctx_full).should eq(options3)
      end

      it "returns nil if no context matches after all backoffs" do
        input = "1234".to_slice
        input3 = "2345".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        ctx_back3 = llm.context_id(input3)
        model = {ctx_back3 => options3}
        llm.load(model)
        llm.next_options(ctx_full).should be_nil
      end

      it "returns nil if model is empty" do
        input = "1234".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        llm.next_options(ctx_full).should be_nil
      end

      it "prefers direct hit over backoff hit" do
        input = "1234".to_slice
        input1 = "123".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        ctx_back1 = llm.context_id(input1)
        model = {
          ctx_full => options1,
          ctx_back1 => options2
        }
        llm.load(model)
        llm.next_options(ctx_full).should eq(options1)
      end

      it "prefers earlier backoff hit over later backoff hit" do
        input = "1234".to_slice
        input1 = "234".to_slice
        input2 = "34".to_slice
        llm = NGramLLM.new(n)
        ctx_full = llm.context_id(input)
        ctx_back1 = llm.context_id(input1)
        ctx_back2 = llm.context_id(input2)
        model = {
          ctx_back1 => options2,
          ctx_back2 => options3
        }
        llm.load(model)
        llm.next_options(ctx_full).should eq(options2)
      end
    end
  end

  context "context_shift" do
    n = 3
    it "shifts and adds" do
      input = "123".to_slice
      llm = NGramLLM.new(n)
      cc_id = llm.context_id(input[0..0])
      shifted_cc_id = llm.context_shift(cc_id, input[1])
      expected_cc_id = llm.context_id(input[0..1])
      shifted_cc_id.should eq(expected_cc_id)
    end

    it "shifts out of bounds and adds" do
      input = "12".to_slice
      llm = NGramLLM.new(n)
      cc_id = llm.context_id(input)
      shifted_cc_id = llm.context_shift(cc_id, "3".to_slice[0])
      expected_cc_id = llm.context_id("23".to_slice)
      shifted_cc_id.should eq(expected_cc_id)
    end
  end

  context "train" do
    n = 3
    it "backoffs properly" do
      input = "1234"
      llm = NGramLLM.new(n)
      llm.train(input) 
      model = llm.@model
      model[llm.context_id("1".to_slice)].should eq({
        "2".to_slice[0] => 1
      })
      model[llm.context_id("12".to_slice)].should eq({
        "3".to_slice[0] => 1
      })
      model[llm.context_id("2".to_slice)].should eq({
        "3".to_slice[0] => 1
      })
      model[llm.context_id("23".to_slice)].should eq({
        "4".to_slice[0] => 1
      })
      model[llm.context_id("3".to_slice)].should eq({
        "4".to_slice[0] => 1
      })
    end
  end

  context "generate" do
    n = 3
    it "generates with direct hit" do
      input = "1234"
      llm = NGramLLM.new(n)
      llm.train(input)
      output = llm.generate("123", 1)
      output.should eq("1234")
    end

    it "generates with backoff hit" do
      input = "1234"
      llm = NGramLLM.new(n)
      llm.train(input)
      output = llm.generate("xx3", 1)
      output.should eq("xx34")
    end
  end
end

