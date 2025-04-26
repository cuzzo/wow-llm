#! /usr/bin/env ruby

require_relative "llm"

# 1. Initialize the Model
if File.exist?(MODEL_FILE)
  # Load existing nnlm from file
  nnlm = NNLM.load_model(MODEL_FILE)
else
  raise "Train the model first"
end


# 2. Predict
puts "\n--- Predictions ---"
#context1 = "Jon Snow looked back in"
context1 = "It took Ned a moment"
pred1 = nnlm.predict_next_word(context1)
puts "Context: #{context1} -> Predicted: #{pred1}" # Likely predicts 'cat' or 'dog'

