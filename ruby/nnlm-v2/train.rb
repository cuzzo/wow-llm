#! /usr/bin/env ruby
# frozen_string_literal: true

require_relative "llm"

training_dir = ARGV[0]

# 2. Initialize the Model
if !File.exist?(MODEL_FILE)
  nnlm = NNLM.new(
    embedding_dim: 32,   # Small embedding size
    context_size: 32,    # Use 32 preceding words
    hidden_size: 20,     # Small hidden layer
    learning_rate: 0.05
  )
  # 3. Build Vocabulary and Initialize Parameters
  nnlm.build_vocabulary(training_dir)
else
  # Load existing nnlm from file
  nnlm = NNLM.load_model(MODEL_FILE)
end


# 4. Train the Model
nnlm.train(training_dir, epochs: 25) # More epochs needed for such small data/model
nnlm.save_model(MODEL_FILE)

# 5. Predict
puts "\n--- Predictions ---"
# context1 = "Jon Snow looked back in"
context1 = "It took Ned a moment"
pred1 = nnlm.predict_next_word(context1)
puts "Context: #{context1} -> Predicted: #{pred1}" # Likely predicts 'cat' or 'dog'
