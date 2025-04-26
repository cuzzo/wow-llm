#! /usr/bin/env ruby

require 'set'
require 'numo/narray' # Use Numo::NArray
require 'msgpack'

require_relative "tokenizer"

DASHES = [150, 151].map(&:chr) # em & en dash
PARAGRAPH = '[PARAGRAPH]'.freeze
MODEL_FILE = "model.msgpack".freeze
TOKEN_FILE = "tokens.json".freeze

training_dir = ARGV[0]

# Helper functions for basic vector/matrix operations on Ruby Arrays
module BasicLinAlg
  def dot_product(vec1, vec2)
    vec1.dot(vec2)
  end

  # vector * matrix
  def multiply_vec_mat(vec, mat)
    vec.dot(mat)
  end

  # matrix * vector (column vector assumed)
  def multiply_mat_vec(mat, vec)
    mat.dot(vec)
  end

  # outer product: vec1 (col) * vec2 (row) -> matrix
  def outer_product(vec1, vec2)
    # Ensure inputs are Numo::NArray vectors if they aren't already
    # (This might not be strictly necessary if you guarantee inputs are NArrays,
    # but adds robustness)
    v1 = vec1.is_a?(Numo::NArray) ? vec1 : Numo::DFloat.cast(vec1)
    v2 = vec2.is_a?(Numo::NArray) ? vec2 : Numo::DFloat.cast(vec2)

    # Check if inputs are vectors (1-dimensional)
    unless v1.ndim == 1 && v2.ndim == 1
      raise ArgumentError, "Inputs to outer_product must be 1-dimensional vectors. Shapes were: #{v1.shape.inspect} and #{v2.shape.inspect}"
    end

    # Reshape v1 to a column vector [m, 1]
    # Reshape v2 to a row vector [1, n]
    # Perform matrix multiplication: [m, 1] . [1, n] -> [m, n]
    v1.reshape(v1.size, 1).dot(v2.reshape(1, v2.size))
  end

  def transpose(mat)
    mat.transpose
  end

  def add_vectors(vec1, vec2)
    vec1 + vec2
  end

  def subtract_vectors(vec1, vec2)
    vec1 - vec2
  end

  def multiply_elementwise(vec1, vec2)
    vec1 * vec2
  end

  def scalar_multiply(scalar, vec)
    vec * scalar
  end

  def tanh(vec)
    Numo::NMath.tanh(vec)
  end

  # Derivative of tanh: 1 - tanh(x)^2
  def dtanh(tanh_output_vec)
    1.0 - (tanh_output_vec ** 2)
  end

  def softmax(vec)
    # Subtract max for numerical stability
    max_val = vec.max || 0.0
    exps = vec.map { |x| Math.exp(x - max_val) }
    sum_exps = exps.sum
    return Numo::DFloat.ones(vec.size) / vec.size if sum_exps == 0 # Handle edge case
    exps / sum_exps
  end
end

class NNLM
  include BasicLinAlg

  attr_reader :word_to_ix, :ix_to_word, :vocab_size, :tokenizer

  def initialize(embedding_dim:, context_size:, hidden_size:, learning_rate: 0.01)
    @embedding_dim = embedding_dim
    @context_size = context_size # Number of preceding words (n-1 grams for predicting nth)
    @hidden_size = hidden_size
    @learning_rate = learning_rate

    @tokenizer = BpeTokenizer.new()

    # Placeholders - vocabulary needs to be built first
    @vocab_size = 0
    @word_to_ix = {}
    @ix_to_word = []

    # Parameters - will be initialized after vocab is built
    @embeddings = Numo::DFloat.new()
    @W_h = nil # Hidden layer weights: input_size x hidden_size
    @b_h = nil # Hidden layer biases: hidden_size
    @W_o = nil # Output layer weights: hidden_size x vocab_size
    @b_o = nil # Output layer biases: vocab_size
  end

  def build_vocabulary(training_dir)
    if File.exist?(TOKEN_FILE)
      load_tokenizer()
    else
      tokenizer.train(get_files(training_dir))
    end

    puts "Building vocabulary..."
    @ix_to_word = @tokenizer.instance_variable_get(:@tokenizer).vocab.keys
    @word_to_ix = @ix_to_word.each_with_index.to_h
    @vocab_size = @ix_to_word.size

    puts "Vocabulary size: #{@vocab_size}"
    _initialize_parameters()
  end

  def _initialize_parameters
    puts "Initializing parameters..."
    input_concat_size = @context_size * @embedding_dim

    @embeddings = Numo::DFloat.cast(@vocab_size.times.map do |i|
      Numo::DFloat.cast(@embedding_dim.times.map { rand * 0.1 - 0.05 })
    end)

    # Ensure PAD embedding is zero? Often helpful.
    @embeddings[@word_to_ix['[PAD]'], true] = Numo::DFloat.cast([0.0] * @embedding_dim)

    # Hidden Layer Weights/Biases
    @W_h = Numo::DFloat.cast(Array.new(input_concat_size) { Array.new(@hidden_size) { rand * 0.1 - 0.05 } })
    @b_h = Numo::DFloat.cast(Array.new(@hidden_size) { rand * 0.1 - 0.05 })

    # Output Layer Weights/Biases
    @W_o = Numo::DFloat.cast(Array.new(@hidden_size) { Array.new(@vocab_size) { rand * 0.1 - 0.05 } })
    @b_o = Numo::DFloat.cast(Array.new(@vocab_size) { rand * 0.1 - 0.05 })

    puts "Parameter initialization complete."
  end

  # --- Forward Pass ---
  # O(C*E*H + H*V) => ContextSize * EmbeddingDim * HiddenSize + HiddenSize * VocabSize
  def forward(context_indices)
    # 1. Projection Layer: Look up and concatenate embeddings
    # Think of embeddings as a dictionary where each word has a unique "meaning vector"
    input_layer = @embeddings[context_indices, true].reshape(@embedding_dim * @context_size)
    # Example: If context_indices = [42, 15] (representing "the cat")
    # And embeddings = { 42 => [0.1, 0.2], 15 => [0.3, 0.4] }
    # Then input_layer = [0.1, 0.2, 0.3, 0.4]
    #
    # The `inputs` represent the Nueral Network's `hidden` knowledge

    # 2. Hidden Layer: Process the word information into a compact representation
    hidden_input = add_vectors(
      multiply_vec_mat(input_layer, @W_h), # Multiply -> # Transform: Apply weights to extract meaningful patterns for each neuron
      @b_h) # Add -> Adjust for the baseline preference of each hidden neuron

    # hidden_layer = [0.1, 0.2, 0.3, 0.4]
    # W_h = matrix of hidden weights   => size: (input_size* x hidden_size => input_size: hidden_size x context_size
    # b_h = vector of hidden biases    => size: hidden_size
    # [ x, y, z ]
    #
    # In this example, we are only compressing 4 total inputs of knowledge into 3.
    # However, in a real nueral network, the context_size maybe be 20 and the dimension size 512
    # And the hidden_size maybe 64
    #
    # So we would compress 10k pieces of information down to 64
    #
    # When you see a 7B `parameter` model, the number of `parameters` is equal to:
    # 20 x 512 x 64 = 655k
    #
    # In our example, we have 2 x 2 x 3 = 12

    hidden_activation = tanh(hidden_input) # Apply tanh to keep values between -1 and 1

    # The tanh function adds non-linearity, allowing the network to learn complex patterns
    # Without this, we'd only capture simple linear relationships between words
    #
    # The hidden input is a simple linear combination
    # Imagine trying to predict whether someone will like a movie
    # The `features` of the movie could be percentage Action, and percentage Romance:
    # (Linear) Prediction = (0.7 × Action) + (0.3 × Romance)
    #
    # Here, 0.7 and 0.3 represent the hidden weights
    # If a movie is 0.5% Action, and 0.5% Romance, with a simple linear combination,
    # we would get:
    #
    # Prediction = (0.7 x 0.5) + (0.3 x 0.5) = 0.35 + 0.15 = 0.5
    #
    # But perhaps things aren't this simple / linear. We need to curve the scores with tanh
    # Movies that are 50/50 may not be liked, but movies that are closer to 100 Romance
    # or 100 Action may be liked.
    #
    # tanh(10,0) = 0.96, (pure action)
    # tanh(0,10) = -0.96, (pure romance)
    # tanh(5,5) = 0, (50/50 action/romance)

    # 3. Output Layer: Transform hidden features into word predictions
    output_scores = add_vectors(
      multiply_vec_mat(hidden_activation, @W_o), # Multiply -> Each hidden feature votes on possible next words
      @b_o) # Add -> Adjust for the baseline preference of each word

    # At this point, we have raw "scores" for each word in our vocabulary
    # But these aren't yet proper probabilities

    # 4. Softmax / Probability Calculation: Convert scores to proper probabilities
    probabilities = softmax(output_scores) # Transform the scores into proper probabilities that sum to 1

    # Softmax does three things:
    # - Makes all values positive (using exponential function)
    # - Amplifies differences (higher scores become much more likely)
    # - Normalizes everything to sum to 1 (creates a valid probability distribution)
    # Result: [0.01, 0.02, 0.8, 0.15, 0.02] = 80% chance the 3rd word comes next

    # Return values needed for backpropagation
    {
      probabilities: probabilities,
      hidden_activation: hidden_activation,
      input_layer: input_layer # concatenated embeddings
    }
  end

  # --- Backward Pass (Backpropagation) ---
  # O(C*E*H + H*V)
  def backward(context_indices, target_index, forward_pass_data)
    probabilities = forward_pass_data[:probabilities]
    hidden_activation = forward_pass_data[:hidden_activation]
    input_layer = forward_pass_data[:input_layer]

    # Initialize gradients (matching parameter structures)
    grad_embeddings = Hash.new { |h, k| h[k] = Numo::DFloat.cast([0.0] * @embedding_dim) }
    grad_W_h = Numo::DFloat.new(@W_h.size) { Numo::DFloat.new(@hidden_size, 0.0) }
    grad_b_h = Numo::DFloat.new(@hidden_size, 0.0)
    grad_W_o = Numo::DFloat.new(@hidden_size) { Numo::DFloat.new(@vocab_size, 0.0) }
    grad_b_o = Numo::DFloat.new(@vocab_size, 0.0)

    # 1. Calculate the main error signal: "How wrong was our prediction?"
    # This is remarkably simple: subtract 1 from the probability of the correct word
    # Example: If we predicted [0.1, 0.2, 0.7] but target_index was 0
    # Then error is [0.1-1, 0.2, 0.7] = [-0.9, 0.2, 0.7]
    # This means: "Increase probability of word 0, decrease probability of words 1 and 2"
    d_output_scores = probabilities.dup
    d_output_scores[target_index] -= 1.0

    # 2. Calculate how to adjust output layer weights
    # For each connection between hidden layer and output layer:
    # - If hidden value was strong AND error was large, make a big adjustment
    # - If either was small, make a smaller adjustment
    grad_W_o = outer_product(hidden_activation, d_output_scores)
    grad_b_o = d_output_scores # Bias gradient is just the error signal

    # 3. Send the error signal backwards to the hidden layer
    # "How much did each hidden neuron contribute to our mistakes?"
    # We multiply the error by the output weights to find out
    d_hidden_input_signal = multiply_vec_mat(d_output_scores, transpose(@W_o)) # Calculate dL/dHiddenActivation * dHiddenActivation/dHiddenInput part 1

    # 4. Account for the tanh function we used
    # Since tanh squished values, we need to "unsquish" the error signal
    # This uses the derivative of tanh: 1 - (activation)²
    # Example: If activation was 0.8, derivative is 1-(0.8)² = 0.36
    # This means neurons closer to 0 can change more than those near -1 or 1
    d_hidden_input = multiply_elementwise(d_hidden_input_signal, dtanh(hidden_activation))

    # 5. Calculate how to adjust hidden layer weights
    # Similar to step 2, but for the connections between input and hidden layers
    grad_W_h = outer_product(input_layer, d_hidden_input)
    grad_b_h = d_hidden_input # Bias gradient

    # 6. Send the error all the way back to the input embeddings
    # "How should each word's embedding change to reduce our error?"
    d_input_layer = multiply_vec_mat(d_hidden_input, transpose(@W_h))

    # 7. Split up the error for each individual word embedding
    # Since we concatenated the embeddings earlier, we now need to separate
    # the error signal for each original word
    context_indices.each_with_index do |word_ix, i|
      start_idx = i * @embedding_dim
      end_idx = start_idx + @embedding_dim - 1

      # Get the portion of error relevant to this word
      embedding_grad_slice = d_input_layer[start_idx..end_idx]

      # Add it to our correction sheet for this word's embedding
      # (We add because the same word might appear multiple times)
      grad_embeddings[word_ix] = add_vectors(grad_embeddings[word_ix], embedding_grad_slice)
    end

    {
      grad_embeddings: grad_embeddings,
      grad_W_h: grad_W_h, grad_b_h: grad_b_h,
      grad_W_o: grad_W_o, grad_b_o: grad_b_o
    }
  end

  # --- Parameter Update ---
  def update_parameters(gradients)
    # Update Embeddings (Sparse update using hash)
    gradients[:grad_embeddings].each do |word_ix, grad|
      # Check shapes before attempting subtraction to prevent errors
      if @embeddings[word_ix, true].shape == grad.shape
        @embeddings[word_ix, true] -= @learning_rate * grad
      else
        # Handle potential shape mismatch if default proc created an unexpected shape
        # This might indicate an issue in how grad_embeddings is populated or initialized
        raise "Shape mismatch during embedding update for index #{word_ix}. Embedding shape: #{@embeddings[word_ix, true].shape}, Gradient shape: #{grad_vec.shape}"
      end
    end

    # Update Hidden Layer (Dense update using matrix/vector subtraction)
    grad_wh = gradients[:grad_W_h]
    grad_bh = gradients[:grad_b_h]
    if @W_h.shape == grad_wh.shape && @b_h.shape == grad_bh.shape
      @W_h -= @learning_rate * grad_wh
      @b_h -= @learning_rate * grad_bh
    else
      raise "Shape mismatch updating hidden layer: W_h=#{@W_h.shape} vs Grad=#{@grad_wh.shape}, b_h=#{@b_h.shape} vs Grad=#{grad_bh.shape}"
    end

    # Update Output Layer (Dense update using matrix/vector subtraction)
    grad_wo = gradients[:grad_W_o]
    grad_bo = gradients[:grad_b_o]
    if @W_o.shape == grad_wo.shape && @b_o.shape == grad_bo.shape
      @W_o -= @learning_rate * grad_wo
      @b_o -= @learning_rate * grad_bo
    else
      raise "Shape mismatch updating output layer: W_o=#{@W_o.shape} vs Grad=#{@grad_wo.shape}, b_o=#{@b_o.shape} vs Grad=#{grad_bo.shape}"
    end
  end

  # w -> for each word
  # O(3*C*E*H + 3*H*V) => O(3*5*10*20 + 3*20*4096) => O(3000 + 245760)
  # embedding_dim: 10, # Small embedding size
  # context_size: 5,   # Use 2 preceding words (trigrams)
  # hidden_size: 20,   # Small hidden layer
  def process_context(input, i)
    context_indices = input[i...(i + @context_size)]
    target_index = input[i + @context_size]

    # Forward pass
    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    # Calculate Loss (Cross-Entropy) - optional for training but good for monitoring
    loss = -Math.log(probabilities[target_index] + 1e-9) # Add epsilon for numerical stability

    # Backward pass
    gradients = backward(context_indices, target_index, forward_data)

    # Update parameters
    update_parameters(gradients)

    loss
  end

  # --- Training Loop ---
  def train(training_dir, epochs: 10)
    raise "Vocabulary not built!" unless @vocab_size > 0

    padding_ix = @word_to_ix["[PAD]"]

    puts "\nStarting training..."
    epochs.times do |epoch|
      total_loss = 0.0
      example_count = 0

      1_0000.times do |batch|
      	 excerpts = get_input(training_dir, batch, 500)

        excerpts.each_with_index do |excerpt|
          # Create context windows and targets
          padded_excerpt = Array.new(@context_size, padding_ix) + excerpt
          (padded_excerpt.size - @context_size).times do |i|
            total_loss += process_context(padded_excerpt, i)
            example_count += 1
          end
        end
	       avg_loss = example_count > 0 ? total_loss / example_count : 0
        puts "Epoch #{epoch + 1}/#{epochs}, Batch #{batch + 1}/1000, Average Loss: #{avg_loss.round(4)}, Perplexity: #{(Math::E**avg_loss).round(4)}"
      end
    end
    puts "Training finished."
  end

  # --- Prediction ---
  def predict_next_word(prompt)
    raise "Vocabulary not built!" unless @vocab_size > 0

    context_indices = @tokenizer.tokenize(prompt)
    puts "CONTEXT INDICES: #{context_indices}"
    raise ArgumentError, "Context size mismatch" unless context_indices.size == @context_size

    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    # Find the index with the highest probability
    predicted_index = probabilities
      .to_a
      .each_with_index
      .max_by { |prob, _ix| prob }
      .last

    @ix_to_word[predicted_index]
  end

  def save_model(filepath)
    puts "Saving model to #{filepath}..."
    model_data = {
      # Hyperparameters
      embedding_dim: @embedding_dim,
      context_size: @context_size,
      hidden_size: @hidden_size,
      vocab_size: @vocab_size,

      # Vocabulary
      word_to_ix: @word_to_ix,
      ix_to_word: @ix_to_word,

      # Parameters
      embeddings: @embeddings.to_a,
      W_h: @W_h.to_a,
      b_h: @b_h.to_a,
      W_o: @W_o.to_a,
      b_o: @b_o.to_a
    }

    begin
      File.open(filepath, 'wb') do |file|
        MessagePack.dump(model_data, file)
      end
      puts "Model saved successfully."
    rescue => e
      puts "Error saving model: #{e.message}"
    end

    @tokenizer.save(TOKEN_FILE)
  end

  def load_tokenizer()
    @tokenizer.load(TOKEN_FILE)
  end

  def self.load_model(filepath)
    puts "Loading model from #{filepath}..."
    begin
      packed_data = File.binread(filepath)
      model_data = MessagePack.unpack(packed_data)

      # 1. Create a new instance with saved hyperparameters
      loaded_model = NNLM.new(
        embedding_dim: model_data["embedding_dim"],
        context_size: model_data["context_size"],
        hidden_size: model_data["hidden_size"]
        # learning_rate is not needed for loading/inference, can use default
      )
      loaded_model.load_tokenizer()

      # 2. Load the state into the new instance
      loaded_model.instance_variable_set(:@vocab_size, model_data["vocab_size"])
      loaded_model.instance_variable_set(:@word_to_ix, model_data["word_to_ix"])
      loaded_model.instance_variable_set(:@ix_to_word, model_data["ix_to_word"])
      loaded_model.instance_variable_set(:@embeddings, Numo::DFloat.cast(model_data["embeddings"]))
      loaded_model.instance_variable_set(:@W_h, Numo::DFloat.cast(model_data["W_h"]))
      loaded_model.instance_variable_set(:@b_h, Numo::DFloat.cast(model_data["b_h"]))
      loaded_model.instance_variable_set(:@W_o, Numo::DFloat.cast(model_data["W_o"]))
      loaded_model.instance_variable_set(:@b_o, Numo::DFloat.cast(model_data["b_o"]))

      puts "Model loaded successfully."
      loaded_model # Return the rehydrated model object
    rescue => e
      puts "Error loading model: #{e.message}"
      nil
    end
  end

  def get_files(training_dir)
    Dir
      .foreach(training_dir)
      .to_a
      .reject { |p| File.basename(p).start_with?(".") }
      .map { |p| File.join(training_dir, p) }
  end

  def get_input(training_dir, batch, batch_size)
    files = get_files(training_dir)

    files
      .map do |f|
        if f == "." || f == ".."
          next acc
        end
        if File.exist?("data/#{File.basename(f)}.msgpack")
          packed_data = File.binread("data/#{File.basename(f)}.msgpack")
          tokens = MessagePack.unpack(packed_data)
          next tokens
        end

        puts "READ FILE: #{f}"
        str = File
          .read(f)
          .downcase
          .chars
          .map do |c|
            if c == "“" || c == "”"
              c = "\""
            elsif c == "’" || c == "‘"
              c = "'"
            elsif DASHES.include?(c)
              c = "-"
            elsif c == '—'
              c = "-"
            elsif c == "{" || c == "["
              c = "("
            elsif c == "}" || c == "}"
              c = ")"
            elsif c == "€"
              c = "$"
            end
            c
          end
          .join("")
          .gsub(/([a-z])'([a-z])/, '\1 \' \2') # handle contractions
          .gsub(/([.,!?;:()\[\]{}""''…`_-])/, ' \1 ') # handle punctation
          .gsub(/(\d) \. (\d)/, '\1.\2') # handle numbers
          .gsub(/\s*\n+\s*/, " #{PARAGRAPH} ")

        puts "TOKENIZE FILE: #{f}"
        tokens = @tokenizer.tokenize(str)
	       puts "STORING #{f}"
        File.open("data/#{File.basename(f)}.msgpack", "wb") { |mf| MessagePack.dump(tokens, mf) }
        tokens
      end
      .map { |input| input[(batch*batch_size)...((batch+1)*batch_size+1)] } # Get batch_size tokens for each book
  end
end
