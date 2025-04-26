#! /usr/bin/env ruby
# frozen_string_literal: true

require "set"
require "msgpack"
require_relative "tokenizer"

class TokenLLM
  attr_reader :n, :weights, :model, :vocab, :token_to_id, :id_to_token

  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(n = 3)
    raise ArgumentError, "n must be at least 2" if n < 2

    @n = n
    @weights = nil

    @model = Hash.new { |h, k| h[k] = Hash.new(0) }
    @context_size = n - 1

    @tokenizer = BpeTokenizer.new()
  end

  def save(model_file, token_file)
    @model.keys.each do |key|
      new_key = key.pack("Q>*") # Convert key to binary string
      @model[new_key] = @model[key] # Move value from old key to new key
      @model.delete(key) # Delete old key to save memory
    end
    File.write(model_file, @model.to_msgpack, mode: "wb")
    @tokenizer.save(token_file)
  end

  def load(model_file, token_file)
    @model = MessagePack.unpack(File.read(model_file, mode: "rb"))
    @model.keys.each do |key|
      new_key = key.unpack("Q>*") # Convert key from binary string to array of int-64
      @model[new_key] = @model[key] # Move value from old key to expected key
      @model.delete(key) # Delete the old key, it's unused.
    end
    @tokenizer.load(token_file)
  end

  # Train the model on a given text
  def train(files)
    @tokenizer.train(files)

    training_data = files
      .reduce("") do |acc, f|
        if f == "." || f == ".."
          next acc
        end
        acc += File.read(f)
        acc
      end

    puts "Tokenizing..."
    tokens = @tokenizer.tokenize(training_data)

    context = [tokens[0]]

    # 2) For each context, set occurence of next char
    (1...tokens.length).each do |i|
      if i % 10_000 == 0
        puts "TRAINING: #{i} of #{tokens.length}"
      end
      next_char = tokens[i]

      # Increment the count for this next_char given the context
      @model[context_id(context)][next_char] += 1

      c2 = context.dup
      (context.length - 1).times do
        c2.shift()
        @model[context_id(c2)][next_char] += 1
      end

      context.shift() if context.length == @context_size
      context.push(next_char)
    end

    puts "Training complete. Model has #{@model.keys.size} contexts."
  end

  # Generate text starting with a prompt
  def generate(prompt, length = 100, temp = 1.0, k = 0.5)
    raise "Model not trained yet!" if @model.empty?
    raise "Prompt must be at least #{@context_size} characters long" if prompt.length < @context_size

    # Start with the prompt (normalized to downcase, since the input is normalized to downcase)
    generated_tokens = @tokenizer.tokenize(prompt)

    # Get the last context_size characters
    current_context = generated_tokens[(-@context_size)..]

    puts "Generating #{length} characters..."
    length.times do
      next_token = weighted_choice(next_options(current_context, temp, k))

      if next_token.nil?
        next_char == fallback_char()
        # Avoid getting stuck if vocab is empty (shouldn't happen if trained)
        break if next_token.nil?
      end

      # Updated output
      generated_tokens << next_token

      # Update the context for the next iteration
      current_context.shift()
      current_context.push(next_token)
    end

    puts "Generation complete."
    @tokenizer
      .detokenize(generated_tokens)
      .gsub(/\s+([.,;:?!\])\}])/, '\1')
  end

  private

  def next_options(context, temp = 1.0, k = 0.5)
    context
      .then { |c| interpolate_options(c) }
      .then { |c| temper_options(c, temp) }
      # .then { |c| smoothen_options(c) }
  end

  # Fallback: If context not seen during training, pick a random character from vocab
  # Or could try backing off to a shorter context (n-1 grams)
  # Random is simpler for now.
  def fallback_char()
    warn "Warning: Context '#{current_context}' not found in model. Choosing random character."
    @tokenizer.vocab.to_a.sample.first
  end

  # Encode 12-bit context tokens into an array 64-bit integer ID (can handle up to 5 tokens)
  def context_id(token_context)
    result = []
    token_context.each_with_index do |token, idx|
      # Shift each token to its position and OR it in
      result[idx / 5] ||= 0
      result[idx / 5] |= (token & 0x0FFF) << ((idx % 5) * 12)
    end
    result
  end

  # Helper method to make a weighted random choice from a hash of {item => weight}
  def weighted_choice(options)
    total_weight = options.values.sum
    random_num = rand * total_weight # Random float between 0 and total_weight

    cumulative_weight = 0
    options.each do |char, weight|
      cumulative_weight += weight
      return char if random_num < cumulative_weight
    end

    # Fallback (should theoretically not be reached if total_weight > 0)
    options.keys.sample
  end

  # Helper method to implement Add-k smoothening to the options for the next choice
  def smoothen_options(options)
    smoothed_weights = {}
    options.values.sum

    # Calculate smoothed weight for every character in the vocabulary
    @vocab.each do |token|
      raw_count = options[token] || 0
      # The weight is the numerator of the Add-k probability formula
      smoothed_weights[token] = raw_count + @smoothing_k
    end

    smoothed_weights
  end

  def temper_options(options, temp)
    # Convert counts to probabilities and apply temperature
    adjusted_weights = {}

    options.each do |token, count|
      # Apply temperature scaling: lower temp = more deterministic, higher = more random
      adjusted_weights[token] = Math.exp(Math.log(count) / temp)
    end

    # Normalize to create a valid probability distribution
    total = adjusted_weights.values.sum
    adjusted_weights.transform_values! { |w| w / total }

    adjusted_weights
  end

  def interpolate_options(context)
    c = context.dup
    (0...@context_size).each.reduce({}) do |acc, i|
      # Get counts for this context
      options = @model[context_id(c)] || {}

     weight = weights()[i]
     options.each do |token, v|
       acc[token] ||= 0
       acc[token] += v * weight
     end

      # Shorten context by removing oldest character
     c.shift

     acc
   end
  end

  def weights(bias_factor = 30.0)
    return @weights unless @weights.nil?

    # Initialize weights array for n-grams of different orders
    # Index 0 will be for the highest order n-gram (full context_size)
    # Last index will be for unigrams
    weights = []

    # Generate exponentially biased weights
    # Higher bias_factor means stronger preference for higher-order n-grams
    for i in 0..@context_size
      # This formula exponentially increases the weight as i decreases
      # (remember i=0 is the highest-order n-gram)
      weights[i] = Math.exp(bias_factor * (1.0 - (i.to_f/@context_size)))
    end

    # Normalize weights to sum to 1.0
    total = weights.sum
    weights.map! { |w| w / total }
    @weights = weights
    @weights
  end
end
