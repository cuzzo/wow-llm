#! /usr/bin/env ruby
# frozen_string_literal: true

require "set"

class NGramLLM
  attr_reader :n, :model, :vocab

  DASHES = [145, 146].map(&:chr) # em & en dash

  # DO NOT INCLUDE MORE THAN 63 CHARS, AT 62 NOW.
  CHARS = (("a".."z").to_a + ("0".."9").to_a + [
        " ", '\n', ".", ",", '"', "'", "-",
        "!", "?", ";", ":", "_",
        "(", ")",
        "/", "\\", "|",
        "@", "#", "$", "%", "%", "*",
        "+", "=", "<", ">"
      ])
      .each_with_index
      .map { |char, i| [char, i + 1] }
      .to_h

  ORDS = CHARS.invert

  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(n = 3)
    raise ArgumentError, "n must be at least 2" if n < 2

    @n = n
    @weights = nil

    @model = Hash.new { |h, k| h[k] = Hash.new(0) }
    @vocab = Set.new # Keep track of all unique characters seen
    @context_size = n - 1
  end

  def load(model)
    @model = model
    @vocab = @model.values.map { |dict| dict.keys }.flatten.to_set
  end

  # Train the model on a given text
  def train(text)
    # Normalize the text to lowercase so that "Th" and "th" are not two different probabilities.
    tokens = tokenize(text)

    # 1) Build vocabulary
    tokens.each { |token| @vocab.add(token) }

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
    generated_text = tokenize(prompt)

    # Get the last context_size characters
    current_context = generated_text[(-@context_size)..]

    puts "Generating #{length} characters..."
    length.times do
      next_char_options = interpolate_options(current_context)

      if !next_char_options.nil? && !next_char_options.empty?
        # Choose the next character based on weighted probability (counts)
        next_char = weighted_choice(smoothen_options(temper_options(next_char_options, temp), k))

      else
        # Fallback: If context not seen during training, pick a random character from vocab
        # Or could try backing off to a shorter context (n-1 grams)
        # Random is simpler for now.

        warn "Warning: Context '#{current_context}' not found in model. Choosing random character."

        next_char = @vocab.to_a.sample

        # Avoid getting stuck if vocab is empty (shouldn't happen if trained)
        break if next_char.nil?
      end

      # Updated output
      generated_text << next_char

      # Update the context for the next iteration
      current_context.shift()
      current_context.push(next_char)
    end

    puts "Generation complete."
    decode(generated_text)
  end

  private

  # Encode an n-gram of 6-bit integers (ascii) into a BigInt of arbitrary size.
  def context_id(byte_context)
    result = 0
    byte_context.each_with_index do |token, idx|
      # Shift each token to its position and OR it in
      result |= (token & 0x3F) << (idx * 6)
    end
    result < 2**64 ?
      result :
      0
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

  def tokenize(text)
    text
      .downcase
      .chars
      .map do |c|
        if c == "“" || c == "”"
          c = "\""
        elsif c == "’" || c == "‘"
          c = "'"
        elsif DASHES.include?(c)
          c = "-"
        elsif c == "{" || c == "["
          c = "("
        elsif c == "}" || c == "}"
          c = ")"
        elsif c == "€"
          c = "$"
        end
        CHARS[c] || 0
      end
  end

  def decode(tokens)
    tokens
      .map { |c| ORDS[c.to_i] || "~" }
      .join("")
  end

  # Helper method to implement Add-k smoothening to the options for the next choice
  def smoothen_options(options, k)
    smoothed_weights = {}
    options.values.sum

    # Calculate smoothed weight for every character in the vocabulary
    @vocab.each do |char_code|
      raw_count = options[char_code] || 0
      # The weight is the numerator of the Add-k probability formula
      smoothed_weights[char_code] = raw_count + k
    end

    smoothed_weights
  end

  def temper_options(options, temp)
    # Convert counts to probabilities and apply temperature
    adjusted_weights = {}

    options.each do |char, count|
      # Apply temperature scaling: lower temp = more deterministic, higher = more random
      adjusted_weights[char] = Math.exp(Math.log(count) / temp)
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
      options.each do |char, v|
        acc[char] ||= 0
        acc[char] += v * weight
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
