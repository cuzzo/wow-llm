#! /usr/bin/env ruby

require 'set'

class NGramLLM
  attr_reader :n, :model, :vocab

  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(n = 3)
    raise ArgumentError, "n must be at least 2" if n < 2
    @n = n

    @model = Hash.new { |h, k| h[k] = Hash.new(0) }
    @vocab = Set.new # Keep track of all unique characters seen
    @context_size = n - 1
  end

  # Due to Ruby JSON formatting, the model is stored with the keys as strings
  # However, it is expected for the keys to be integers.
  # Here, we convert the keys from strings back into integers.
  # This was not a problem in the previous version.
  # The keys WERE strings, not encoded integers.
  def load(model)
    @model = Hash[model.map do |k, v|
      [k.to_i, Hash[v.map { |k2, v2| [k2.to_i, v2] }]]
    end]
    @vocab = @model.values.map { |dict| dict.keys }.flatten.uniq
  end

  # Train the model on a given text
  def train(text)
    # Normalize the text to lowercase so that "Th" and "th" are not two different probabilities.
    tokens = tokenize(text)

    # 1) Build vocabulary
    tokens.each { |token| @vocab.add(token) }

    context = tokens[0...@context_size]

    # 2) For each context, set occurence of next char
    (tokens.length - @context_size).times do |i|
      if i % 10_000 == 0
        puts "TRAINING: #{i} of #{tokens.length}"
      end
      next_char = tokens[i + @context_size]

      # Increment the count for this next_char given the context
      @model[context_id(context)][next_char] += 1

      # Add backoff options
      c2 = context.dup
      (@context_size - 1).times do
        c2.shift()
        @model[context_id(c2)][next_char] += 1
      end

      # Special case for the beginning of the input
      if i == 0
        c2 = context.dup
        (@context_size - 1).times do
          p_char = c2.pop()
          @model[context_id(c2)][p_char] += 1
        end
      end

      context.shift()
      context.push(next_char)
    end

    puts "Training complete. Model has #{@model.keys.size} contexts."
  end

  # Generate text starting with a prompt
  def generate(prompt, length = 100)
    raise "Model not trained yet!" if @model.empty?
    raise "Prompt must be at least #{@context_size} characters long" if prompt.length < @context_size

    # Start with the prompt (normalized to downcase, since the input is normalized to downcase)
    generated_text = tokenize(prompt)

    # Get the last context_size characters
    current_context = generated_text[(-@context_size)..]

    puts "Generating #{length} characters..."
    length.times do
      next_char_options = options_for_current_context(current_context)

      if !next_char_options.empty?
        # Choose the next character based on weighted probability (counts)
        next_char = weighted_choice(next_char_options)

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
    generated_text.pack('C*')
  end

  private

  def options_for_current_context(context)
    # Get the possible next characters and their counts for the current context
    next_options = @model[context_id(context)]

    # If context not found, back-off progressively, one character a time, until one is found
    c2 = (next_options.nil? || next_options.empty?) ? context.dup : nil
    while (next_options.nil? || next_options.empty?) && c2.length > 0 do
      c2.shift()
      next_options = @model[context_id(c2)]
    end

    next_options || {}
  end

  # TODO: Encode and decode functions
  def tokenize(text)
    text.downcase.bytes
      #.map do |byte|
      #  if byte < 145
      #    byte
      #  elsif byte in [145, 146] # fance single quotes
      #    39 # single quote
      #  elsif byte in [147, 148] # fancy quotes
      #    22 # simple quote
      #  elsif byte == 152 # fancy tilde
      #    126 # tilde
      #  elsif byte == 128 # euro sign
      #    36 # dollar sign
      #  elsif byte == 160 # Non-breaking space
      #    32 # Space
      #  else
      #    157 # Unused / unknown
      #  end
      #end
  end

  # Encode an n-gram of 8-bit integers into a Bigint of arbitrary size.
  def context_id(byte_context)
    result = 0
    byte_context.each_with_index do |token, idx|
      # Shift each token to its position and OR it in
      result |= (token & 0xFF) << (idx * 8)
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
end
