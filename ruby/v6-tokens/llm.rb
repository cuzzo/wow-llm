#! /usr/bin/env ruby
# frozen_string_literal: true

require "set"

class TokenLLM
  attr_reader :n, :weights, :model, :vocab, :token_to_id, :id_to_token

  DASHES = [150, 151].map(&:chr) # em & en dash
  PARAGRAPH = "[PARAGRAPH]".freeze
  PARAGRAPH_STR = "\n\n".freeze

  CAPITAL_MARKER = [".", "!", "?", "\n\n"]
  PUNCTUATION = [".", ",", "!", "?", ";", ":", "(", ")", "-"]
  QUOTES = ['"', "'"]

  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(n = 3)
    raise ArgumentError, "n must be at least 2" if n < 2

    @n = n
    @weights = nil

    @model = Hash.new { |h, k| h[k] = Hash.new(0) }
    @vocab = Set.new # Keep track of all unique characters seen
    @context_size = n - 1

    # Maps for token conversion
    @token_to_id = {}
    @id_to_token = {}
    @next_token_id = 0
  end

  def load(model, tokens)
    @model = model
    @vocab = @model.values.map { |dict| dict.keys }.flatten.to_set
    @token_to_id = tokens
    @id_to_token = @token_to_id.invert
    @next_token_id = tokens.length
  end

  # Train the model on a given text
  def train(text)
    puts "Tokenizing..."
    tokens = tokenize(text)
    puts "Unique tokens: #{@next_token_id}"

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
      next_char = weighted_choice(next_options(current_context, temp, k))

      if next_char.nil?
        next_char == fallback_char()
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
    detokenize(generated_text)
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
    @vocab.to_a.sample
  end

  # Encode 16-bit context tokens into an array 64-bit integer ID (can handle up to 4 tokens)
  def context_id(token_context)
    result = []
    token_context.each_with_index do |token, idx|
      # Shift each token to its position and OR it in
      result[idx / 4] ||= 0
      result[idx / 4] |= (token & 0xFFFF) << ((idx % 4) * 16)
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
    current_context_total_count = options.values.sum

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
      weights[i] = Math.exp(bias_factor * (1.0 - i.to_f/@context_size))
    end

    # Normalize weights to sum to 1.0
    total = weights.sum
    weights.map! { |w| w / total }
    @weights = weights
    @weights
  end

  # Convert a token to an ID (or assign a new ID if not seen before)
  def get_token_id(token)
    unless @token_to_id.has_key?(token)
      @token_to_id[token] = @next_token_id
      @id_to_token[@next_token_id] = token
      @vocab.add(@next_token_id)
      @next_token_id += 1
      raise "Token overlfow, too many tokens, >2^16 unique tokens." if @next_token_id > 2**16
    end
    @token_to_id[token]
  end

  # Tokenize text into token IDs
  # TODO: HANDLE NEW LINES!
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
        elsif c == "—"
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
      .split(/\s+/)
      .map { |token| get_token_id(token) }
  end

  # Convert token IDs back to text
  # TODO: HYPHENATED WORDS NOT HANDLED WELL
  def detokenize(token_ids)
    p2_word = ""
    prev_word = ""

    token_ids
      .map { |id| @id_to_token[id] }
      .map do |word|
        # Convert paragraphs symbols into paragraphs.
        if word == PARAGRAPH
          word = PARAGRAPH_STR
        end

        # Titleize words at the beginning of prompts, sentances, paragraphs.
        if prev_word == "" || CAPITAL_MARKER.include?(prev_word) || (QUOTES.include?(prev_word) && CAPITAL_MARKER.include?(p2_word))
          word = word[0].upcase + word[1..-1]
        end

        # Add space to the beginning of a word
        if word != PARAGRAPH_STR
          # Quotes
          if QUOTES.include?(prev_word) || QUOTES.include?(word) || PUNCTUATION.include?(word)
            if QUOTES.include?(prev_word) && PUNCTUATION.include?(p2_word)
              word = " " + word
            end
          # Everything else
          else
            if prev_word != "" && prev_word != PARAGRAPH_STR
              word = " " + word
            end
          end
        end

        p2_word = prev_word
        prev_word = word
        word
      end
      .join("")
  end
end
