#! /usr/bin/env ruby

require 'set'

class NGramLLM
  attr_reader :n, :model, :vocab

  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(n = 3)
    raise ArgumentError, "n must be at least 2" if n < 2
    @n = n

    # @model stores the counts: { context => { next_char => count } }
    # Example for n=3: { "th" => { "e" => 10, "a" => 3 }, "he" => { " " => 5, "l" => 8 } }
    @model = Hash.new { |h, k| h[k] = Hash.new(0) }
    @vocab = Set.new # Keep track of all unique characters seen
    @context_size = n - 1
  end

  def load(model)
    @model = model
  end

  # Train the model on a given text
  def train(text)
    # Normalize the text to lowercase so that "Th" and "th" are not two different probabilities.
    text = text.downcase

    # 1) Build vocabulary
    text.each_char { |char| @vocab.add(char) }

    # 2) For each context, set occurence of next char
    (text.length - @context_size).times do |i|
      if i % 10_000 == 0
        puts "TRAINING: #{i} of #{text.length}"
      end
      context = text[i...(i + @context_size)]
      next_char = text[i + @context_size]

      # Increment the count for this next_char given the context
      @model[context][next_char] += 1
    end

    puts "Training complete. Model has #{@model.keys.size} contexts."
  end

  # Generate text starting with a prompt
  def generate(prompt, length = 100)
    raise "Model not trained yet!" if @model.empty?
    raise "Prompt must be at least #{@context_size} characters long" if prompt.length < @context_size

    # Start with the prompt (normalized to downcase, since the input is normalized to downcase)
    generated_text = prompt.downcase

    # Get the last context_size characters
    current_context = generated_text[(-@context_size)..]
    puts "Current context: #{current_context}"

    puts "Generating #{length} characters..."
    length.times do
      # Get the possible next characters and their counts for the current context
      next_char_options = @model[current_context]

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
      current_context = generated_text[(-@context_size)..]
    end

    puts "Generation complete."
    generated_text
  end

  private

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
