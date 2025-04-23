require "set"

class NGramLLM
  # --- Instance Variables ---
  getter n : Int32
  getter model : Hash(UInt64, Hash(UInt8, Int64)) # { context => { next_char => count } }
  getter vocab : Set(UInt8)

  @context_size : Int32
  @context_mask : UInt64

  # --- Initialization ---
  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(@n = 3)
    raise ArgumentError.new("n must be at least 2") if @n < 2
    raise ArgumentError.new("n is temporarily limited to 9") if @n > 9

    @context_size = @n - 1
    #@context_mask = UInt64::MAX << @context_size * 8
    @context_mask = UInt64::MAX >> (64 - (@context_size * 8))

    # Initialize the nested hash:
    # The outer hash maps String context to the inner Hash.
    # The inner hash maps the next Char to its count (Int64), defaulting to 0.
    @model = Hash(UInt64, Hash(UInt8, Int64)).new do |h, k|
      h[k] = Hash(UInt8, Int64).new(0_i64)
    end

    # Initialize the vocabulary set
    @vocab = Set(UInt8).new
  end

  # --- Load Model (Simple Assignment) ---
  # Allows replacing the model with a pre-loaded one.
  # Ensure the loaded model has the correct type structure.
  def load(model : Hash(UInt64, Hash(UInt8, Int64)))
    @model = model
    @vocab = @model.values.map { |dict| dict.keys }.flatten.to_set
  end

  # --- Training ---
  # Train the model on a given text string.
  def train(text : String)
    # Normalize the text to lowercase
    text = text.downcase.to_slice

    # 1) Build vocabulary
    text.each { |char| @vocab.add(char) }

    # Ensure text is long enough to extract at least one n-gram
    return if text.size <= @context_size

    puts "Starting training..."

    # 2) For each context, set occurence of next char
    (1...text.bytesize).each do |i|
      # Print progress periodically
      if i % 100_000 == 0
        puts "TRAINING: #{i} of #{text.bytesize}"
      end

      start_index = Math.max(0, i - @context_size)
      end_index = i - 1

      # Extract context (String of length @context_size)
      context = text[start_index..end_index]
      # Extract the character following the context
      next_char = text[i]

      cc_id = context_id(context)

      # Increment the count for this next_char given the context.
      # The default proc on the inner hash handles initialization if needed.
      @model[cc_id][next_char] += 1

      # Backoff
      (1...context.bytesize).each do |i|
        backoff_cc_id = context_id_backoff(cc_id, i)
        @model[backoff_cc_id][next_char] += 1
      end
    end

    puts "Training complete. Model has #{@model.size} contexts."
  end

  # --- Generation ---
  # Generate text starting with a prompt.
  def generate(prompt : String, length : Int32 = 100) : String
    raise "Model not trained yet! Vocabulary is empty." if @vocab.empty?
    # We check vocab emptiness as a proxy for training, since @model might be loaded.
    raise "Prompt must be at least #{@context_size} characters long" if prompt.size < @context_size

    # Use String::Builder for efficient string concatenation in the loop
    builder = String::Builder.new

    # Start with the prompt (normalized to downcase)
    generated_text_base = prompt.downcase.to_slice
    builder << prompt 

    # Get the initial context (last @context_size characters)
    cc_id = context_id(generated_text_base[-@context_size..])

    puts "Generating #{length} characters..."
    length.times do
      options = next_options(cc_id)
      next_char = nil # Initialize next_char as nil

      # Check if context was found and has successor characters
      if options && !options.empty?
        # Choose the next character based on weighted probability (counts)
        next_char = weighted_choice(options)
      else
        # Fallback: Context not found or has no successors recorded.
        # Pick a random character from the entire vocabulary.
        STDERR.puts "Warning: Context '#{cc_id}' not found or has no successors in model. Choosing random character."

        # Convert Set to Array to sample. Use `?` as sample can return nil if array is empty.
        next_char = @vocab.to_a.sample

        # If vocab somehow became empty or sample failed, break generation.
        break unless next_char
      end

      # If we failed to get a character (e.g., fallback failed), stop.
      break unless next_char

      # Append the chosen character
      builder << next_char.unsafe_chr

      cc_id = context_shift(cc_id, next_char)
    end

    puts "Generation complete."
    builder.to_s # Return the final generated string
  end

  # --- Private Helper Methods ---
  # Helper method to make a weighted random choice from a hash of {item => weight}.
  # Input: Hash where keys are Char and values are Int64 counts (weights).
  # Output: A chosen Char, or nil if the options hash is empty.
  private def weighted_choice(options : Hash(UInt8, Int64)) : UInt8?
    return nil if options.empty?

    total_weight = options.values.sum

    # If total weight is zero or negative (shouldn't happen with positive counts),
    # fallback to random sample among keys.
    return options.keys.sample if total_weight <= 0

    # Generate a random float between 0.0 and total_weight
    random_num = rand * total_weight

    cumulative_weight = 0_i64
    options.each do |char, weight|
      cumulative_weight += weight
      # Compare the random float against the cumulative weight
      return char if random_num < cumulative_weight.to_f64
    end

    # Fallback (should theoretically not be reached if total_weight > 0)
    # Use `sample?` which returns nil if keys are empty (already checked).
    options.keys.sample
  end

  def context_id(context : Slice(UInt8)) : UInt64
    result = 0_u64

    context.each_with_index do |byte, index|
      index = context.bytesize - index - 1
      result |= (byte.to_u64 << (index * 8))
    end

    result
  end

  # Get the possible next characters and their counts for the current context.
  # Use `?` for safe navigation in case the context key doesn't exist in the model.
  def next_options(cc_id : UInt64) : Hash(UInt8, Int64)?
    options = @model[cc_id]?
    return options if !options.nil?

    # Backoff
    (1...@context_size).each do |i| 
      backoff_cc_id = context_id_backoff(cc_id, i)
      options = @model[backoff_cc_id]?
      return options if !options.nil?
    end

    nil
  end

  def context_shift(cc_id : UInt64, next_char : UInt8) : UInt64
    cc_id = (cc_id << 8) | next_char.to_u64
    cc_id &= @context_mask
    cc_id
  end

  private def context_id_backoff(cc_id : UInt64, i : Int32) : UInt64
    cc_id = cc_id << i * 8
    cc_id &= @context_mask
    cc_id = cc_id >> i * 8
    cc_id
  end
end
