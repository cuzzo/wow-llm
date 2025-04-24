require "set"
require "msgpack"
require "io/memory"

require "./tokeniza_bindings"

alias Token = UInt16
alias ContextId = UInt128
alias Options = Hash(Token, Int64)
alias WeightedOptions = Hash(Token, Float64)
alias Model = Hash(ContextId, Options)
alias StorableModel = Hash(Bytes, Options)

class NGramLLM
  # --- Instance Variables ---
  getter n : Int32
  getter model : Model
  getter vocab : Set(Token)

  @context_size : Int32
  @context_mask : ContextId
  @weights : Array(Float64)
  @tokenizer : RustTokenizerInterface

  DASHES = [150, 151].map { |c| c.chr } # em & en dash
  PARAGRAPH = "[PARAGRAPH]"
  TOKEN_BITS = 12
  CONTEXT_BITS = 128

  # --- Initialization ---
  # token_file: The path to the BPETokenizer json file
  # n: The order of the n-gram (e.g., 3 for trigrams)
  def initialize(@tokenizer, @n = 3)
    raise ArgumentError.new("n must be at least 2") if @n < 2
    raise ArgumentError.new("n is temporarily limited to 11") if @n > 11

    @context_size = @n - 1
    @context_mask = ContextId::MAX >> (CONTEXT_BITS - (@context_size * TOKEN_BITS))
    @weights = _weights(30.0)

    # Initialize the nested hash:
    # The outer hash maps String context to the inner Hash.
    # The inner hash maps the next Char to its count (Int64), defaulting to 0.
    @model = Model.new do |h, k|
      h[k] = Options.new(0_i64)
    end

    # Initialize the vocabulary set
    @vocab = Set(Token).new
  end

  def save(model_file)
    File.write(model_file, MessagePack.pack(get_storage_model()), mode: "wb")
  end

  def load(model_file)
    all_bytes = File.open(model_file, "rb") do |file|
      io_memory = IO::Memory.new
      IO.copy(file, io_memory)
      io_memory.to_slice
    end
    @model = convert_storage_model(StorableModel.from_msgpack(all_bytes))
    @vocab = @model.values.map { |dict| dict.keys }.flatten.to_set
  end

  # --- Training ---
  # Train the model on a given text string.
  def train(text : String)
    tokens = @tokenizer.encode(text).map { |t| t.to_u16 }

    # 1) Build vocabulary
    tokens.each { |token| @vocab.add(token) }

    # Ensure text is long enough to extract at least one n-gram
    return if tokens.size <= @context_size

    puts "Starting training..."

    cc_id = context_id([tokens[0]])

    # 2) For each context, set occurence of next char
    (1...tokens.size).each do |i|
      # Print progress periodically
      if i % 100_000 == 0
        puts "TRAINING: #{i} of #{tokens.size}"
      end

      next_token = tokens[i]

      # Increment the count for this next_token given the context.
      # The default proc on the inner hash handles initialization if needed.
      @model[cc_id][next_token] += 1

      # Backoff
      (1...Math.min(i, @context_size)).each do |i|
        backoff_cc_id = context_id_backoff(cc_id, i)
        @model[backoff_cc_id][next_token] += 1
      end

      cc_id = context_shift(cc_id, next_token)
    end

    puts "Training complete. Model has #{@model.size} contexts."
  end

  # --- Generation ---
  # Generate text starting with a prompt.
  def generate(prompt : String, length : Int32 = 100) : String
    raise "Model not trained yet! Vocabulary is empty." if @vocab.empty?
    # We check vocab emptiness as a proxy for training, since @model might be loaded.
    raise "Prompt must be at least #{@context_size} characters long" if prompt.size < @context_size

    # Start with the prompt (normalized to downcase)
    generated_text = @tokenizer.encode(prompt).map { |t| t.to_u16 }

    tokens = Array(Token).new(generated_text.size + length)
    tokens.concat(generated_text)

    # Get the initial context (last @context_size characters)
    puts "GENERATED TEXT: #{generated_text}"
    cc_id = context_id(generated_text[-@context_size..])

    puts "Generating #{length} characters..."
    length.times do
      options = next_options(cc_id)
      next_token = weighted_choice(options)

      if next_token.nil?
        next_token = fallback(cc_id)
        # If vocab somehow became empty or sample failed, break generation.
        break unless next_token
      end

      tokens << next_token

      cc_id = context_shift(cc_id, next_token)
    end

    puts "Generation complete."
    @tokenizer.decode(tokens.map { |t| t.to_u32 } )
  end

  # --- Private Helper Methods ---
  # Helper method to make a weighted random choice from a hash of {item => weight}.
  # Input: Hash where keys are Char and values are Int64 counts (weights).
  # Output: A chosen Char, or nil if the options hash is empty.
  private def weighted_choice(options : WeightedOptions) : Token?
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

  def context_id(context : Array(Token)) : ContextId
    result = 0_u128

    context.each_with_index do |token, index|
      index = context.size - index - 1
      result |= (token.to_u128 << (index * TOKEN_BITS))
    end

    result
  end

  # Get the possible next characters and their counts for the current context.
  def next_options(cc_id : ContextId) : WeightedOptions
    interpolate_options(cc_id)
  end

  def context_shift(cc_id : ContextId, next_token : Token) : ContextId
    cc_id = (cc_id << TOKEN_BITS) | next_token.to_u128
    cc_id &= @context_mask
    cc_id
  end

  private def context_id_backoff(cc_id : ContextId, i : Int32) : ContextId
    cc_id = cc_id << (i * TOKEN_BITS)
    cc_id &= @context_mask
    cc_id = cc_id >> (i * TOKEN_BITS)
    cc_id
  end

  def interpolate_options(cc_id : ContextId) : WeightedOptions
    (0...@context_size)
      .each
      .reduce({} of Token => Float64) do |acc, i|
        cc_id = context_id_backoff(cc_id, i)

        # Get counts for this context
        options = @model[cc_id]?
        next acc if options.nil?
        
        weight = @weights[i]
        options.each do |token, v|
          acc[token] ||= 0.0
          acc[token] += v * weight
        end

        acc
     end  
  end

  private def _weights(bias_factor : Float64) : Array(Float64)
    # Initialize weights array for n-grams of different orders
    # Index 0 will be for the highest order n-gram (full context_size)
    # Last index will be for unigrams
    weights = Array(Float64).new(@context_size, 0)
    
    # Generate exponentially biased weights
    # Higher bias_factor means stronger preference for higher-order n-grams
    (0...@context_size).each do |i|
      # This formula exponentially increases the weight as i decreases
      # (remember i=0 is the highest-order n-gram)
      weights[i] = Math.exp(bias_factor * (1.0 - i.to_f/@context_size))
    end
    
    # Normalize weights to sum to 1.0
    total = weights.sum
    weights.map! { |w| w / total }
    weights
  end

  # Fallback: Context not found or has no successors recorded.
  # Pick a random character from the entire vocabulary.
  private def fallback(cc_id : ContextId) : Token?
    STDERR.puts "Warning: Context '#{cc_id}' not found or has no successors in model. Choosing random character."
    @vocab.to_a.sample
  end

  private def fmt_context_id(cc_id : ContextId) : Array(Token)
    mask = (Token::MAX).to_u128
    (0...@context_size)
      .map do |i|
        shift = i * TOKEN_BITS
        ((cc_id & (mask << shift)) >> shift).to_u32
      end
      .reverse
  end

  private def get_storage_model() : StorableModel
    storable = StorableModel.new
    @model.each do |key, value|
      # Use IO::Memory to easily write the UInt128 bytes
      io = IO::Memory.new(16) # Allocate 16 bytes for UInt128
      io.write_bytes(key, IO::ByteFormat::BigEndian) # Or LittleEndian, but be consistent!
      key_bytes = io.to_slice
      storable[key_bytes] = value
    end
    storable
  end
  
  private def convert_storage_model(raw_model : StorableModel) : Model
    model = Model.new
    raw_model.each do |key_bytes, value|
      # Ensure the byte slice is exactly 16 bytes
      raise "Invalid key size: expected 16 bytes, got #{key_bytes.size}" if key_bytes.size != 16
  
      # Use IO::Memory to easily read the UInt128 bytes
      io = IO::Memory.new(key_bytes)
      key = io.read_bytes(UInt128, IO::ByteFormat::BigEndian) # Use the SAME endianness as serialization!
      model[key] = value
    end
    model
  end 
end
