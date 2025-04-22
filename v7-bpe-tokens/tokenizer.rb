require "tokenizers"

class BpeTokenizer
  # Training parameters
  TARGET_VOCAB_SIZE = 4096 # Desired size of the vocabulary after training
  MIN_FREQUENCY = 2      # Minimum number of times a pair must appear to be merged
  # Define special tokens required by many models
  SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"].freeze

  def initialize  
    # --- Tokenizer and Model Setup ---
    # 1. Create a BPE Model. This defines the core algorithm.
    #    Options like dropout can be passed here if needed: BPE.new(dropout: 0.1)
    @bpe_model = Tokenizers::Models::BPE.new
  
    # 2. Create a base Tokenizer instance configured to use the BPE model.
    @tokenizer = Tokenizers::Tokenizer.new(@bpe_model)
    @tokenizer.decoder = Tokenizers::Decoders::BPEDecoder.new()
  
    # 3. Configure Pre-tokenization (How text is split *before* BPE merging)
    #    - Whitespace: Simple split on whitespace. Good for space-separated languages.
    #    - ByteLevel: Treats the input as raw bytes. Handles any language/character.
    # Let's use Whitespace for this example.
    @tokenizer.pre_tokenizer = Tokenizers::PreTokenizers::Whitespace.new
  end

  def train(files, show_progress = true)
    # --- Trainer Setup ---
    # Create and configure the BpeTrainer
    bpe_trainer = Tokenizers::Trainers::BpeTrainer.new(
      vocab_size: TARGET_VOCAB_SIZE,
      min_frequency: MIN_FREQUENCY,
      special_tokens: SPECIAL_TOKENS,
      end_of_word_suffix: "</w>",
      show_progress: show_progress,
    )

    @tokenizer.train(files, bpe_trainer)
  end

  def tokenize(text) 
    @tokenizer.encode(text).ids
  end

  def detokenize(tokens)
    @tokenizer.decode(tokens)
  end
  
  def save(fname)  
    @tokenizer.save(fname, pretty: true) # pretty: true for human-readable JSON
  end

  def load(fname)
    @tokenizer = Tokenizers::Tokenizer.from_file(fname)
    @tokenizer.decoder = Tokenizers::Decoders::BPEDecoder.new()
  end
end
