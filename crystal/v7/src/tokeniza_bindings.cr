require "./tokeniza_bindings"

# Low-level bindings to the Rust library
@[Link(ldflags: "-L/root/dev/wow-llm/rust/tokeniza/target/release -ltokeniza")] # Adjust path and lib name if needed
lib RustLib
  # --- Types ---
  # Opaque pointer to the Rust Tokenizer object
  alias TokenizerT = Void*

  # Struct matching the Rust CU32Array
  struct CU32Array
    data : UInt32* # Corresponds to uint32_t*
    len : LibC::SizeT    # Corresponds to size_t
  end

  # --- Functions ---
  # Training and Loading
  fun rust_tokenizer_train(dir_path : LibC::Char*, output_file : LibC::Char*) : TokenizerT
  fun rust_tokenizer_load(file_path : LibC::Char*) : TokenizerT
  fun rust_tokenizer_free(tokenizer_ptr : TokenizerT)

  # Encoding
  fun rust_tokenizer_encode(tokenizer_ptr : TokenizerT, text : LibC::Char*) : CU32Array
  fun rust_free_u32_array(arr : CU32Array)

  # Decoding
  fun rust_tokenizer_decode(tokenizer_ptr : TokenizerT, ids_ptr : UInt32*, len : LibC::SizeT) : LibC::Char*
  fun rust_free_string(s : LibC::Char*)
end

require "file_utils" # For creating directories if needed

abstract class RustTokenizerInterface
  #abstract def self.train(dir_path : String) : self
  #abstract def self.load(file_path : String) : self
  abstract def finalize
  abstract def encode(text : String) : Array(UInt32)  
  abstract def decode(token_ids : Array(UInt32)) : String
end

# Crystal wrapper class for the Rust tokenizer
class RustTokenizer < RustTokenizerInterface
  # Hold the pointer to the Rust tokenizer object
  @ptr : RustLib::TokenizerT

  # Class method to train a tokenizer
  def self.train(dir_path : String, output_file : String) : self
    raise ArgumentError.new("Training directory does not exist: #{dir_path}") unless Dir.exists?(dir_path)

    ptr = RustLib.rust_tokenizer_train(dir_path.to_unsafe, output_file.to_unsafe)
    if ptr.null?
      raise "Failed to train Rust tokenizer (check Rust stderr for details)"
    end
    new(ptr) # Create a new instance with the valid pointer
  end

  # Class method to load a tokenizer
  def self.load(file_path : String) : self
    raise ArgumentError.new("Tokenizer file does not exist: #{file_path}") unless File.exists?(file_path)

    ptr = RustLib.rust_tokenizer_load(file_path.to_unsafe)
    if ptr.null?
      raise "Failed to load Rust tokenizer from #{file_path} (check Rust stderr for details)"
    end
    new(ptr) # Create a new instance with the valid pointer
  end

  # Private initializer, use .train or .load
  private def initialize(@ptr : RustLib::TokenizerT)
  end

  # Ensure the Rust memory is freed when the Crystal object is garbage collected
  def finalize
    puts "Crystal finalize called for tokenizer" # For debugging
    RustLib.rust_tokenizer_free(@ptr) unless @ptr.null?
    @ptr = Pointer(Void).null # Avoid double free
  end

  # Encode text
  def encode(text : String) : Array(UInt32)
    raise "Tokenizer already freed" if @ptr.null?

    # Call the Rust function
    c_array = RustLib.rust_tokenizer_encode(@ptr, text.to_unsafe)

    # Check for error (null data pointer)
    if c_array.data.null? && c_array.len == 0
        raise "Rust tokenizer encode failed (check Rust stderr)"
    end

    # Convert the C array to a Crystal Array
    # Create a Slice from the C pointer and length
    slice = Slice.new(c_array.data, c_array.len.to_i32) # Use to_i32 for Slice size
    result = slice.to_a # Convert Slice to Array

    # IMPORTANT: Free the memory allocated by Rust for the array data
    RustLib.rust_free_u32_array(c_array)

    result
  end

  # Decode token IDs
  def decode(token_ids : Array(UInt32)) : String
    raise "Tokenizer already freed" if @ptr.null?

    # Call the Rust function, passing the array's pointer and size
    c_string_ptr = RustLib.rust_tokenizer_decode(@ptr, token_ids.to_unsafe, token_ids.size.to_u64) # Use to_u64 for size_t

    # Check for error
    if c_string_ptr.null?
      raise "Rust tokenizer decode failed (check Rust stderr)"
    end

    # Convert the C string to a Crystal String
    result = String.new(c_string_ptr)

    # IMPORTANT: Free the memory allocated by Rust for the string
    RustLib.rust_free_string(c_string_ptr)

    result
  end
end
