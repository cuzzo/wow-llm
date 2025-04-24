require "./tokeniza_bindings"
require "msgpack"

# --- Example Usage ---

training_dir = "../../train"

# --- Option 1: Load an existing tokenizer (assuming you trained and saved one) ---
# First, train and save using Rust if needed, or ensure tokenizer.json exists
# For this example, let's assume tokenizer.json was created by a previous Rust run
tokenizer_file = "tokenizer.json"
# Ensure the file exists for loading example
unless File.exists?(tokenizer_file)
    puts "tokenizer.json not found, attempting to create one via training..."
    begin
      temp_tokenizer = RustTokenizer.train(training_dir)
    rescue e
      puts "Could not train/create tokenizer.json: #{e}. Exiting load example."
      exit(1) # Exit if we can't ensure the file exists
    end
end

# Now try loading
tokenizer = RustTokenizer.load(tokenizer_file)

files = Dir
  .entries(training_dir)
  .reject { |p| File.basename(p).starts_with?(".") }

puts "TOKENIZING THESE FILES: #{files}"

training_data = files
  .reduce("") do |acc, f|
    if f == "." || f == ".."
      next acc
    end
    acc += File.read(File.join(training_dir, f))
    acc
  end

begin
  encoded_ids = tokenizer.encode(training_data)
  File.write("training-tokens.msgpack", MessagePack.pack(encoded_ids), mode: "wb")
          
  puts "Data written to: training-tokens.msgpack"
rescue e
  puts "Error during encode/decode: #{e}"
end
