require "./llm"
require "./tokeniza_bindings"

training_dir = ARGV[0]
n = ARGV[1].to_i

output_model_file = "model.msgpack"
output_token_file = "tokenizer.json"

puts "TRAINING BPE TOKENIZER..."
tokenizer = RustTokenizer.train(training_dir, output_token_file)

files = Dir
  .entries(training_dir)
  .reject { |p| File.basename(p).starts_with?(".") }

puts "TRAINING ON THESE FILES: #{files}"

training_data = files
  .reduce("") do |acc, f|
    if f == "." || f == ".."
      next acc
    end
    acc += File.read(File.join(training_dir, f))
    acc
  end


llm = NGramLLM.new(tokenizer, ARGV[1].to_i)
llm.train(training_data)
llm.save(output_model_file)

puts "MODEL WRITTEN TO: #{output_model_file}"

