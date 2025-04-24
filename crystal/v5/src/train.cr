require "./llm"
require "msgpack"

training_dir = ARGV[0]
n = ARGV[1].to_i

output_file = "model.msgpack"

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


llm = NGramLLM.new(ARGV[1].to_i)
llm.train(training_data)

File.write(output_file, MessagePack.pack(llm.model), mode: "wb")

puts "MODEL WRITTEN TO: #{output_file}"

