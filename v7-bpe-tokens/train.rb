#! /usr/bin/env ruby

require_relative("llm")
require "msgpack"

training_dir = ARGV[0]
n = ARGV[1].to_i

output_model_file = "models/model.#{n}.msgpack"
output_token_file = "models/tokens.#{n}.json"

files = Dir
  .foreach(training_dir)
  .to_a
  .reject { |p| File.basename(p).start_with?(".") }
  .map { |p| File.join(training_dir, p) }

puts "TRAINING ON THESE FILES: #{files}"

llm = TokenLLM.new(ARGV[1].to_i)
llm.train(files)
llm.save(output_model_file, output_token_file)

puts "MODEL WRITTEN TO: #{output_token_file}"

