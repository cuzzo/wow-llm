#! /usr/bin/env ruby

require_relative("llm")

start_prompt = ARGV[0]
token_count = ARGV[1].to_i
n = ARGV[2].to_i
model_file = "models/model.#{n}.msgpack"
token_file = "models/tokens.#{n}.json"

llm = TokenLLM.new(n)
llm.load(model_file, token_file)

puts "START PROMPT: #{start_prompt}"
generated_output = llm.generate(start_prompt, token_count)

puts "\n--- Generated Text ---"
puts generated_output
puts "----------------------"
