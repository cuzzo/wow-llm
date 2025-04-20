#! /usr/bin/env ruby

require_relative("llm")
require "json"

n = ARGV[2].to_i

llm = NGramLLM.new(n)
llm.load(JSON.parse(File.read("model.#{n}.json")))

start_prompt = ARGV[0]
token_count = ARGV[1].to_i

puts "START PROMPT: #{start_prompt}"

generated_output = llm.generate(start_prompt, token_count)

puts "\n--- Generated Text ---"
puts generated_output
puts "----------------------"
