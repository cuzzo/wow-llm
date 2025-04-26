#! /usr/bin/env ruby
# frozen_string_literal: true

require_relative("llm")
require "json"

training_data = ARGV[0]
puts "TRAINING DATA: #{training_data}"

llm = NGramLLM.new(3)
llm.train(training_data)
puts JSON.pretty_generate(llm.model)

start_prompt = ARGV[1]
puts "START PROMPT: #{start_prompt}"

generated_output = llm.generate(start_prompt, 200) # Generate 200 characters

puts "\n--- Generated Text ---"
puts generated_output
puts "----------------------"
