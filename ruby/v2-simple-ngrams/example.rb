#! /usr/bin/env ruby
# frozen_string_literal: true

require_relative("llm")
require "json"

puts "DIR: #{Dir.foreach(ARGV[0]).to_a}"

training_data = Dir
  .foreach(ARGV[0])
  .to_a
  .reduce("") do |acc, f|
    if f == "." || f == ".."
      next acc
    end
    acc += File.read(File.join(ARGV[0], f))
    acc
  end

puts "TRAINING DATA: #{training_data[1...100]}..."

llm = NGramLLM.new(3)
llm.train(training_data)

File.write("model.json", llm.model.to_json)

start_prompt = ARGV[1]
puts "START PROMPT: #{start_prompt}"

generated_output = llm.generate(start_prompt, 200) # Generate 200 characters

puts "\n--- Generated Text ---"
puts generated_output
puts "----------------------"
