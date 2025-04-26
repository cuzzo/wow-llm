#! /usr/bin/env ruby
# frozen_string_literal: true

require_relative("llm")
require "msgpack"

start_prompt = ARGV[0]
token_count = ARGV[1].to_i
n = ARGV[2].to_i
model_file = ARGV[3]

llm = NGramLLM.new(n)
llm.load(MessagePack.unpack(File.read(model_file, mode: "rb")))

puts "START PROMPT: #{start_prompt}"
generated_output = llm.generate(start_prompt, token_count)

puts "\n--- Generated Text ---"
puts generated_output
puts "----------------------"
