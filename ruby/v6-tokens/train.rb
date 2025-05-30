#! /usr/bin/env ruby
# frozen_string_literal: true

require_relative("llm")
require "msgpack"

training_dir = ARGV[0]
n = ARGV[1].to_i
data_len = ARGV.size > 2 ? ARGV[2].to_i : nil

output_model_file = data_len.nil? ? "model.#{n}.msgpack" : "model.#{n}-#{data_len}.msgpack"
output_token_file = data_len.nil? ? "tokens.#{n}.msgpack" : "tokens.#{n}-#{data_len}.msgpack"

files = Dir
  .foreach(training_dir)
  .to_a
  .reject { |p| File.basename(p).start_with?(".") }

puts "TRAINING ON THESE FILES: #{files}"

training_data = files
  .reduce("") do |acc, f|
    if f == "." || f == ".."
      next acc
    end
    acc += File.read(File.join(ARGV[0], f))
    acc
  end


llm = TokenLLM.new(ARGV[1].to_i)

data_len = data_len || training_data.size
llm.train(training_data[0...data_len])

File.write("models/#{output_model_file}", llm.model.to_msgpack, mode: "wb")
File.write("models/#{output_token_file}", llm.token_to_id.to_msgpack, mode: "wb")

puts "MODEL WRITTEN TO: models/#{output_token_file}"
