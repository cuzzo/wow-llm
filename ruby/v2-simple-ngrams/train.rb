#! /usr/bin/env ruby

require_relative("llm")
require "json"

puts "TRAINING ON THESE FILES: #{Dir.foreach(ARGV[0]).to_a}"

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

data_len = ARGV.size > 2 ? ARGV[2].to_i : training_data.size

llm = NGramLLM.new(ARGV[1].to_i)
llm.train(training_data[0...data_len])

File.write("model.#{ARGV[1]}-#{ARGV[2]}.json", llm.model.to_json)

puts "MODEL WRITTEN TO: model.json"
