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

llm = NGramLLM.new(ARGV[1].to_i)
llm.train(training_data)

File.write("model.#{ARGV[1]}.json", llm.model.to_json)

puts "MODEL WRITTEN TO: model.json"


