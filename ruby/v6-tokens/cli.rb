#! /usr/bin/env ruby

require_relative("llm")
require "msgpack"

n = ARGV[0].to_i
model_file = "models/model.#{n}.msgpack"
token_file = "models/tokens.#{n}.msgpack"

puts "LOADING MODEL..."
llm = TokenLLM.new(n)
llm.load(
  MessagePack.unpack(File.read(model_file, mode: "rb")),
  MessagePack.unpack(File.read(token_file, mode: "rb"))
)

def help()
  puts "ENTER INPUT"
  puts "  FORMAT (N:T:K: <prompt>)"
  puts "  EXAMPLE"
  puts "> 500:1.0:0.5: once there was"
  puts ""
  puts "ENTER .q OR .quit TO QUIT"
  puts "ENTER .h OR .help FOR HELP"
  puts ""
end

help()

loop do
  puts "> "
  user_input = STDIN.gets

  if user_input.nil?
    puts "\nInput stream closed. Exiting."
    break # Exit the loop if gets returns nil
  end

  user_input = user_input.chomp

  if user_input.downcase == ".quit" || user_input.downcase == ".q"
    puts "Exit command received. Goodbye!"
    break # Exit the loop
  elsif user_input.empty?
    next
  elsif user_input.downcase == ".help" || user_input.downcase == ".h"
    help()
  else
    token_count, temp, k, prompt = user_input.split(":", 4)

    begin
      generated_output = llm.generate(prompt.chomp, token_count.to_i, temp.to_f, k.to_f)
      puts "\n--- Generated Text ---"
      puts generated_output
      puts "----------------------"
    rescue Error => e
      puts "ERROR"
      puts e
    end
  end
end
