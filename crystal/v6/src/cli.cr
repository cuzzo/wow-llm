#! /usr/bin/env ruby

require "./llm"

n = ARGV[0].to_i
model_file = "model.msgpack"
token_file = "tokens.msgpack"

def help()
  puts "ENTER INPUT"
  puts "  FORMAT (N: <prompt>)"
  puts "  EXAMPLE"
  puts "> 500: once there was"
  puts ""
  puts "ENTER .q OR .quit TO QUIT"
  puts "ENTER .h OR .help FOR HELP"
  puts ""
end 

puts "LOADING MODEL..."
llm = NGramLLM.new(n)
llm.load(model_file, token_file)

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
    token_count, prompt = user_input.split(":", 2)

    begin 
      generated_output = llm.generate(prompt, token_count.to_i)
      puts "\n--- Generated Text ---"
      puts generated_output
      puts "----------------------"
    rescue e
      puts "ERROR"
      puts e
    end
  end 
end

