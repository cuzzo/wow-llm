# Optimized Ngrams

The brunt of the work done in training is copying with String slicing:

```bash
ruby --yjit --yjit-stats v2-simple-ngrams/train.rb train 4 100000
```

```
context_cache_hit_rate:           74 (45.7%)
live_page_count:                   1
freed_page_count:                  0
code_gc_count:                     0
num_gc_obj_refs:                  23
object_shape_count:              236
side_exit_count:                   1
total_exit_count:            208,814
total_insns_count:         7,331,750
vm_insns_count:            1,253,590
yjit_insns_count:          6,078,161
ratio_in_yjit:                 82.9%
avg_len_in_yjit:                29.1
Top-1 most frequent exit ops (100.0% of exits):
  1 (100.0%): leave
Top-4 most frequent C calls (51.0% of C calls):
  99,968 (49.0%): String#[]
   4,223 ( 2.1%): Class#new
      18 ( 0.0%): Integer#to_s
       9 ( 0.0%): Kernel#puts
Top-1 most frequent ISEQ calls (99.6% of ISEQ calls):
```

On a single book, this can take >11M for a modestly sized book. To train on all 5 books would take greater than an hour.

Since we'll want to regularly train our model as we iterate over it to improve it, this is simply too slow.

To get around this, we optimze the `train` method in a few ways:

## NGram Packing

We pack the n-gram strings into an integer. This works while our model is simple, but will become a problem later.

This is done mainly by the new function:

```ruby
  # Encode an n-gram of 3 bytes into a 32-bit integer.
  def context_id(byte_context)
    ([0] + byte_context)
      .pack('C4')
      .unpack('N')
      .first
  end
```

This looks like black magic, but it is quite simple:

```
input = "hello..."
context = [104, 101, 108]  # ["h", "e", "l"] as bytes instead of strings
next_char = 108 # "l" as byte instead of string

([0] + context) # [0, 104, 101, 108]
  .pack("C4") # pack into a the string of 4 Chars => "\0hel"
  .unpack("N") # unpack into an integer (formatted to send over the network) => [6841708]
  .first # 6841708

## Push/pop instead of constant substring

Due to all the overhead of the String object, it's inefficient to constantly splice the string as we did before:

```ruby
      context = text[i...(i + @context_size)]
```

Instead, each time we iterate through the string, we will add the current character to the end of our context, and then we will remove the first character from the context. This is much faster:

```ruby
      context.shift()
      context.push(next_char) 
```

## Results

Prior to optimization, training on a single book took:

```
> time ruby --yjit v2-simple-ngrams/train.rb train 4 

real    19m20.365s
user    18m48.708s
sys     0m1.837s
```

Post-optimization, it takes:

```
> time ruby --yjit v3-ngrams-optimzed/train.rb train 4

real    0m2.014s
user    0m1.919s
sys     0m0.042s
```

This is ~99.8% faster.

## Tradeoffs

This design will only work for ascii-based text. It will not work for unicode.

This is an acceptable trade-off at this time, as training on several Gigabytes of text -- even on a fast comptuer -- could take days.

