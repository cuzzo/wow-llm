#! /usr/bin/env ruby
# frozen_string_literal: true

require_relative "llm"

# Renders embeddings 2x2, 2 input x 2 context size, 3 hidden size, 3 output NN
# Each stage represents in 3 lines
# Input = 22 chars
# Ouput = 8 chars
class NNLMPresenter < NNLM
  def build_vocabulary(vocab)
    @ix_to_word = vocab 
    @word_to_ix = @ix_to_word.each_with_index.to_h
    @vocab_size = @ix_to_word.size

    _initialize_parameters()
  end

  # decimal format
  # Sn.ff
  def df(d)
    sprintf("%.2f", d).rjust(5, " ")
  end

  # Inline vector format, to print matrixes
  def vf(v)
    "[ " + v.map { |v| df(v) }.join(", ") + " ]"
  end

  def format_vector(vec, height, label)
    pad_top = (height - vec.size) / 2
    pad_bot = height - vec.size - pad_top

    vec_rows = vec.map { |d| " #{df(d)} " }
    rows = ([""] * pad_top) + vec_rows + ([""] * pad_bot)
    len = vec_rows.map(&:length).max
    label = label.center(len, " ")
    rows.unshift(label)
    rows.map { |r| r.ljust(len, " ") }
  end

  def format_matrix(mat, height, label)
    pad_top = (height - mat.size) / 2
    pad_bot = height - mat.size - pad_top

    mat_rows = mat.map { |v| " #{vf(v)} " }
    rows = ([""] * pad_top) + mat_rows + ([""] * pad_bot)
    len = mat_rows.map(&:length).max
    label = label.center(len, " ")
    rows.unshift(label)
    rows.map { |r| r.ljust(len, " ") }
  end


  def format_sign(sign, times, height)
    sign = " #{sign} "
    sl = sign.length
    pad_top = (height - times) / 2
    pad_bot = height - times - pad_top

    rows = ([""] * (pad_top + 1)) + ([sign] * times) + ([""] * pad_bot)
    rows.map { |r| r.ljust(sl, " ") }
  end

  def present_forward(context_indices)
    puts " ~~~ ~~~~~~~ ~~~ "
    puts " ~~~ FORWARD ~~~ "
    puts " ~~~ ~~~~~~~ ~~~ "

    il = input_layer(context_indices)
    hi = hidden_input(il)
    ha = tanh(hi)
    os = output_scores(ha)
    sm = softmax(os)

    rows = [
        format_embeddings(context_indices),
        format_sign("=>", 2, 4),
        format_vector(il, 4, "IL"),
        format_sign("x", 2, 4),
        format_matrix(@W_h, 4, "HW"),
        format_sign("+", 1, 4),
        format_vector(@b_h, 4, "HB"),
        format_sign("=>", 1, 4), 
        format_vector(hi, 4, "HI"),
        format_sign("~tanh~", 1, 4),
        format_vector(ha, 4, "HA"),
        format_sign("x", 1, 4),
        format_matrix(@W_o, 4, "OW"),
        format_sign("+", 1, 4),
        format_vector(@b_o, 4, "OB"),
        format_sign("=>", 1, 4),
        format_vector(os, 4, "RAW"),
        format_sign("~softmax~", 1, 4),
        format_vector(sm, 4, "PRED"),
      ]
      .transpose()
      .map(&:join)

    puts rows
  end

  def present_backward(context_indices, target_idx)
    puts " ~~~ ~~~~~~~~ ~~~ "
    puts " ~~~ BACKWARD ~~~ "
    puts " ~~~ ~~~~~~~~ ~~~ "

    il = input_layer(context_indices)
    hi = hidden_input(il)
    ha = tanh(hi)
    os = output_scores(ha)
    sm = softmax(os)

    es = sm.dup
    es[target_idx] -= 1

    dhis = multiply_vec_mat(es, transpose(@W_o))
    dhi = multiply_elementwise(dhis, dtanh(ha))

    dil = multiply_vec_mat(dhi, transpose(@W_h))
    grad_embeddings = gradient_embeddings(dhi,  dil, context_indices)

    delta_embeddings = [
      format_vector(es, 4, "ErrS"),
      format_sign("x", 2, 4),
      format_matrix(transpose(@W_o), 4, "TRANS(OH)"),
      format_sign("=>", 2, 4),
      format_vector(dhis, 4, "dHIS"),
      format_sign("x", 2, 4),
      format_vector(dtanh(ha), 4, "dT(ha)"),
      format_sign("=>", 2, 4),
      format_vector(dhi, 4, "dHI"),
      format_sign("x", 2, 4),
      format_matrix(transpose(@W_h), 4, "TRANS(OW)"),
      format_sign("=>", 2, 4),
      format_vector(dil, 4, "dIL")
      ]
      .transpose()
      .map(&:join)

    grad_W_h = outer_product(input_layer(context_indices), dhi)
    grad_b_h = dhi

    delta_hidden_weights = [
        format_vector(input_layer(context_indices), 4, "IL"),
        format_sign("~outer product~", 1, 4),
        format_vector(dhi, 4, "dHI"),
        format_sign("=>", 2, 4),
        format_matrix(grad_W_h, 4, "dWH")
      ]
      .transpose()
      .map(&:join)

    grad_W_o = outer_product(ha, es)
    grad_b_o = es

    delta_output_weights = [
        format_vector(es, 4, "ErrS"),
        format_sign("~outer product~", 1, 4),
        format_vector(ha, 4, "ha"),
        format_sign("=>", 2, 4),
        format_matrix(grad_W_h, 4, "dWO")
      ]
      .transpose()
      .map(&:join)


    puts ""
    puts " -> GRAD EMBEDDINGS"
    puts delta_embeddings
    puts ""
    puts " --> FOR EACH EMBEDDING"
    puts " -----> APPLY ERROR (SCALED FOR LEARNING RATE)"

    puts ""
    puts " -> GRAD OUTPUT BIAS (ERROR SIGNAL)"
    puts ""
    puts format_vector(grad_b_o, 4, "dBO")

    # GRAD W_o
    puts ""
    puts " -> GRAD OUTPUT WEIGHTS"
    puts ""
    puts delta_output_weights 

    # GRAD W_h
    puts ""
    puts " -> GRAD HIDDEN WEIGHTS"
    puts ""
    puts delta_hidden_weights 

    puts ""
    puts " -> GRAD HIDDEN BIAS (dHI)"
    puts ""
    puts format_vector(grad_b_h, 4, "dWH")

  end

  def gradient_embeddings(dhi, dil, context_indices)
    grad_embeddings = Hash.new { |h, k| h[k] = Array.new(@embedding_dim, 0.0) }
    context_indices.each_with_index do |word_ix, i|
      start_idx = i * @embedding_dim
      end_idx = start_idx + @embedding_dim - 1

      # Get the portion of error relevant to this word
      embedding_grad_slice = dil[start_idx..end_idx]

      # Add it to our correction sheet for this word's embedding
      # (We add because the same word might appear multiple times)
      grad_embeddings[word_ix] = add_vectors(grad_embeddings[word_ix], embedding_grad_slice)
    end
  end

  def format_embeddings(context_indices)
    idx_0 = ("x" * context_indices.select { |idx| idx == 0 }.size).rjust(2, " ")
    idx_1 = ("x" * context_indices.select { |idx| idx == 1 }.size).rjust(2, "-")
    [
      "EMEDDINGS".center(20),
      "",
      idx_0 + " " + vf(@embeddings[0]),
      idx_1 + " " + vf(@embeddings[1]),
      ""
    ].map { |l| l.ljust(20, " ") }
  end

  # WEIGHTS ARE MATRIXES
  # BIASES ARE VECTORS
  def to_a
    [
        format_matrix(@embeddings.values.first(4), 4, "EMBEDDINGS"),
        format_sign(" ", 1, 4),
        format_matrix(@W_h, 4, "W_h"),
        format_sign(" ", 1, 4),
        format_matrix(@W_o, 4, "W_o"),
        format_sign(" ", 1, 4),
        format_vector(@b_h, 4, "b_h"),
        format_sign(" ", 1, 4),
        format_vector(@b_o.first(4), 4, "b_o")
      ]
      .transpose()
      .map(&:join)
  end

  def input_layer(context_indices)
    context_indices.map { |ix| @embeddings[ix] }.flatten
  end

  def hidden_input(input_layer)
    add_vectors(
      multiply_vec_mat(input_layer, @W_h), # Multiply -> # Transform: Apply weights to extract meaningful patterns for each neuron
      @b_h) # Add -> Adjust for the baseline preference of each hidden neuron
  end

  def output_scores(hidden_activation)
    add_vectors(
      multiply_vec_mat(hidden_activation, @W_o), # Multiply -> Each hidden feature votes on possible next words
      @b_o) # Add -> Adjust for the baseline preference of each word
  end

  def process(context_indices, target_index)
    # Forward pass
    forward_data = forward(context_indices)
    probabilities = forward_data[:probabilities]

    # Calculate Loss (Cross-Entropy) - optional for training but good for monitoring
    loss = -Math.log(probabilities[target_index] + 1e-9) # Add epsilon for numerical stability

    # Backward pass
    gradients = backward(context_indices, target_index, forward_data)

    # Update parameters
    update_parameters(gradients)

    loss
  end
end


def truth_table_example
  epochs = 2_000
  update_size = 10

  nnlm = NNLMPresenter.new(
    embedding_dim: 2, # Each "word" has 2 dimensions
    context_size: 2, # Use 2 words
    hidden_size: 3, # Small hidden layer
    learning_rate: 0.05
  )
  
  nnlm.build_vocabulary(["false", "true"])
  
  #nnlm.present_forward(context_indices)
  #nnlm.present_backward(context_indices, 0)
  
  total_loss = 0
  (1..epochs).each do |i|
    if i % update_size == 0
      puts nnlm.to_a
    end

    # 0 & 0 = 0
    # 0 & 1 = 0
    # 1 & 0 = 0
    # 1 & 1 = 1
    total_loss += nnlm.process([0, 0], 0)
    total_loss += nnlm.process([0, 1], 0)
    total_loss += nnlm.process([1, 0], 0)
    total_loss += nnlm.process([1, 1], 1)
  
    if i % update_size == 0
      loss = total_loss / update_size.to_f / 4.0
      puts "Epoch #{i}/#{epochs}, Average Loss: #{loss.round(4)}, Perplexity: #{(Math::E**loss).round(4)}, Correct: #{nnlm.df(Math::E**(-loss)*100)}%"
      puts ""
      total_loss = 0
    end
  end
end

def language_example
  epochs = 2_000
  update_size = 100
  nnlm = NNLMPresenter.new(
    embedding_dim: 2, # Each "word" has 3 dimensions
    context_size: 2, # Use 2 words
    hidden_size: 3, # Small hidden layer
    learning_rate: 0.05
  )

  nnlm.build_vocabulary(["not", "tall", "is", "short", "green", "UNK", "girl", "boy"])
  
  total_loss = 0
  (1..epochs).each do |i|
    if i % update_size == 0
      puts nnlm.to_a
    end
 
    total_loss += nnlm.process([0, 1], 3) # not tall
    total_loss += nnlm.process([1, 0], 5) # tall not
    total_loss += nnlm.process([2, 1], 1) # is tall
    total_loss += nnlm.process([1, 2], 5) # tall is
    total_loss += nnlm.process([2, 3], 3) # is short
    total_loss += nnlm.process([3, 2], 5) # short is
    total_loss += nnlm.process([2, 4], 4) # is green
    total_loss += nnlm.process([0, 4], 5) # is UNK
    total_loss += nnlm.process([6, 1], 5) # UNK tall
    total_loss += nnlm.process([6, 4], 5) # UNK green
    total_loss += nnlm.process([6, 4], 5) # UNK green

    total_loss += nnlm.process([1, 6], 1) # tall boy
    total_loss += nnlm.process([1, 7], 1) # tall girl
  
  
    if i % update_size == 0
      loss = total_loss / update_size.to_f / 13.0
      puts "Epoch #{i}/#{epochs}, Average Loss: #{loss.round(4)}, Perplexity: #{(Math::E**loss).round(4)}, Correct: #{nnlm.df(Math::E**(-loss)*100)}%"
      puts ""
      total_loss = 0
    end
  end
end 

nnlm = NNLMPresenter.new(
  embedding_dim: 2, # Each "word" has 3 dimensions
  context_size: 2, # Use 2 words
  hidden_size: 3, # Small hidden layer
  learning_rate: 0.05
)
nnlm.build_vocabulary(["false", "true"])

nnlm.present_forward([0, 0])
nnlm.present_backward([0, 0], 0)
#language_example()
