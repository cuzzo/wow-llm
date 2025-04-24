use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Result, TokenizerBuilder};
use tokenizers::Tokenizer;
use tokenizers::TokenizerImpl;
use std::path::Path;
use std::fs::{self};

fn train(dir : String) -> Result<TokenizerImpl<tokenizers::models::bpe::BPE, tokenizers::normalizers::Sequence, tokenizers::decoders::byte_level::ByteLevel, tokenizers::decoders::byte_level::ByteLevel, tokenizers::decoders::byte_level::ByteLevel>> {

    let data_dir = Path::new(&dir);

    let vocab_size: usize = 4096; // Desired vocabulary size
    let min_frequency: u64 = 2;   // Minimum frequency for pairs to be merged

    // Define special tokens (UNK = Unknown, CLS = Classification, etc.)
    let special_tokens = vec![
        AddedToken::from("[UNK]", true),
        AddedToken::from("[CLS]", true),
        AddedToken::from("[SEP]", true),
        AddedToken::from("[PAD]", true),
        AddedToken::from("[MASK]", true),
    ];

    // --- Tokenizer Training ---
    // 1. Build Trainer
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(min_frequency)
        .special_tokens(special_tokens)
        .build();

    // 2. Build Tokenizer
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    // 3. Get list of training files
    let training_files: Vec<String> = fs::read_dir(data_dir)? // Use ? for error handling
    .filter_map(|entry_result| {
        // Map over Result<DirEntry, io::Error>
        entry_result.ok().and_then(|entry| {
            let path = entry.path();
            // Ensure it's a file and convert path to String
            if path.is_file() {
                path.to_str().map(String::from) // Converts PathBuf -> Option<&str> -> Option<String>
            } else {
                None // Ignore directories or entries that aren't files
            }
        })
    })
    .collect();

    println!("Starting BPE training...");
    // 4. Train the tokenizer model using the specified trainer and files
    let pretty = false;
    tokenizer
        .train_from_files(
            &mut trainer,
            training_files,
        )?
        .save("tokenizer.json", pretty)?;

    println!("Training complete.");

    Ok(tokenizer)
}

fn load(token_file : String) -> Result<Tokenizer> {
    Ok(Tokenizer::from_file(token_file).unwrap())
}

fn encode(tokenizer : &Tokenizer, s : String) -> Vec<u32> {
    return tokenizer.encode(s, false).unwrap().get_ids().to_vec();
}

fn decode(tokenizer : &Tokenizer, tokens : Vec<u32>) -> String {
    return tokenizer.decode(&tokens, false).unwrap();
}

fn main() -> Result<()> {
    // --- Configuration ---
    //let tokenizer = train("/root/dev/wow-llm/train".to_string()).unwrap();
    let tokenizer = load("tokenizer.json".to_string()).unwrap();

    // --- Optional: Test Encoding ---
    let test_sentence = "This is a test sentence using the trained tokenizer.";
    println!("\nTesting encoding for: '{}'", test_sentence);

    let encoding = encode(&tokenizer, test_sentence.to_string());
    println!("{:?}", encoding);
    println!("{:?}", decode(&tokenizer, encoding));
    
    println!("\nProgram finished successfully.");
    
    Ok(())
}

