use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder}; 
use tokenizers::Tokenizer;
use tokenizers::TokenizerImpl;
use std::path::Path;
use std::fs::{self};
use std::os::raw::{c_char};
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;
use std::mem;
use std::error::Error;
use libc::{size_t}; // Use libc types for FFI

type BoxedError = Box<dyn Error + Send + Sync + 'static>;
type Result<T> = std::result::Result<T, BoxedError>;

// --- Internal Helper Function (Original Logic) ---
// Note: The return type is simplified here as TokenizerImpl is complex across FFI.
// We return a boxed Tokenizer which is easier to manage via pointers.
fn train_internal(dir : &str, output_path : &str) -> Result<TokenizerImpl<tokenizers::models::bpe::BPE, tokenizers::normalizers::Sequence, tokenizers::decoders::byte_level::ByteLevel, tokenizers::decoders::byte_level::ByteLevel, tokenizers::decoders::byte_level::ByteLevel>> {
    let data_dir = Path::new(dir);
    let vocab_size: usize = 4096;
    let min_frequency: u64 = 2;

    let special_tokens = vec![
        AddedToken::from("[UNK]", true),
        AddedToken::from("[CLS]", true),
        AddedToken::from("[SEP]", true),
        AddedToken::from("[PAD]", true),
        AddedToken::from("[MASK]", true),
    ];

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(min_frequency)
        .special_tokens(special_tokens)
        .build();

    // Build a Tokenizer (using the concrete BPE model)
    let model = BPE::default();
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(model) // Pass the specific model instance
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        // Post-processor and decoder often match the pre-tokenizer for byte level
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?; // Build into a generic Tokenizer first

    let training_files: Vec<String> = fs::read_dir(data_dir)?
        .filter_map(|entry_result| {
            entry_result.ok().and_then(|entry| {
                let path = entry.path();
                if path.is_file() {
                    path.to_str().map(String::from)
                } else {
                    None
                }
            })
        })
        .collect();

    if training_files.is_empty() {
         // Handle the case where no files are found to avoid panic
         eprintln!("Error: No training files found in directory: {}", dir);
         // Consider returning a specific error type if TokenizerResult allows
         return Err(Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "No training files found")));
    }

    println!("Starting BPE training on files: {:?}", training_files);

    // Train the tokenizer model
    let pretty = false;
    tokenizer
        .train_from_files(&mut trainer, training_files)? // No need for .save here if we return the tokenizer object
        .save(output_path, pretty)?;

    println!("Training complete.");
    Ok(tokenizer) // Return the trained tokenizer directly
}

fn load_internal(token_file : &str) -> Result<Tokenizer> {
    Ok(Tokenizer::from_file(token_file)?)
}

// --- FFI Exposed Functions ---

/// Creates and trains a tokenizer from files in a directory.
/// Returns a pointer to the tokenizer object on success, or null on error.
/// The caller is responsible for freeing the tokenizer using `rust_tokenizer_free`.
#[unsafe(no_mangle)]
pub extern "C" fn rust_tokenizer_train(dir_path: *const c_char, output_file : *const c_char) -> *mut Tokenizer {
    if dir_path.is_null() {
        eprintln!("rust_tokenizer_train: received null directory path");
        return ptr::null_mut();
    }
    let dir_cstr = unsafe { CStr::from_ptr(dir_path) };

    let output_file_cstr: &CStr = unsafe {
        CStr::from_ptr(output_file)
    };

    let output_file_str: &str = match output_file_cstr.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FFI Error: Failed to convert output_file C string to UTF-8: {}", e);
            // Handle the UTF-8 conversion error
            return ptr::null_mut();
        }
    };


    match dir_cstr.to_str() {
        Ok(dir_str) => {
            match train_internal(dir_str, output_file_str) {
                Ok(tokenizer) => Box::into_raw(Box::new(tokenizer.into())),
                Err(e) => {
                    eprintln!("rust_tokenizer_train: failed: {}", e);
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
            eprintln!("rust_tokenizer_train: invalid UTF-8 directory path: {}", e);
            ptr::null_mut()
        }
    }
}

/// Loads a tokenizer from a file.
/// Returns a pointer to the tokenizer object on success, or null on error.
/// The caller is responsible for freeing the tokenizer using `rust_tokenizer_free`.
#[unsafe(no_mangle)]
pub extern "C" fn rust_tokenizer_load(file_path: *const c_char) -> *mut Tokenizer {
     if file_path.is_null() {
        eprintln!("rust_tokenizer_load: received null file path");
        return ptr::null_mut();
    }
    let file_cstr = unsafe { CStr::from_ptr(file_path) };
     match file_cstr.to_str() {
        Ok(file_str) => {
            match load_internal(file_str) {
                Ok(tokenizer) => Box::into_raw(Box::new(tokenizer)),
                Err(e) => {
                    eprintln!("rust_tokenizer_load: failed: {}", e);
                    ptr::null_mut()
                }
            }
        }
        Err(e) => {
             eprintln!("rust_tokenizer_load: invalid UTF-8 file path: {}", e);
             ptr::null_mut()
        }
    }
}

/// Frees the memory allocated for a tokenizer object.
#[unsafe(no_mangle)]
pub extern "C" fn rust_tokenizer_free(tokenizer_ptr: *mut Tokenizer) {
    if !tokenizer_ptr.is_null() {
        unsafe {
            drop(Box::from_raw(tokenizer_ptr));
        }
        println!("Rust tokenizer freed."); // For debugging
    }
}

/// Represents a C-compatible array of u32.
#[repr(C)]
pub struct CU32Array {
    pub data: *mut u32,
    pub len: size_t,
}

/// Encodes a string using the tokenizer.
/// Returns a CU32Array struct containing the token IDs.
/// The caller is responsible for freeing the array's data using `rust_free_u32_array`.
/// Returns a CU32Array with null data and zero len on error or null input.
#[unsafe(no_mangle)]
pub extern "C" fn rust_tokenizer_encode(tokenizer_ptr: *const Tokenizer, text: *const c_char) -> CU32Array {
    let default_error_array = CU32Array { data: ptr::null_mut(), len: 0 };

    if tokenizer_ptr.is_null() || text.is_null() {
         eprintln!("rust_tokenizer_encode: received null tokenizer or text pointer");
         return default_error_array;
    }

    let tokenizer = unsafe { &*tokenizer_ptr };
    let text_cstr = unsafe { CStr::from_ptr(text) };

    match text_cstr.to_str() {
        Ok(text_str) => {
            match tokenizer.encode(text_str, false) { // Add '.map_err' if encode can fail beyond panics
                Ok(encoding) => {
                    let mut ids = encoding.get_ids().to_vec();
                    ids.shrink_to_fit(); // Ensure capacity matches length
                    let len = ids.len();
                    let data = ids.as_mut_ptr();
                    mem::forget(ids); // Prevent Rust from freeing the memory now

                    // Ensure we return uint32_t compatible data
                    // Rust u32 and C uint32_t are usually the same, but explicit cast is safer
                    CU32Array { data: data as *mut u32, len: len as size_t }
                }
                Err(e) => {
                    eprintln!("rust_tokenizer_encode: encoding failed: {}", e);
                    default_error_array
                }
            }
        }
        Err(e) => {
            eprintln!("rust_tokenizer_encode: invalid UTF-8 text: {}", e);
            default_error_array
        }
    }
}

/// Frees the memory allocated for the data pointer within a CU32Array.
#[unsafe(no_mangle)]
pub extern "C" fn rust_free_u32_array(arr: CU32Array) {
    if !arr.data.is_null() {
        unsafe {
            // Reconstruct the Vec and let it drop to free the memory
            let _ = Vec::from_raw_parts(arr.data, arr.len, arr.len);
             println!("Rust u32 array freed."); // For debugging
        }
    }
}


/// Decodes a sequence of token IDs back into a string.
/// Returns a C string (char*).
/// The caller is responsible for freeing the returned string using `rust_free_string`.
/// Returns a null pointer on error or null input.
#[unsafe(no_mangle)]
pub extern "C" fn rust_tokenizer_decode(tokenizer_ptr: *const Tokenizer, ids_ptr: *const u32, len: size_t) -> *mut c_char {
    if tokenizer_ptr.is_null() || ids_ptr.is_null() {
         eprintln!("rust_tokenizer_decode: received null tokenizer or ids pointer");
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &*tokenizer_ptr };
    let ids_slice = unsafe { slice::from_raw_parts(ids_ptr as *const u32, len as usize) }; // Cast back to u32 slice

     match tokenizer.decode(ids_slice, false) {
        Ok(decoded_string) => {
            match CString::new(decoded_string) {
                Ok(c_string) => c_string.into_raw(),
                Err(e) => {
                     eprintln!("rust_tokenizer_decode: failed to create CString (contains null byte?): {}", e);
                     ptr::null_mut()
                }
            }
        }
        Err(e) => {
             eprintln!("rust_tokenizer_decode: decoding failed: {}", e);
             ptr::null_mut()
        }
    }
}

/// Frees a C string that was allocated by Rust.
#[unsafe(no_mangle)]
pub extern "C" fn rust_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
             println!("Rust string freed."); // For debugging
        }
    }
}

