use proc_macro::TokenStream;
use proc_macro2::TokenTree;
use quote::quote;

/// We need to copy the implementation of cudaGetDeviceCount here because we can't use `cuda-lib` as a crate dependency
/// if we want to use this macro in `cuda-lib` itself.
fn cuda_get_device_count() -> Result<i32, u32> {
    let mut count: i32 = 0;
    let ret = unsafe { cudart_sys::cudaGetDeviceCount(&raw mut count) };
    match ret {
        0 => Ok(count),
        _ => Err(ret),
    }
}

#[proc_macro_attribute]
/// Skip the test if no GPUs are available.
pub fn gpu_test(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(input);

    // instead of panicking, we should skip the test, an error is likely because
    // CUDA stuff is unavailable anyway
    let available_gpus: Result<(), String> = match cuda_get_device_count() {
        Ok(count) if count > 0 => Ok(()),
        Ok(_) => Err("no GPUs available".to_string()),
        Err(e) => Err(format!("error getting device count, cuda error code: {}", e)),
    };

    let output = if let Err(e) = available_gpus {
        quote! {
            #[ignore = #e]
            #input
        }
    } else {
        let input = append_gpu_suffix(input);
        quote! {
            #input
        }
    };

    TokenStream::from(output)
}

/// Append _GPU to the function name. we can use this to filter out GPU tests.
fn append_gpu_suffix(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let mut next_is_fn = false;
    let mut output = proc_macro2::TokenStream::new();
    for token in input.into_iter() {
        let append_token = match &token {
            TokenTree::Ident(ident) if next_is_fn => {
                next_is_fn = false;
                let span = ident.span();
                let new_fn_name = ident.to_string() + "_GPU";
                TokenTree::Ident(proc_macro2::Ident::new(&new_fn_name, span))
            }
            TokenTree::Ident(ident) if ident == "fn" => {
                next_is_fn = true;
                token
            }
            _ => token,
        };
        output.extend([append_token]);
    }
    output
}
