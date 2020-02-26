@echo off
cargo build --release --target=wasm32-unknown-unknown
copy target\wasm32-unknown-unknown\release\wasm_nn.wasm .\wasm\wasm_nn.wasm