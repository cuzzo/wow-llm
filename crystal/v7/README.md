# Important

Set `LD_LIBRARY_PATH` to run Crystal:

```bash
LD_LIBRARY_PATH="/absolute/path/to/wow-llm/rust/tokeniza/target/release:$LD_LIBRARY_PATH"
```

e.g:

```bash
LD_LIBRARY_PATH="/root/dev/wow-llm/rust/tokeniza/target/release:$LD_LIBRARY_PATH" crystal tokeniza_test.cr 
```
