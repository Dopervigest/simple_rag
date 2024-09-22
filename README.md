# Very simplified version of a RAG system

This system is designed to help choosing an appropriate activity based on user input. It uses the following elements::
- Quantized version of Llama-3 as a causal model;
- All-MiniLM-L6-v2 as an encoding model;
- `data.json` as a substitute for database;

## Usage
Run the following command:
```python
python inference.py
```
