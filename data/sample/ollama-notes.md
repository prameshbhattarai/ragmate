# Ollama notes

## What Ollama is

Ollama runs open language models locally and serves them over an HTTP API on port 11434. Models are
downloaded once with `ollama pull <name>` and then run entirely on the machine, so no data leaves it.

## Models used by ragmate

The chat model is gemma3:4b, a small model from Google's Gemma family that fits comfortably in the
memory of a laptop. The embedding model is nomic-embed-text, which produces 768-dimensional vectors
and supports long inputs. Embedding models are much smaller and faster than chat models.

## Useful commands

`ollama list` shows the downloaded models. `ollama ps` shows which models are loaded in memory right
now and how much memory they use. `ollama stop <name>` unloads a model. A model stays loaded for five
minutes after its last request by default, which can be changed with the OLLAMA_KEEP_ALIVE setting.

## Context window

The context window is how many tokens the model can read at once, including the prompt, the
retrieved chunks and the answer. ragmate sets it with OLLAMA_NUM_CTX. A larger context window lets
more chunks fit into the prompt but uses more memory and makes every request slower.
