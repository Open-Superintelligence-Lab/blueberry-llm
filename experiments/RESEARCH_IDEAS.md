I've thought about 2 research ideas for DeepSeek Sparse Attention, is there any intuition here on how they would perform? Or is there no way to know until I try?

## 1. Learnable k

Instead of always looking at the previous 2048 tokens, train model to predict optimal k (number of previous tokens to look at) based on the current token.

Advantages: Using a small k for simple or repetitive text (skimming, boilerplate), a large k for complex reasoning or summarization, and a medium k for general tasks.

Disadvantage: More parameters and training, selecting discrete k is non differentiable, possibly need reinforcement learning (1. make LLM perform better and 2. make k as small as possible)

## 2. Top-p

Select most probable previous tokens until their probability adds up to p (eg. 80%)

- Good for tasks where it's not clear from the current token how large k should be.
- The main advantage over learnable k is that it doesn't need to make the final decision based on just the current token and instead it looks how "uncertain" the model is, that is if a lot of previous tokens show similar probability of being relevant, it will pick many of them, or a few tokens are peaking, it will just take context from those few.

