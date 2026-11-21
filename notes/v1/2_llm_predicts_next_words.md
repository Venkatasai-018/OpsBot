AI Predicts next words it doesnt have memory

AI Predicts the next words based on probability , i doesnt have any memory

It Provides Paragraphs(CHATGPT) actually it predicts token by token based on previous token in context.

Model doesnt have any memory ,it has patterns which is stored is called weights(Parameters)

Inference-> chatting

An LLM is fundamentally autoregressive token predictior. During Inference LLM generates the text by repeatdely predicting the next token based on tokens which are available in context window.

LLM doesnt have any persistent memory , what appears to be memory is which is actually present in the context which may include system,user prompts , reterival context , conversation history.

During Inference Model may losses the context if the information is not present in context (if Context window Exceeds previous data/token might be deleted)
