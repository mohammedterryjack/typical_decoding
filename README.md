# typical_decoding
a numpy implementation of Typical Decoding

### Example Use:

logits = [0.2,0.3,0.5,0.3,0.2,0.01]
tokens = ["hi","this","is","typical","decoding","."]

TypicalDecoding.filter_logits(logits,mass_threshold=.4)
>> [0.2, 0.3, -inf, 0.3, 0.2, -inf]

token_id = TypicalDecoding.sample_index(logits,mass_threshold=.4)
tokens[token_id]
>> `this`

token_id = TypicalDecoding.best_index(logits,mass_threshold=.4)
tokens[token_id] 
>> `this`
