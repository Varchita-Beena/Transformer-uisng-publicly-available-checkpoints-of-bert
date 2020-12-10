# bert2bert
A transformer model using publicly available bert (for both encoder and decoder part) checkpoints<br>
Rather than initializing the embedding layers, attention layers (query, key, value matrices) and other matrices in transformer randomly I have used the matrices of pre-trained bert which is available publicly (bert uncased_L-12_H-768_A-12)
## References
Leveraging Pre-trained Checkpoints for Sequence Generation Tasks. https://arxiv.org/pdf/1907.12461.pdf <br>
https://www.tensorflow.org/tutorials/text/transformer
