# Swift Student Challenge 2020 Submission

This repository contains my WWDC20 Swift Student Challenge submission.

The submission has been granted a Swift Student Challenge award.

## About

This playground uses a seq2seq model with attention to predict replies in a chat conversation.

A model that has been trained on the Chatterbot dataset has been added to the playground as a resource.

### Technologies Used

- SwiftUI for the chat user interface
- [DL4S](https://github.com/palle-k/DL4S.git) (Deep Learning for Swift) is used as the deep learning framework that has been used to train and deploy the chatbot network
- Combine is used for the view model
- The output sequence is predicted using beam search

