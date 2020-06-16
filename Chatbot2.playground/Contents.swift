/*:
# Neural Chatbot
### THIS PLAYGROUND TAKES A WHILE TO COMPILE

This playground demonstrates a chatbot based on a recurrent neural network, called Seq2Seq.

The network consumes messages that have been entered by the user and predicts a reply,
based on example replies it has seen in the training phase.

It has been trained on the English chatterbot corpus, which is a small corpus of message replies.

The chatbot is written on top of DL4S, which is a library for automatic differentiation and tensor operations,
that I've written over the past year. DL4S uses Accelerate or Intel's MKL under the hood, has an API similar to Numpy or PyTorch and is fully written in Swift.

#### Implementation Details
 
- Most implementation details are in the Sources folder.
- The Seq2Seq implementation as well as the Chatbot view model are contained in the Chatbot.min.swift file.
- The user interface is implemented in the Swift files in ChatbotUI.
- The trained model and the vocabulary are stored in the Resources folder.
- The DL4S library has been stripped down.

#### Steps for Improvements

- Increase model size and train on a larger corpus (would exceed 25MB limit)
- Use transformer networks instead of RNNs

 */

import Foundation
import PlaygroundSupport
import SwiftUI

guard let vocabFile = Bundle.main.url(forResource: "chatterbot_vocab", withExtension: "txt") else {
    fatalError("Cannot load vocabulary")
}
let vocab = try Language(contentsOf: vocabFile)

guard let modelFile = Bundle.main.url(forResource: "model", withExtension: "json") else {
    fatalError("Cannot load model")
}
let modelData = try Data(contentsOf: modelFile)
let decoder = JSONDecoder()
decoder.dataDecodingStrategy = .base64
let model = try decoder.decode(Seq2Seq<CPU>.self, from: modelData)

let chatbot = Chatbot(model: model, language: vocab)
chatbot.receive("""
Hello!
I am a Chatbot based on a neural network called Seq2Seq.
You can say things to me or ask me questions.

I am not perfect and may not be able to answer everything correctly.

If you don't know what to say, try:
• How are you?
• Are you experiencing an energy shortage?
• Which is your favorite soccer club?
• Tell me a joke
• What are your hobbies?
""")

let rootView = ConversationView(chatbot: chatbot)
let hostingView = NSHostingView(rootView: rootView)
hostingView.frame = CGRect(x: 0, y: 0, width: 480, height: 720)

PlaygroundPage.current.setLiveView(hostingView)
