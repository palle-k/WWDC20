//
//  File.swift
//  
//
//  Created by Palle Klewitz on 10.05.20.
//

import Foundation


public protocol StateType: Equatable {
    associatedtype Element
    func appending(_ element: Element) -> Self
}


public struct Hypothesis<State: StateType> {
    public var state: State
    public var logProbability: Double
    public var isCompleted: Bool
    public var beamLength: Int
    
    public init(initialState: State) {
        self.state = initialState
        self.logProbability = 0 // log(1)
        self.isCompleted = false
        self.beamLength = 0
    }
    
    public init(state: State, logProbability: Double, isCompleted: Bool, beamLength: Int) {
        self.state = state
        self.logProbability = logProbability
        self.isCompleted = isCompleted
        self.beamLength = beamLength
    }
    
    public func extended(with element: State.Element, logProbability: Double, isCompleted: Bool) -> Hypothesis<State> {
        guard !self.isCompleted else {
            return self
        }
        return Hypothesis(state: self.state.appending(element), logProbability: self.logProbability + logProbability, isCompleted: isCompleted, beamLength: self.beamLength + 1)
    }
}

public struct BeamSearchContext<State: StateType> {
    private(set) var hypotheses: [Hypothesis<State>]
    private var newHypotheses: [Hypothesis<State>] = []
    
    public let maxBeamLength: Int
    public let beamCount: Int
    
    /// Hypotheses, sorted by likelihood descending
    public var bestHypotheses: [Hypothesis<State>] {
        return hypotheses
            .sorted(by: {$0.logProbability < $1.logProbability})
            .reversed()
    }
    
    public var isCompleted: Bool {
        return hypotheses.allSatisfy {
            $0.isCompleted || $0.beamLength >= maxBeamLength
        }
    }
    
    public init(beamCount: Int, maxLength: Int, initialState: State) {
        self.beamCount = beamCount
        self.maxBeamLength = maxLength
        self.hypotheses = [Hypothesis(initialState: initialState)]
    }
    
    public mutating func endIteration() {
        hypotheses = newHypotheses
            .sorted(by: {$0.logProbability < $1.logProbability})
            .reversed()
            .prefix(beamCount)
            .collect(Array.init)
        newHypotheses = []
    }
    
    public mutating func add(_ hypothesis: Hypothesis<State>) {
        newHypotheses.append(hypothesis)
    }
}
import Combine
import Foundation

public enum Message {
    case sent(String)
    case received(String)
    case writing
}

public class Chatbot: ObservableObject {
    private let model: Seq2Seq<CPU>
    private let language: Language
    
    @Published public var conversation: [Message] = []
    private var fixedMessages: [Message] = []
    private var messageQueue: [String] = []
    private var isRunning = false
    private let queue = DispatchQueue(label: "io.palle.Chatbot")
    
    public init(model: Seq2Seq<CPU>, language: Language) {
        self.model = model
        self.language = language
    }
    
    public func receive(_ message: String) {
        fixedMessages.append(.received(message))
        conversation = fixedMessages + (isRunning ? [.writing] : [])
    }
    
    public func send(_ message: String) {
        fixedMessages.append(.sent(message))
        conversation = fixedMessages + (isRunning ? [.writing] : [])
        messageQueue.append(message)
        run()
    }
    
    public func run() {
        guard !isRunning else {
            return
        }
        guard let message = messageQueue.first else {
            isRunning = false
            return
        }
        messageQueue.removeFirst()
        isRunning = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + .seconds(Int.random(in: 1 ... 2))) {
            self.conversation = self.fixedMessages + [.writing]
            
            self.queue.async {
                let tok = self.language.indexSequence(from: message)
                let hypotheses = self.model.callAsFunction(inputs: tok, beamCount: 4, maxLength: 30)
                let sent = self.language.formattedSentence(from: hypotheses[0])
                
                DispatchQueue.main.asyncAfter(deadline: .now() + .seconds(Int.random(in: 1 ... 3))) {
                    self.fixedMessages += [.received(sent)]
                    self.conversation = self.fixedMessages
                    self.isRunning = false
                    self.run()
                }
            }
        }
    }
}
//
//  Language.swift
//  Seq2Seq-DL4S
//
//  Created by Palle Klewitz on 15.03.19.
//  Copyright (c) 2019 Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

import Foundation

public struct Language {
    public static let padding: Int = -1
    public static let startOfSentence: Int = 0
    public static let endOfSentence: Int = 1
    public static let unknown: Int = 2
    
    public var wordToIndex: [String: Int]
    public var words: [String]
    
    public init<S: Sequence>(fromExamples examples: S) where S.Element == String {
        let examples = examples.map {$0.lowercased()}
        
        let words = examples
            .lazy
            .flatMap {
                $0.components(separatedBy: .whitespaces)
            }
            .filter {!$0.isEmpty}
        
        let frequencies: [String: Int] = words.reduce(into: [:], {$0[$1, default: 0] += 1})

        let uniqueWords = frequencies
            .keys
            .sorted(by: {frequencies[$0, default: 0] < frequencies[$1, default: 0]})
            .reversed()
        
        self.words = ["<s>", "</s>", "<unk>"] + uniqueWords
        self.wordToIndex = Dictionary(uniqueKeysWithValues: self.words.enumerated().lazy.map {($1, $0)})
    }
    
    public init(words: [String]) {
        self.words = words
        self.wordToIndex = Dictionary(uniqueKeysWithValues: self.words.enumerated().map {($1, $0)})
    }
    
    public init(contentsOf url: URL) throws {
        self.words = try String(data: Data(contentsOf: url), encoding: .utf8)!
            .split(whereSeparator: {$0.isNewline})
            .map(String.init)
        self.wordToIndex = Dictionary(uniqueKeysWithValues: self.words.enumerated().map {($1, $0)})
    }
    
    public func limited(toWordCount wordCount: Int) -> Language {
        return Language(words: Array(words[..<min(wordCount, words.count)]))
    }
    
    public func write(to url: URL) throws {
        try words.joined(separator: "\n")
            .data(using: .utf8)!
            .write(to: url)
    }
    
    public static func cleanup(_ string: String) -> String {
        string
            .lowercased()
            .replacingOccurrences(of: #"([.,?!();\-_$°+/:])"#, with: " $1 ", options: .regularExpression)
            .replacingOccurrences(of: #"[„"”“‟‘»«]"#, with: " \" ", options: .regularExpression)
            .replacingOccurrences(of: #"[–—－]"#, with: " - ", options: .regularExpression)
            .replacingOccurrences(of: #"[0-9]+"#, with: "<num>", options: .regularExpression)
    }
    
    public func formattedSentence(from sequence: [Int32]) -> String {
        let words = sequence.filter {$0 >= 2}.map(Int.init).compactMap {self.words[safe: $0]}
        
        var result: String = ""
        
        for w in words {
            if let f = w.first, Set(".,!?").contains(f) {
                result.append(w)
            } else if result.isEmpty {
                result.append(w)
            } else {
                result.append(" \(w)")
            }
        }
        
        return result
    }
    
    public func indexSequence(from sentence: String) -> [Int32] {
        let sentence = Language.cleanup(sentence)
        let words = sentence.components(separatedBy: .whitespaces).filter {!$0.isEmpty}
        let indices = words.map {wordToIndex[$0] ?? Language.unknown}.map(Int32.init)
        return indices
    }
    
    public func wordSequence(from indexSequence: [Int32]) -> [String] {
        return indexSequence.map(Int.init).compactMap {words[safe: $0]}
    }
}
//
//  File.swift
//  
//
//  Created by Palle Klewitz on 05.05.20.
//

import Foundation


public struct TanhAttention<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameters: [Tensor<Element, Device>] {
        return [W_h, W_s, b, v]
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \Self.W_h, \Self.W_s, \Self.b, \Self.v
    ]}
    
    var W_h: Tensor<Element, Device>
    var W_s: Tensor<Element, Device>
    var b: Tensor<Element, Device>
    var v: Tensor<Element, Device>
    
    public let encoderHiddenSize: Int
    public let decoderHiddenSize: Int
    public let latentSize: Int
    
    public init(encoderHiddenSize: Int, decoderHiddenSize: Int, latentSize: Int) {
        self.encoderHiddenSize = encoderHiddenSize
        self.decoderHiddenSize = decoderHiddenSize
        self.latentSize = latentSize
        
        W_h = Tensor<Element, Device>(normalDistributedWithShape: decoderHiddenSize, latentSize, mean: 0, stdev: Element(1 / sqrt(Float(decoderHiddenSize))), requiresGradient: true)
        W_s = Tensor<Element, Device>(normalDistributedWithShape: encoderHiddenSize, latentSize, mean: 0, stdev: Element(1 / sqrt(Float(encoderHiddenSize))), requiresGradient: true)
        b = Tensor<Element, Device>(normalDistributedWithShape: latentSize, mean: 0, stdev: Element(1 / sqrt(Float(latentSize))), requiresGradient: true)
        v = Tensor<Element, Device>(normalDistributedWithShape: latentSize, mean: 0, stdev: Element(1 / sqrt(Float(latentSize))), requiresGradient: true)
    }
    
    public func callAsFunction(_ inputs: (encoderStateSequence: Tensor<Element, Device>, decoderState: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        let encoderStateSequence = inputs.0 // [seqlen, batchSize, encHS]
        let decoderState = inputs.1 // [batchSize, decHS]
        
        // let seqlen = encoderStateSequence.shape[0]
        let batchSize = decoderState.shape[0]
        
        let encIn = encoderStateSequence.view(as: -1, encoderHiddenSize)
        
        let encScore = encIn.matrixMultiplied(with: W_s).view(as: [-1, batchSize, latentSize]) // [seqlen * batchSize, latentSize]
        let decScore = decoderState.matrixMultiplied(with: W_h) // [batchSize, latentSize]
        
        let scores = tanh(encScore + decScore + b) // [seqlen, batchSize, latentSize]
            .view(as: -1, latentSize) // [seqlen * batchSize, latentSize]
            .matrixMultiplied(with: v.view(as: -1, 1)) // [seqlen * batchSize, 1]
            .view(as: -1, batchSize) // [seqlen, batchSize]
        
        return softmax(scores, axis: 0)
    }
}


public struct AttentionCombine<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}
    
    public func callAsFunction(_ inputs: (query: Tensor<Element, Device>, values: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        let scores = inputs.0 // [seqlen, batchsize]
        let weights = inputs.1 // [seqlen, batchSize, hiddenSize]
        
        let scoredWeights = (scores.unsqueezed(at: 2) * weights).reduceSum(along: 0) // [batchSize, hiddenSize]
        
        return scoredWeights
    }
}


public struct Seq2Seq<Device: DeviceType>: LayerType, Codable {
    public typealias Inputs = Tensor<Int32, Device>
    public typealias Outputs = (logits: Tensor<Float, Device>, tokenSequence: [Int])
    public typealias Parameter = Float
    public typealias Device = Device
    
    private var embedding: Embedding<Float, Device>
    private var encoderRNN: GRU<Float, Device>
    private var decoderRNN: GRU<Float, Device>
    private var attention: TanhAttention<Float, Device>
    private var attentionCombine: AttentionCombine<Float, Device>
    private var decoderPost: Dense<Float, Device>
    private var output: LogSoftmax<Float, Device>
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Float, Device>>] {
        Array([
            embedding.parameterPaths.map((\Self.embedding).appending(path:)),
            encoderRNN.parameterPaths.map((\Self.encoderRNN).appending(path:)),
            decoderRNN.parameterPaths.map((\Self.decoderRNN).appending(path:)),
            attention.parameterPaths.map((\Self.attention).appending(path:)),
            attentionCombine.parameterPaths.map((\Self.attentionCombine).appending(path:)),
            decoderPost.parameterPaths.map((\Self.decoderPost).appending(path:))
        ].joined())
    }
    
    public var parameters: [Tensor<Float, Device>] {
        Array([
            embedding.parameters,
            encoderRNN.parameters,
            decoderRNN.parameters,
            attention.parameters,
            attentionCombine.parameters,
            decoderPost.parameters
        ].joined())
    }
    
    public init(vocabSize: Int, embedDim: Int, hiddenDim: Int) {
        embedding = Embedding(inputFeatures: vocabSize, outputSize: embedDim)
        encoderRNN = GRU(inputSize: embedDim, hiddenSize: hiddenDim)
        decoderRNN = GRU(inputSize: embedDim + hiddenDim, hiddenSize: hiddenDim)
        attention = TanhAttention(encoderHiddenSize: hiddenDim, decoderHiddenSize: hiddenDim, latentSize: hiddenDim / 2)
        attentionCombine = AttentionCombine()
        decoderPost = Dense(inputSize: hiddenDim, outputSize: embedDim)
        output = LogSoftmax()
    }
    
    private func decodeStep(encoderSequence: Tensor<Float, Device>, state: Tensor<Float, Device>, input: Tensor<Int32, Device>) -> (Tensor<Float, Device>, Tensor<Float, Device>, Int) {
        let embedded = embedding(input)
        
        let query = attention((encoderStateSequence: encoderSequence, decoderState: state))
        let attended = attentionCombine((query: query, values: encoderSequence))
        
        let rnnInput = stack([embedded, attended], along: 1)
        let (rnnOut, _) = decoderRNN(rnnInput.view(as: [1, 1, -1]), state: state)
        
        let latentOutput = leakyRelu(decoderPost(rnnOut.view(as: [1, -1])), leakage: 0.2) //[batchSize, embedDim]
        
        let logits = latentOutput.matrixMultiplied(with: embedding.embeddingMatrix, transposeOther: true) // [batchSize, embedDim] x [embedDim, vocabSize]  ==> [batchSize, vocabSize]
        let out = output(logits)
        
        return (rnnOut, out, out.flattened().argmax())
    }
    
    private func decodeSequence(encoderSequence: Tensor<Float, Device>, state: Tensor<Float, Device>) -> (Tensor<Float, Device>, [Int]) {
        var logitSequence: [Tensor<Float, Device>] = []
        var tokenSequence: [Int] = []
        var state = state
        var currentToken = Language.startOfSentence
        
        while currentToken != Language.endOfSentence && logitSequence.count < 32 {
            let logits: Tensor<Float, Device>
            (state, logits, currentToken) = decodeStep(encoderSequence: encoderSequence, state: state, input: Tensor([Int32(currentToken)]))
            tokenSequence.append(currentToken)
            logitSequence.append(logits.squeezed(at: 0))
        }
        
        return (
            Tensor(stacking: logitSequence.map {$0.unsqueezed(at: 0)}),
            tokenSequence
        )
    }
    
    private func decodeForcedSequence(encoderSequence: Tensor<Float, Device>, state: Tensor<Float, Device>, forcedSequence: [Int32], scheduledSamplingRate: Float = 0) -> (Tensor<Float, Device>, [Int]) {
        let inputSequence = Tensor<Int32, Device>([Int32(Language.startOfSentence)] + forcedSequence)
        var logitSequence: [Tensor<Float, Device>] = []
        var tokenSequence: [Int] = []
        var currentToken = Language.startOfSentence
        var state = state
        
        for i in forcedSequence.indices {
            let logits: Tensor<Float, Device>
            let input: Tensor<Int32, Device>
            if Float.random(in: 0 ... 1) <= scheduledSamplingRate {
                input = Tensor([Int32(currentToken)])
            } else {
                input = inputSequence[i].unsqueezed(at: 0)
            }
            (state, logits, currentToken) = decodeStep(encoderSequence: encoderSequence, state: state, input: input)
            
            tokenSequence.append(currentToken)
            logitSequence.append(logits.squeezed(at: 0))
        }
        
        return (
            Tensor(stacking: logitSequence.map {$0.unsqueezed(at: 0)}),
            tokenSequence
        )
    }
    
    public func callAsFunction(_ inputs: Tensor<Int32, Device>) -> (logits: Tensor<Float, Device>, tokenSequence: [Int]) {
        let flat = inputs.flattened()
        let emb = embedding(flat)
        let rnnIn = emb.view(as: inputs.shape + [1, -1])
        let (finalEncoderState, makeEncoderSequence) = encoderRNN(rnnIn)
        let (logitSeq, tokens) = decodeSequence(encoderSequence: makeEncoderSequence(), state: finalEncoderState)
        return (logitSeq, tokens)
    }
    
    public func callAsFunction(_ inputs: Tensor<Int32, Device>, forcedSequence: [Int32], scheduledSamplingRate: Float = 0) -> (logits: Tensor<Float, Device>, tokenSequence: [Int]) {
        let flat = inputs.flattened()
        let emb = embedding(flat)
        let rnnIn = emb.view(as: inputs.shape + [1, -1])
        let (finalEncoderState, makeEncoderSequence) = encoderRNN(rnnIn)
        let (logitSeq, tokens) = decodeForcedSequence(encoderSequence: makeEncoderSequence(), state: finalEncoderState, forcedSequence: forcedSequence)
        return (logitSeq, tokens)
    }
}


struct Seq2SeqAttentionBeamState<Element: NumericType, Device: DeviceType>: StateType {
    typealias Element = (word: Int32, hiddenState: Tensor<Element, Device>)
    
    var indices: [Int32]
    var hiddenState: Tensor<Element, Device>
    
    func appending(_ element: (word: Int32, hiddenState: Tensor<Element, Device>)) -> Seq2SeqAttentionBeamState<Element, Device> {
        return Seq2SeqAttentionBeamState(indices: indices + [element.word], hiddenState: element.hiddenState)
    }
}



public extension Seq2Seq {
    func callAsFunction(inputs: [Int32], beamCount: Int, maxLength: Int) -> [[Int32]] {
        let inputs = Tensor<Int32, Device>(inputs).flattened()
        let emb = embedding(inputs)
        let rnnIn = emb.view(as: inputs.shape + [1, -1])
        let (finalEncoderState, makeEncoderSequence) = encoderRNN(rnnIn)
        let encoderSequence = makeEncoderSequence()
        
        var context = BeamSearchContext(
            beamCount: beamCount,
            maxLength: maxLength,
            initialState: Seq2SeqAttentionBeamState(indices: [Int32(Language.startOfSentence)], hiddenState: finalEncoderState)
        )
        
        while !context.isCompleted {
            for hypothesis in context.hypotheses {
                if hypothesis.isCompleted {
                    context.add(hypothesis)
                    continue
                }
                
                let (state, out, _) = decodeStep(encoderSequence: encoderSequence, state: hypothesis.state.hiddenState, input: Tensor([hypothesis.state.indices.last!]))
                
                let best = out
                    .elements
                    .enumerated()
                    .top(count: beamCount, by: {$0.element < $1.element})
                
                for (idx, prob) in best {
                    context.add(
                        hypothesis.extended(
                            with: (
                                word: Int32(idx),
                                hiddenState: state
                            ),
                            logProbability: Double(element: prob),
                            isCompleted: idx == Language.endOfSentence
                        )
                    )
                }
            }
            
            context.endIteration()
        }
        
        return context.bestHypotheses.map {$0.state.indices}
    }
}
import Foundation

public extension Sequence {
    func collect<Collector>(_ collect: (Self) throws -> Collector) rethrows -> Collector {
        return try collect(self)
    }
}

public extension Sequence {
    func top(count: Int, by comparator: (Element, Element) -> Bool) -> [Element] {
        // return sorted(by: comparator).reversed().prefix(count).collect(Array.init)
        return reduce(into: []) { acc, value in
            var i = acc.count - 1
            while i >= 0 {
                if comparator(acc[i], value) {
                    i -= 1
                } else {
                    break
                }
            }
            if i < count - 1 {
                acc.insert(value, at: i + 1)
                
                if acc.count > count {
                    acc.removeLast()
                }
            }
        }
    }
}


public extension Collection {
    subscript(safe index: Index) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}


public protocol LearningRateScheduler {
    func learningRate(atStep step: Int) -> Float
}

public struct NoamScheduler: LearningRateScheduler {
    public let warmupSteps: Int
    public let modelDim: Int
    
    public init(warmupSteps: Int, modelDim: Int) {
        self.warmupSteps = warmupSteps
        self.modelDim = modelDim
    }
    
    public func learningRate(atStep step: Int) -> Float {
        let step = Float(step)
        let warmupSteps = Float(self.warmupSteps)
        let modelDim = Float(self.modelDim)
        
        return 1 / sqrt(modelDim) * min(1 / sqrt(step), step * pow(warmupSteps, -1.5))
    }
}

func makeEncoderMasks<Device: DeviceType>(sequenceLengths: [Int]) -> Tensor<Float, Device> {
    let maxInLen = sequenceLengths.reduce(0, max)
    let batchSize = sequenceLengths.count
    
    return Tensor<Float, Device>(sequenceLengths.map {
        Array(repeating: 0, count: $0) + Array(repeating: 1, count: maxInLen - $0)
    }).view(as: batchSize, 1, 1, maxInLen) // TODO: Check if maxLen in 3rd or 4th position
}

func makeDecoderMasks<Device: DeviceType>(sequenceLengths: [Int]) -> Tensor<Float, Device> {
    let batchSize = sequenceLengths.count
    let maxLen = sequenceLengths.reduce(0, max)
    
    let decoderSeqMask = Tensor<Float, Device>(sequenceLengths.map {
        Array(repeating: 0, count: $0) + Array(repeating: 1, count: maxLen - $0)
    })
    
    let decoderCausalMask = Tensor<Float, Device>(repeating: 1, shape: maxLen, maxLen)
        .bandMatrix(belowDiagonal: -1, aboveDiagonal: nil) // [maxLen, maxLen]
    
    return 1 - relu(1 - decoderSeqMask.view(as: batchSize, 1, 1, maxLen) - decoderCausalMask)
}
