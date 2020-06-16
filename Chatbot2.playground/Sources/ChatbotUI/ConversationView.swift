//
//  ConversationView.swift
//  ChatbotUI
//
//  Created by Palle Klewitz on 05.05.20.
//  Copyright © 2020 Palle Klewitz. All rights reserved.
//

import SwiftUI

#if os(macOS)
extension NSTextField {
    open override var focusRingType: NSFocusRingType {
        get { .none }
        set { }
    }
}
#endif

public struct ConversationView: View {
    @ObservedObject private var chatbot: Chatbot
    @Environment(\.colorScheme) private var colorScheme: ColorScheme
    @State private var currentMessage: String = ""
    
    public init(chatbot: Chatbot) {
        self.chatbot = chatbot
    }
    
    public var body: some View {
        VStack {
            ScrollView {
                ForEach(chatbot.conversation.indices.reversed(), id: \.self) { idx in
                    MessageView(message: self.chatbot.conversation[idx])
                    .scaleEffect(x: 1, y: -1, anchor: .center)
                }
            }
            .scaleEffect(x: 1, y: -1, anchor: .center)
            
            HStack {
                TextField("Write a Messsage", text: $currentMessage, onCommit: send)
                .textFieldStyle(PlainTextFieldStyle())
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                    .fill(colorScheme == .dark ? Color.black : Color.white)
                )
                .background(
                    RoundedRectangle(cornerRadius: 16)
                    .stroke(Color(white: colorScheme == .dark ? 0.3 : 0.85), lineWidth: 2)
                )
                
                Button(action: send) {
                    Text("Send")
                    .frame(width: 40)
                    .foregroundColor(Color.white)
                    .padding(8)
                    .background(Capsule().fill(Color.blue))
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding(8)
            .background(Color(white: colorScheme == .dark ? 0.2 : 0.95))
            .border(Color(white: colorScheme == .dark ? 0.3 : 0.85), width: 1)
        }
        .background(colorScheme == .dark ? Color.black : Color.white)
    }
    
    private func send() {
        let trimmed = currentMessage.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            chatbot.send(trimmed)
            currentMessage = ""
        }
    }
}
