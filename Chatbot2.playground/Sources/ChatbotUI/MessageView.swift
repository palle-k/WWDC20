//
//  MessageView.swift
//  ChatbotUI
//
//  Created by Palle Klewitz on 05.05.20.
//  Copyright © 2020 Palle Klewitz. All rights reserved.
//

import SwiftUI

extension View {
    func asAny() -> AnyView {
        return AnyView(self)
    }
}

public struct MessageView: View {
    public var message: Message
    
    public var body: AnyView {
        switch (message) {
        case .sent(let msg):
            return HStack {
                Spacer()
                
                Text(msg)
                .font(.system(size: 15))
                .foregroundColor(.white)
                .padding(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.blue)
                )
            }
            .padding()
            .asAny()
            
        case .received(let msg):
            return HStack {
                Text(msg)
                .foregroundColor(.black)
                .font(.system(size: 15))
                .padding(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color(white: 0.9))
                )
                
                Spacer()
            }
            .padding()
            .asAny()
            
        case .writing:
            return HStack {
                HStack(spacing: 6) {
                    ForEach(0 ..< 3) { i in
                        Circle()
                        .fill(Color.gray)
                        .frame(width: 8, height: 8)
                    }
                    
                }
                .frame(height: 19)
                .padding(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color(white: 0.9))
                )
                
                Spacer()
            }
            .padding()
            .asAny()
        }
    }
}
