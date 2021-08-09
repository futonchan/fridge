//
//  Veget.swift
//  FridgeChecker2
//
//  Created by 矢野大暉 on 2021/05/06.
//

import Foundation
struct Vegets: Codable {
    var vegets: [Veget]
}

struct Veget: Codable {
    var veget: String        // 名前
    var count: Int            // 個数
}
