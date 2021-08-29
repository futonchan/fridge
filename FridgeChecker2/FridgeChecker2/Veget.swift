//
//  Veget.swift
//  FridgeChecker2
//
//  Created by 矢野大暉 on 2021/05/06.
//

import Foundation
struct Veget: Codable {
    let id: Int        // id
    let name: String            // 野菜名
    let comment: String
    let count: Int
    let weight: Float
    let created_at: String
    let updated_at: String
}
