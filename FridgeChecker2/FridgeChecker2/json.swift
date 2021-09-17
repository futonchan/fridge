//
//  json.swift
//  FridgeChecker2
//
//  Created by 矢野大暉 on 2021/05/06.
//
//

// 野菜の一覧を取得(GET)したときのJSONオブジェクト構造

import Foundation

struct Veget: Codable {
    let name: String            // 野菜名
    let comment: String?
    let count: Int?
    let weight: Float?
    let input_date: String
    let expiry_date: String?
}

struct ResponseMetadata: Codable {
    let RequestId: String
    let HTTPStatusCode: Int
    let HTTPHeaders: HTTPHeaders
    let RetryAttempts: Int
}

struct HTTPHeaders: Codable {
    let server: String
    let date: String
    let contentType: String
    let contentLength: String
    let connection: String
    let xAmznRequestid: String
    let xAmzCrc32: String
    
    enum CodingKeys: String, CodingKey {
        case server
        case date
        case contentType = "content-type"
        case contentLength = "content-length"
        case connection
        case xAmznRequestid = "x-amzn-requestid"
        case xAmzCrc32 = "x-amz-crc32"
    }
}

struct GetJsonObject: Codable {
    let Items: [Veget]
    let Count: Int
    let ScannedCount: Int
    let ResponseMetadata: ResponseMetadata
}
